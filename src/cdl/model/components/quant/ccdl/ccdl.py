import abc
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ccdl import cascaded_cdl_topk_forward, cascaded_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


def group_tensor_into_blocks(
    theta: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    k = theta.shape[-1]

    extra = k % block_size
    pad_size = 0
    if extra:
        pad_size = block_size - extra
        theta = F.pad(theta, (0, pad_size))

    num_blocks = (k + pad_size) // block_size
    cdl_style_theta = theta.reshape(*theta.shape[:-1], num_blocks, block_size)
    return cdl_style_theta, pad_size


@dataclass
class CascadedCdlTensor(QTensor):
    theta_format: Format
    theta_indices: torch.Tensor
    q_format: Format
    q_indices: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int

    def dequantize(self) -> torch.Tensor:
        theta = self.theta_format.level[self.theta_indices.long()]
        q = self.q_format.level[self.q_indices.long()]
        alpha = self.alpha.to(theta.dtype)
        x = (theta * q * alpha).flatten(-2)[..., : self.k_dim]
        return x

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(-2)[..., : self.k_dim]


@dataclass
class CascadedCdlBypassTensor(QTensor):
    theta: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None


class CascadedCdlQuant(Quant):
    tensor: type[CascadedCdlTensor] = CascadedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.k_dim = input_shape[-1]
        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.q_format = q_format
        self.q_topk = q_topk
        self.block_size = block_size

        num_blocks = (self.k_dim + block_size - 1) // block_size

        self.q_num_quantized_params = math.prod(input_shape[:-1]) * num_blocks
        self.theta_num_quantized_params = math.prod(input_shape[:-1]) * self.k_dim

        q_shape = input_shape[:-1] + (num_blocks, 1)

        self.log_theta_kappa = nn.Parameter(torch.full((), math.log(2)))
        self.log_q_kappa = nn.Parameter(torch.full((), math.log(2)))
        self.log_q = nn.Parameter(torch.full(q_shape, math.log(1)))
        self.log_alpha = nn.Parameter(torch.full((), math.log(1)))

        self.initialized = False
        self.bypassing = False

    @property
    def theta_kappa(self) -> torch.Tensor:
        return self.log_theta_kappa.exp()

    @property
    def q_kappa(self) -> torch.Tensor:
        return self.log_q_kappa.exp()

    @property
    def q(self) -> torch.Tensor:
        return self.log_q.exp()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @abc.abstractmethod
    def get_q_lr(self, eta: float) -> float: ...

    @abc.abstractmethod
    def get_alpha_lr(self, eta: float) -> float: ...

    def get_theta_kappa_lr(self, eta: float) -> float:
        return eta / self.theta_num_quantized_params**0.5

    def get_q_kappa_lr(self, eta: float) -> float:
        return eta / self.q.numel() ** 0.5

    @torch.no_grad()
    def init_q_from_block_means(self, block_means: torch.Tensor) -> None:
        q_tensor = 2 * block_means / 2 ** ((self.theta_format.bits - 1) / 2)
        log_q_tensor = q_tensor.clamp_min(torch.finfo(torch.float32).tiny).log()
        self.log_q.copy_(log_q_tensor)

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta_indices, q_indices, grad_proxy, _, _ = cascaded_cdl_topk_forward(
            theta,
            self.theta_format,
            self.theta_kappa,
            self.theta_topk,
            self.q,
            self.q_format,
            self.q_kappa,
            self.q_topk,
            self.alpha,
            True,
        )
        return theta_indices, q_indices, grad_proxy

    def forward(self, x: torch.Tensor) -> QTensor:
        if self.bypassing:
            return CascadedCdlBypassTensor(x)

        assert self.initialized
        if x.ndim != len(self.input_shape):
            raise ValueError(
                f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
            )
        for declared, actual in zip(self.input_shape, x.shape):
            if declared != 1 and declared != actual:
                raise ValueError(
                    f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
                )

        theta, _ = group_tensor_into_blocks(x, self.block_size)

        if not self.training:
            theta_indices, q_indices = cascaded_nearest_round(
                theta,
                self.theta_format,
                self.q,
                self.q_format,
                self.alpha,
            )
            grad_proxy = None
        else:
            theta_indices, q_indices, grad_proxy = self.training_forward(theta)

        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            self.alpha.detach(),
            grad_proxy,
            self.k_dim,
        )


class CascadedCdlQuantForActivation(CascadedCdlQuant):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
    ):
        super().__init__(
            input_shape,
            theta_format,
            theta_topk,
            q_format,
            q_topk,
            block_size,
        )

    def get_q_lr(self, eta: float) -> float:
        n = self.block_size
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2**bits))

    def get_alpha_lr(self, eta: float) -> float:
        n = self.theta_num_quantized_params
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2**bits))


class CascadedCdlQuantForWeight(CascadedCdlQuant):
    def __init__(
        self,
        weight: torch.Tensor,
        theta_format: Format,
        q_format: Format,
        block_size: int,
    ):
        out_dim, k_dim = weight.shape
        super().__init__(
            (out_dim, k_dim),
            theta_format,
            theta_format.n,
            q_format,
            q_format.n,
            block_size,
        )
        self.init_q_from_weight(weight)
        self.initialized = True

    def get_q_lr(self, eta: float) -> float:
        n = self.block_size
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2 ** (bits - 1)))

    def get_alpha_lr(self, eta: float) -> float:
        n = self.theta_num_quantized_params
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2 ** (bits - 1)))

    @torch.no_grad()
    def init_q_from_weight(self, weight: torch.Tensor) -> None:
        blocked, pad_size = group_tensor_into_blocks(weight, self.block_size)
        block_sums = blocked.abs().sum(dim=-1, keepdim=True)

        per_block_len = blocked.new_full((blocked.shape[1],), self.block_size)
        if pad_size:
            per_block_len[-1] -= pad_size
        block_means = block_sums / per_block_len[None, :, None]

        self.init_q_from_block_means(block_means)


class CascadedCdlQuantMixin(BatchNormBatchStatsMixin):
    def get_weight_quant_counts(self) -> int:
        return sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForWeight)
        )

    def get_activation_quant_counts(self) -> int:
        return sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation)
        )

    @torch.no_grad()
    def quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, CascadedCdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.theta_kappa"] = m.theta_kappa.item()
            out[f"{name}.q"] = m.q.detach().float().mean().item()
            out[f"{name}.q_kappa"] = m.q_kappa.item()
            out[f"{name}.alpha"] = m.alpha.item()
        return out

    @torch.no_grad()
    def get_param_groups(
        self,
        learning_rate: float,
        train_alpha: bool,
        train_kappa: bool,
    ) -> tuple[list[dict], set[nn.Parameter]]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[float, list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CascadedCdlQuant):
                continue

            customized_param_set.update(m.parameters())

            q_lr = m.get_q_lr(learning_rate)
            buckets.setdefault(q_lr, []).append(m.log_q)

            if train_alpha:
                alpha_lr = m.get_alpha_lr(learning_rate)
                buckets.setdefault(alpha_lr, []).append(m.log_alpha)
            else:
                m.log_alpha.requires_grad_(False)

            if train_kappa:
                theta_kappa_lr = m.get_theta_kappa_lr(learning_rate)
                q_kappa_lr = m.get_q_kappa_lr(learning_rate)
                buckets.setdefault(theta_kappa_lr, []).append(m.log_theta_kappa)
                buckets.setdefault(q_kappa_lr, []).append(m.log_q_kappa)
            else:
                m.log_theta_kappa.requires_grad_(False)
                m.log_q_kappa.requires_grad_(False)

        groups = [
            {"params": params, "lr": lr, "weight_decay": 0.0}
            for lr, params in buckets.items()
        ]
        return groups, customized_param_set

    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m for m in self.modules() if isinstance(m, CascadedCdlQuantForActivation)
        ]
        if not modules:
            return

        stats: dict[CascadedCdlQuantForActivation, tuple[torch.Tensor, int, int]] = {}

        def hook(m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            blocked, _ = group_tensor_into_blocks(x, m.block_size)
            reduce_dims = tuple(
                i for i in range(blocked.ndim - 2) if m.input_shape[i] == 1
            ) + (blocked.ndim - 1,)
            block_sums = blocked.abs().sum(dim=reduce_dims)
            positions = x.numel() // math.prod(m.input_shape)
            previous = stats.get(m)
            if previous is None:
                stats[m] = (block_sums, positions, x.shape[-1])
                return
            sums, positions_total, width = previous
            assert width == x.shape[-1]
            stats[m] = (sums + block_sums, positions_total + positions, width)

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.training
        self.eval()
        for m in modules:
            m.bypassing = True
        try:
            with self.batchnorm_batch_stats():
                for x in samples:
                    self(x)
        finally:
            for h in handles:
                h.remove()
            for m in modules:
                m.bypassing = False
            self.train(was_training)

        for m in modules:
            assert m in stats
            sums, positions, width = stats[m]
            counts = sums.new_full((sums.numel(),), m.block_size)
            pad_size = -width % m.block_size
            if pad_size:
                counts[-1] -= pad_size
            block_means = (sums.flatten() / (positions * counts)).reshape(
                m.input_shape[:-1] + (-1, 1)
            )
            m.init_q_from_block_means(block_means)
            m.initialized = True
