import abc
from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ccdl import cascaded_cdl_topk_forward, cascaded_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


def group_tensor_into_blocks(
    theta: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    assert len(theta.shape) == 2
    out_dim, k = theta.shape

    extra = k % block_size
    pad_size = 0
    if extra:
        pad_size = block_size - extra
        theta = F.pad(theta, (0, pad_size))

    num_blocks = (k + pad_size) // block_size
    cdl_style_theta = theta.reshape(out_dim, num_blocks, block_size)
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
        x = (theta * q * self.alpha).flatten(1)[:, : self.k_dim]
        return x

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(1)[:, : self.k_dim]


@dataclass
class CascadedCdlBypassTensor(QTensor):
    theta: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None


class CascadedCdlQuant(Quant):
    def __init__(
        self,
        out_dim_or_broadcast: int,
        k_dim: int,
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
    ):
        super().__init__()
        self.k_dim = k_dim
        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.q_format = q_format
        self.q_topk = q_topk
        self.block_size = block_size

        extra = k_dim % block_size
        num_blocks = (k_dim + extra) // block_size

        self.q_num_quantized_params = out_dim_or_broadcast * num_blocks
        self.theta_num_quantized_params = out_dim_or_broadcast * k_dim

        q_shape = (out_dim_or_broadcast, num_blocks, 1)

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

    def forward(self, x: torch.Tensor) -> QTensor:
        if self.bypassing:
            return CascadedCdlBypassTensor(x)

        assert self.initialized
        theta, _ = group_tensor_into_blocks(x, self.block_size)

        if not self.training:
            theta_indices, q_indices = cascaded_nearest_round(
                theta,
                self.theta_format,
                self.q,
                self.q_format,
                self.alpha,
            )
            return CascadedCdlTensor(
                self.theta_format,
                theta_indices,
                self.q_format,
                q_indices,
                self.alpha.detach(),
                None,
                self.k_dim,
            )

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

        return CascadedCdlTensor(
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
        k_dim: int,
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
    ):
        super().__init__(
            1,
            k_dim,
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
            out_dim,
            k_dim,
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
        weight_decay: float,
        skip_alpha: bool,
        skip_kappa: bool,
    ) -> list[dict]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CascadedCdlQuant):
                continue

            customized_param_set.add(m.log_q)
            customized_param_set.add(m.log_alpha)
            customized_param_set.add(m.log_theta_kappa)
            customized_param_set.add(m.log_q_kappa)

            q_lr = m.get_q_lr(learning_rate)
            buckets.setdefault((q_lr, 0.0), []).append(m.log_q)

            if not skip_alpha:
                alpha_lr = m.get_alpha_lr(learning_rate)
                buckets.setdefault((alpha_lr, 0.0), []).append(m.log_alpha)

            if not skip_kappa:
                theta_kappa_lr = m.get_theta_kappa_lr(learning_rate)
                q_kappa_lr = m.get_q_kappa_lr(learning_rate)
                buckets.setdefault((theta_kappa_lr, 0.0), []).append(m.log_theta_kappa)
                buckets.setdefault((q_kappa_lr, 0.0), []).append(m.log_q_kappa)

        decay = []
        no_decay = []
        for param in self.parameters():
            if param in customized_param_set:
                continue
            if param.ndim <= 1:
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in buckets.items()
        ]

        if decay:
            groups.append(
                {"params": decay, "lr": learning_rate, "weight_decay": weight_decay}
            )

        if no_decay:
            groups.append(
                {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0}
            )

        return groups

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
            assert x.ndim == 2
            blocked, _ = group_tensor_into_blocks(x.detach(), m.block_size)
            block_sums = blocked.abs().sum(dim=(0, 2))
            previous = stats.get(m)
            if previous is None:
                stats[m] = (block_sums, x.shape[0], x.shape[1])
                return
            sums, rows, width = previous
            assert width == x.shape[1]
            stats[m] = (sums + block_sums, rows + x.shape[0], width)

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
            sums, rows, width = stats[m]
            counts = sums.new_full((sums.numel(),), m.block_size)
            pad_size = -width % m.block_size
            if pad_size:
                counts[-1] -= pad_size
            block_means = (sums / (rows * counts)).reshape(1, -1, 1)
            m.init_q_from_block_means(block_means)
            m.initialized = True
