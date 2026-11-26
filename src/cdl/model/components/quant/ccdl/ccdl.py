import abc
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ccdl import cascaded_cdl_topk_forward, cascaded_nearest_round
from cdl.model.components.quant.block import (
    group_blocks_by_dims,
    ungroup_blocks,
)
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


@dataclass
class CascadedCdlTensor(QTensor):
    theta_format: Format
    theta_indices: torch.Tensor
    q_format: Format
    q_indices: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    block_mask: Sequence[bool]
    orig_shape: Sequence[int]
    raw: torch.Tensor

    @property
    def shape(self) -> Sequence[int]:
        return self.orig_shape

    def dequantize(self) -> torch.Tensor:
        theta = self.theta_format.level[self.theta_indices.long()]
        q = self.q_format.level[self.q_indices.long()]
        alpha = self.alpha.to(theta.dtype)
        return ungroup_blocks(theta * q * alpha, self.block_mask, self.orig_shape)

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return ungroup_blocks(self.grad_proxy, self.block_mask, self.orig_shape)

    def source(self) -> torch.Tensor:
        return self.raw


@dataclass
class CascadedCdlBypassTensor(QTensor):
    theta: torch.Tensor

    @property
    def shape(self) -> Sequence[int]:
        return self.theta.shape

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None

    def source(self) -> torch.Tensor:
        return self.theta


class CascadedCdlQuant(Quant):
    tensor: type[CascadedCdlTensor] = CascadedCdlTensor

    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
        kappa_forward: float,
        theta_kappa_backward: float | None,
        q_kappa_backward: float | None,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.block_dims = tuple(block_dims)
        if len(self.block_dims) != len(self.input_shape):
            raise ValueError(
                f"expected block_dims of length {len(self.input_shape)}, got {len(self.block_dims)}"
            )
        if not any(self.block_dims):
            raise ValueError(
                f"block_dims must block at least one dim, got {self.block_dims}"
            )
        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.q_format = q_format
        self.q_topk = q_topk
        self.block_size = block_size

        q_shape: list[int] = []
        for s, is_block in zip(self.input_shape, self.block_dims, strict=True):
            if is_block:
                nb = (s + block_size - 1) // block_size
                q_shape.extend((nb, 1))
            else:
                q_shape.append(s)

        self.theta_num_quantized_params = math.prod(self.input_shape)
        self.q_num_quantized_params = math.prod(q_shape)
        self.q_block_numel = block_size ** sum(self.block_dims)

        self.log_theta_kappa = nn.Parameter(torch.full((), math.log(kappa_forward)))
        self.log_q_kappa = nn.Parameter(torch.full((), math.log(kappa_forward)))
        self.register_buffer(
            "theta_kappa_backward",
            torch.tensor(theta_kappa_backward, dtype=torch.float32)
            if theta_kappa_backward is not None
            else None,
            persistent=False,
        )
        self.register_buffer(
            "q_kappa_backward",
            torch.tensor(q_kappa_backward, dtype=torch.float32)
            if q_kappa_backward is not None
            else None,
            persistent=False,
        )
        self.log_q = nn.Parameter(torch.full(q_shape, math.log(1)))
        self.log_alpha = nn.Parameter(torch.full((), math.log(1)))
        self.register_buffer(
            "log_q_init", self.log_q.detach().clone(), persistent=False
        )

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
    def copy_log_q(self, log_q_tensor: torch.Tensor) -> None:
        self.log_q.copy_(log_q_tensor)
        self.log_q_init.copy_(log_q_tensor)

    @torch.no_grad()
    def init_q_from_block_amax(self, block_amax: torch.Tensor) -> None:
        q_tensor = block_amax / self.theta_format.level[-1]
        log_q_tensor = q_tensor.clamp_min(torch.finfo(torch.float32).tiny).log()
        self.copy_log_q(log_q_tensor)

    def group_tensor_into_blocks(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[bool, ...], tuple[int, ...]]:
        return group_blocks_by_dims(x, self.block_dims, self.block_size)

    def eval_forward(self, x: torch.Tensor) -> QTensor:
        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        theta_indices, q_indices = cascaded_nearest_round(
            theta,
            self.theta_format,
            self.q,
            self.q_format,
            self.alpha,
        )
        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            self.alpha.detach(),
            None,
            block_mask,
            orig_shape,
            x,
        )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        theta_indices, q_indices, grad_proxy, _, _ = cascaded_cdl_topk_forward(
            theta,
            self.theta_format,
            self.theta_kappa,
            self.theta_kappa_backward,
            self.theta_topk,
            None,
            self.q,
            self.q_format,
            self.q_kappa,
            self.q_kappa_backward,
            self.q_topk,
            None,
            self.alpha,
            True,
        )
        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            self.alpha.detach(),
            grad_proxy,
            block_mask,
            orig_shape,
            x,
        )

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

        if not self.training:
            return self.eval_forward(x)

        return self.training_forward(x)


class CascadedCdlQuantForActivation(CascadedCdlQuant):
    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        q_topk: int,
        block_size: int,
        kappa_forward: float,
        theta_kappa_backward: float | None = None,
        q_kappa_backward: float | None = None,
    ):
        super().__init__(
            input_shape,
            block_dims,
            theta_format,
            theta_topk,
            q_format,
            q_topk,
            block_size,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )

    def get_q_lr(self, eta: float) -> float:
        n = self.q_block_numel
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
        block_dims: Sequence[bool],
        theta_format: Format,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        theta_kappa_backward: float | None = None,
        q_kappa_backward: float | None = None,
    ):
        super().__init__(
            tuple(weight.shape),
            block_dims,
            theta_format,
            theta_format.n,
            q_format,
            q_format.n,
            block_size,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )
        self.init_q_from_weight(weight)
        self.initialized = True

    def get_q_lr(self, eta: float) -> float:
        n = self.q_block_numel
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2 ** (bits - 1)))

    def get_alpha_lr(self, eta: float) -> float:
        n = self.theta_num_quantized_params
        bits = self.theta_format.bits
        return eta / math.sqrt(n * (2 ** (bits - 1)))

    @torch.no_grad()
    def init_q_from_weight(self, weight: torch.Tensor) -> None:
        blocked, block_mask, _ = self.group_tensor_into_blocks(weight)
        reduce_dims = tuple(i for i, is_block in enumerate(block_mask) if is_block)
        self.init_q_from_block_amax(blocked.abs().amax(dim=reduce_dims, keepdim=True))


class CascadedCdlQuantMixin(BatchNormBatchStatsMixin):
    def cascaded_cdl_get_weight_quant_counts(self) -> int:
        return sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForWeight)
        )

    def cascaded_cdl_get_activation_quant_counts(self) -> int:
        return sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation)
        )

    @torch.no_grad()
    def cascaded_cdl_quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, CascadedCdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.theta_kappa"] = m.theta_kappa.item()
            out[f"{name}.q"] = m.q.detach().float().mean().item()
            rel = (m.log_q - m.log_q_init).float().abs().mean()
            out[f"{name}.q_rel"] = rel.item()
            out[f"{name}.q_kappa"] = m.q_kappa.item()
            out[f"{name}.alpha"] = m.alpha.item()
        return out

    @torch.no_grad()
    def cascaded_cdl_get_param_groups(
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
    def cascaded_cdl_init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m for m in self.modules() if isinstance(m, CascadedCdlQuantForActivation)
        ]
        if not modules:
            return

        stats: dict[CascadedCdlQuantForActivation, torch.Tensor] = {}

        def hook(m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            blocked, block_mask, _ = m.group_tensor_into_blocks(x)
            reduce_dims = tuple(
                i
                for i, (is_block, qs) in enumerate(
                    zip(block_mask, m.q.shape, strict=True)
                )
                if is_block or qs == 1
            )
            block_amax = blocked.abs().amax(dim=reduce_dims, keepdim=True)
            previous = stats.get(m)
            stats[m] = (
                block_amax if previous is None else torch.maximum(previous, block_amax)
            )

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
            m.init_q_from_block_amax(stats[m])
            m.initialized = True
