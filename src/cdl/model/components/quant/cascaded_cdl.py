import abc
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import Quant
from cdl.api.tensor import QuantizedTensor
from cdl.core.v2 import cascaded_cdl_topk_forward, cascaded_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


def group_tensor_into_blocks(
    theta: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, int]:
    out_dim = theta.shape[0]
    flat_weight = theta.reshape(out_dim, -1)
    flat_dim = flat_weight.shape[-1]

    pad_size = 0
    extra = flat_dim % block_size
    if extra:
        pad_size = block_size - extra
        flat_weight = F.pad(flat_weight, (0, pad_size))
        flat_dim += pad_size

    num_blocks = flat_dim // block_size
    blocked_weight = flat_weight.reshape(out_dim, num_blocks, block_size)
    return blocked_weight, pad_size


@dataclass
class CascadedCdlBypassQuantizedTensor(QuantizedTensor):
    theta: torch.Tensor

    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CascadedCdlBypassQuantizedTensor is an unquantized calibration value"
        )

    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CascadedCdlBypassQuantizedTensor is an unquantized calibration value"
        )

    def dequantize(self) -> torch.Tensor:
        return self.theta


@dataclass
class CascadedCdlQuantizedTensor(QuantizedTensor):
    theta_indices: torch.Tensor
    theta_format: Format
    q_indices: torch.Tensor
    q_format: Format
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    dtype: torch.dtype
    k: int

    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CascadedCdlQuantizedTensor stores CDL integer levels and has no "
            "packed FP4 kernel view"
        )

    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "CascadedCdlQuantizedTensor stores CDL integer levels and has no "
            "packed FP8 kernel view"
        )

    def dequantize(self) -> torch.Tensor:
        theta = self.theta_format.level[self.theta_indices.long()]
        q = self.q_format.level[self.q_indices.long()]
        x = (theta * q * self.alpha).flatten(1)[:, : self.k]

        if self.grad_proxy is None:
            return x.to(self.dtype)

        grad_proxy = self.grad_proxy.flatten(1)[:, : self.k]
        return (grad_proxy + (x - grad_proxy).detach()).to(self.dtype)


class CascadedCdlQuant(Quant, abc.ABC):
    def __init__(
        self,
        theta_format: Format,
        theta_kappa: float,
        theta_topk: int,
        theta_num_quantized_params: int,
        q: torch.Tensor,
        q_format: Format,
        q_kappa: float,
        q_topk: int,
        alpha: float,
        block_size: int,
    ):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if theta_topk > theta_format.n:
            raise ValueError(
                f"top-{theta_topk} exceeds the {theta_format.n} quantization levels"
            )
        if q_topk > q_format.n:
            raise ValueError(
                f"top-{q_topk} exceeds the {q_format.n} quantization levels"
            )

        q_tensor = torch.as_tensor(q).detach().clone()
        if not torch.isfinite(q_tensor).all():
            raise ValueError("the initial q must contain only finite values")
        q_tensor = q_tensor.clamp_min(torch.finfo(torch.float32).tiny)

        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.block_size = block_size
        self.theta_num_quantized_params = theta_num_quantized_params
        self.q_format = q_format
        self.q_topk = q_topk
        self.log_theta_kappa = nn.Parameter(torch.full((), math.log(theta_kappa)))
        self.log_q = nn.Parameter(q_tensor.log())
        self.log_q_kappa = nn.Parameter(torch.full((), math.log(q_kappa)))
        self.alpha = nn.Parameter(torch.full((), alpha, dtype=q_tensor.dtype))

    @property
    def theta_kappa(self) -> torch.Tensor:
        return self.log_theta_kappa.exp()

    @property
    def q(self) -> torch.Tensor:
        return self.log_q.exp()

    @property
    def q_kappa(self) -> torch.Tensor:
        return self.log_q_kappa.exp()

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_theta_kappa_lr(self, eta: float) -> float:
        if self.theta_num_quantized_params == 0:
            raise RuntimeError("quantizer not initialized")
        return eta / self.theta_num_quantized_params**0.5

    def scale_q_kappa_lr(self, eta: float) -> float:
        return eta / self.q.numel() ** 0.5

    def forward(self, x: torch.Tensor) -> QuantizedTensor:
        if x.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} expects a 2-D [rows, k] tensor, "
                f"got ndim={x.ndim}"
            )
        blocked, _ = group_tensor_into_blocks(x, self.block_size)
        if blocked.shape[1] != self.q.shape[-2]:
            raise ValueError(
                "input width is incompatible with the initialized block layout: "
                f"expected {self.q.shape[-2]} blocks of {self.block_size}, "
                f"got {blocked.shape[1]} blocks from width {x.shape[1]}"
            )
        if self.q.ndim >= 2 and self.q.shape[0] not in (1, x.shape[0]):
            raise ValueError(
                "input row count is incompatible with the initialized scales: "
                f"expected {self.q.shape[0]} rows, got {x.shape[0]}"
            )

        if not self.training:
            theta_indices, q_indices = cascaded_nearest_round(
                blocked,
                self.theta_format,
                self.q,
                self.q_format,
                self.alpha,
            )
            return CascadedCdlQuantizedTensor(
                theta_indices=theta_indices,
                theta_format=self.theta_format,
                q_indices=q_indices,
                q_format=self.q_format,
                alpha=self.alpha.detach(),
                grad_proxy=None,
                dtype=x.dtype,
                k=x.shape[1],
            )

        theta_indices, q_indices, grad_proxy, _, _ = cascaded_cdl_topk_forward(
            blocked,
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
        return CascadedCdlQuantizedTensor(
            theta_indices=theta_indices,
            theta_format=self.theta_format,
            q_indices=q_indices,
            q_format=self.q_format,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            dtype=x.dtype,
            k=x.shape[1],
        )


class CascadedCdlQuantForWeight(CascadedCdlQuant):
    def __init__(
        self,
        theta: torch.Tensor,
        theta_format: Format,
        theta_kappa: float,
        q_format: Format,
        q_kappa: float,
        alpha: float,
        block_size: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if theta.ndim != 2:
            raise ValueError("theta must be in shape [out_feat, k]")
        if theta.shape[1] == 0:
            raise ValueError("theta must have at least one element per row")

        blocked, pad_size = group_tensor_into_blocks(theta.detach(), block_size)
        block_sums = blocked.abs().sum(dim=-1, keepdim=True)
        block_counts = blocked.new_full((blocked.shape[1],), block_size)
        if pad_size:
            block_counts[-1] -= pad_size
        block_means = block_sums / block_counts[None, :, None]
        q_tensor = (2 * block_means / 2 ** ((theta_format.bits - 1) / 2)).clamp_min(
            torch.finfo(torch.float32).tiny
        )

        super().__init__(
            theta_format,
            theta_kappa,
            theta_format.n,
            theta.numel(),
            q_tensor,
            q_format,
            q_kappa,
            q_format.n,
            alpha,
            block_size,
        )

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.block_size * 2 ** (self.theta_format.bits - 1)) ** 0.5


class CascadedCdlQuantForActivation(CascadedCdlQuant):
    def __init__(
        self,
        theta_format: Format,
        theta_kappa: float,
        theta_topk: int,
        q_format: Format,
        q_kappa: float,
        q_topk: int,
        alpha: float,
        block_size: int,
    ):
        super().__init__(
            theta_format,
            theta_kappa,
            theta_topk,
            0,
            torch.ones(()),
            q_format,
            q_kappa,
            q_topk,
            alpha,
            block_size,
        )
        self.initialized = False
        self.bypassing = False

    def scale_q_lr(self, eta: float) -> float:
        if not self.initialized:
            raise RuntimeError("quantizer not initialized")
        return eta / (self.block_size * 2**self.theta_format.bits) ** 0.5

    @torch.no_grad()
    def set_q_from_block_means(self, block_means: torch.Tensor, width: int) -> None:
        q = (
            2
            * block_means.to(device=self.log_q.device, dtype=self.log_q.dtype)
            / 2 ** ((self.theta_format.bits - 1) / 2)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        log_q = q.reshape(1, -1, 1).log()

        if tuple(self.log_q.shape) != tuple(log_q.shape):
            self.log_q = nn.Parameter(log_q)
        else:
            self.log_q.copy_(log_q)

        self.theta_num_quantized_params = width
        self.initialized = True

    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        if input.ndim != 2:
            raise ValueError(
                "CascadedCdlQuantForActivation expects a 2-D [rows, k] tensor, "
                f"got ndim={input.ndim}"
            )

        if self.bypassing:
            return CascadedCdlBypassQuantizedTensor(input)

        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        if input.shape[1] != self.theta_num_quantized_params:
            raise ValueError(
                "activation width must match calibration: expected "
                f"{self.theta_num_quantized_params}, got {input.shape[1]}"
            )
        return super().forward(input)


class CdlQuantMixin(BatchNormBatchStatsMixin):
    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation) and not m.initialized
        ]
        if not modules:
            return

        stats: dict[
            CascadedCdlQuantForActivation,
            tuple[torch.Tensor, int, int],
        ] = {}

        def hook(mod: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            if x.ndim != 2:
                raise ValueError(
                    f"activation inputs must be in shape [rows, k], got ndim={x.ndim}"
                )

            blocked, _ = group_tensor_into_blocks(x.detach(), mod.block_size)
            block_sums = blocked.abs().sum(dim=(0, 2))
            previous = stats.get(mod)
            if previous is None:
                stats[mod] = (block_sums, x.shape[0], x.shape[1])
                return

            sums, rows, width = previous
            if width != x.shape[1]:
                raise ValueError(
                    "all calibration samples for an activation quantizer must "
                    "have the same width"
                )
            stats[mod] = (sums + block_sums, rows + x.shape[0], width)

        def block_means(m: CascadedCdlQuantForActivation) -> tuple[torch.Tensor, int]:
            if m not in stats:
                raise RuntimeError("some quantizers were never reached by samples")
            sums, rows, width = stats[m]
            counts = sums.new_full((sums.numel(),), m.block_size)
            pad_size = -width % m.block_size
            if pad_size:
                counts[-1] -= pad_size
            return sums / (rows * counts), width

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.training
        parameter = next(self.parameters(), None)
        if parameter is None:
            raise RuntimeError("the model has no parameters")
        device = parameter.device
        self.eval()
        for m in modules:
            m.bypassing = True
        try:
            with self.batchnorm_batch_stats():
                for x in samples:
                    self(x.to(device))
        finally:
            for h in handles:
                h.remove()
            for m in modules:
                m.bypassing = False
            self.train(was_training)

        for m in modules:
            means, width = block_means(m)
            m.set_q_from_block_means(means, width)

    @torch.no_grad()
    def get_param_groups(
        self, learning_rate: float, weight_decay: float, freeze_kappa: bool
    ) -> list[dict]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CascadedCdlQuant):
                continue

            if m.theta_num_quantized_params == 0:
                raise RuntimeError("quantizer not initialized")

            customized_param_set.add(m.log_q)
            q_lr = m.scale_q_lr(learning_rate)
            buckets.setdefault((q_lr, 0.0), []).append(m.log_q)

            customized_param_set.add(m.log_theta_kappa)
            customized_param_set.add(m.log_q_kappa)
            if freeze_kappa:
                m.log_theta_kappa.requires_grad_(False)
                m.log_q_kappa.requires_grad_(False)
            else:
                theta_kappa_lr = m.scale_theta_kappa_lr(learning_rate)
                q_kappa_lr = m.scale_q_kappa_lr(learning_rate)
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
    def floor_quant(self, min_q: float) -> None:
        for m in self.modules():
            if not isinstance(m, CascadedCdlQuant):
                continue
            m.log_q.clamp_(min=math.log(min_q))
            m.log_theta_kappa.clamp_(min=-20.0, max=20.0)
            m.log_q_kappa.clamp_(min=-20.0, max=20.0)

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

    def get_quant_counts(self) -> tuple[int, int]:
        n_weight = sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForWeight)
        )
        n_activation = sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation)
        )
        return n_weight, n_activation
