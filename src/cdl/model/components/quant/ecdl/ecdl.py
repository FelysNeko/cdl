import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ecdl import extended_cdl_topk_forward, extended_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


def group_tensor_into_blocks(
    theta: torch.Tensor,
    block_dims: Sequence[bool],
    block_size: int,
) -> tuple[torch.Tensor, tuple[bool, ...], tuple[int, ...]]:
    orig_shape = tuple(theta.shape)
    pads = tuple(
        (-s % block_size) if is_block else 0
        for s, is_block in zip(orig_shape, block_dims, strict=True)
    )
    if any(pads):
        pad_args: list[int] = []
        for p in reversed(pads):
            pad_args.extend((0, p))
        theta = F.pad(theta, pad_args)

    blocked_shape: list[int] = []
    block_mask: list[bool] = []
    for s, is_block in zip(theta.shape, block_dims):
        if is_block:
            blocked_shape.extend((s // block_size, block_size))
            block_mask.extend((False, True))
        else:
            blocked_shape.append(s)
            block_mask.append(False)

    return theta.reshape(blocked_shape), tuple(block_mask), orig_shape


def ungroup_blocks(
    tensor: torch.Tensor,
    block_mask: Sequence[bool],
    orig_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    merged_shape: list[int] = []
    for size, is_block in zip(tensor.shape, block_mask, strict=True):
        if is_block:
            merged_shape[-1] *= size
        else:
            merged_shape.append(size)
    out = tensor.reshape(merged_shape)
    if orig_shape is not None and tuple(out.shape) != tuple(orig_shape):
        out = out[tuple(slice(None, d) for d in orig_shape)]
    return out


def squeeze_block_scale(
    scale: torch.Tensor,
    block_mask: Sequence[bool],
) -> torch.Tensor:
    dims = tuple(i for i, is_block in enumerate(block_mask) if is_block)
    return scale.squeeze(dims) if dims else scale


def check_triton_block_dims(block_dims: Sequence[bool]) -> bool:
    if not any(block_dims[:-1]) and block_dims[-1]:
        return False
    if (
        len(block_dims) >= 2
        and not any(block_dims[:-2])
        and block_dims[-2]
        and block_dims[-1]
    ):
        return True
    raise NotImplementedError(
        f"triton kernel only supports 1D or 2D trailing block scaling, got block_dims={block_dims}"
    )


@dataclass
class ExtendedCdlTensor(QTensor):
    theta_format: Format
    theta_indices: torch.Tensor
    q_format: Format
    q_indices: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    block_mask: Sequence[bool]
    orig_shape: Sequence[int]

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


@dataclass
class ExtendedCdlBypassTensor(QTensor):
    theta: torch.Tensor

    @property
    def shape(self) -> Sequence[int]:
        return self.theta.shape

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None


class ExtendedCdlQuant(Quant):
    tensor: type[ExtendedCdlTensor] = ExtendedCdlTensor

    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        kappa_backward: float | None,
        scale_mode: Literal["ocp", "nvidia"],
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.block_dims = tuple(block_dims)
        if len(self.block_dims) != len(input_shape):
            raise ValueError(
                f"expected block_dims of length {len(input_shape)}, got {len(self.block_dims)}"
            )
        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.q_format = q_format
        self.block_size = block_size
        self.num_quantized_params = math.prod(input_shape)
        self.scale_mode = scale_mode

        self.log_kappa = nn.Parameter(torch.full((), math.log(kappa_forward)))
        if kappa_backward is not None:
            self.register_buffer(
                "kappa_backward",
                torch.tensor(kappa_backward, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer("kappa_backward", None, persistent=False)

        self.bypassing = False

    @property
    def kappa(self) -> torch.Tensor:
        return self.log_kappa.exp()

    def eval_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        theta_indices, q_indices, global_scale = extended_nearest_round(
            theta,
            block_mask,
            self.theta_format,
            self.q_format,
            self.scale_mode,
        )
        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            global_scale.detach(),
            None,
            block_mask,
            orig_shape,
        )

    def training_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        theta_indices, q_indices, grad_proxy, global_scale, _ = (
            extended_cdl_topk_forward(
                theta,
                block_mask,
                self.theta_format,
                self.theta_topk,
                self.q_format,
                self.kappa,
                self.kappa_backward,
                True,
                self.scale_mode,
            )
        )
        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            global_scale.detach(),
            grad_proxy,
            block_mask,
            orig_shape,
        )

    def forward(self, x: torch.Tensor) -> QTensor:
        if self.bypassing:
            return ExtendedCdlBypassTensor(x)

        if x.ndim != len(self.input_shape):
            raise ValueError(
                f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
            )
        for declared, actual in zip(self.input_shape, x.shape):
            if declared != 1 and declared != actual:
                raise ValueError(
                    f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
                )

        theta, block_mask, orig_shape = group_tensor_into_blocks(
            x, self.block_dims, self.block_size
        )

        if not self.training:
            return self.eval_forward(theta, block_mask, orig_shape)

        return self.training_forward(theta, block_mask, orig_shape)


class ExtendedCdlQuantForActivation(ExtendedCdlQuant):
    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        kappa_backward: float | None,
        scale_mode: Literal["ocp", "nvidia"],
    ):
        super().__init__(
            input_shape,
            block_dims,
            theta_format,
            theta_topk,
            q_format,
            block_size,
            kappa_forward,
            kappa_backward,
            scale_mode,
        )


class ExtendedCdlQuantForWeight(ExtendedCdlQuant):
    def __init__(
        self,
        weight: torch.Tensor,
        block_dims: Sequence[bool],
        theta_format: Format,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        kappa_backward: float | None,
        scale_mode: Literal["ocp", "nvidia"],
    ):
        super().__init__(
            tuple(weight.shape),
            block_dims,
            theta_format,
            theta_format.n,
            q_format,
            block_size,
            kappa_forward,
            kappa_backward,
            scale_mode,
        )


class ExtendedCdlQuantMixin(BatchNormBatchStatsMixin):
    def extended_cdl_get_weight_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, ExtendedCdlQuantForWeight)
        )

    def extended_cdl_get_activation_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, ExtendedCdlQuantForActivation)
        )

    @torch.no_grad()
    def extended_cdl_quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, ExtendedCdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.kappa"] = m.kappa.item()
        return out

    @torch.no_grad()
    def extended_cdl_get_param_groups(
        self,
        learning_rate: float,
        train_kappa: bool,
    ) -> tuple[list[dict], set[nn.Parameter]]:
        customized_param_set: set[nn.Parameter] = set()
        params: list[nn.Parameter] = []

        for m in self.modules():
            if not isinstance(m, ExtendedCdlQuant):
                continue

            customized_param_set.update(m.parameters())

            if train_kappa:
                params.append(m.log_kappa)
            else:
                m.log_kappa.requires_grad_(False)

        groups = (
            [{"params": params, "lr": learning_rate, "weight_decay": 0.0}]
            if params
            else []
        )
        return groups, customized_param_set
