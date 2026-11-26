import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ecdl import extended_cdl_topk_forward, extended_nearest_round
from cdl.model.components.quant.block import (
    group_blocks_by_dims,
    ungroup_blocks,
)
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


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
    x: torch.Tensor

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
        return self.x


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

    def source(self) -> torch.Tensor:
        return self.theta


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

    def group_tensor_into_blocks(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[bool, ...], tuple[int, ...]]:
        return group_blocks_by_dims(x, self.block_dims, self.block_size)

    def eval_forward(self, x: torch.Tensor) -> QTensor:
        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
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
            x,
        )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
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
            x,
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

        if not self.training:
            return self.eval_forward(x)

        return self.training_forward(x)


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
