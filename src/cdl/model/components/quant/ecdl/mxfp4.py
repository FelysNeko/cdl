from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import E8M0
from cdl.core.kernel import mxfp4_ecdl_quant
from cdl.core.kernel.dequant import mxfp4_dequant
from cdl.core.kernel.quant import mxfp4_quant
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
    squeeze_block_scale,
    ungroup_blocks,
)


def _check_triton_block_dims(block_dims: tuple[bool, ...]) -> None:
    if any(block_dims[:-1]) or not block_dims[-1]:
        raise NotImplementedError(
            f"triton kernel only supports 1D last-dim block scaling, got block_dims={block_dims}"
        )


class Mxfp4ExtendedCdlTensor(ExtendedCdlTensor):
    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 32
        packed = ungroup_blocks(
            self.theta_format.pack(self.theta_indices), self.block_mask
        )
        scale = squeeze_block_scale(
            self.q_format.pack(self.q_indices), self.block_mask
        )
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedMxfp4ExtendedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    block_mask: Sequence[bool]
    orig_shape: Sequence[int]
    dtype: torch.dtype

    @property
    def shape(self) -> Sequence[int]:
        return self.orig_shape

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.mxfp4()
        lead = packed.shape[:-1]
        out = mxfp4_dequant(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        )
        return out.reshape(*lead, -1)[..., : self.orig_shape[-1]].to(self.dtype)

    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.packed_theta.shape[-1] == 16
        return (
            ungroup_blocks(self.packed_theta, self.block_mask),
            squeeze_block_scale(self.packed_scale, self.block_mask),
            self.alpha.float().reshape(1),
        )

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return ungroup_blocks(self.grad_proxy, self.block_mask, self.orig_shape)


class Mxfp4ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Mxfp4ExtendedCdlTensor

    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_topk: int,
        dtype: torch.dtype,
        kappa_forward: float,
        kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            block_dims,
            E2M1(dtype),
            theta_topk,
            E8M0(dtype),
            32,
            kappa_forward,
            kappa_backward,
            scale_mode="ocp",
        )
        self.kernel = kernel
        if kernel == "triton":
            _check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(theta, block_mask, orig_shape)

        lead = theta.shape[:-2]
        nb, tpb = theta.shape[-2:]
        packed, scale, alpha = mxfp4_quant(theta.reshape(-1, nb * tpb), None)
        return PackedMxfp4ExtendedCdlTensor(
            packed_theta=packed.reshape(*lead, nb, 16),
            packed_scale=scale.reshape(*lead, nb, 1),
            alpha=alpha.detach(),
            grad_proxy=None,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )

    def training_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta, block_mask, orig_shape)

        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy, global_scale = mxfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedMxfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )


class Mxfp4ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Mxfp4ExtendedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        block_dims: Sequence[bool],
        dtype: torch.dtype,
        kappa_forward: float,
        kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            weight,
            block_dims,
            E2M1(dtype),
            E8M0(dtype),
            32,
            kappa_forward,
            kappa_backward,
            scale_mode="ocp",
        )
        self.kernel = kernel
        if kernel == "triton":
            _check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(theta, block_mask, orig_shape)

        lead = theta.shape[:-2]
        nb, tpb = theta.shape[-2:]
        packed, scale, alpha = mxfp4_quant(theta.reshape(-1, nb * tpb), None)
        return PackedMxfp4ExtendedCdlTensor(
            packed_theta=packed.reshape(*lead, nb, 16),
            packed_scale=scale.reshape(*lead, nb, 1),
            alpha=alpha.detach(),
            grad_proxy=None,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )

    def training_forward(
        self,
        theta: torch.Tensor,
        block_mask: Sequence[bool],
        orig_shape: Sequence[int],
    ) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta, block_mask, orig_shape)

        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy, global_scale = mxfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedMxfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )
