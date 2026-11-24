from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp8 import E4M3, E8M0
from cdl.core.kernel import mxfp8_ecdl_quant
from cdl.core.kernel.dequant import mxfp8_dequant
from cdl.core.kernel.quant import mxfp8_quant
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
    check_triton_block_dims,
    squeeze_block_scale,
    ungroup_blocks,
)


class Mxfp8ExtendedCdlTensor(ExtendedCdlTensor):
    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.theta_indices.shape[-1] == 32
        packed = ungroup_blocks(
            self.theta_format.pack(self.theta_indices), self.block_mask
        )
        scale = squeeze_block_scale(
            self.q_format.pack(self.q_indices), self.block_mask
        )
        return packed, scale, None


@dataclass
class PackedMxfp8ExtendedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    grad_proxy: torch.Tensor | None
    block_mask: Sequence[bool]
    orig_shape: Sequence[int]
    dtype: torch.dtype

    @property
    def shape(self) -> Sequence[int]:
        return self.orig_shape

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.mxfp8()
        lead = packed.shape[:-1]
        out = mxfp8_dequant(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        ).reshape(*lead, -1)
        if tuple(out.shape) != tuple(self.orig_shape):
            out = out[tuple(slice(None, d) for d in self.orig_shape)]
        return out.to(self.dtype)

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.packed_theta.shape[-1] == 32
        return (
            ungroup_blocks(self.packed_theta, self.block_mask),
            squeeze_block_scale(self.packed_scale, self.block_mask),
            None,
        )

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return ungroup_blocks(self.grad_proxy, self.block_mask, self.orig_shape)


class Mxfp8ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Mxfp8ExtendedCdlTensor

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
            E4M3(dtype),
            theta_topk,
            E8M0(dtype),
            32,
            kappa_forward,
            kappa_backward,
            scale_mode="ocp",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
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

        if self.triton_block_2d:
            lead = theta.shape[:-4]
            nb_r, tpb, nb_c, _ = theta.shape[-4:]
            packed, scale, _ = mxfp8_quant(theta.reshape(-1, nb_c * tpb), None, True)
            packed_theta = packed.reshape(*lead, nb_r, tpb, nb_c, 32)
            packed_scale = scale.reshape(*lead, nb_r, 1, nb_c, 1)
        else:
            lead = theta.shape[:-2]
            nb, tpb = theta.shape[-2:]
            packed, scale, _ = mxfp8_quant(theta.reshape(-1, nb * tpb), None, False)
            packed_theta = packed.reshape(*lead, nb, 32)
            packed_scale = scale.reshape(*lead, nb, 1)

        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
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
        packed_theta, packed_scale, grad_proxy, _ = mxfp8_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
            self.triton_block_2d,
        )

        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )


class Mxfp8ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Mxfp8ExtendedCdlTensor

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
            E4M3(dtype),
            E8M0(dtype),
            32,
            kappa_forward,
            kappa_backward,
            scale_mode="ocp",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
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

        if self.triton_block_2d:
            lead = theta.shape[:-4]
            nb_r, tpb, nb_c, _ = theta.shape[-4:]
            packed, scale, _ = mxfp8_quant(theta.reshape(-1, nb_c * tpb), None, True)
            packed_theta = packed.reshape(*lead, nb_r, tpb, nb_c, 32)
            packed_scale = scale.reshape(*lead, nb_r, 1, nb_c, 1)
        else:
            lead = theta.shape[:-2]
            nb, tpb = theta.shape[-2:]
            packed, scale, _ = mxfp8_quant(theta.reshape(-1, nb * tpb), None, False)
            packed_theta = packed.reshape(*lead, nb, 32)
            packed_scale = scale.reshape(*lead, nb, 1)
        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
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
        packed_theta, packed_scale, grad_proxy, _ = mxfp8_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
            self.triton_block_2d,
        )

        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
        )
