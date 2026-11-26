from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp4 import E2M1SingleZero
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel import nvfp4_ecdl_quant
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.core.kernel.quant import nvfp4_quant
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
    check_triton_block_dims,
    squeeze_block_scale,
    ungroup_blocks,
)


class Nvfp4ExtendedCdlTensor(ExtendedCdlTensor):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 16
        packed = ungroup_blocks(
            self.theta_format.pack(self.theta_indices), self.block_mask
        )
        scale = squeeze_block_scale(self.q_format.pack(self.q_indices), self.block_mask)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedNvfp4ExtendedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    block_mask: Sequence[bool]
    orig_shape: Sequence[int]
    dtype: torch.dtype
    x: torch.Tensor

    @property
    def shape(self) -> Sequence[int]:
        return self.orig_shape

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.nvfp4()
        lead = packed.shape[:-1]
        out = nvfp4_dequant(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        ).reshape(*lead, -1)
        if tuple(out.shape) != tuple(self.orig_shape):
            out = out[tuple(slice(None, d) for d in self.orig_shape)]
        return out.to(self.dtype)

    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.packed_theta.shape[-1] == 8
        return (
            ungroup_blocks(self.packed_theta, self.block_mask),
            squeeze_block_scale(self.packed_scale, self.block_mask),
            self.alpha.float().reshape(1),
        )

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return ungroup_blocks(self.grad_proxy, self.block_mask, self.orig_shape)

    def source(self) -> torch.Tensor:
        return self.x


class Nvfp4ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Nvfp4ExtendedCdlTensor

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
            E2M1SingleZero(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            16,
            kappa_forward,
            kappa_backward,
            scale_mode="nvidia",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        if self.triton_block_2d:
            lead = theta.shape[:-4]
            nb_r, tpb, nb_c, _ = theta.shape[-4:]
            packed, scale, alpha = nvfp4_quant(
                theta.reshape(-1, nb_c * tpb), None, True
            )
            packed_theta = packed.reshape(*lead, nb_r, tpb, nb_c, 8)
            packed_scale = scale.reshape(*lead, nb_r, 1, nb_c, 1)
        else:
            lead = theta.shape[:-2]
            nb, tpb = theta.shape[-2:]
            packed, scale, alpha = nvfp4_quant(theta.reshape(-1, nb * tpb), None, False)
            packed_theta = packed.reshape(*lead, nb, 8)
            packed_scale = scale.reshape(*lead, nb, 1)
        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=alpha.detach(),
            grad_proxy=None,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy, global_scale = nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
            self.triton_block_2d,
        )

        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )


class Nvfp4ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Nvfp4ExtendedCdlTensor

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
            E2M1SingleZero(dtype),
            UE4M3ExcludeZero(dtype),
            16,
            kappa_forward,
            kappa_backward,
            scale_mode="nvidia",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        if self.triton_block_2d:
            lead = theta.shape[:-4]
            nb_r, tpb, nb_c, _ = theta.shape[-4:]
            packed, scale, alpha = nvfp4_quant(
                theta.reshape(-1, nb_c * tpb), None, True
            )
            packed_theta = packed.reshape(*lead, nb_r, tpb, nb_c, 8)
            packed_scale = scale.reshape(*lead, nb_r, 1, nb_c, 1)
        else:
            lead = theta.shape[:-2]
            nb, tpb = theta.shape[-2:]
            packed, scale, alpha = nvfp4_quant(theta.reshape(-1, nb * tpb), None, False)
            packed_theta = packed.reshape(*lead, nb, 8)
            packed_scale = scale.reshape(*lead, nb, 1)

        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=alpha.detach(),
            grad_proxy=None,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy, global_scale = nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
            self.triton_block_2d,
        )

        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )
