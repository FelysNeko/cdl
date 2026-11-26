from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp4 import E2M1SingleZero
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel import nvfp4_ccdl_quant
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.model.components.quant.block import (
    check_triton_block_dims,
    squeeze_block_scale,
    ungroup_blocks,
)
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Nvfp4CascadedCdlTensor(CascadedCdlTensor):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 16
        packed = ungroup_blocks(
            self.theta_format.pack(self.theta_indices), self.block_mask
        )
        scale = squeeze_block_scale(self.q_format.pack(self.q_indices), self.block_mask)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedNvfp4CascadedCdlTensor(QTensor):
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
        if scale.ndim >= 3 and any(
            s == 1 and p != 1
            for s, p in zip(scale.shape[:-2], packed.shape[:-2], strict=True)
        ):
            scale = scale.expand(*packed.shape[:-2], *scale.shape[-2:])
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


class Nvfp4CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        input_shape: Sequence[int],
        block_dims: Sequence[bool],
        theta_topk: int,
        q_topk: int,
        dtype: torch.dtype,
        kappa_forward: float,
        theta_kappa_backward: float | None = None,
        q_kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            block_dims,
            E2M1SingleZero(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            q_topk,
            16,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy = nvfp4_ccdl_quant(
            theta,
            self.theta_kappa,
            self.theta_kappa_backward,
            self.theta_topk,
            self.q,
            self.q_kappa,
            self.q_kappa_backward,
            self.q_topk,
            self.alpha,
            self.kernel_seed,
            self.triton_block_2d,
        )
        return PackedNvfp4CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )


class Nvfp4CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        block_dims: Sequence[bool],
        dtype: torch.dtype,
        kappa_forward: float,
        theta_kappa_backward: float | None = None,
        q_kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ):
        super().__init__(
            weight,
            block_dims,
            E2M1SingleZero(dtype),
            UE4M3ExcludeZero(dtype),
            16,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )
        self.kernel = kernel
        if kernel == "triton":
            self.triton_block_2d = check_triton_block_dims(self.block_dims)
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, x: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(x)

        theta, block_mask, orig_shape = self.group_tensor_into_blocks(x)
        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy = nvfp4_ccdl_quant(
            theta,
            self.theta_kappa,
            self.theta_kappa_backward,
            self.theta_topk,
            self.q,
            self.q_kappa,
            self.q_kappa_backward,
            self.q_topk,
            self.alpha,
            self.kernel_seed,
            self.triton_block_2d,
        )
        return PackedNvfp4CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            block_mask=block_mask,
            orig_shape=orig_shape,
            dtype=self.theta_format.level.dtype,
            x=x,
        )
