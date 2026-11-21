from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel import nvfp4_ccdl_quant
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Nvfp4CascadedCdlTensor(CascadedCdlTensor):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 16
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedNvfp4CascadedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int
    dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.nvfp4()
        lead = packed.shape[:-1]
        packed = packed.reshape(-1, packed.shape[-1])
        scale = scale.reshape(-1, scale.shape[-1])
        if scale.shape[0] != packed.shape[0]:
            scale = scale.expand(packed.shape[0], -1)
        out = nvfp4_dequant(packed, scale, alpha)
        return out.reshape(*lead, -1)[..., : self.k_dim].to(self.dtype)

    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.packed_theta.shape[-1] == 8
        return (
            self.packed_theta.flatten(-2),
            self.packed_scale.flatten(-2),
            self.alpha.float().reshape(1),
        )

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(-2)[..., : self.k_dim]


class Nvfp4CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_topk: int,
        q_topk: int,
        dtype: torch.dtype,
        kappa: float,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            E2M1(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            q_topk,
            16,
            kappa,
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy = nvfp4_ccdl_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )
        return PackedNvfp4CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )


class Nvfp4CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa: float,
        kernel: Literal["torch", "triton"] = "torch",
    ):
        super().__init__(weight, E2M1(dtype), UE4M3ExcludeZero(dtype), 16, kappa)
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy = nvfp4_ccdl_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )
        return PackedNvfp4CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )
