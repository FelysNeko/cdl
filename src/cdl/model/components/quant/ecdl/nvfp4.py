from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel import nvfp4_ecdl_quant
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.core.kernel.quant import nvfp4_quant
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
)


class Nvfp4ExtendedCdlTensor(ExtendedCdlTensor):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 16
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedNvfp4ExtendedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int
    dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.nvfp4()
        lead = packed.shape[:-1]
        out = nvfp4_dequant(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        )
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


class Nvfp4ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Nvfp4ExtendedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_topk: int,
        dtype: torch.dtype,
        kappa_forward: float,
        kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            E2M1(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            16,
            kappa_forward,
            kappa_backward,
            scale_mode="nvidia",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(theta)

        lead = theta.shape[:-2]
        nb, tpb = theta.shape[-2:]
        packed, scale, alpha = nvfp4_quant(theta.reshape(-1, nb * tpb))
        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed.reshape(*lead, nb, 8),
            packed_scale=scale.reshape(*lead, nb, 1),
            alpha=alpha.detach(),
            grad_proxy=None,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy, global_scale = nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )


class Nvfp4ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Nvfp4ExtendedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa_forward: float,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            weight,
            E2M1(dtype),
            UE4M3ExcludeZero(dtype),
            16,
            kappa_forward=kappa_forward,
            scale_mode="nvidia",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def eval_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().eval_forward(theta)

        lead = theta.shape[:-2]
        nb, tpb = theta.shape[-2:]
        packed, scale, alpha = nvfp4_quant(theta.reshape(-1, nb * tpb))
        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed.reshape(*lead, nb, 8),
            packed_scale=scale.reshape(*lead, nb, 1),
            alpha=alpha.detach(),
            grad_proxy=None,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy, global_scale = nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedNvfp4ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=global_scale.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )
