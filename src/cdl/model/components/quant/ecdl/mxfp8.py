from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp8 import E4M3, E8M0
from cdl.core.kernel import mxfp8_ecdl_quant
from cdl.core.kernel.dequant import mxfp8_dequant
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
)


class Mxfp8ExtendedCdlTensor(ExtendedCdlTensor):
    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.theta_indices.shape[-1] == 32
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        return packed, scale, None


@dataclass
class PackedMxfp8ExtendedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int
    dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.mxfp8()
        lead = packed.shape[:-1]
        out = mxfp8_dequant(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        )
        return out.reshape(*lead, -1)[..., : self.k_dim].to(self.dtype)

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.packed_theta.shape[-1] == 32
        return self.packed_theta.flatten(-2), self.packed_scale.flatten(-2), None

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(-2)[..., : self.k_dim]


class Mxfp8ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Mxfp8ExtendedCdlTensor

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
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy, _ = mxfp8_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )


class Mxfp8ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Mxfp8ExtendedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa_forward: float,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            weight,
            E4M3(dtype),
            E8M0(dtype),
            32,
            kappa_forward=kappa_forward,
            scale_mode="ocp",
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        packed_theta, packed_scale, grad_proxy, _ = mxfp8_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.kernel_seed,
            self.theta_topk,
        )

        return PackedMxfp8ExtendedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )
