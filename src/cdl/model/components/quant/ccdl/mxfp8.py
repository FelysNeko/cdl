from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor
from cdl.core.format.fp8 import E4M3, E8M0
from cdl.core.kernel import mxfp8_ccdl_quant
from cdl.core.kernel.dequant import mxfp8_dequant
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Mxfp8CascadedCdlTensor(CascadedCdlTensor):
    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.theta_indices.shape[-1] == 32
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


@dataclass
class PackedMxfp8CascadedCdlTensor(QTensor):
    packed_theta: torch.Tensor
    packed_scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int
    dtype: torch.dtype

    @property
    def shape(self) -> Sequence[int]:
        return (*self.packed_theta.shape[:-2], self.k_dim)

    def dequantize(self) -> torch.Tensor:
        packed, scale, alpha = self.mxfp8()
        lead = packed.shape[:-1]
        packed = packed.reshape(-1, packed.shape[-1])
        scale = scale.reshape(-1, scale.shape[-1])
        if scale.shape[0] != packed.shape[0]:
            scale = scale.expand(packed.shape[0], -1)
        out = mxfp8_dequant(packed, scale, alpha)
        return out.reshape(*lead, -1)[..., : self.k_dim].to(self.dtype)

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert self.packed_theta.shape[-1] == 32
        return (
            self.packed_theta.flatten(-2),
            self.packed_scale.flatten(-2),
            self.alpha.float().reshape(1),
        )

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(-2)[..., : self.k_dim]


class Mxfp8CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Mxfp8CascadedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
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
            E4M3(dtype),
            theta_topk,
            E8M0(dtype),
            q_topk,
            32,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy = mxfp8_ccdl_quant(
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
        )
        return PackedMxfp8CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )


class Mxfp8CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Mxfp8CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa_forward: float,
        theta_kappa_backward: float | None = None,
        q_kappa_backward: float | None = None,
        kernel: Literal["torch", "triton"] = "torch",
    ):
        super().__init__(
            weight,
            E4M3(dtype),
            E8M0(dtype),
            32,
            kappa_forward,
            theta_kappa_backward,
            q_kappa_backward,
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(self, theta: torch.Tensor) -> QTensor:
        if self.kernel == "torch":
            return super().training_forward(theta)

        self.kernel_seed.add_(1)
        packed_theta, packed_scale, grad_proxy = mxfp8_ccdl_quant(
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
        )
        return PackedMxfp8CascadedCdlTensor(
            packed_theta=packed_theta,
            packed_scale=packed_scale,
            alpha=self.alpha.detach(),
            grad_proxy=grad_proxy,
            k_dim=self.k_dim,
            dtype=self.theta_format.level.dtype,
        )
