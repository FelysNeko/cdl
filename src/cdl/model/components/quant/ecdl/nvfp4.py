from typing import Literal

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel import nvfp4_ecdl_quant
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


class Nvfp4ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Nvfp4ExtendedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_topk: int,
        dtype: torch.dtype,
        kappa_forward: float,
        kappa_backward: float,
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
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        return nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.beta,
            self.kernel_seed,
            self.theta_topk,
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
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        return nvfp4_ecdl_quant(
            theta,
            self.kappa,
            self.kappa_backward,
            self.beta,
            self.kernel_seed,
            self.theta_topk,
        )
