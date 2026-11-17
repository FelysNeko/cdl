from typing import Literal

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import E8M0
from cdl.model.components.quant.ecdl.ecdl import (
    ExtendedCdlQuantForActivation,
    ExtendedCdlQuantForWeight,
    ExtendedCdlTensor,
)


class Mxfp4ExtendedCdlTensor(ExtendedCdlTensor):
    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 32
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


class Mxfp4ExtendedCdlQuantForActivation(ExtendedCdlQuantForActivation):
    tensor = Mxfp4ExtendedCdlTensor

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

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        raise NotImplementedError(
            "triton ECDL awaits the kernel scale-mode rework; use kernel='torch'"
        )


class Mxfp4ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Mxfp4ExtendedCdlTensor

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

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        raise NotImplementedError(
            "triton ECDL awaits the kernel scale-mode rework; use kernel='torch'"
        )
