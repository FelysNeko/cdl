from typing import Literal

import torch

from cdl.core.format.fp8 import E4M3, E8M0
from cdl.core.kernel import mxfp8_ccdl_quant
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Mxfp8CascadedCdlTensor(CascadedCdlTensor):
    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 32
        packed = self.theta_format.pack(self.theta_indices).flatten(-2)
        scale = self.q_format.pack(self.q_indices).flatten(-2)
        alpha = self.alpha.float().reshape(1)
        return packed, scale, alpha


class Mxfp8CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Mxfp8CascadedCdlTensor

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
            input_shape, E4M3(dtype), theta_topk, E8M0(dtype), q_topk, 32, kappa
        )
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        return mxfp8_ccdl_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )


class Mxfp8CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Mxfp8CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa: float,
        kernel: Literal["torch", "triton"] = "torch",
    ):
        super().__init__(weight, E4M3(dtype), E8M0(dtype), 32, kappa)
        self.kernel = kernel
        if kernel == "triton":
            self.register_buffer(
                "kernel_seed", torch.zeros((), dtype=torch.int64), persistent=False
            )

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel == "torch":
            return super().training_forward(theta)
        return mxfp8_ccdl_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )
