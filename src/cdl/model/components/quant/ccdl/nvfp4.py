from typing import Literal

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.core.kernel.nvfp4 import nvfp4_quant
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


class Nvfp4CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_topk: int,
        q_topk: int,
        dtype: torch.dtype,
        kernel: Literal["torch", "triton"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            E2M1(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            q_topk,
            16,
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
        return nvfp4_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )


class Nvfp4CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kernel: Literal["torch", "triton"] = "torch",
    ):
        super().__init__(weight, E2M1(dtype), UE4M3ExcludeZero(dtype), 16)
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
        return nvfp4_quant(
            theta,
            self.q,
            self.theta_kappa,
            self.q_kappa,
            self.alpha,
            self.kernel_seed,
            self.theta_topk,
            self.q_topk,
        )
