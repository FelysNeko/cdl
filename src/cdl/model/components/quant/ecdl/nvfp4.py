from typing import Literal

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
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
        kappa: float,
        kernel: Literal["torch"] = "torch",
    ) -> None:
        super().__init__(
            input_shape,
            E2M1(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            16,
            kappa=kappa,
        )
        if kernel != "torch":
            raise ValueError(
                "extended CDL has no Triton kernel yet; set model.kernel to 'torch'"
            )


class Nvfp4ExtendedCdlQuantForWeight(ExtendedCdlQuantForWeight):
    tensor = Nvfp4ExtendedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
        kappa: float,
        kernel: Literal["torch"] = "torch",
    ) -> None:
        super().__init__(
            weight,
            E2M1(dtype),
            UE4M3ExcludeZero(dtype),
            16,
            kappa=kappa,
        )
        if kernel != "torch":
            raise ValueError(
                "extended CDL has no Triton kernel yet; set model.kernel to 'torch'"
            )
