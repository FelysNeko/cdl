import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Nvfp4CascadedCdlTensor(CascadedCdlTensor):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 16
        packed = self.theta_format.pack(self.theta_indices).flatten(1)
        scale = self.q_format.pack(self.q_indices).flatten(1)
        return packed, scale, self.alpha.float().reshape(1)


class Nvfp4CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        k_dim: int,
        theta_topk: int,
        q_topk: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            k_dim,
            E2M1(dtype),
            theta_topk,
            UE4M3ExcludeZero(dtype),
            q_topk,
            16,
        )


class Nvfp4CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Nvfp4CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
    ):
        super().__init__(
            weight,
            E2M1(dtype),
            UE4M3ExcludeZero(dtype),
            16,
        )
