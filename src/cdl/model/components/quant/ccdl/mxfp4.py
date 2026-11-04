import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import E8M0
from cdl.model.components.quant.ccdl.ccdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CascadedCdlTensor,
)


class Mxfp4CascadedCdlTensor(CascadedCdlTensor):
    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.theta_indices.shape[-1] == 32
        packed = self.theta_format.pack(self.theta_indices).flatten(1)
        scale = self.q_format.pack(self.q_indices).flatten(1)
        return packed, scale, self.alpha.float().reshape(1)


class Mxfp4CascadedCdlQuantForActivation(CascadedCdlQuantForActivation):
    tensor = Mxfp4CascadedCdlTensor

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
            E8M0(dtype),
            q_topk,
            32,
        )


class Mxfp4CascadedCdlQuantForWeight(CascadedCdlQuantForWeight):
    tensor = Mxfp4CascadedCdlTensor

    def __init__(
        self,
        weight: torch.Tensor,
        dtype: torch.dtype,
    ):
        super().__init__(
            weight,
            E2M1(dtype),
            E8M0(dtype),
            32,
        )
