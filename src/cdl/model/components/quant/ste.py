from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor, Quant
from cdl.core.kernel.dequant import mxfp4_dequant, mxfp8_dequant, nvfp4_dequant
from cdl.core.kernel.quant import mxfp4_quant, mxfp8_quant, nvfp4_quant

type Simulation = Literal["nvfp4", "mxfp4", "mxfp8"]


@dataclass
class SteTensor(QTensor):
    theta: torch.Tensor
    simulation: Simulation | None

    @property
    def leading_shape(self) -> torch.Size:
        return self.theta.shape[:-1]

    @property
    def k(self) -> int:
        return self.theta.shape[-1]

    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            packed, scale, alpha = nvfp4_quant(self.theta.reshape(-1, self.k))
            packed = packed.reshape(*self.leading_shape, packed.shape[-1])
            scale = scale.reshape(*self.leading_shape, scale.shape[-1])
            return packed, scale, alpha

    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            packed, scale, alpha = mxfp4_quant(self.theta.reshape(-1, self.k))
            packed = packed.reshape(*self.leading_shape, packed.shape[-1])
            scale = scale.reshape(*self.leading_shape, scale.shape[-1])
            return packed, scale, alpha

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            packed, scale, alpha = mxfp8_quant(self.theta.reshape(-1, self.k))
            packed = packed.reshape(*self.leading_shape, packed.shape[-1])
            scale = scale.reshape(*self.leading_shape, scale.shape[-1])
            return packed, scale, alpha

    def dequantize(self) -> torch.Tensor:
        if self.simulation is None:
            return self.theta

        if self.simulation == "nvfp4":
            packed, scale, alpha = self.nvfp4()
            out = nvfp4_dequant(
                packed.reshape(-1, packed.shape[-1]),
                scale.reshape(-1, scale.shape[-1]),
                alpha,
            )
        elif self.simulation == "mxfp4":
            packed, scale, alpha = self.mxfp4()
            out = mxfp4_dequant(
                packed.reshape(-1, packed.shape[-1]),
                scale.reshape(-1, scale.shape[-1]),
                alpha,
            )
        elif self.simulation == "mxfp8":
            packed, scale, alpha = self.mxfp8()
            out = mxfp8_dequant(
                packed.reshape(-1, packed.shape[-1]),
                scale.reshape(-1, scale.shape[-1]),
                alpha,
            )
        else:
            raise ValueError(f"{self.simulation} is not supported")

        return out.reshape(*self.leading_shape, out.shape[-1])[..., : self.k].to(
            self.theta.dtype
        )

    def proxy(self) -> torch.Tensor | None:
        return self.theta


class SteQuant(Quant):
    def __init__(self, simulation: Simulation | None):
        super().__init__()
        self.simulation = simulation

    def forward(self, x: torch.Tensor) -> QTensor:
        return SteTensor(theta=x, simulation=self.simulation)
