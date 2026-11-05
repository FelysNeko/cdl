from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from cdl.api.quant import QTensor, Quant
from cdl.core.mm.mxfp4 import mxfp4_dequant
from cdl.core.mm.mxfp8 import mxfp8_dequant
from cdl.model.wrapper.flashinfer import (
    mxfp4_quantize,
    mxfp8_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
)

type Simulation = Literal["nvfp4", "mxfp4", "mxfp8"]


@dataclass
class SteTensor(QTensor):
    theta: torch.Tensor
    simulation: Simulation | None

    def get_padded_theta(self, multiple: int) -> torch.Tensor:
        pad = -self.theta.shape[-1] % multiple
        if pad:
            return F.pad(self.theta, (0, pad))
        return self.theta

    @torch.no_grad()
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.get_padded_theta(16)
        amax = flat.float().abs().amax()
        global_scale = torch.where(
            amax > 0,
            amax.new_full((), 448.0 * 6.0) / amax,
            amax.new_full((), torch.finfo(torch.float32).max),
        ).reshape(1)
        packed, scale = nvfp4_quantize(flat, global_scale)
        return packed, scale, (1.0 / global_scale).float()

    @torch.no_grad()
    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.get_padded_theta(32)
        packed, scale = mxfp4_quantize(flat)
        ones = torch.ones(1, device=flat.device, dtype=torch.float32)
        return packed, scale, ones

    @torch.no_grad()
    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.get_padded_theta(32)
        packed, scale = mxfp8_quantize(flat)
        ones = torch.ones(1, device=flat.device, dtype=torch.float32)
        return packed.view(torch.float8_e4m3fn), scale, ones

    def dequantize(self) -> torch.Tensor:
        if self.simulation is None:
            return self.theta

        k = self.theta.shape[-1]
        if self.simulation == "nvfp4":
            packed, scale, alpha = self.nvfp4()
            out = nvfp4_dequantize(packed, scale, alpha)
        elif self.simulation == "mxfp4":
            packed, scale, alpha = self.mxfp4()
            out = mxfp4_dequant(packed, scale, alpha)
        elif self.simulation == "mxfp8":
            packed, scale, alpha = self.mxfp8()
            out = mxfp8_dequant(packed, scale, alpha)
        else:
            raise ValueError(f"{self.simulation} is not supported")

        return out[:, :k].to(self.theta.dtype)

    def proxy(self) -> torch.Tensor | None:
        return self.theta


class SteQuant(Quant):
    def __init__(self, simulation: Simulation | None):
        super().__init__()
        self.simulation = simulation

    def forward(self, x: torch.Tensor) -> QTensor:
        return SteTensor(theta=x, simulation=self.simulation)
