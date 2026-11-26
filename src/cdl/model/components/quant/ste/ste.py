from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from cdl.api.quant import QTensor, Quant
from cdl.core.kernel.dequant import mxfp4_dequant, mxfp8_dequant, nvfp4_dequant
from cdl.core.kernel.quant import mxfp4_quant, mxfp8_quant, nvfp4_quant

type Simulation = Literal["nvfp4", "mxfp4", "mxfp8"]
type QuantFn = Callable[
    [torch.Tensor, torch.Tensor | None, bool],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
]


@dataclass
class SteTensor(QTensor):
    theta: torch.Tensor
    simulation: Simulation | None
    block_2d: bool

    @property
    def shape(self) -> Sequence[int]:
        return self.theta.shape

    def apply_quantize_fn(
        self, quant_fn: QuantFn
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            if self.block_2d:
                assert self.theta.ndim == 2
                return quant_fn(self.theta, None, True)
            lead = self.theta.shape[:-1]
            k = self.theta.shape[-1]
            packed, scale, alpha = quant_fn(self.theta.reshape(-1, k), None, False)
            packed = packed.reshape(*lead, packed.shape[-1])
            scale = scale.reshape(*lead, scale.shape[-1])
            return packed, scale, alpha

    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, scale, alpha = self.apply_quantize_fn(nvfp4_quant)
        assert alpha is not None
        return packed, scale, alpha

    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed, scale, alpha = self.apply_quantize_fn(mxfp4_quant)
        assert alpha is not None
        return packed, scale, alpha

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.apply_quantize_fn(mxfp8_quant)

    def dequantize(self) -> torch.Tensor:
        if self.simulation is None:
            return self.theta

        if self.simulation == "nvfp4":
            packed, scale, alpha = self.nvfp4()
            dequant_fn = nvfp4_dequant
        elif self.simulation == "mxfp4":
            packed, scale, alpha = self.mxfp4()
            dequant_fn = mxfp4_dequant
        elif self.simulation == "mxfp8":
            packed, scale, alpha = self.mxfp8()
            dequant_fn = mxfp8_dequant
        else:
            raise ValueError(f"{self.simulation} is not supported")

        lead = packed.shape[:-1]
        out = dequant_fn(
            packed.reshape(-1, packed.shape[-1]),
            scale.reshape(-1, scale.shape[-1]),
            alpha,
        ).reshape(*lead, -1)
        if tuple(out.shape) != tuple(self.theta.shape):
            out = out[tuple(slice(None, d) for d in self.theta.shape)]
        return out.to(self.theta.dtype)

    def proxy(self) -> torch.Tensor | None:
        return self.theta

    def source(self) -> torch.Tensor:
        return self.theta


class SteQuant(Quant):
    def __init__(self, simulation: Simulation | None, block_2d: bool):
        super().__init__()
        self.simulation = simulation
        self.block_2d = block_2d

    def forward(self, x: torch.Tensor) -> QTensor:
        return SteTensor(theta=x, simulation=self.simulation, block_2d=self.block_2d)
