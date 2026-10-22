from dataclasses import dataclass

import torch

from cdl.api.quant import Quant
from cdl.api.tensor import QuantizedTensor


@dataclass
class DenseQuantizedTensor(QuantizedTensor):
    theta: torch.Tensor

    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "DenseQuantizedTensor has no FP4 indices or scales, so converting "
            "it to FP4 is lossy and has no gradient"
        )

    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "DenseQuantizedTensor has no FP8 indices or scales, so converting "
            "it to FP8 is lossy and has no gradient"
        )

    def dequantize(self) -> torch.Tensor:
        return self.theta


class IdentityQuant(Quant):
    def forward(self, x: torch.Tensor) -> QuantizedTensor:
        return DenseQuantizedTensor(x)
