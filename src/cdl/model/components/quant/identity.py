from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cdl.api.quant import QTensor, Quant


@dataclass
class IdentityTensor(QTensor):
    theta: torch.Tensor

    @property
    def shape(self) -> Sequence[int]:
        return self.theta.shape

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None


class IdentityQuant(Quant):
    def forward(self, x: torch.Tensor) -> IdentityTensor:
        return IdentityTensor(theta=x)
