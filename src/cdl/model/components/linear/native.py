import torch
import torch.nn.functional as F
from torch import nn


class NativeQuantizedLinear(nn.Linear):
    def __init__(self, *args, quant: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quant(self.weight), self.bias)
