import torch
import torch.nn.functional as F
from torch import nn


class NativeQuantizedLinear(nn.Linear):
    def __init__(self, *args, quant: nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.quant = quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quant(self.weight), self.bias)
