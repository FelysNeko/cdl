import torch
import torch.nn.functional as F
from torch import nn


class NativeQuantizedLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: nn.Module,
        weight_quant: nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            self.activation_quant(x),
            self.weight_quant(self.weight),
            self.bias,
        )
