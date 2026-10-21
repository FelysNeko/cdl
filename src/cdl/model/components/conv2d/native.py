import torch
import torch.nn.functional as F
from torch import nn


class NativeQuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        activation_quant: nn.Module,
        weight_quant: nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            self.activation_quant(x),
            self.weight_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
