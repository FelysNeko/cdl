from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import Quant


class NativeQuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Literal["native", "fp4", "fp8"] = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation_quant(x)
        weight = self.weight_quant(self.weight)
        if self.backend != "native":
            raise NotImplementedError(
                f"{self.backend} quantized convolution is not implemented"
            )
        return F.conv2d(
            activation.dequantize(),
            weight.dequantize(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
