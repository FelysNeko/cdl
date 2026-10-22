from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import Quant


class NativeQuantizedLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Literal["native", "fp4", "fp8"] = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation_quant(x)
        weight = self.weight_quant(self.weight)
        if self.backend != "native":
            raise NotImplementedError(
                f"{self.backend} quantized linear is not implemented"
            )
        return F.linear(
            activation.dequantize(),
            weight.dequantize(),
            self.bias,
        )
