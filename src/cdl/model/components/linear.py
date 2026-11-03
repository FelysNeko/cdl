from typing import Literal

import torch
from torch import nn

from cdl.api.quant import Quant


class QLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        mm: Literal["native", "nvfp4", "mxfp4", "mxfp8"] = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.mm = mm

    @property
    def available_backend_fn(self) -> dict[str]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation_quant.forward(x)
        weight = self.weight_quant.forward(x)

        out = self.available_backend_fn[self.backend](activation, weight)

        if self.bias:
            out = out + self.bias

        return out
