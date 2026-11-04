from typing import Literal

import torch
from torch import nn

from cdl.api.quant import MatMul, Quant
from cdl.core.mm import native


class QLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Literal["native", "nvfp4", "mxfp4", "mxfp8"],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.backend = backend

    @property
    def available_backend_fn(self) -> dict[str, MatMul]:
        return {"native": native.matmul}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation_quant(x)
        weight = self.weight_quant(self.weight)

        out = self.available_backend_fn[self.backend](activation, weight)

        if self.bias is not None:
            out = out + self.bias

        return out
