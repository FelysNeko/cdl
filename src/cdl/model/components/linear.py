from typing import Literal

import torch
from torch import nn

from cdl.api.quant import MatMul, Quant
from cdl.core.mm import mxfp4, mxfp8, native, nvfp4

type Backend = Literal["native", "nvfp4", "mxfp4", "mxfp8"]


class QLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Backend,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.backend = backend

    @property
    def available_backend_fn(self) -> dict[str, MatMul]:
        return {
            "native": native.matmul,
            "nvfp4": nvfp4.matmul,
            "mxfp4": mxfp4.matmul,
            "mxfp8": mxfp8.matmul,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.activation_quant(x)
        weight = self.weight_quant(self.weight)

        out = self.available_backend_fn[self.backend](activation, weight)
        if self.bias is not None:
            out = out + self.bias

        return out


class QLinearMixin(nn.Module):
    def switch_backend(self, backend: Backend) -> None:
        for m in self.modules():
            if isinstance(m, QLinear):
                m.backend = backend
