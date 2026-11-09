from typing import Any, Literal

import torch
from torch import nn

from cdl.api.quant import MatMul, Quant
from cdl.core.mm import mxfp4, mxfp8, native, nvfp4
from cdl.model.components.quant.ste import SteQuant

type Backend = Literal["native", "nvfp4", "mxfp4", "mxfp8"]


class QLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation_quant = SteQuant(None)
        self.weight_quant = SteQuant(None)
        self.backend = "native"

    def set_activation_quant(self, quant: Quant) -> None:
        self.activation_quant = quant

    def set_weight_quant(self, quant: Quant) -> None:
        self.weight_quant = quant

    def set_backend(self, backend: Backend) -> None:
        self.backend = backend

    @property
    def available_backend_fn(self) -> dict[str, MatMul]:
        return {
            "native": native.matmul,
            "nvfp4": nvfp4.matmul,
            "mxfp4": mxfp4.matmul,
            "mxfp8": mxfp8.matmul,
        }

    @staticmethod
    def from_linear(linear: nn.Linear) -> "QLinear":
        new = QLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        new.weight.copy_(linear.weight)
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.shape[-1])
        activation = self.activation_quant(x2d)
        weight = self.weight_quant(self.weight)

        out = self.available_backend_fn[self.backend](activation, weight)
        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*x.shape[:-1], -1)


class QLinearMixin(nn.Module):
    def q_linear_switch_backend(self, backend: Backend) -> None:
        for m in self.modules():
            if isinstance(m, QLinear):
                m.set_backend(backend)
