from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import MatMul, Quant
from cdl.core.mm import mxfp4, mxfp8, native, nvfp4
from cdl.model.components.quant.ste import SteQuant

type Backend = Literal["native", "nvfp4", "mxfp4", "mxfp8"]


class QConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        patches = F.unfold(
            x, self.kernel_size, self.dilation, self.padding, self.stride
        )
        rows = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        activation = self.activation_quant(rows)
        weight = self.weight_quant(self.weight.flatten(1))

        out = self.available_backend_fn[self.backend](activation, weight)
        if self.bias is not None:
            out = out + self.bias

        oh = (
            height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        ow = (
            width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        out = out.reshape(batch, oh, ow, self.out_channels).permute(0, 3, 1, 2)
        return out


class QConv2dMixin(nn.Module):
    def q_conv2d_switch_backend(self, backend: Backend) -> None:
        for m in self.modules():
            if isinstance(m, QConv2d):
                m.set_backend(backend)
