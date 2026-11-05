from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import MatMul, Quant
from cdl.core.mm import mxfp4, mxfp8, native, nvfp4

type Backend = Literal["native", "nvfp4", "mxfp4", "mxfp8"]


class QConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Backend,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.groups != 1:
            raise ValueError("QConv2d only supports groups=1")
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
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
        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"convolution input must be 3-D or 4-D, got {x.ndim}-D")

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
        return out.squeeze(0) if unbatched else out
