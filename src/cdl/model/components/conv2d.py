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
        im2col: bool = False,
        backend: Literal["native", "fp4", "fp8"] = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if im2col and self.groups != 1:
            raise ValueError("im2col convolution only supports groups=1")
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.im2col = im2col
        self.backend = backend

    def im2col_for_activation(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            x,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )
        return patches.transpose(1, 2).reshape(-1, patches.shape[1])

    @property
    def im2col_weight(self) -> torch.Tensor:
        return self.weight.flatten(1)

    def im2col_reshape_output(
        self, output: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = x.shape
        out_height = (
            height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        out_width = (
            width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        return output.reshape(batch, out_height, out_width, self.out_channels).permute(
            0, 3, 1, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend != "native":
            raise NotImplementedError(
                f"{self.backend} quantized convolution is not implemented"
            )

        if not self.im2col:
            activation = self.activation_quant(x)
            weight = self.weight_quant(self.weight)
            return F.conv2d(
                activation.dequantize(),
                weight.dequantize(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"convolution input must be 3-D or 4-D, got {x.ndim}-D")

        im2col_activation = self.im2col_for_activation(x)
        activation = self.activation_quant(im2col_activation).dequantize()
        weight = self.weight_quant(self.im2col_weight).dequantize()

        output = activation @ weight.T
        if self.bias is not None:
            output = output + self.bias

        output = self.im2col_reshape_output(output, x)
        return output.squeeze(0) if unbatched else output
