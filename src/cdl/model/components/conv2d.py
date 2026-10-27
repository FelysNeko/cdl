from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import Quant
from cdl.core.kernel.matmul import fused_cdl_matmul, gemm_mode, layer_pads
from cdl.core.matmul import matmul

Backend = Literal["native", "fp4", "fused"]


class NativeQuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        im2col: bool = False,
        backend: Backend = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if im2col and self.groups != 1:
            raise ValueError("im2col convolution only supports groups=1")
        if backend in ("fp4", "fused") and not im2col:
            raise ValueError(f"{backend} backend requires im2col=True")
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

    def _pad_gemm_layout(self, im2col_x: torch.Tensor, w: torch.Tensor):
        """Pad im2col x/w to the widths the selected GEMM layout requires.

        `w` must already be the flattened [out, kdim] im2col weight.
        """
        kpad, npad = layer_pads(self.activation_quant, w.shape[1], self.weight.shape[0])
        if kpad:
            im2col_x = F.pad(im2col_x, (0, kpad))
        if kpad or npad:
            w = F.pad(w, (0, kpad, 0, npad))
        return im2col_x, w

    def fused_im2col_forward(self, im2col_x: torch.Tensor) -> torch.Tensor:
        quant_a, quant_w = self.activation_quant, self.weight_quant
        mode = gemm_mode(quant_a)
        if mode is None or gemm_mode(quant_w) != mode:
            raise NotImplementedError(
                "the fused backend requires both quantizers on the same "
                "supported format pair (nvfp4, mxfp4 or mxfp8)"
            )
        im2col_x, w = self._pad_gemm_layout(im2col_x, self.im2col_weight)
        out = fused_cdl_matmul(
            im2col_x.contiguous(), quant_a, w, quant_w, quant_a.kernel_seed
        )
        return out[:, : self.weight.shape[0]]

    def im2col_forward(self, im2col_x: torch.Tensor) -> torch.Tensor:
        n, k_raw = self.weight.shape[0], im2col_x.shape[1]
        im2col_x, w = self._pad_gemm_layout(im2col_x, self.im2col_weight)

        activation = self.activation_quant(im2col_x)
        weight = self.weight_quant(w)

        if self.backend == "native":
            dense_w = weight.dequantize()[:n, :k_raw]
            return activation.dequantize()[:, :k_raw] @ dense_w.T

        if self.backend in ("fp4", "fused"):
            # Eval-mode fused falls back to the deterministic nearest-round
            # path produced by the quantizers above.
            return matmul(activation, weight)[:, :n]

        raise NotImplementedError(
            f"{self.backend} quantized convolution is not implemented"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.im2col:
            if self.backend != "native":
                raise NotImplementedError(
                    f"{self.backend} quantized convolution is not implemented"
                )
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

        if self.backend == "fused" and self.training:
            output = self.fused_im2col_forward(im2col_activation)
        else:
            output = self.im2col_forward(im2col_activation)

        if self.bias is not None:
            output = output + self.bias

        output = self.im2col_reshape_output(output, x)
        return output.squeeze(0) if unbatched else output
