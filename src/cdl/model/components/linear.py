from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import Quant
from cdl.core.kernel.matmul import fused_cdl_matmul, gemm_mode, layer_pads
from cdl.core.matmul import matmul

Backend = Literal["native", "fp4", "fused"]


class NativeQuantizedLinear(nn.Linear):
    def __init__(
        self,
        *args,
        activation_quant: Quant,
        weight_quant: Quant,
        backend: Backend = "native",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation_quant = activation_quant
        self.weight_quant = weight_quant
        self.backend = backend

    def fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        quant_a, quant_w = self.activation_quant, self.weight_quant
        mode = gemm_mode(quant_a)
        if mode is None or gemm_mode(quant_w) != mode:
            raise NotImplementedError(
                "the fused backend requires both quantizers on the same "
                "supported format pair (nvfp4, mxfp4 or mxfp8)"
            )
        if x.ndim != 2:
            raise NotImplementedError(
                f"the fused backend expects a 2-D input, got {x.ndim}-D"
            )
        x, w = self._pad_gemm_layout(x, self.weight)
        out = fused_cdl_matmul(x.contiguous(), quant_a, w, quant_w, quant_a.kernel_seed)
        out = out[:, : self.weight.shape[0]]
        if self.bias is not None:
            out = out + self.bias
        return out

    def _pad_gemm_layout(self, x: torch.Tensor, w: torch.Tensor):
        """Pad x/w to the widths the selected GEMM layout requires."""
        kpad, npad = layer_pads(
            self.activation_quant, self.weight.shape[1], self.weight.shape[0]
        )
        if kpad:
            x = F.pad(x, (0, kpad))
        if kpad or npad:
            w = F.pad(w, (0, kpad, 0, npad))
        return x, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "fused" and self.training:
            return self.fused_forward(x)

        n, k_raw = self.weight.shape[0], x.shape[1]
        x, w = self._pad_gemm_layout(x, self.weight)

        activation = self.activation_quant(x)
        weight = self.weight_quant(w)

        if self.backend == "native":
            dense_w = weight.dequantize()[:n, :k_raw]
            return F.linear(activation.dequantize()[:, :k_raw], dense_w, self.bias)

        if self.backend in ("fp4", "fused"):
            # Eval-mode fused falls back to the deterministic nearest-round
            # path produced by the quantizers above.
            out = matmul(activation, weight)[:, :n]
            if self.bias is not None:
                out = out + self.bias
            return out

        raise NotImplementedError(f"{self.backend} quantized linear is not implemented")
