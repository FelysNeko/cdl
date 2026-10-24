from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.quant import Quant
from cdl.core.kernel.matmul import fusable_quant, fused_cdl_matmul
from cdl.core.matmul import matmul

Backend = Literal["native", "fp4", "fp8", "fused"]


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
        if not (fusable_quant(quant_a) and fusable_quant(quant_w)):
            raise NotImplementedError(
                "the fused backend requires cascaded CDL quantizers on the "
                "E2M1/UE4M3ExcludeZero formats with block size 16"
            )
        if x.ndim != 2:
            raise NotImplementedError(
                f"the fused backend expects a 2-D input, got {x.ndim}-D"
            )
        if x.shape[1] % 16 or self.weight.shape[1] % 16:
            raise NotImplementedError(
                "the fused backend requires the contraction dimension to be "
                f"a multiple of 16, got {x.shape[1]}"
            )
        out = fused_cdl_matmul(
            x.contiguous(), quant_a, self.weight, quant_w, quant_a.kernel_seed
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "fused" and self.training:
            return self.fused_forward(x)

        activation = self.activation_quant(x)
        weight = self.weight_quant(self.weight)

        if self.backend == "native":
            return F.linear(
                activation.dequantize(),
                weight.dequantize(),
                self.bias,
            )

        if self.backend in ("fp4", "fused"):
            # Eval-mode fused falls back to the deterministic nearest-round
            # path produced by the quantizers above.
            out = matmul(activation, weight)
            if self.bias is not None:
                out = out + self.bias
            return out

        raise NotImplementedError(f"{self.backend} quantized linear is not implemented")
