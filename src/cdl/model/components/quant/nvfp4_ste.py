from dataclasses import dataclass

import flashinfer
import torch
import torch.nn.functional as F

from cdl.api.quant import Quant
from cdl.api.tensor import QuantizedTensor


@torch.library.custom_op("cdl::nvfp4_quantize", mutates_args=())
def nvfp4_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, k] with k % 32 == 0 (mm_fp4 TMA alignment), bfloat16, row-major.
    # Returns (packed fp4 [rows, k / 2] uint8, block scales [rows, k / 16] uint8)
    # in the linear (non-swizzled) scale layout.
    packed, scale = flashinfer.fp4_quantize(
        x, global_scale, is_sf_swizzled_layout=False
    )
    return packed, scale


@nvfp4_quantize.register_fake
def _(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = x.shape
    return (
        torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device),
        torch.empty(rows, k // 16, dtype=torch.uint8, device=x.device),
    )


@torch.library.custom_op("cdl::nvfp4_dequantize", mutates_args=())
def nvfp4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> torch.Tensor:
    # packed: [rows, k / 2] uint8, scale: [rows, k / 16] uint8 (linear layout).
    # Returns the dequantized bfloat16 tensor of shape [rows, k].
    return flashinfer.nvfp4_kv_dequantize(
        packed, scale, global_scale_inv, output_dtype=torch.bfloat16
    )


@nvfp4_dequantize.register_fake
def _(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> torch.Tensor:
    rows = packed.shape[0]
    k = packed.shape[-1] * 2
    return torch.empty(rows, k, dtype=torch.bfloat16, device=packed.device)


@torch.no_grad()
def nvfp4_pack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Quantizes a [rows, k] tensor to the NVFP4 kernel layout: pads k to a
    # multiple of 32 (mm_fp4 TMA alignment), derives the per-tensor global
    # scale from the amax, and returns (packed fp4 [rows, k' / 2] uint8,
    # linear block scales [rows, k' / 16] uint8, global_scale_inv [1]).
    k = x.shape[-1]
    flat = x.bfloat16()
    r = -k % 32
    if r:
        flat = F.pad(flat, (0, r))

    amax = flat.float().abs().amax()
    global_scale = torch.where(
        amax > 0,
        amax.new_full((), 448.0 * 6.0) / amax,
        amax.new_full((), torch.finfo(torch.float32).max),
    ).reshape(1)
    packed, scale = nvfp4_quantize(flat, global_scale)
    return packed, scale, (1.0 / global_scale).float()


@dataclass
class Nvfp4SteQuantizedTensor(QuantizedTensor):
    packed: torch.Tensor
    scale: torch.Tensor
    global_scale_inv: torch.Tensor
    grad_proxy: torch.Tensor
    k: int
    dtype: torch.dtype

    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.packed, self.scale, self.global_scale_inv

    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Nvfp4SteQuantizedTensor has no fp8 representation")

    def dequantize(self) -> torch.Tensor:
        dense = nvfp4_dequantize(self.packed, self.scale, self.global_scale_inv)
        dense = dense[..., : self.k].to(self.dtype)
        return self.grad_proxy + (dense - self.grad_proxy).detach()

    def proxy(self) -> torch.Tensor:
        return self.grad_proxy


class Nvfp4SteQuant(Quant):
    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        if input.ndim != 2:
            raise ValueError(
                f"Nvfp4SteQuant expects a 2-D [rows, k] tensor, got ndim={input.ndim}"
            )

        packed, scale, global_scale_inv = nvfp4_pack(input)
        return Nvfp4SteQuantizedTensor(
            packed=packed,
            scale=scale,
            global_scale_inv=global_scale_inv,
            grad_proxy=input,
            k=input.shape[-1],
            dtype=input.dtype,
        )
