from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cdl.api.quant import Quant
from cdl.api.tensor import QuantizedTensor
from cdl.core.format.fp8 import E4M3, E8M0
from cdl.core.matmul import mxfp4_dequant, mxfp8_dequant, unswizzle_nvfp4_scale_factors
from cdl.model.wrapper.flashinfer import (
    mxfp4_quantize,
    mxfp8_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
)


def _formats(name: str):
    from cdl.core.format.fp4 import E2M1
    from cdl.core.format.fp8 import UE4M3ExcludeZero

    if name == "nvfp4":
        return E2M1, UE4M3ExcludeZero, 16
    if name == "mxfp4":
        return E2M1, E8M0, 32
    if name == "mxfp8":
        return E4M3, E8M0, 32
    raise ValueError(f"unknown STE format {name!r}; expected nvfp4, mxfp4 or mxfp8")


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
class SteQuantizedTensor(QuantizedTensor):
    packed: torch.Tensor
    scale: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor
    k: int
    dtype: torch.dtype
    theta_format: object
    q_format: object
    block_size: int

    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.block_size == 16:
            return self.packed, self.scale, self.alpha
        packed = self.packed
        scale = self.scale
        pad = -(packed.shape[-1] * 2) % 32
        if pad:
            packed = F.pad(packed, (0, pad // 2))
            scale = F.pad(scale, (0, pad // 32))
        return packed, scale, self.alpha.reshape(1)

    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.packed, self.scale

    def dequantize(self) -> torch.Tensor:
        if self.block_size == 16:
            dense = nvfp4_dequantize(self.packed, self.scale, self.alpha)
        elif self.theta_format.bits == 8:
            dense = mxfp8_dequant(self.packed, self.scale)
        else:
            dense = mxfp4_dequant(self.packed, self.scale, self.alpha)
        dense = dense[..., : self.k].to(self.dtype)
        return self.grad_proxy + (dense - self.grad_proxy).detach()

    def proxy(self) -> torch.Tensor:
        return self.grad_proxy


class Nvfp4SteQuant(Quant):
    """Straight-through-estimator quantizer for the block-scaled formats.

    `format_name` selects nvfp4 (E2M1/UE4M3 scales, block 16, global scale),
    mxfp4 (E2M1/E8M0, block 32) or mxfp8 (E4M3/E8M0, block 32). Per-block
    scales are derived from the block amax (spec semantics, no CDL).
    """

    def __init__(self, format_name: str = "nvfp4") -> None:
        super().__init__()
        self.format_name = format_name
        theta_format, q_format, block = _formats(format_name)
        self.theta_format = theta_format(torch.float32)
        self.q_format = q_format(torch.float32)
        self.block_size = block

    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        if input.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} expects a 2-D [rows, k] tensor, got "
                f"ndim={input.ndim}"
            )
        if self.format_name == "nvfp4":
            packed, scale, alpha = nvfp4_pack(input)
        elif self.format_name == "mxfp4":
            k = input.shape[-1]
            flat = input.bfloat16()
            r = -k % 32
            if r:
                flat = F.pad(flat, (0, r))
            packed, swizzled = mxfp4_quantize(flat)
            scale = unswizzle_nvfp4_scale_factors(swizzled)[
                : flat.shape[0], : flat.shape[-1] // 32
            ]
            alpha = torch.ones(1, device=input.device, dtype=torch.float32)
        else:
            k = input.shape[-1]
            flat = input.bfloat16()
            r = -k % 128  # b12x mm_mxfp8 requires K % 128
            if r:
                flat = F.pad(flat, (0, r))
            packed, scale = mxfp8_quantize(flat)
            alpha = torch.ones(1, device=input.device, dtype=torch.float32)
        return SteQuantizedTensor(
            packed=packed,
            scale=scale,
            alpha=alpha,
            grad_proxy=input,
            k=input.shape[-1],
            dtype=input.dtype,
            theta_format=self.theta_format,
            q_format=self.q_format,
            block_size=self.block_size,
        )
