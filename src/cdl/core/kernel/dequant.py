import functools

import torch
import triton
import triton.language as tl


@functools.cache
def _e8m0_lut(device: torch.device) -> torch.Tensor:
    return torch.exp2(torch.arange(256, dtype=torch.float32, device=device) - 127.0)


@functools.cache
def _e2m1_mag_lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=device,
    )


@functools.cache
def _e4m3_lut(device: torch.device) -> torch.Tensor:
    return (
        torch.arange(256, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .float()
    )


@triton.jit
def _fp4_dequant_kernel(
    packed_ptr,
    scale_ptr,
    alpha_ptr,
    out_ptr,
    K,
    NB,
    mag_ptr,
    sc_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs < K
    nbytes = tl.load(packed_ptr + row * (K // 2) + offs // 2, mask=kmask, other=0)
    nib = (nbytes >> ((offs % 2) * 4)) & 0xF
    mag = tl.load(mag_ptr + (nib & 7))
    val = mag * tl.where(nib >= 8, -1.0, 1.0)
    sb = tl.load(scale_ptr + row * NB + offs // GROUP_SIZE, mask=kmask, other=0)
    sc = tl.load(sc_ptr + sb)
    alpha = tl.load(alpha_ptr)
    tl.store(out_ptr + row * K + offs, (val * sc * alpha).to(tl.bfloat16), mask=kmask)


@triton.jit
def _mxfp8_dequant_kernel(
    packed_ptr,
    scale_ptr,
    alpha_ptr,
    out_ptr,
    K,
    NB,
    val_ptr,
    sc_ptr,
    HAS_ALPHA: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs < K
    b = tl.load(packed_ptr + row * K + offs, mask=kmask, other=0)
    val = tl.load(val_ptr + b)
    sb = tl.load(scale_ptr + row * NB + offs // 32, mask=kmask, other=0)
    sc = tl.load(sc_ptr + sb)
    out = val * sc
    if HAS_ALPHA:
        out = out * tl.load(alpha_ptr)
    tl.store(out_ptr + row * K + offs, out.to(tl.bfloat16), mask=kmask)


def _fp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    group_size: int,
    sc_lut: torch.Tensor,
    block_k: int = 1024,
) -> torch.Tensor:
    rows, half_k = packed.shape
    k = half_k * 2
    out = torch.empty((rows, k), dtype=torch.bfloat16, device=packed.device)
    if rows == 0 or k == 0:
        return out
    _fp4_dequant_kernel[(rows, triton.cdiv(k, block_k))](
        packed,
        scale,
        alpha,
        out,
        k,
        k // group_size,
        _e2m1_mag_lut(packed.device),
        sc_lut,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def nvfp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    block_k: int = 1024,
) -> torch.Tensor:
    return _fp4_dequant(packed, scale, alpha, 16, _e4m3_lut(packed.device), block_k)


def mxfp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    block_k: int = 1024,
) -> torch.Tensor:
    return _fp4_dequant(packed, scale, alpha, 32, _e8m0_lut(packed.device), block_k)


def mxfp8_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor | None = None,
    block_k: int = 1024,
) -> torch.Tensor:
    rows, k = packed.shape
    out = torch.empty((rows, k), dtype=torch.bfloat16, device=packed.device)
    if rows == 0 or k == 0:
        return out
    has_alpha = alpha is not None
    _mxfp8_dequant_kernel[(rows, triton.cdiv(k, block_k))](
        packed.view(torch.uint8),
        scale,
        alpha if has_alpha else scale,
        out,
        k,
        k // 32,
        _e4m3_lut(packed.device),
        _e8m0_lut(packed.device),
        HAS_ALPHA=has_alpha,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
