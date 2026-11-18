import torch
import triton
import triton.language as tl


@triton.jit
def _e8m0_from_byte(sb):
    # Byte 0 is IEEE fp32 subnormal 2**-127 (tl.exp2(-127) flushes to 0 under FTZ).
    u = sb.to(tl.uint32)
    return tl.cast(tl.where(u == 0, 0x00400000, u << 23), tl.float32, bitcast=True)


@triton.jit
def _e4m3_from_byte(b):
    return tl.cast(b.to(tl.uint8), tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _e2m1_from_nib(nib):
    m = (nib & 7).to(tl.float32)
    mag = tl.where(m < 4.0, m * 0.5, tl.where(m < 7.0, m - 2.0, 6.0))
    return mag * tl.where(nib >= 8, -1.0, 1.0)


@triton.jit
def _fp4_dequant_kernel(
    packed_ptr,
    scale_ptr,
    alpha_ptr,
    out_ptr,
    K,
    NB,
    GROUP_SIZE: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs < K
    nbytes = tl.load(packed_ptr + row * (K // 2) + offs // 2, mask=kmask, other=0)
    nib = (nbytes >> ((offs % 2) * 4)) & 0xF
    val = _e2m1_from_nib(nib)
    sb = tl.load(scale_ptr + row * NB + offs // GROUP_SIZE, mask=kmask, other=0)
    sc = _e8m0_from_byte(sb) if SCALE_IS_E8M0 else _e4m3_from_byte(sb)
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
    HAS_ALPHA: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    kb = tl.program_id(1)
    offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = offs < K
    b = tl.load(packed_ptr + row * K + offs, mask=kmask, other=0)
    val = _e4m3_from_byte(b)
    sb = tl.load(scale_ptr + row * NB + offs // 32, mask=kmask, other=0)
    sc = _e8m0_from_byte(sb)
    out = val * sc
    if HAS_ALPHA:
        out = out * tl.load(alpha_ptr)
    tl.store(out_ptr + row * K + offs, out.to(tl.bfloat16), mask=kmask)


def _fp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    group_size: int,
    scale_is_e8m0: bool,
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
        GROUP_SIZE=group_size,
        SCALE_IS_E8M0=scale_is_e8m0,
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
    return _fp4_dequant(packed, scale, alpha, 16, False, block_k)


def mxfp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    block_k: int = 1024,
) -> torch.Tensor:
    return _fp4_dequant(packed, scale, alpha, 32, True, block_k)


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
        HAS_ALPHA=has_alpha,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
