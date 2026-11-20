import torch
import triton
import triton.language as tl

from cdl.core.kernel.formats.ordinals import (
    e2m1_triton_ordinal,
    ue4m3_triton_ordinal,
)

FP32_TINY = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def _quant_kernel(
    x_ptr,
    gmax_ptr,
    alpha_ptr,
    packed_ptr,
    scale_ptr,
    K,
    NB,
    PACK_STRIDE,
    TPB: tl.constexpr,
    BB: tl.constexpr,
    IS_FP4: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
    EMAX: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    gb = tl.program_id(1)
    offs_b = gb * BB + tl.arange(0, BB)
    bmask = offs_b < NB
    cols = offs_b[:, None] * TPB + tl.arange(0, TPB)[None, :]
    raw = tl.load(x_ptr + r * K + cols, mask=bmask[:, None] & (cols < K), other=0.0).to(
        tl.float32
    )

    block_absmax = tl.max(tl.abs(raw), axis=1)
    if SCALE_IS_E8M0:
        # OCP MX Spec Section 6.3: shared exponent = floor(log2(block_amax)) - emax.
        bits = tl.cast(tl.maximum(block_absmax, FP32_TINY), tl.uint32, bitcast=True)
        floor_log2 = (bits >> 23).to(tl.int32) - 127
        q_byte = tl.minimum(tl.maximum(floor_log2 - EMAX + 127, 0), 254).to(tl.uint8)
        u = q_byte.to(tl.uint32)
        # Byte 0 is IEEE fp32 subnormal 2**-127 (avoiding FTZ flush from exp2).
        s = tl.cast(tl.where(u == 0, 0x00400000, u << 23), tl.float32, bitcast=True)
    else:
        qv = tl.div_rn(block_absmax * 448.0, tl.load(gmax_ptr).to(tl.float32))
        q_byte = (ue4m3_triton_ordinal(qv, 126) + 1).to(tl.uint8)
        q_hat = tl.cast(q_byte, tl.float8e4nv, bitcast=True).to(tl.float32)
        s = q_hat * tl.load(alpha_ptr).to(tl.float32)

    tl.store(scale_ptr + r * NB + offs_b, q_byte, mask=bmask)

    v = tl.div_rn(raw, s[:, None])
    if IS_FP4:
        ti = e2m1_triton_ordinal(v, 16)
        code = tl.where(ti < 8, 15 - ti, ti - 8)
        lo, hi = tl.split(tl.reshape(code, (BB, TPB // 2, 2)))
        pcols = offs_b[:, None] * (TPB // 2) + tl.arange(0, TPB // 2)[None, :]
        tl.store(
            packed_ptr + r * PACK_STRIDE + pcols,
            (lo | (hi << 4)).to(tl.uint8),
            mask=bmask[:, None],
        )
    else:
        packed_fp8 = tl.cast(tl.minimum(tl.maximum(v, -448.0), 448.0), tl.float8e4nv)
        tl.store(packed_ptr + r * PACK_STRIDE + cols, packed_fp8, mask=bmask[:, None])


def _quant(
    x: torch.Tensor,
    group_size: int,
    is_fp4: bool,
    scale_is_e8m0: bool,
    emax: int,
    block_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor [rows, k], got shape {tuple(x.shape)}")
    x = x.contiguous()
    rows, k = x.shape
    nb = triton.cdiv(k, group_size)
    pack_stride = nb * (group_size // 2 if is_fp4 else group_size)
    packed_dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn

    packed = torch.empty((rows, pack_stride), dtype=packed_dtype, device=x.device)
    scale = torch.empty((rows, nb), dtype=torch.uint8, device=x.device)

    tiny = torch.finfo(torch.float32).tiny
    if scale_is_e8m0:
        alpha = torch.ones(1, dtype=torch.float32, device=x.device) if is_fp4 else None
        global_absmax = x
        alpha_ptr = x
    else:
        global_absmax = (
            x.abs().amax().float().clamp_min(tiny)
            if rows > 0 and k > 0
            else torch.full((), tiny, dtype=torch.float32, device=x.device)
        )
        alpha = (global_absmax / (448.0 * 6.0)).reshape(1)
        alpha_ptr = alpha

    if rows == 0 or k == 0:
        return packed, scale, alpha

    _quant_kernel[(rows, triton.cdiv(nb, block_chunk))](
        x,
        global_absmax,
        alpha_ptr,
        packed,
        scale,
        k,
        nb,
        pack_stride,
        TPB=group_size,
        BB=block_chunk,
        IS_FP4=is_fp4,
        SCALE_IS_E8M0=scale_is_e8m0,
        EMAX=emax,
        num_warps=4,
    )
    return packed, scale, alpha


def nvfp4_quant(
    x: torch.Tensor, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 16, True, False, 2, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp4_quant(
    x: torch.Tensor, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 32, True, True, 2, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp8_quant(
    x: torch.Tensor, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, None]:
    packed, scale, _ = _quant(x, 32, False, True, 8, block_chunk)
    return packed, scale, None
