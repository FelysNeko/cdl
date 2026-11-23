import torch
import triton
import triton.language as tl

from cdl.core.kernel.formats.ordinals import (
    e2m1_triton_ordinal,
    ue4m3_triton_ordinal,
)

FP32_TINY = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def _e8m0_scale(block_absmax, EMAX: tl.constexpr):
    # OCP MX Spec Section 6.3: shared exponent = floor(log2(block_amax)) - emax.
    bits = tl.cast(tl.maximum(block_absmax, FP32_TINY), tl.uint32, bitcast=True)
    floor_log2 = (bits >> 23).to(tl.int32) - 127
    q_byte = tl.minimum(tl.maximum(floor_log2 - EMAX + 127, 0), 254).to(tl.uint8)
    u = q_byte.to(tl.uint32)
    # Byte 0 is IEEE fp32 subnormal 2**-127 (avoiding FTZ flush from exp2).
    s = tl.cast(tl.where(u == 0, 0x00400000, u << 23), tl.float32, bitcast=True)
    return q_byte, s


@triton.jit
def _fp4_stochastic_round(v, seed, offset):
    u = tl.rand(seed, offset)
    sgn = (tl.cast(v, tl.uint32, bitcast=True) >> 31) != 0
    av = tl.minimum(tl.abs(v), 6.0)
    scaled = tl.where(av < 2.0, av * 2.0, av)
    lo_f = tl.floor(scaled)
    lo_mag = tl.where(
        av < 2.0,
        lo_f.to(tl.int32),
        tl.where(av < 4.0, lo_f.to(tl.int32) + 2, tl.where(av < 6.0, 6, 7)),
    )
    frac = tl.where(
        av < 2.0,
        scaled - lo_f,
        tl.where(av < 4.0, av - lo_f, tl.where(av < 6.0, (av - 4.0) * 0.5, 0.0)),
    )
    mag = tl.minimum(lo_mag + (u < frac).to(tl.int32), 7)
    return tl.where(sgn, 8 + mag, mag)


@triton.jit
def _fp8_stochastic_round(v, seed, offset):
    u = tl.rand(seed, offset)
    sgn = (tl.cast(v, tl.uint32, bitcast=True) >> 31) != 0
    av = tl.minimum(tl.abs(v), 448.0)
    is_sub = av < 0.015625
    sub_lo = tl.floor(av * 512.0) * (1.0 / 512.0)
    sub_hi = sub_lo + (1.0 / 512.0)
    bits = tl.cast(tl.maximum(av, 0.015625), tl.uint32, bitcast=True)
    norm_lo_bits = bits & 0xFFF00000
    norm_lo = tl.cast(norm_lo_bits, tl.float32, bitcast=True)
    norm_hi = tl.minimum(
        tl.cast(norm_lo_bits + 0x00100000, tl.float32, bitcast=True), 448.0
    )
    lo = tl.where(is_sub, sub_lo, norm_lo)
    hi = tl.where(is_sub, sub_hi, norm_hi)
    denom = tl.maximum(hi - lo, FP32_TINY)
    frac = tl.where(hi > lo, (av - lo) / denom, 0.0)
    mag = tl.where(u < frac, hi, lo)
    return tl.cast(tl.where(sgn, -mag, mag), tl.float8e4nv)


@triton.jit
def _fp4_quant_kernel(
    x_ptr,
    gmax_ptr,
    alpha_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    K,
    NB,
    PACK_STRIDE,
    TPB: tl.constexpr,
    BB: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    r = tl.program_id(0).to(tl.int64)
    offs_b = tl.program_id(1) * BB + tl.arange(0, BB)
    bmask = offs_b < NB
    cols = offs_b[:, None] * TPB + tl.arange(0, TPB)[None, :]
    raw = tl.load(
        x_ptr + r * K + cols, mask=bmask[:, None] & (cols < K), other=0.0
    ).to(tl.float32)

    block_absmax = tl.max(tl.abs(raw), axis=1)
    if SCALE_IS_E8M0:
        q_byte, s = _e8m0_scale(block_absmax, 2)
    else:
        qv = tl.div_rn(block_absmax * 448.0, tl.load(gmax_ptr).to(tl.float32))
        q_byte = (ue4m3_triton_ordinal(qv, 126) + 1).to(tl.uint8)
        q_hat = tl.cast(q_byte, tl.float8e4nv, bitcast=True).to(tl.float32)
        s = q_hat * tl.load(alpha_ptr).to(tl.float32)

    tl.store(scale_ptr + r * NB + offs_b, q_byte, mask=bmask)
    v = tl.div_rn(raw, s[:, None])
    if STOCHASTIC:
        code = _fp4_stochastic_round(v, tl.load(seed_ptr).to(tl.int32), r * K + cols)
    else:
        ti = e2m1_triton_ordinal(v, 16)
        code = tl.where(ti < 8, 15 - ti, ti - 8)

    lo, hi = tl.split(tl.reshape(code, (BB, TPB // 2, 2)))
    pcols = offs_b[:, None] * (TPB // 2) + tl.arange(0, TPB // 2)[None, :]
    tl.store(
        packed_ptr + r * PACK_STRIDE + pcols,
        (lo | (hi << 4)).to(tl.uint8),
        mask=bmask[:, None],
    )


@triton.jit
def _fp8_quant_kernel(
    x_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    K,
    NB,
    PACK_STRIDE,
    TPB: tl.constexpr,
    BB: tl.constexpr,
    EMAX: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    r = tl.program_id(0).to(tl.int64)
    offs_b = tl.program_id(1) * BB + tl.arange(0, BB)
    bmask = offs_b < NB
    cols = offs_b[:, None] * TPB + tl.arange(0, TPB)[None, :]
    raw = tl.load(
        x_ptr + r * K + cols, mask=bmask[:, None] & (cols < K), other=0.0
    ).to(tl.float32)

    q_byte, s = _e8m0_scale(tl.max(tl.abs(raw), axis=1), EMAX)
    tl.store(scale_ptr + r * NB + offs_b, q_byte, mask=bmask)
    v = tl.div_rn(raw, s[:, None])
    if STOCHASTIC:
        out = _fp8_stochastic_round(v, tl.load(seed_ptr).to(tl.int32), r * K + cols)
    else:
        out = tl.cast(tl.minimum(tl.maximum(v, -448.0), 448.0), tl.float8e4nv)
    tl.store(packed_ptr + r * PACK_STRIDE + cols, out, mask=bmask[:, None])


def _quant(
    x: torch.Tensor,
    group_size: int,
    is_fp4: bool,
    scale_is_e8m0: bool,
    emax: int,
    seed: torch.Tensor | None,
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

    grid = (rows, triton.cdiv(nb, block_chunk))
    if is_fp4:
        _fp4_quant_kernel[grid](
            x,
            global_absmax,
            alpha_ptr,
            seed if seed is not None else x,
            packed,
            scale,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            BB=block_chunk,
            SCALE_IS_E8M0=scale_is_e8m0,
            STOCHASTIC=seed is not None,
            num_warps=4,
        )
    else:
        _fp8_quant_kernel[grid](
            x,
            seed if seed is not None else x,
            packed,
            scale,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            BB=block_chunk,
            EMAX=emax,
            STOCHASTIC=seed is not None,
            num_warps=4,
        )
    return packed, scale, alpha


def nvfp4_quant(
    x: torch.Tensor, seed: torch.Tensor | None, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 16, True, False, 2, seed, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp4_quant(
    x: torch.Tensor, seed: torch.Tensor | None, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 32, True, True, 2, seed, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp8_quant(
    x: torch.Tensor, seed: torch.Tensor | None, block_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor, None]:
    packed, scale, _ = _quant(x, 32, False, True, 8, seed, block_chunk)
    return packed, scale, None
