import torch
import triton
import triton.language as tl

from cdl.core.kernel.formats.ordinals import (
    e2m1_single_zero_triton_ordinal,
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
def _fp4_scale(block_absmax, gmax_ptr, alpha_ptr, SCALE_IS_E8M0: tl.constexpr):
    if SCALE_IS_E8M0:
        return _e8m0_scale(block_absmax, 2)
    qv = tl.div_rn(block_absmax * 448.0, tl.load(gmax_ptr).to(tl.float32))
    q_byte = (ue4m3_triton_ordinal(qv, 126) + 1).to(tl.uint8)
    q_hat = tl.cast(q_byte, tl.float8e4nv, bitcast=True).to(tl.float32)
    return q_byte, q_hat * tl.load(alpha_ptr).to(tl.float32)


@triton.jit
def _fp4_quant_code(v, seed_ptr, offset, STOCHASTIC: tl.constexpr):
    if STOCHASTIC:
        return _fp4_stochastic_round(v, tl.load(seed_ptr).to(tl.int32), offset)
    ti = e2m1_single_zero_triton_ordinal(v, 15)
    return tl.where(ti < 7, 15 - ti, ti - 7)


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


def _absmax_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=4),
        triton.Config({"BLOCK": 4096}, num_warps=8),
        triton.Config({"BLOCK": 8192}, num_warps=8),
    ]


@triton.autotune(
    configs=_absmax_configs(),
    key=["N_BUCKET"],
    cache_results=True,
)
@triton.jit
def _absmax_atomic_kernel(
    x_ptr, out_u32_ptr, N, N_BUCKET: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < N, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(v), axis=0)
    tl.atomic_max(out_u32_ptr, tl.cast(m, tl.uint32, bitcast=True))


def _quant_1d_configs() -> list[triton.Config]:
    return [
        triton.Config({"BM": 2, "BB": 1}, num_warps=2),
        triton.Config({"BM": 4, "BB": 1}, num_warps=2),
        triton.Config({"BM": 8, "BB": 1}, num_warps=2),
        triton.Config({"BM": 8, "BB": 2}, num_warps=4),
        triton.Config({"BM": 8, "BB": 4}, num_warps=4),
        triton.Config({"BM": 16, "BB": 2}, num_warps=4),
        triton.Config({"BM": 16, "BB": 4}, num_warps=4, num_stages=2),
        triton.Config({"BM": 16, "BB": 4}, num_warps=4, num_stages=3),
        triton.Config({"BM": 16, "BB": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 16, "BB": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 32, "BB": 4}, num_warps=4),
        triton.Config({"BM": 32, "BB": 8}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32, "BB": 8}, num_warps=8, num_stages=3),
    ]


@triton.jit
def _fp4_quant_kernel_impl(
    x_ptr,
    gmax_ptr,
    alpha_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    ROWS,
    K,
    NB,
    PACK_STRIDE,
    TPB: tl.constexpr,
    ROWS_BUCKET: tl.constexpr,
    NB_BUCKET: tl.constexpr,
    BM: tl.constexpr,
    BB: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_r * BM + tl.arange(0, BM)
    bs = pid_b * BB + tl.arange(0, BB)
    rmask = rows < ROWS
    bmask = bs < NB
    rbmask = rmask[:, None] & bmask[None, :]
    cols = bs[:, None] * TPB + tl.arange(0, TPB)[None, :]
    offs = rows[:, None, None].to(tl.int64) * K + cols[None, :, :]
    mask = rbmask[:, :, None] & (cols[None, :, :] < K)
    raw = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    block_absmax = tl.max(tl.abs(raw), axis=2)
    q_byte, s = _fp4_scale(block_absmax, gmax_ptr, alpha_ptr, SCALE_IS_E8M0)

    tl.store(
        scale_ptr + rows[:, None].to(tl.int64) * NB + bs[None, :],
        q_byte,
        mask=rbmask,
    )
    v = tl.div_rn(raw, s[:, :, None])
    code = _fp4_quant_code(v, seed_ptr, offs, STOCHASTIC)

    lo, hi = tl.split(tl.reshape(code, (BM, BB, TPB // 2, 2)))
    pcols = bs[:, None] * (TPB // 2) + tl.arange(0, TPB // 2)[None, :]
    poffs = rows[:, None, None].to(tl.int64) * PACK_STRIDE + pcols[None, :, :]
    tl.store(
        packed_ptr + poffs,
        (lo | (hi << 4)).to(tl.uint8),
        mask=rbmask[:, :, None],
    )


_fp4_quant_kernel = triton.autotune(
    configs=_quant_1d_configs(),
    key=["ROWS_BUCKET", "NB_BUCKET", "TPB", "SCALE_IS_E8M0", "STOCHASTIC"],
    cache_results=True,
)(_fp4_quant_kernel_impl)


@triton.jit
def _fp8_quant_kernel_impl(
    x_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    ROWS,
    K,
    NB,
    PACK_STRIDE,
    TPB: tl.constexpr,
    ROWS_BUCKET: tl.constexpr,
    NB_BUCKET: tl.constexpr,
    BM: tl.constexpr,
    BB: tl.constexpr,
    EMAX: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_r * BM + tl.arange(0, BM)
    bs = pid_b * BB + tl.arange(0, BB)
    rmask = rows < ROWS
    bmask = bs < NB
    rbmask = rmask[:, None] & bmask[None, :]
    cols = bs[:, None] * TPB + tl.arange(0, TPB)[None, :]
    offs = rows[:, None, None].to(tl.int64) * K + cols[None, :, :]
    mask = rbmask[:, :, None] & (cols[None, :, :] < K)
    raw = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    q_byte, s = _e8m0_scale(tl.max(tl.abs(raw), axis=2), EMAX)
    tl.store(
        scale_ptr + rows[:, None].to(tl.int64) * NB + bs[None, :],
        q_byte,
        mask=rbmask,
    )
    v = tl.div_rn(raw, s[:, :, None])
    if STOCHASTIC:
        out = _fp8_stochastic_round(v, tl.load(seed_ptr).to(tl.int32), offs)
    else:
        out = tl.cast(tl.minimum(tl.maximum(v, -448.0), 448.0), tl.float8e4nv)
    tl.store(
        packed_ptr + rows[:, None, None].to(tl.int64) * PACK_STRIDE + cols[None, :, :],
        out,
        mask=rbmask[:, :, None],
    )


_fp8_quant_kernel = triton.autotune(
    configs=_quant_1d_configs(),
    key=["ROWS_BUCKET", "NB_BUCKET", "TPB", "EMAX", "STOCHASTIC"],
    cache_results=True,
)(_fp8_quant_kernel_impl)


@triton.jit
def _fp4_quant_kernel_2d(
    x_ptr,
    gmax_ptr,
    alpha_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    ROWS,
    K,
    NB_C,
    PACK_STRIDE,
    TPB: tl.constexpr,
    SCALE_IS_E8M0: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    tid = tl.program_id(0)
    br = (tid // NB_C).to(tl.int64)
    bc = (tid % NB_C).to(tl.int64)
    rows_offs = br * TPB + tl.arange(0, TPB)
    cols_offs = bc * TPB + tl.arange(0, TPB)
    mask = (rows_offs[:, None] < ROWS) & (cols_offs[None, :] < K)
    raw = tl.load(
        x_ptr + rows_offs[:, None] * K + cols_offs[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    block_absmax = tl.max(tl.abs(raw))
    q_byte, s = _fp4_scale(block_absmax, gmax_ptr, alpha_ptr, SCALE_IS_E8M0)

    tl.store(scale_ptr + tid, q_byte)
    v = tl.div_rn(raw, s)
    code = _fp4_quant_code(
        v,
        seed_ptr,
        (rows_offs[:, None] * K + cols_offs[None, :]).to(tl.int32),
        STOCHASTIC,
    )

    lo, hi = tl.split(tl.reshape(code, (TPB, TPB // 2, 2)))
    pcols = bc * (TPB // 2) + tl.arange(0, TPB // 2)
    tl.store(
        packed_ptr + rows_offs[:, None] * PACK_STRIDE + pcols[None, :],
        (lo | (hi << 4)).to(tl.uint8),
    )


@triton.jit
def _fp8_quant_kernel_2d(
    x_ptr,
    seed_ptr,
    packed_ptr,
    scale_ptr,
    ROWS,
    K,
    NB_C,
    PACK_STRIDE,
    TPB: tl.constexpr,
    EMAX: tl.constexpr,
    STOCHASTIC: tl.constexpr = False,
):
    tid = tl.program_id(0)
    br = (tid // NB_C).to(tl.int64)
    bc = (tid % NB_C).to(tl.int64)
    rows_offs = br * TPB + tl.arange(0, TPB)
    cols_offs = bc * TPB + tl.arange(0, TPB)
    mask = (rows_offs[:, None] < ROWS) & (cols_offs[None, :] < K)
    raw = tl.load(
        x_ptr + rows_offs[:, None] * K + cols_offs[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    q_byte, s = _e8m0_scale(tl.max(tl.abs(raw)), EMAX)
    tl.store(scale_ptr + tid, q_byte)
    v = tl.div_rn(raw, s)
    if STOCHASTIC:
        out = _fp8_stochastic_round(
            v,
            tl.load(seed_ptr).to(tl.int32),
            (rows_offs[:, None] * K + cols_offs[None, :]).to(tl.int32),
        )
    else:
        out = tl.cast(tl.minimum(tl.maximum(v, -448.0), 448.0), tl.float8e4nv)
    tl.store(packed_ptr + rows_offs[:, None] * PACK_STRIDE + cols_offs[None, :], out)


def _global_absmax(x: torch.Tensor) -> torch.Tensor:
    g_u32 = torch.full((), 0x00800000, dtype=torch.uint32, device=x.device)
    n_el = x.numel()
    if n_el > 0:
        n_bucket = 0 if n_el <= 131072 else (1 if n_el <= 2097152 else 2)
        _absmax_atomic_kernel[lambda meta: (triton.cdiv(n_el, meta["BLOCK"]),)](
            x, g_u32, n_el, N_BUCKET=n_bucket
        )
    return g_u32.view(torch.float32)


def _launch_quant_2d(
    x: torch.Tensor,
    global_absmax: torch.Tensor,
    alpha_ptr: torch.Tensor,
    seed_arg: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    rows: int,
    k: int,
    nb_r: int,
    nb: int,
    pack_stride: int,
    group_size: int,
    is_fp4: bool,
    scale_is_e8m0: bool,
    emax: int,
    stochastic: bool,
) -> None:
    grid = (nb_r * nb,)
    if is_fp4:
        _fp4_quant_kernel_2d[grid](
            x,
            global_absmax,
            alpha_ptr,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            SCALE_IS_E8M0=scale_is_e8m0,
            STOCHASTIC=stochastic,
            num_warps=4,
        )
    else:
        _fp8_quant_kernel_2d[grid](
            x,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            EMAX=emax,
            STOCHASTIC=stochastic,
            num_warps=4,
        )


def _launch_quant_1d(
    x: torch.Tensor,
    global_absmax: torch.Tensor,
    alpha_ptr: torch.Tensor,
    seed_arg: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    rows: int,
    k: int,
    nb: int,
    pack_stride: int,
    group_size: int,
    is_fp4: bool,
    scale_is_e8m0: bool,
    emax: int,
    stochastic: bool,
    block_chunk: int | None,
) -> None:
    rows_bucket = 0 if rows <= 512 else (1 if rows <= 2048 else 2)
    nb_bucket = 0 if nb <= 1 else (1 if nb <= 4 else 2)
    if block_chunk is not None:
        bm = 16 if group_size == 16 else 8
        bb = block_chunk
        grid = (triton.cdiv(rows, bm), triton.cdiv(nb, bb))
        if is_fp4:
            _fp4_quant_kernel_impl[grid](
                x,
                global_absmax,
                alpha_ptr,
                seed_arg,
                packed,
                scale,
                rows,
                k,
                nb,
                pack_stride,
                TPB=group_size,
                ROWS_BUCKET=rows_bucket,
                NB_BUCKET=nb_bucket,
                BM=bm,
                BB=bb,
                SCALE_IS_E8M0=scale_is_e8m0,
                STOCHASTIC=stochastic,
                num_warps=4,
            )
        else:
            _fp8_quant_kernel_impl[grid](
                x,
                seed_arg,
                packed,
                scale,
                rows,
                k,
                nb,
                pack_stride,
                TPB=group_size,
                ROWS_BUCKET=rows_bucket,
                NB_BUCKET=nb_bucket,
                BM=bm,
                BB=bb,
                EMAX=emax,
                STOCHASTIC=stochastic,
                num_warps=4,
            )
        return

    if is_fp4:
        _fp4_quant_kernel[
            lambda meta: (triton.cdiv(rows, meta["BM"]), triton.cdiv(nb, meta["BB"]))
        ](
            x,
            global_absmax,
            alpha_ptr,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            ROWS_BUCKET=rows_bucket,
            NB_BUCKET=nb_bucket,
            SCALE_IS_E8M0=scale_is_e8m0,
            STOCHASTIC=stochastic,
        )
    else:
        _fp8_quant_kernel[
            lambda meta: (triton.cdiv(rows, meta["BM"]), triton.cdiv(nb, meta["BB"]))
        ](
            x,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb,
            pack_stride,
            TPB=group_size,
            ROWS_BUCKET=rows_bucket,
            NB_BUCKET=nb_bucket,
            EMAX=emax,
            STOCHASTIC=stochastic,
        )


def _quant(
    x: torch.Tensor,
    group_size: int,
    is_fp4: bool,
    scale_is_e8m0: bool,
    emax: int,
    seed: torch.Tensor | None,
    block_2d: bool,
    block_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor [rows, k], got shape {tuple(x.shape)}")
    x = x.contiguous()
    rows, k = x.shape
    nb = triton.cdiv(k, group_size)
    pack_stride = nb * (group_size // 2 if is_fp4 else group_size)
    packed_dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn

    if block_2d:
        nb_r = triton.cdiv(rows, group_size)
        out_rows = nb_r * group_size
        packed = torch.empty(
            (out_rows, pack_stride), dtype=packed_dtype, device=x.device
        )
        scale = torch.empty((nb_r, nb), dtype=torch.uint8, device=x.device)
    else:
        nb_r = 0
        packed = torch.empty((rows, pack_stride), dtype=packed_dtype, device=x.device)
        scale = torch.empty((rows, nb), dtype=torch.uint8, device=x.device)

    if scale_is_e8m0:
        alpha = torch.ones(1, dtype=torch.float32, device=x.device) if is_fp4 else None
        global_absmax = x
        alpha_ptr = x
    else:
        global_absmax = _global_absmax(x)
        alpha = (global_absmax * (1.0 / (448.0 * 6.0))).reshape(1)
        alpha_ptr = alpha

    if rows == 0 or k == 0:
        return packed, scale, alpha

    seed_arg = seed if seed is not None else x
    stochastic = seed is not None
    if block_2d:
        _launch_quant_2d(
            x,
            global_absmax,
            alpha_ptr,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb_r,
            nb,
            pack_stride,
            group_size,
            is_fp4,
            scale_is_e8m0,
            emax,
            stochastic,
        )
    else:
        _launch_quant_1d(
            x,
            global_absmax,
            alpha_ptr,
            seed_arg,
            packed,
            scale,
            rows,
            k,
            nb,
            pack_stride,
            group_size,
            is_fp4,
            scale_is_e8m0,
            emax,
            stochastic,
            block_chunk,
        )
    return packed, scale, alpha


def nvfp4_quant(
    x: torch.Tensor,
    seed: torch.Tensor | None,
    block_2d: bool,
    block_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 16, True, False, 2, seed, block_2d, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp4_quant(
    x: torch.Tensor,
    seed: torch.Tensor | None,
    block_2d: bool,
    block_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, scale, alpha = _quant(x, 32, True, True, 2, seed, block_2d, block_chunk)
    assert alpha is not None
    return packed, scale, alpha


def mxfp8_quant(
    x: torch.Tensor,
    seed: torch.Tensor | None,
    block_2d: bool,
    block_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    packed, scale, _ = _quant(x, 32, False, True, 8, seed, block_2d, block_chunk)
    return packed, scale, None
