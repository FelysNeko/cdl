import triton
import triton.language as tl

from cdl.core.kernel.primitives import (
    cdl_proxy_coeffs,
    cdl_sample,
)

FP32_TINY = tl.constexpr(1.1754943508222875e-38)


@triton.jit
def _ecdl_absmax(
    th_ptr,
    bamax_ptr,
    K,
    NB,
    TPB: tl.constexpr,
    BB: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB
    idx = (offs_b[:, None] * TPB + tl.arange(0, TPB)[None, :]).to(tl.int32)
    raw = tl.load(th_ptr + r * K + idx, mask=bmask[:, None], other=0.0)
    block_absmax = tl.max(tl.abs(raw.to(tl.float32)), axis=1)
    tl.store(bamax_ptr + r * NB + offs_b, block_absmax, mask=bmask)


@triton.jit
def _ecdl_scale(
    block_absmax,
    gmax_ptr,
    qmax,
    ql_ptr,
    al,
    q_ordinal,
    bmask,
    QN: tl.constexpr,
    MODE: tl.constexpr,
    EMAX: tl.constexpr,
):
    if MODE == 1:
        # OCP MX scales: byte = floor(log2(block_amax)) - emax + 127, single
        # pass (no global-amax reduction).  q_hat comes from the q level
        # table, not exp2: byte 0 maps to the fp32-subnormal 2**-127, which
        # tl.exp2 flushes to zero.  The bamax/gmax/qmax/q_ordinal arguments
        # are dead in this branch.
        clamped = tl.maximum(block_absmax, FP32_TINY)
        bits = tl.cast(clamped, tl.uint32, bitcast=True)
        floor_log2 = (bits >> 23).to(tl.int32) - 127
        q_indices = tl.minimum(tl.maximum(floor_log2 - EMAX + 127, 0), 254)
        q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
        return q_indices, q_hat * al
    if MODE == 0:
        global_absmax = tl.maximum(tl.load(gmax_ptr).to(tl.float32), FP32_TINY)
        qv = block_absmax * qmax / global_absmax
        q_indices = q_ordinal(qv, QN)
        q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
        return q_indices, q_hat * al
    q_indices = q_ordinal(tl.full(block_absmax.shape, 1.0, tl.float32), QN)
    q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
    return q_indices, q_hat * al


@triton.jit
def _ecdl_fwd(
    th_ptr,
    bamax_ptr,
    gmax_ptr,
    qmax,
    thl_ptr,
    thg_ptr,
    ql_ptr,
    kt_ptr,
    al_ptr,
    seed_ptr,
    pt_ptr,
    ps_ptr,
    K,
    NB,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    QN: tl.constexpr,
    BB: tl.constexpr,
    MODE: tl.constexpr,
    EMAX: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)

    if MODE == 1:
        block_absmax = tl.max(tl.abs(raw.to(tl.float32)), axis=1)
    elif MODE == 0:
        block_absmax = tl.load(bamax_ptr + r * NB + offs_b, mask=bmask, other=0.0).to(
            tl.float32
        )
    else:
        block_absmax = tl.zeros((BB,), dtype=tl.float32)

    q_indices, s = _ecdl_scale(
        block_absmax, gmax_ptr, qmax, ql_ptr, al, q_ordinal, bmask, QN, MODE, EMAX
    )
    tl.store(
        ps_ptr + r * NB + offs_b, (q_indices + SCALE_BIAS).to(tl.uint8), mask=bmask
    )

    v = raw.to(tl.float32) / s[:, None]
    seed = tl.load(seed_ptr).to(tl.int32)
    u_t = tl.rand(seed, r * K + idx)
    theta_indices = cdl_sample(v, thl_ptr, thg_ptr, kt, u_t, TN, TOPK, t_ordinal)
    if TN == 16:
        code = tl.where(theta_indices < 8, 15 - theta_indices, theta_indices - 8)
        lo, hi = tl.split(tl.reshape(code, (BB, TPB // 2, 2)))
        packed = (lo | (hi << 4)).to(tl.uint8)
        pidx = (offs_b[:, None] * (TPB // 2) + tl.arange(0, TPB // 2)[None, :]).to(
            tl.int32
        )
        tl.store(pt_ptr + r * (K // 2) + pidx, packed, mask=lmask)
    else:
        packed = tl.where(
            theta_indices < 127, 254 - theta_indices, theta_indices - 127
        ).to(tl.uint8)
        tl.store(pt_ptr + r * K + idx, packed, mask=lmask)


@triton.jit
def _ecdl_bwd(
    g_ptr,
    th_ptr,
    ps_ptr,
    ql_ptr,
    thl_ptr,
    thg_ptr,
    kt_ptr,
    al_ptr,
    dth_ptr,
    part_ptr,
    K,
    NB,
    GS,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    BB: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
    COMPUTE_SCALARS: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    qi = tl.load(ps_ptr + r * NB + offs_b, mask=bmask, other=0).to(tl.int32)
    if SCALE_BIAS != 0:
        qi = tl.maximum(qi - SCALE_BIAS, 0)
    q_hat = tl.load(ql_ptr + qi, mask=bmask, other=0.0).to(tl.float32)
    # Masked lanes use s = 1 so v stays finite; their q_hat is 0 so the
    # zero gradient cannot poison the d(alpha) partial with 0 * inf NaNs.
    s = tl.where(bmask, q_hat * al, 1.0)
    q_hat = tl.where(bmask, q_hat, 0.0)

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    g = tl.load(g_ptr + r * GS + idx, mask=lmask, other=0.0).to(tl.float32)
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    v = raw.to(tl.float32) / s[:, None]

    dth_coeff, ds_coeff, dk_term = cdl_proxy_coeffs(
        v, thl_ptr, thg_ptr, kt, TN, TOPK, t_ordinal
    )
    dth = g * dth_coeff
    tl.store(dth_ptr + r * K + idx, dth.to(dth_ptr.dtype.element_ty), mask=lmask)

    if COMPUTE_SCALARS:
        ds = g * ds_coeff
        dkt = tl.sum(tl.sum(g * s[:, None] * dk_term, 1), 0)
        dal = tl.sum(tl.sum(ds * q_hat[:, None], 1), 0)

        pbase = (r * tl.cdiv(NB, BB) + gb) * 2
        tl.store(part_ptr + pbase + 0, dkt)
        tl.store(part_ptr + pbase + 1, dal)


@triton.jit
def _ecdl_absmax_2d(
    th_ptr,
    bamax_ptr,
    H,
    K,
    NB_R,
    NB_C,
    TPB: tl.constexpr,
):
    tid = tl.program_id(0)
    tiles_per_m = NB_R * NB_C
    m_idx = (tid // tiles_per_m).to(tl.int64)
    rem = tid % tiles_per_m
    br = (rem // NB_C).to(tl.int64)
    bc = (rem % NB_C).to(tl.int64)

    row_offs = m_idx * H + br * TPB + tl.arange(0, TPB)
    col_offs = bc * TPB + tl.arange(0, TPB)
    raw = tl.load(th_ptr + row_offs[:, None] * K + col_offs[None, :])
    block_absmax = tl.max(tl.abs(raw.to(tl.float32)))
    tl.store(bamax_ptr + tid, block_absmax)


@triton.jit
def _ecdl_fwd_2d(
    th_ptr,
    bamax_ptr,
    gmax_ptr,
    qmax,
    thl_ptr,
    thg_ptr,
    ql_ptr,
    kt_ptr,
    al_ptr,
    seed_ptr,
    pt_ptr,
    ps_ptr,
    H,
    K,
    NB_R,
    NB_C,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    QN: tl.constexpr,
    MODE: tl.constexpr,
    EMAX: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
):
    tid = tl.program_id(0)
    tiles_per_m = NB_R * NB_C
    m_idx = (tid // tiles_per_m).to(tl.int64)
    rem = tid % tiles_per_m
    br = (rem // NB_C).to(tl.int64)
    bc = (rem % NB_C).to(tl.int64)

    kt = tl.load(kt_ptr)
    al = tl.load(al_ptr)

    row_offs = m_idx * H + br * TPB + tl.arange(0, TPB)
    col_offs = bc * TPB + tl.arange(0, TPB)
    offs = row_offs[:, None] * K + col_offs[None, :]
    raw = tl.load(th_ptr + offs)

    if MODE == 1:
        block_absmax = tl.max(tl.abs(raw.to(tl.float32)))
        clamped = tl.maximum(block_absmax, FP32_TINY)
        bits = tl.cast(clamped, tl.uint32, bitcast=True)
        floor_log2 = (bits >> 23).to(tl.int32) - 127
        q_indices = tl.minimum(tl.maximum(floor_log2 - EMAX + 127, 0), 254)
        q_hat = tl.load(ql_ptr + q_indices).to(tl.float32)
        s = q_hat * al
    elif MODE == 0:
        block_absmax = tl.load(bamax_ptr + tid).to(tl.float32)
        global_absmax = tl.maximum(tl.load(gmax_ptr).to(tl.float32), FP32_TINY)
        qv = block_absmax * qmax / global_absmax
        q_indices = q_ordinal(qv, QN)
        q_hat = tl.load(ql_ptr + q_indices).to(tl.float32)
        s = q_hat * al
    else:
        q_indices = q_ordinal(tl.full((), 1.0, tl.float32), QN)
        q_hat = tl.load(ql_ptr + q_indices).to(tl.float32)
        s = q_hat * al

    tl.store(ps_ptr + tid, (q_indices + SCALE_BIAS).to(tl.uint8))

    v = raw.to(tl.float32) / s
    seed = tl.load(seed_ptr).to(tl.int32)
    u_t = tl.rand(seed, offs.to(tl.int32))
    theta_indices = cdl_sample(v, thl_ptr, thg_ptr, kt, u_t, TN, TOPK, t_ordinal)
    if TN == 16:
        code = tl.where(theta_indices < 8, 15 - theta_indices, theta_indices - 8)
        lo, hi = tl.split(tl.reshape(code, (TPB, TPB // 2, 2)))
        packed = (lo | (hi << 4)).to(tl.uint8)
        pcol_offs = bc * (TPB // 2) + tl.arange(0, TPB // 2)
        tl.store(pt_ptr + row_offs[:, None] * (K // 2) + pcol_offs[None, :], packed)
    else:
        packed = tl.where(
            theta_indices < 127, 254 - theta_indices, theta_indices - 127
        ).to(tl.uint8)
        tl.store(pt_ptr + offs, packed)


@triton.jit
def _ecdl_bwd_2d(
    g_ptr,
    th_ptr,
    ps_ptr,
    ql_ptr,
    thl_ptr,
    thg_ptr,
    kt_ptr,
    al_ptr,
    dth_ptr,
    part_ptr,
    H,
    K,
    NB_R,
    NB_C,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
    COMPUTE_SCALARS: tl.constexpr,
):
    tid = tl.program_id(0)
    tiles_per_m = NB_R * NB_C
    m_idx = (tid // tiles_per_m).to(tl.int64)
    rem = tid % tiles_per_m
    br = (rem // NB_C).to(tl.int64)
    bc = (rem % NB_C).to(tl.int64)

    kt = tl.load(kt_ptr)
    al = tl.load(al_ptr)

    qi = tl.load(ps_ptr + tid).to(tl.int32)
    if SCALE_BIAS != 0:
        qi = tl.maximum(qi - SCALE_BIAS, 0)
    q_hat = tl.load(ql_ptr + qi).to(tl.float32)
    s = q_hat * al

    row_offs = m_idx * H + br * TPB + tl.arange(0, TPB)
    col_offs = bc * TPB + tl.arange(0, TPB)
    offs = row_offs[:, None] * K + col_offs[None, :]
    g = tl.load(g_ptr + offs).to(tl.float32)
    raw = tl.load(th_ptr + offs)
    v = raw.to(tl.float32) / s

    dth_coeff, ds_coeff, dk_term = cdl_proxy_coeffs(
        v, thl_ptr, thg_ptr, kt, TN, TOPK, t_ordinal
    )
    dth = g * dth_coeff
    tl.store(dth_ptr + offs, dth.to(dth_ptr.dtype.element_ty))

    if COMPUTE_SCALARS:
        ds = g * ds_coeff
        dkt = tl.sum(g * s * dk_term)
        dal = tl.sum(ds * q_hat)

        pbase = tid.to(tl.int64) * 2
        tl.store(part_ptr + pbase + 0, dkt)
        tl.store(part_ptr + pbase + 1, dal)
