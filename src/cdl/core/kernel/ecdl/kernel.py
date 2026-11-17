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
    ti_ptr,
    qi_ptr,
    s_ptr,
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
        # OCP MX scales: byte = floor(log2(block_amax)) - emax + 127, single
        # pass (no global-amax reduction).  q_hat comes from the q level
        # table, not exp2: byte 0 maps to the fp32-subnormal 2**-127, which
        # tl.exp2 flushes to zero.  The bamax/gmax/qmax/q_ordinal arguments
        # are dead in this branch.
        block_absmax = tl.max(tl.abs(raw.to(tl.float32)), axis=1)
        clamped = tl.maximum(block_absmax, FP32_TINY)
        bits = tl.cast(clamped, tl.uint32, bitcast=True)
        floor_log2 = (bits >> 23).to(tl.int32) - 127
        q_indices = tl.minimum(tl.maximum(floor_log2 - EMAX + 127, 0), 254)
        q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
        s = q_hat * al
    elif MODE == 0:
        block_absmax = tl.load(
            bamax_ptr + r * NB + offs_b, mask=bmask, other=0.0
        ).to(tl.float32)
        global_absmax = tl.maximum(tl.load(gmax_ptr).to(tl.float32), FP32_TINY)
        qv = block_absmax * qmax / global_absmax
        q_indices = q_ordinal(qv, QN)
        q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
        s = q_hat * al
    else:
        # Unknown mode: brute fallback, block and global scales both 1.
        q_indices = q_ordinal(tl.full((BB,), 1.0, tl.float32), QN)
        q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
        s = q_hat * al

    tl.store(s_ptr + r * NB + offs_b, s, mask=bmask)
    tl.store(qi_ptr + r * NB + offs_b, q_indices.to(tl.uint8), mask=bmask)

    v = raw.to(tl.float32) / s[:, None]
    seed = tl.load(seed_ptr).to(tl.int32)
    u_t = tl.rand(seed, r * K + idx)
    theta_indices = cdl_sample(v, thl_ptr, thg_ptr, kt, u_t, TN, TOPK, t_ordinal)
    tl.store(ti_ptr + r * K + idx, theta_indices.to(tl.int16), mask=lmask)


@triton.jit
def _ecdl_bwd(
    g_ptr,
    th_ptr,
    s_ptr,
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
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    s = tl.load(s_ptr + r * NB + offs_b, mask=bmask, other=1.0)
    # Masked lanes load s = 1; with a tiny alpha their q_hat = 1/alpha
    # overflows and 0 * inf = NaN would poison the d(alpha) partial.
    q_hat = tl.where(bmask, s / al, 0.0)

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
    ds = g * ds_coeff
    tl.store(dth_ptr + r * K + idx, dth.to(dth_ptr.dtype.element_ty), mask=lmask)

    dkt = tl.sum(tl.sum(g * s[:, None] * dk_term, 1), 0)
    dal = tl.sum(tl.sum(ds * q_hat[:, None], 1), 0)

    pbase = (r * tl.cdiv(NB, BB) + gb) * 2
    tl.store(part_ptr + pbase + 0, dkt)
    tl.store(part_ptr + pbase + 1, dal)
