import triton
import triton.language as tl

from cdl.core.kernel.primitives import (
    cdl_proxy_coeffs,
    cdl_proxy_coeffs_truncated,
    cdl_sample,
    cdl_sample_truncated,
)


@triton.jit
def _ccdl_fwd(
    th_ptr,
    q_ptr,
    QS,
    thl_ptr,
    thg_ptr,
    ql_ptr,
    qg_ptr,
    kt_ptr,
    kq_ptr,
    al_ptr,
    seed_ptr,
    ti_ptr,
    qi_ptr,
    s_ptr,
    M,
    K,
    NB,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    QT: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    QN: tl.constexpr,
    BB: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    kq = tl.load(kq_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0).to(tl.float32)
    seed = tl.load(seed_ptr).to(tl.int32)
    uq = tl.rand(seed, M * K + r * QS + offs_b)
    q_indices = cdl_sample_truncated(qv, ql_ptr, qg_ptr, kq, uq, QN, QT, q_ordinal)
    q_hat = tl.load(ql_ptr + q_indices, mask=bmask, other=0.0).to(tl.float32)
    s = q_hat * al
    tl.store(s_ptr + r * NB + offs_b, s, mask=bmask)
    tl.store(qi_ptr + r * NB + offs_b, q_indices.to(tl.uint8), mask=bmask)

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    v = raw.to(tl.float32) / s[:, None]
    u_t = tl.rand(seed, r * K + idx)
    theta_indices = cdl_sample(v, thl_ptr, thg_ptr, kt, u_t, TN, TOPK, t_ordinal)
    tl.store(ti_ptr + r * K + idx, theta_indices.to(tl.int16), mask=lmask)


@triton.jit
def _ccdl_bwd(
    g_ptr,
    th_ptr,
    q_ptr,
    QS,
    s_ptr,
    thl_ptr,
    thg_ptr,
    ql_ptr,
    qg_ptr,
    kt_ptr,
    kq_ptr,
    al_ptr,
    dth_ptr,
    dq_ptr,
    part_ptr,
    M,
    K,
    NB,
    GS,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    QT: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    QN: tl.constexpr,
    BB: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    kq = tl.load(kq_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    s = tl.load(s_ptr + r * NB + offs_b, mask=bmask, other=1.0)
    q_hat = s / al

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

    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0).to(tl.float32)
    dq_coeff, _, dkq_term = cdl_proxy_coeffs_truncated(
        qv, ql_ptr, qg_ptr, kq, QN, QT, q_ordinal
    )
    gq = al * tl.sum(tl.where(lmask, ds, 0.0), 1)
    dq = gq * dq_coeff
    dkq = tl.sum(tl.where(bmask, gq * dkq_term, 0.0), 0)
    tl.store(dq_ptr + r * NB + offs_b, dq, mask=bmask)

    pbase = (r * tl.cdiv(NB, BB) + gb) * 3
    tl.store(part_ptr + pbase + 0, dkt)
    tl.store(part_ptr + pbase + 1, dkq)
    tl.store(part_ptr + pbase + 2, dal)
