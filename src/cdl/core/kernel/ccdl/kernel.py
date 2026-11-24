import triton
import triton.language as tl

from cdl.core.kernel.primitives import (
    cdl_proxy_coeffs,
    cdl_proxy_coeffs_truncated,
    cdl_sample,
    cdl_sample_truncated,
)


def _ccdl_fwd_configs() -> list[triton.Config]:
    return [
        triton.Config({"BB": 1}, num_warps=2),
        triton.Config({"BB": 2}, num_warps=2),
        triton.Config({"BB": 2}, num_warps=4),
        triton.Config({"BB": 4}, num_warps=4),
        triton.Config({"BB": 8}, num_warps=4),
        triton.Config({"BB": 8}, num_warps=8),
    ]


def _ccdl_bwd_configs() -> list[triton.Config]:
    return [
        triton.Config({"BB": 1}, num_warps=2),
        triton.Config({"BB": 2}, num_warps=2),
        triton.Config({"BB": 2}, num_warps=4),
        triton.Config({"BB": 4}, num_warps=2),
        triton.Config({"BB": 4}, num_warps=4),
        triton.Config({"BB": 8}, num_warps=4),
    ]


@triton.autotune(
    configs=_ccdl_fwd_configs(),
    key=["M_BUCKET", "NB_BUCKET", "TPB", "TN", "TOPK", "QN", "QT", "SCALE_BIAS"],
    cache_results=True,
)
@triton.jit
def _ccdl_fwd(
    th_ptr,
    thl_ptr,
    thg_ptr,
    kt_ptr,
    t_ordinal,
    q_ptr,
    QS,
    ql_ptr,
    qg_ptr,
    kq_ptr,
    q_ordinal,
    al_ptr,
    seed_ptr,
    pt_ptr,
    ps_ptr,
    M,
    K,
    NB,
    M_BUCKET: tl.constexpr,
    NB_BUCKET: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    QN: tl.constexpr,
    QT: tl.constexpr,
    BB: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
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
    # Packed scale byte: SCALE_BIAS=1 for UE4M3 (excludes zero), 0 for E8M0.
    tl.store(
        ps_ptr + r * NB + offs_b, (q_indices + SCALE_BIAS).to(tl.uint8), mask=bmask
    )

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    v = raw.to(tl.float32) / s[:, None]
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


@triton.autotune(
    configs=_ccdl_bwd_configs(),
    key=[
        "M_BUCKET",
        "NB_BUCKET",
        "TPB",
        "TN",
        "TOPK",
        "QN",
        "QT",
        "SCALE_BIAS",
        "NEED_DQ",
        "NEED_DKT",
        "NEED_DKQ",
        "NEED_DAL",
    ],
    reset_to_zero=["part_ptr"],
    cache_results=True,
)
@triton.jit
def _ccdl_bwd(
    g_ptr,
    th_ptr,
    thl_ptr,
    thg_ptr,
    kt_ptr,
    t_ordinal,
    q_ptr,
    QS,
    ql_ptr,
    qg_ptr,
    kq_ptr,
    q_ordinal,
    ps_ptr,
    PSS,
    al_ptr,
    dth_ptr,
    dq_ptr,
    part_ptr,
    K,
    NB,
    GS,
    M_BUCKET: tl.constexpr,
    NB_BUCKET: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    QN: tl.constexpr,
    QT: tl.constexpr,
    BB: tl.constexpr,
    SCALE_BIAS: tl.constexpr,
    NEED_DQ: tl.constexpr,
    NEED_DKT: tl.constexpr,
    NEED_DKQ: tl.constexpr,
    NEED_DAL: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    kq = tl.load(kq_ptr)
    al = tl.load(al_ptr)

    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    # The block scale is rebuilt from the stored scale byte (PSS == 0 means one
    # scale row is shared by every theta row).  Masked lanes use s = 1 so v
    # stays finite; their q_hat is 0 so the zero gradient cannot poison the
    # d(alpha) partial with 0 * inf NaNs.
    qi = tl.load(ps_ptr + r * PSS + offs_b, mask=bmask, other=0).to(tl.int32)
    if SCALE_BIAS != 0:
        qi = tl.maximum(qi - SCALE_BIAS, 0)
    q_hat = tl.load(ql_ptr + qi, mask=bmask, other=0.0).to(tl.float32)
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

    ds = g * ds_coeff

    if NEED_DKT:
        dkt = tl.sum(tl.sum(g * s[:, None] * dk_term, 1), 0)
    else:
        dkt = 0.0
    if NEED_DAL:
        dal = tl.sum(tl.sum(ds * q_hat[:, None], 1), 0)
    else:
        dal = 0.0

    if NEED_DQ or NEED_DKQ:
        qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0).to(tl.float32)
        dq_coeff, _, dkq_term = cdl_proxy_coeffs_truncated(
            qv, ql_ptr, qg_ptr, kq, QN, QT, q_ordinal
        )
        gq = al * tl.sum(tl.where(lmask, ds, 0.0), 1)
        if NEED_DQ:
            dq = gq * dq_coeff
            tl.store(dq_ptr + r * NB + offs_b, dq, mask=bmask)
        if NEED_DKQ:
            dkq = tl.sum(tl.where(bmask, gq * dkq_term, 0.0), 0)
        else:
            dkq = 0.0
    else:
        dkq = 0.0

    if NEED_DKT or NEED_DKQ or NEED_DAL:
        slot = ((r * 17 + gb) & 63) * 3
        tl.atomic_add(part_ptr + slot + 0, dkt)
        tl.atomic_add(part_ptr + slot + 1, dkq)
        tl.atomic_add(part_ptr + slot + 2, dal)
