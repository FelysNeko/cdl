"""Fused cascaded-CDL quantization Triton kernels and their custom ops.

`_cdl_quant_fwd` fuses, per tensor: the q-level CDL top-k pass (UE4M3
alphabet), the theta-level CDL top-k pass (E2M1 alphabet), top-k sampling,
theta index packing into NVFP4 bytes, and the per-block scale bytes (both
the linear layout consumed by nvfp4_dequantize and the 128x4 swizzled layout
consumed by mm_fp4). The soft proxy is never materialized.

`_cdl_quant_bwd` recomputes the deterministic pmf and applies the
closed-form chain rule, so gradients match the reference autograd graph
(wrt theta, q, kappa_theta, kappa_q, alpha) while touching [M, K] data once.

Both kernels assume the companded path of cascaded_cdl_topk_forward
(companded=True). Grid is (M, ceil(NB / BB)) with NB = K // 16; each program
covers BB consecutive blocks of 16 elements.

Env knobs:
- CDL_NUM_WARPS: triton num_warps for the kernels (default 2).
- CDL_BB: blocks per program (default 32).

FORCE_U (see set_force_u) pins the sampling quantile for deterministic
tests; None (default) samples from the philox stream.
"""

import os

import torch
import triton
import triton.language as tl

from cdl.core.kernel.tables import e2m1_tables, ue4m3_tables

LOG2E = tl.constexpr(1.4426950408889634)

FORCE_U: float | None = None
NUM_WARPS = int(os.environ.get("CDL_NUM_WARPS", "2"))
BB_CONST = int(os.environ.get("CDL_BB", "32"))


def set_force_u(u: float | None) -> None:
    """Pin the sampling quantile for both fused kernels (testing only)."""
    global FORCE_U
    FORCE_U = u


@triton.jit
def _cdl_quant_fwd(
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
    pk_ptr,
    sc_ptr,
    sw_ptr,
    s_ptr,
    M,
    NB,
    K,
    PKB,
    SCS,
    NC4,
    TOPK: tl.constexpr,
    QN: tl.constexpr,
    QT: tl.constexpr,
    BB: tl.constexpr,
    CU: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    kt = tl.load(kt_ptr)
    kq = tl.load(kq_ptr)
    al = tl.load(al_ptr)
    b0 = gb * BB
    offs_b = b0 + tl.arange(0, BB)
    bmask = offs_b < NB

    # ---- q pass: companded CDL top-k over the UE4M3 alphabet (BB-wide) ----
    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0)
    u8 = tl.cast(tl.minimum(tl.maximum(qv, 2.0**-9), 448.0), tl.float8e4nv)
    key = tl.cast(u8, tl.uint8, bitcast=True).to(tl.int32)
    key = tl.minimum(tl.maximum(key - 1, 0), 125)
    klev = tl.load(ql_ptr + key, mask=bmask, other=0.0)
    left = tl.maximum(key - 1, 0)
    kgap = tl.where(
        qv >= klev,
        tl.load(qg_ptr + key, mask=bmask, other=1.0),
        tl.load(qg_ptr + left, mask=bmask, other=1.0),
    )
    c = key + (qv - klev) / kgap
    start = tl.minimum(tl.maximum(key - QT // 2, 0), QN - QT)

    # The softmax max is -kq * dmin^2 (nearest window level). FMA contraction
    # makes the max-subtraction inexact for huge -kq*d^2, so the exponent is
    # clamped at 0 and the nearest level's weight is forced to exactly 1.
    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + (QT - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin = t_near - c
    m = -kq * dmin * dmin
    hq = QT // 2
    if kq > 0.05:
        hq = tl.minimum(hq.to(tl.float32), tl.sqrt(60.0 / kq)).to(tl.int32) + 1
    t_lo = tl.maximum(start, t_near - hq)
    z = tl.zeros((BB,), tl.float32)
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        d = c - t
        inw = (t >= start) & (t <= start + (QT - 1))
        e = tl.where(
            t == t_near,
            1.0,
            tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E),
        )
        z += tl.where(inw, e, 0.0)

    seed = tl.load(seed_ptr).to(tl.int32)
    if CU < 0.0:
        uq = tl.rand(seed, M * K + r * QS + offs_b)
    else:
        uq = tl.full((BB,), CU, tl.float32)

    # Left-tail levels (j < t_lo) all satisfy cdf <= u when u*z exceeds their
    # total mass (bounded by (t_lo-start)*exp at t_lo-1); otherwise zero.
    e_left = tl.math.exp2(
        tl.minimum(-kq * (t_lo - 1 - c) * (t_lo - 1 - c) - m, 0.0) * LOG2E
    )
    off = tl.where(uq * z >= e_left * (t_lo - start + 1.0), t_lo - start, 0)
    acc = tl.zeros((BB,), tl.float32)
    cnt = tl.zeros((BB,), tl.int32)
    uz = uq * z
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        d = c - t
        inw = (t >= start) & (t <= start + (QT - 1))
        e = tl.where(
            t == t_near,
            1.0,
            tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E),
        )
        acc += tl.where(inw, e, 0.0)
        cnt += ((acc <= uz) & inw).to(tl.int32)
    sub = tl.minimum(off + cnt, QT - 1)
    chosen = start + sub

    s = tl.load(ql_ptr + chosen, mask=bmask, other=0.0) * al
    tl.store(s_ptr + r * NB + offs_b, s, mask=bmask)
    sbyte = (chosen + 1).to(tl.uint8)
    tl.store(sc_ptr + r * SCS + offs_b, sbyte, mask=bmask)
    # 128x4 swizzled scale layout (CUTLASS): write at the final address.
    r128 = r // 128
    i4 = (r % 128) // 32
    i32 = r % 32
    sw_off = (((r128 * NC4 + offs_b // 4) * 32 + i32) * 4 + i4) * 4 + offs_b % 4
    tl.store(sw_ptr + sw_off, sbyte, mask=bmask)

    # ---- theta pass: companded CDL top-k over E2M1 ((BB, 8, 2) lanes) ----
    ar8 = tl.arange(0, 8)
    ar2 = tl.arange(0, 2)
    idx = (offs_b[:, None, None] * 16 + ar8[None, :, None] * 2 + ar2[None, None, :]).to(
        tl.int32
    )
    lmask = bmask[:, None, None]
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    if raw.dtype == tl.float32:
        sgn = (tl.cast(raw, tl.uint32, bitcast=True) >> 31) != 0
    else:
        sgn = (tl.cast(raw, tl.uint16, bitcast=True) >> 15) != 0
    v = raw.to(tl.float32) / s[:, None, None]

    av = tl.minimum(tl.abs(v), 6.0)
    mag = (
        (av > 0.25).to(tl.int32)
        + (av > 0.75).to(tl.int32)
        + (av > 1.25).to(tl.int32)
        + (av > 1.75).to(tl.int32)
        + (av > 2.5).to(tl.int32)
        + (av > 3.5).to(tl.int32)
        + (av > 5.0).to(tl.int32)
    )
    key_t = tl.where(sgn, 7 - mag, 8 + mag)
    klev_t = tl.load(thl_ptr + key_t, mask=lmask, other=0.0)
    left_t = tl.maximum(key_t - 1, 0)
    kgap_t = tl.where(
        v >= klev_t,
        tl.load(thg_ptr + key_t, mask=lmask, other=1.0),
        tl.load(thg_ptr + left_t, mask=lmask, other=1.0),
    )
    c_t = key_t.to(tl.float32) + (v - klev_t) / kgap_t
    start_t = tl.minimum(tl.maximum(key_t - TOPK // 2, 0), 16 - TOPK)

    t_near_t = tl.minimum(
        tl.maximum(tl.floor(c_t + 0.5), start_t.to(tl.float32)),
        (start_t + (TOPK - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin_t = t_near_t - c_t
    m_t = -kt * dmin_t * dmin_t
    z_t = tl.zeros((BB, 8, 2), tl.float32)
    for j in tl.range(0, TOPK):
        t_t = (start_t + j).to(tl.float32)
        d = c_t - t_t
        e = tl.where(
            t_t == t_near_t,
            1.0,
            tl.math.exp2(tl.minimum(-kt * d * d - m_t, 0.0) * LOG2E),
        )
        z_t += e

    if CU < 0.0:
        u_t = tl.rand(seed, r * K + idx)
    else:
        u_t = tl.full((BB, 8, 2), CU, tl.float32)

    acc_t = tl.zeros((BB, 8, 2), tl.float32)
    cnt_t = tl.zeros((BB, 8, 2), tl.int32)
    uz_t = u_t * z_t
    for j in tl.range(0, TOPK):
        t_t = (start_t + j).to(tl.float32)
        d = c_t - t_t
        e = tl.where(
            t_t == t_near_t,
            1.0,
            tl.math.exp2(tl.minimum(-kt * d * d - m_t, 0.0) * LOG2E),
        )
        acc_t += e
        cnt_t += ((acc_t <= uz_t) & (t_t <= start_t + (TOPK - 1))).to(tl.int32)
    sub_t = tl.minimum(cnt_t, TOPK - 1)
    chosen_t = start_t + sub_t
    code = tl.where(chosen_t < 8, 15 - chosen_t, chosen_t - 8)
    lo, hi = tl.split(tl.reshape(code, (BB, 8, 2)))
    byte = (lo | (hi << 4)).to(tl.uint8)
    pk_off = r * PKB + offs_b[:, None] * 8 + ar8[None, :]
    tl.store(pk_ptr + pk_off, byte, mask=bmask[:, None])


@triton.jit
def _cdl_quant_bwd(
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
    NB,
    K,
    GS,
    TOPK: tl.constexpr,
    QN: tl.constexpr,
    QT: tl.constexpr,
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

    # ---- q pass recompute (deterministic pmf; no sampling needed) ----
    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0)
    u8 = tl.cast(tl.minimum(tl.maximum(qv, 2.0**-9), 448.0), tl.float8e4nv)
    key = tl.cast(u8, tl.uint8, bitcast=True).to(tl.int32)
    key = tl.minimum(tl.maximum(key - 1, 0), 125)
    klev = tl.load(ql_ptr + key, mask=bmask, other=0.0)
    left = tl.maximum(key - 1, 0)
    kgap = tl.where(
        qv >= klev,
        tl.load(qg_ptr + key, mask=bmask, other=1.0),
        tl.load(qg_ptr + left, mask=bmask, other=1.0),
    )
    c = key + (qv - klev) / kgap
    start = tl.minimum(tl.maximum(key - QT // 2, 0), QN - QT)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + (QT - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin = t_near - c
    m = -kq * dmin * dmin
    hq = QT // 2
    if kq > 0.05:
        hq = tl.minimum(hq.to(tl.float32), tl.sqrt(60.0 / kq)).to(tl.int32) + 1
    t_lo = tl.maximum(start, t_near - hq)
    z = tl.zeros((BB,), tl.float32)
    sl = tl.zeros((BB,), tl.float32)
    sa = tl.zeros((BB,), tl.float32)
    sal = tl.zeros((BB,), tl.float32)
    sb = tl.zeros((BB,), tl.float32)
    sbl = tl.zeros((BB,), tl.float32)
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        d = c - t
        inw = (t >= start) & (t <= start + (QT - 1))
        e = tl.where(
            t == t_near,
            1.0,
            tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E),
        )
        w = tl.where(inw, e, 0.0)
        lv = tl.load(
            ql_ptr + tl.minimum(t, start + (QT - 1)).to(tl.int32), mask=bmask, other=0.0
        )
        a = -2.0 * kq * d
        z += w
        sl += w * lv
        sa += w * a
        sal += w * a * lv
        sb += w * d * d
        sbl += w * d * d * lv

    s = tl.load(s_ptr + r * NB + offs_b, mask=bmask, other=1.0)
    qhat = s / al

    # ---- theta pass gradients ----
    ar8 = tl.arange(0, 8)
    ar2 = tl.arange(0, 2)
    idx = (offs_b[:, None, None] * 16 + ar8[None, :, None] * 2 + ar2[None, None, :]).to(
        tl.int32
    )
    lmask = bmask[:, None, None]
    g = tl.load(g_ptr + r * GS + idx, mask=lmask, other=0.0).to(tl.float32)
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    if raw.dtype == tl.float32:
        sgn = (tl.cast(raw, tl.uint32, bitcast=True) >> 31) != 0
    else:
        sgn = (tl.cast(raw, tl.uint16, bitcast=True) >> 15) != 0
    v = raw.to(tl.float32) / s[:, None, None]

    av = tl.minimum(tl.abs(v), 6.0)
    mag = (
        (av > 0.25).to(tl.int32)
        + (av > 0.75).to(tl.int32)
        + (av > 1.25).to(tl.int32)
        + (av > 1.75).to(tl.int32)
        + (av > 2.5).to(tl.int32)
        + (av > 3.5).to(tl.int32)
        + (av > 5.0).to(tl.int32)
    )
    key_t = tl.where(sgn, 7 - mag, 8 + mag)
    klev_t = tl.load(thl_ptr + key_t, mask=lmask, other=0.0)
    left_t = tl.maximum(key_t - 1, 0)
    kgap_t = tl.where(
        v >= klev_t,
        tl.load(thg_ptr + key_t, mask=lmask, other=1.0),
        tl.load(thg_ptr + left_t, mask=lmask, other=1.0),
    )
    c_t = key_t.to(tl.float32) + (v - klev_t) / kgap_t
    start_t = tl.minimum(tl.maximum(key_t - TOPK // 2, 0), 16 - TOPK)

    t_near_t = tl.minimum(
        tl.maximum(tl.floor(c_t + 0.5), start_t.to(tl.float32)),
        (start_t + (TOPK - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin_t = t_near_t - c_t
    m_t = -kt * dmin_t * dmin_t
    z_t = tl.zeros((BB, 8, 2), tl.float32)
    sl_t = tl.zeros((BB, 8, 2), tl.float32)
    sa_t = tl.zeros((BB, 8, 2), tl.float32)
    sal_t = tl.zeros((BB, 8, 2), tl.float32)
    sb_t = tl.zeros((BB, 8, 2), tl.float32)
    sbl_t = tl.zeros((BB, 8, 2), tl.float32)
    for j in tl.range(0, TOPK):
        t_t = (start_t + j).to(tl.float32)
        d = c_t - t_t
        e = tl.where(
            t_t == t_near_t,
            1.0,
            tl.math.exp2(tl.minimum(-kt * d * d - m_t, 0.0) * LOG2E),
        )
        lv = tl.load(thl_ptr + start_t + j, mask=lmask, other=0.0)
        a = -2.0 * kt * d
        z_t += e
        sl_t += e * lv
        sa_t += e * a
        sal_t += e * a * lv
        sb_t += e * d * d
        sbl_t += e * d * d * lv

    p_lt = sl_t / z_t
    gc = s[:, None, None] * (sal_t / z_t - p_lt * (sa_t / z_t))
    gv = gc / kgap_t
    dth = g * gv / s[:, None, None]
    ds = g * (gv * (-v / s[:, None, None]) + p_lt)
    tl.store(dth_ptr + r * K + idx, dth.to(dth_ptr.dtype.element_ty), mask=lmask)

    dkt = tl.sum(
        tl.sum(
            tl.sum(g * s[:, None, None] * (p_lt * (sb_t / z_t) - sbl_t / z_t), 2), 1
        ),
        0,
    )
    dal = tl.sum(tl.sum(tl.sum(ds * qhat[:, None, None], 2), 1), 0)
    gq = al * tl.sum(tl.sum(ds, 2), 1)

    # ---- q pass gradients ----
    dq = gq * (sal / z - (sl / z) * (sa / z)) / kgap
    dkq = tl.sum(tl.where(bmask, gq * ((sl / z) * (sb / z) - sbl / z), 0.0), 0)
    tl.store(dq_ptr + r * NB + offs_b, dq, mask=bmask)

    pbase = (r * tl.cdiv(NB, BB) + gb) * 4
    tl.store(part_ptr + pbase + 0, dkt)
    tl.store(part_ptr + pbase + 1, dkq)
    tl.store(part_ptr + pbase + 2, dal)


@torch.library.custom_op("cdl::cdl_quant_fwd", mutates_args=())
def cdl_quant_fwd(
    theta: torch.Tensor,
    log_q: torch.Tensor,
    theta_level: torch.Tensor,
    theta_gap: torch.Tensor,
    q_level: torch.Tensor,
    q_gap: torch.Tensor,
    kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    topk: int,
    qtopk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused cascaded-CDL quantization to the NVFP4 kernel layout.

    Returns (packed fp4 [ceil(M/16)*16, PKB] uint8, linear scale
    [ceil(M/16)*16, SCS] uint8, swizzled scale [128pad, SCS4] uint8,
    block scale s [M, NB] float32). theta must be contiguous [M, K] with
    K % 16 == 0; log_q is the already-exp'd per-block scale [M, NB, 1] or
    [M, NB].
    """
    m, k = theta.shape
    nb = k // 16
    lq = log_q.reshape(-1, nb)
    if lq.shape[0] not in (1, m):
        raise ValueError(f"q rows {lq.shape[0]} must be 1 or match theta rows {m}")
    qs = nb if lq.shape[0] == m else 0
    pkb = (k + 31) // 32 * 16
    nbs = (k + 31) // 32 * 2  # padded scale cols so K' = pkb*2 stays TMA-aligned
    m16 = (m + 15) // 16 * 16
    packed = theta.new_zeros(m16, pkb, dtype=torch.uint8)
    scale = theta.new_zeros((m16, nbs), dtype=torch.uint8)
    nbs4 = (nbs + 3) // 4 * 4
    swizzled = theta.new_zeros(((m + 127) // 128 * 128, nbs4), dtype=torch.uint8)
    s = theta.new_empty((m, nb), dtype=torch.float32)
    qn = q_level.numel()
    _cdl_quant_fwd[(m, (nb + BB_CONST - 1) // BB_CONST)](
        theta,
        lq,
        qs,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        kappa,
        q_kappa,
        alpha,
        seed,
        packed,
        scale,
        swizzled,
        s,
        m,
        nb,
        k,
        pkb,
        nbs,
        nbs4 // 4,
        TOPK=topk,
        QN=qn,
        QT=qtopk,
        BB=BB_CONST,
        CU=-1.0 if FORCE_U is None else FORCE_U,
        num_warps=NUM_WARPS,
    )
    return packed, scale, swizzled, s


@cdl_quant_fwd.register_fake
def _(
    theta,
    log_q,
    theta_level,
    theta_gap,
    q_level,
    q_gap,
    kappa,
    q_kappa,
    alpha,
    seed,
    topk,
    qtopk,
):
    m, k = theta.shape
    nb = k // 16
    pkb = (k + 31) // 32 * 16
    nbs = (k + 31) // 32 * 2
    return (
        theta.new_empty(((m + 15) // 16 * 16, pkb), dtype=torch.uint8),
        theta.new_empty(((m + 15) // 16 * 16, nbs), dtype=torch.uint8),
        theta.new_empty(
            ((m + 127) // 128 * 128, (nbs + 3) // 4 * 4), dtype=torch.uint8
        ),
        theta.new_empty((m, nb), dtype=torch.float32),
    )


@torch.library.custom_op("cdl::cdl_quant_bwd", mutates_args=())
def cdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    log_q: torch.Tensor,
    s: torch.Tensor,
    theta_level: torch.Tensor,
    theta_gap: torch.Tensor,
    q_level: torch.Tensor,
    q_gap: torch.Tensor,
    kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    topk: int,
    qtopk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic backward of the fused quantization.

    grad_proxy is dL/dproxy with row width possibly padded (GS); returns
    (dtheta [M, K], dq [M, NB] float32, scalar sums [3] = dL/dkappa_theta,
    dL/dkappa_q, dL/dalpha for the exp'd values).
    """
    m, k = theta.shape
    nb = k // 16
    lq = log_q.reshape(-1, nb)
    if lq.shape[0] not in (1, m):
        raise ValueError(f"q rows {lq.shape[0]} must be 1 or match theta rows {m}")
    qs = nb if lq.shape[0] == m else 0
    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    dq = theta.new_empty((m, nb), dtype=torch.float32)
    ngb = (nb + BB_CONST - 1) // BB_CONST
    partials = theta.new_empty((m * ngb, 4), dtype=torch.float32)
    qn = q_level.numel()
    _cdl_quant_bwd[(m, ngb)](
        grad_proxy,
        theta,
        lq,
        qs,
        s,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        kappa,
        q_kappa,
        alpha,
        dtheta,
        dq,
        partials,
        m,
        nb,
        k,
        grad_proxy.shape[1],
        TOPK=topk,
        QN=qn,
        QT=qtopk,
        BB=BB_CONST,
        num_warps=NUM_WARPS,
    )
    return dtheta, dq, partials[:, :3].sum(0)


@cdl_quant_bwd.register_fake
def _(
    grad_proxy,
    theta,
    log_q,
    s,
    theta_level,
    theta_gap,
    q_level,
    q_gap,
    kappa,
    q_kappa,
    alpha,
    topk,
    qtopk,
):
    m, k = theta.shape
    nb = k // 16
    return (
        theta.new_empty((m, k), dtype=theta.dtype),
        theta.new_empty((m, nb), dtype=torch.float32),
        theta.new_empty((3,), dtype=torch.float32),
    )


def _tables(device):
    return (*e2m1_tables(device), *ue4m3_tables(device))
