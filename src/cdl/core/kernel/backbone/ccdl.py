import math
import os

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
NUM_WARPS = int(os.environ.get("CDL_NUM_WARPS", "1"))
BB = int(os.environ.get("CDL_BB", "32"))


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
    ti_ptr,
    qi_ptr,
    s_ptr,
    M,
    K,
    NB,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
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

    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0).to(tl.float32)
    key = q_ordinal(qv, QN)
    klev = tl.load(ql_ptr + key, mask=bmask, other=0.0).to(tl.float32)
    left = tl.maximum(key - 1, 0)
    kgap = tl.where(
        qv >= klev,
        tl.load(qg_ptr + key, mask=bmask, other=1.0).to(tl.float32),
        tl.load(qg_ptr + left, mask=bmask, other=1.0).to(tl.float32),
    )
    c = key.to(tl.float32) + (qv - klev) / kgap
    start = tl.minimum(tl.maximum(key - QT // 2, 0), QN - QT)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + (QT - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin = t_near - c
    m = -kq * dmin * dmin
    hq = QT // 2
    if kq > 0.05:
        hq = tl.minimum(QT // 2, tl.sqrt(60.0 / kq)).to(tl.int32) + 1
    t_lo = tl.maximum(start, t_near - hq)

    seed = tl.load(seed_ptr).to(tl.int32)
    uq = tl.rand(seed, M * K + r * QS + offs_b)

    z = tl.zeros((BB,), tl.float32)
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        d = c - t
        inw = (t >= start) & (t <= start + (QT - 1))
        e = tl.where(
            t == t_near, 1.0, tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E)
        )
        z += tl.where(inw, e, 0.0)

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
            t == t_near, 1.0, tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E)
        )
        acc += tl.where(inw, e, 0.0)
        cnt += ((acc <= uz) & inw).to(tl.int32)
    sub = tl.minimum(off + cnt, QT - 1)
    chosen = start + sub

    s = tl.load(ql_ptr + chosen, mask=bmask, other=0.0).to(tl.float32) * al
    tl.store(s_ptr + r * NB + offs_b, s, mask=bmask)
    tl.store(qi_ptr + r * NB + offs_b, chosen.to(tl.uint8), mask=bmask)

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    v = raw.to(tl.float32) / s[:, None]
    key_t = t_ordinal(v, TN)
    klev_t = tl.load(thl_ptr + key_t, mask=lmask, other=0.0).to(tl.float32)
    left_t = tl.maximum(key_t - 1, 0)
    kgap_t = tl.where(
        v >= klev_t,
        tl.load(thg_ptr + key_t, mask=lmask, other=1.0).to(tl.float32),
        tl.load(thg_ptr + left_t, mask=lmask, other=1.0).to(tl.float32),
    )
    c_t = key_t.to(tl.float32) + (v - klev_t) / kgap_t
    start_t = tl.minimum(tl.maximum(key_t - TOPK // 2, 0), TN - TOPK)

    t_near_t = tl.minimum(
        tl.maximum(tl.floor(c_t + 0.5), start_t.to(tl.float32)),
        (start_t + (TOPK - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin_t = t_near_t - c_t
    m_t = -kt * dmin_t * dmin_t
    z_t = tl.zeros((BB, TPB), tl.float32)
    for j in tl.range(0, TOPK):
        t_t = (start_t + j).to(tl.float32)
        d = c_t - t_t
        z_t += tl.where(
            t_t == t_near_t,
            1.0,
            tl.math.exp2(tl.minimum(-kt * d * d - m_t, 0.0) * LOG2E),
        )

    u_t = tl.rand(seed, r * K + idx)

    acc_t = tl.zeros((BB, TPB), tl.float32)
    cnt_t = tl.zeros((BB, TPB), tl.int32)
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
    tl.store(ti_ptr + r * K + idx, chosen_t.to(tl.int16), mask=lmask)


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
    K,
    NB,
    GS,
    q_ordinal,
    t_ordinal,
    TOPK: tl.constexpr,
    TPB: tl.constexpr,
    TN: tl.constexpr,
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

    qv = tl.load(q_ptr + r * QS + offs_b, mask=bmask, other=0.0).to(tl.float32)
    key = q_ordinal(qv, QN)
    klev = tl.load(ql_ptr + key, mask=bmask, other=0.0).to(tl.float32)
    left = tl.maximum(key - 1, 0)
    kgap = tl.where(
        qv >= klev,
        tl.load(qg_ptr + key, mask=bmask, other=1.0).to(tl.float32),
        tl.load(qg_ptr + left, mask=bmask, other=1.0).to(tl.float32),
    )
    c = key.to(tl.float32) + (qv - klev) / kgap
    start = tl.minimum(tl.maximum(key - QT // 2, 0), QN - QT)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + (QT - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin = t_near - c
    m = -kq * dmin * dmin
    hq = QT // 2
    if kq > 0.05:
        hq = tl.minimum(QT // 2, tl.sqrt(60.0 / kq)).to(tl.int32) + 1
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
            t == t_near, 1.0, tl.math.exp2(tl.minimum(-kq * d * d - m, 0.0) * LOG2E)
        )
        w = tl.where(inw, e, 0.0)
        lv = tl.load(
            ql_ptr + tl.minimum(t, start + (QT - 1)).to(tl.int32),
            mask=bmask,
            other=0.0,
        ).to(tl.float32)
        a = -2.0 * kq * d
        z += w
        sl += w * lv
        sa += w * a
        sal += w * a * lv
        sb += w * d * d
        sbl += w * d * d * lv

    s = tl.load(s_ptr + r * NB + offs_b, mask=bmask, other=1.0)
    qhat = s / al

    ar_t = tl.arange(0, TPB)
    idx = (offs_b[:, None] * TPB + ar_t[None, :]).to(tl.int32)
    lmask = bmask[:, None]
    g = tl.load(g_ptr + r * GS + idx, mask=lmask, other=0.0).to(tl.float32)
    raw = tl.load(th_ptr + r * K + idx, mask=lmask, other=0.0)
    v = raw.to(tl.float32) / s[:, None]
    key_t = t_ordinal(v, TN)
    klev_t = tl.load(thl_ptr + key_t, mask=lmask, other=0.0).to(tl.float32)
    left_t = tl.maximum(key_t - 1, 0)
    kgap_t = tl.where(
        v >= klev_t,
        tl.load(thg_ptr + key_t, mask=lmask, other=1.0).to(tl.float32),
        tl.load(thg_ptr + left_t, mask=lmask, other=1.0).to(tl.float32),
    )
    c_t = key_t.to(tl.float32) + (v - klev_t) / kgap_t
    start_t = tl.minimum(tl.maximum(key_t - TOPK // 2, 0), TN - TOPK)

    t_near_t = tl.minimum(
        tl.maximum(tl.floor(c_t + 0.5), start_t.to(tl.float32)),
        (start_t + (TOPK - 1)).to(tl.float32),
    ).to(tl.int32)
    dmin_t = t_near_t - c_t
    m_t = -kt * dmin_t * dmin_t
    z_t = tl.zeros((BB, TPB), tl.float32)
    sl_t = tl.zeros((BB, TPB), tl.float32)
    sa_t = tl.zeros((BB, TPB), tl.float32)
    sal_t = tl.zeros((BB, TPB), tl.float32)
    sb_t = tl.zeros((BB, TPB), tl.float32)
    sbl_t = tl.zeros((BB, TPB), tl.float32)
    for j in tl.range(0, TOPK):
        t_t = (start_t + j).to(tl.float32)
        d = c_t - t_t
        e = tl.where(
            t_t == t_near_t,
            1.0,
            tl.math.exp2(tl.minimum(-kt * d * d - m_t, 0.0) * LOG2E),
        )
        lv = tl.load(thl_ptr + start_t + j, mask=lmask, other=0.0).to(tl.float32)
        a = -2.0 * kt * d
        z_t += e
        sl_t += e * lv
        sa_t += e * a
        sal_t += e * a * lv
        sb_t += e * d * d
        sbl_t += e * d * d * lv

    p_lt = sl_t / z_t
    gc = s[:, None] * (sal_t / z_t - p_lt * (sa_t / z_t))
    gv = gc / kgap_t
    dth = g * gv / s[:, None]
    ds = g * (gv * (-v / s[:, None]) + p_lt)
    tl.store(dth_ptr + r * K + idx, dth.to(dth_ptr.dtype.element_ty), mask=lmask)

    dkt = tl.sum(tl.sum(g * s[:, None] * (p_lt * (sb_t / z_t) - sbl_t / z_t), 1), 0)
    dal = tl.sum(tl.sum(ds * qhat[:, None], 1), 0)
    gq = al * tl.sum(tl.where(lmask, ds, 0.0), 1)

    dq = gq * (sal / z - (sl / z) * (sa / z)) / kgap
    dkq = tl.sum(tl.where(bmask, gq * ((sl / z) * (sb / z) - sbl / z), 0.0), 0)
    tl.store(dq_ptr + r * NB + offs_b, dq, mask=bmask)

    pbase = (r * tl.cdiv(NB, BB) + gb) * 4
    tl.store(part_ptr + pbase + 0, dkt)
    tl.store(part_ptr + pbase + 1, dkq)
    tl.store(part_ptr + pbase + 2, dal)


def _flatten_q(lead: torch.Size, nb: int, q: torch.Tensor) -> tuple[torch.Tensor, int]:
    m = math.prod(lead)
    if q.numel() == nb:
        return q.reshape(1, nb), 0
    if q.numel() == m * nb:
        return q.reshape(m, nb), nb
    return (q * q.new_ones(*lead, nb, 1)).reshape(m, nb), nb


def quant_fwd(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_level: torch.Tensor,
    theta_gap: torch.Tensor,
    q_level: torch.Tensor,
    q_gap: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    topk: int,
    tn: int,
    qn: int,
    qt: int,
    q_ordinal,
    t_ordinal,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    m = math.prod(lead)
    theta = theta.reshape(m, nb, tpb)
    k = nb * tpb
    lq, qs = _flatten_q(lead, nb, q)
    theta_indices = theta.new_empty((m, k), dtype=torch.int16)
    q_indices = torch.empty((m, nb), dtype=torch.uint8, device=theta.device)
    s = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    _cdl_quant_fwd[(m, triton.cdiv(nb, BB))](
        theta,
        lq,
        qs,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        theta_indices,
        q_indices,
        s,
        m,
        k,
        nb,
        q_ordinal,
        t_ordinal,
        TOPK=topk,
        TPB=tpb,
        TN=tn,
        QN=qn,
        QT=qt,
        BB=BB,
        num_warps=NUM_WARPS,
    )
    theta_indices = theta_indices.reshape(*lead, nb, tpb)
    if qs == 0:
        q_indices = q_indices[:1].reshape(q.shape)
    else:
        q_indices = q_indices.reshape(*lead, nb, 1)
    s = s.reshape(*lead, nb)
    return theta_indices, q_indices, s


def quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    theta_level: torch.Tensor,
    theta_gap: torch.Tensor,
    q_level: torch.Tensor,
    q_gap: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    topk: int,
    tn: int,
    qn: int,
    qt: int,
    q_ordinal,
    t_ordinal,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    m = math.prod(lead)
    theta = theta.reshape(m, nb, tpb)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    s = s.reshape(m, nb)
    lq, qs = _flatten_q(lead, nb, q)
    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    dq = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    ngb = triton.cdiv(nb, BB)
    partials = torch.empty((m * ngb, 4), dtype=torch.float32, device=theta.device)
    _cdl_quant_bwd[(m, ngb)](
        g,
        theta,
        lq,
        qs,
        s,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        theta_kappa,
        q_kappa,
        alpha,
        dtheta,
        dq,
        partials,
        m,
        k,
        nb,
        k,
        q_ordinal,
        t_ordinal,
        TOPK=topk,
        TPB=tpb,
        TN=tn,
        QN=qn,
        QT=qt,
        BB=BB,
        num_warps=NUM_WARPS,
    )
    return dtheta, dq, partials[:, :3].sum(0)


def _dq_grad(dq: torch.Tensor, theta: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    full = dq.reshape(*theta.shape[:-2], -1)
    lead_pad = q.ndim - full.ndim - 1
    full = full.reshape((1,) * max(lead_pad, 0) + full.shape + (1,))
    dims = tuple(
        i for i, size in enumerate(q.shape) if size == 1 and full.shape[i] != 1
    )
    if dims:
        full = full.sum(dim=dims, keepdim=True)
    return full.reshape(q.shape)


class CdlQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fwd,
        bwd,
        theta: torch.Tensor,
        q: torch.Tensor,
        theta_kappa: torch.Tensor,
        q_kappa: torch.Tensor,
        alpha: torch.Tensor,
        seed: torch.Tensor,
        theta_topk: int,
        q_topk: int,
    ):
        seed.add_(1)
        theta_indices, q_indices, s = fwd(
            theta, q, theta_kappa, q_kappa, alpha, seed, theta_topk, q_topk
        )
        ctx.save_for_backward(theta, q, s, theta_kappa, q_kappa, alpha)
        ctx.bwd = bwd
        ctx.args = (theta_topk, q_topk)
        carrier = theta.new_zeros((), dtype=torch.float32).expand(theta.shape)
        return theta_indices, q_indices, carrier

    @staticmethod
    def backward(ctx, g_ti, g_qi, g_proxy):
        theta, q, s, theta_kappa, q_kappa, alpha = ctx.saved_tensors
        theta_topk, q_topk = ctx.args
        dtheta, dq, sums = ctx.bwd(
            g_proxy.contiguous(),
            theta,
            q,
            s,
            theta_kappa,
            q_kappa,
            alpha,
            theta_topk,
            q_topk,
        )
        return (
            None,
            None,
            dtheta.reshape(theta.shape),
            _dq_grad(dq, theta, q),
            sums[0],
            sums[1],
            sums[2],
            None,
            None,
            None,
        )
