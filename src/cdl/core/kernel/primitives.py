"""Shape-agnostic Triton device functions for CDL top-k quantization.

Everything here operates on register tensors of arbitrary shape and never
touches ``program_id``.  The kernels in ``ccdl`` / ``ecdl`` are responsible for
memory layout, masks and program ids.

The full-window variants are used for theta (weight / activation) CDL.  The
truncated variants reproduce the q-side optimisation of cascaded CDL: for a
large inverse temperature most of the top-k mass lies near the nearest level,
so only ``[t_lo, t_lo + 2*hq]`` is evaluated and the left tail is approximated
when sampling.
"""

import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def cdl_window(v, level_ptr, gap_ptr, TN: tl.constexpr, TOPK: tl.constexpr, ordinal):
    """Companded window around the nearest level.

    Returns the continuous position ``c``, the first index of the top-k window
    and the gap used by the local linearisation.
    """

    key = ordinal(v, TN)
    klev = tl.load(level_ptr + key)
    left = tl.maximum(key - 1, 0)
    kgap = tl.where(
        v >= klev,
        tl.load(gap_ptr + key),
        tl.load(gap_ptr + left),
    )
    c = key.to(tl.float32) + (v - klev) / kgap
    start = tl.minimum(tl.maximum(key - TOPK // 2, 0), TN - TOPK)
    return c, start, kgap


@triton.jit
def cdl_weight(c, t, t_near, m, kappa):
    """Unnormalised CDL weight of level ``t`` (1.0 at the nearest level)."""

    d = c - t
    return tl.where(
        t == t_near,
        1.0,
        tl.math.exp2(tl.minimum(-kappa * d * d - m, 0.0) * LOG2E),
    )


@triton.jit
def cdl_sample(
    v,
    level_ptr,
    gap_ptr,
    kappa,
    u,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    ordinal,
):
    """Sample an index from the full top-k window given a uniform ``u``."""

    c, start, _ = cdl_window(v, level_ptr, gap_ptr, TN, TOPK, ordinal)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + TOPK - 1).to(tl.float32),
    ).to(tl.int32)
    m = -kappa * (t_near - c) * (t_near - c)

    z = tl.zeros_like(c)
    for j in tl.range(0, TOPK):
        t = (start + j).to(tl.float32)
        z += cdl_weight(c, t, t_near, m, kappa)

    acc = tl.zeros_like(c)
    cnt = tl.zeros_like(c).to(tl.int32)
    uz = u * z
    for j in tl.range(0, TOPK):
        t = (start + j).to(tl.float32)
        acc += cdl_weight(c, t, t_near, m, kappa)
        cnt += (acc <= uz).to(tl.int32)
    return start + tl.minimum(cnt, TOPK - 1)


@triton.jit
def cdl_sample_truncated(
    v,
    level_ptr,
    gap_ptr,
    kappa,
    u,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    ordinal,
):
    """Sample an index using the truncated q-side window (cascaded CDL)."""

    c, start, _ = cdl_window(v, level_ptr, gap_ptr, TN, TOPK, ordinal)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + TOPK - 1).to(tl.float32),
    ).to(tl.int32)
    m = -kappa * (t_near - c) * (t_near - c)
    hq = TOPK // 2
    if kappa > 0.05:
        hq = tl.minimum(TOPK // 2, tl.sqrt(60.0 / kappa)).to(tl.int32) + 1
    t_lo = tl.maximum(start, t_near - hq)

    z = tl.zeros_like(c)
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        inw = (t >= start) & (t <= start + (TOPK - 1))
        z += tl.where(inw, cdl_weight(c, t, t_near, m, kappa), 0.0)

    d_left = t_lo - 1 - c
    e_left = tl.math.exp2(tl.minimum(-kappa * d_left * d_left - m, 0.0) * LOG2E)
    off = tl.where(u * z >= e_left * (t_lo - start + 1.0), t_lo - start, 0)

    acc = tl.zeros_like(c)
    cnt = tl.zeros_like(c).to(tl.int32)
    uz = u * z
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        inw = (t >= start) & (t <= start + (TOPK - 1))
        acc += tl.where(inw, cdl_weight(c, t, t_near, m, kappa), 0.0)
        cnt += ((acc <= uz) & inw).to(tl.int32)
    sub = tl.minimum(off + cnt, TOPK - 1)
    return start + sub


@triton.jit
def cdl_proxy_coeffs(
    v,
    level_ptr,
    gap_ptr,
    kappa,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    ordinal,
):
    """Gradient coefficients of the CDL proxy over the full top-k window.

    Returns ``(dth_coeff, ds_coeff, dk_term)`` such that, for upstream gradient
    ``g`` and scale ``s``::

        dtheta = g * dth_coeff
        ds     = g * ds_coeff
        dkappa = sum(g * s * dk_term)
    """

    c, start, kgap = cdl_window(v, level_ptr, gap_ptr, TN, TOPK, ordinal)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + TOPK - 1).to(tl.float32),
    ).to(tl.int32)
    m = -kappa * (t_near - c) * (t_near - c)

    z = tl.zeros_like(c)
    sl = tl.zeros_like(c)
    sa = tl.zeros_like(c)
    sal = tl.zeros_like(c)
    sb = tl.zeros_like(c)
    sbl = tl.zeros_like(c)
    for j in tl.range(0, TOPK):
        t = (start + j).to(tl.float32)
        d = c - t
        e = cdl_weight(c, t, t_near, m, kappa)
        lv = tl.load(level_ptr + (start + j)).to(tl.float32)
        a = -2.0 * kappa * d
        z += e
        sl += e * lv
        sa += e * a
        sal += e * a * lv
        sb += e * d * d
        sbl += e * d * d * lv

    p_lt = sl / z
    dth_coeff = (sal / z - p_lt * (sa / z)) / kgap
    ds_coeff = p_lt - v * dth_coeff
    dk_term = p_lt * (sb / z) - sbl / z
    return dth_coeff, ds_coeff, dk_term


@triton.jit
def cdl_proxy_coeffs_truncated(
    v,
    level_ptr,
    gap_ptr,
    kappa,
    TN: tl.constexpr,
    TOPK: tl.constexpr,
    ordinal,
):
    """Gradient coefficients using the truncated q-side window."""

    c, start, kgap = cdl_window(v, level_ptr, gap_ptr, TN, TOPK, ordinal)

    t_near = tl.minimum(
        tl.maximum(tl.floor(c + 0.5), start.to(tl.float32)),
        (start + TOPK - 1).to(tl.float32),
    ).to(tl.int32)
    m = -kappa * (t_near - c) * (t_near - c)
    hq = TOPK // 2
    if kappa > 0.05:
        hq = tl.minimum(TOPK // 2, tl.sqrt(60.0 / kappa)).to(tl.int32) + 1
    t_lo = tl.maximum(start, t_near - hq)

    z = tl.zeros_like(c)
    sl = tl.zeros_like(c)
    sa = tl.zeros_like(c)
    sal = tl.zeros_like(c)
    sb = tl.zeros_like(c)
    sbl = tl.zeros_like(c)
    for j in tl.range(0, 2 * hq + 1):
        t = (t_lo + j).to(tl.float32)
        inw = (t >= start) & (t <= start + (TOPK - 1))
        d = c - t
        e = cdl_weight(c, t, t_near, m, kappa)
        ti = tl.minimum(t, (start + TOPK - 1).to(tl.float32)).to(tl.int32)
        lv = tl.load(level_ptr + ti).to(tl.float32)
        a = -2.0 * kappa * d
        w = tl.where(inw, e, 0.0)
        z += w
        sl += w * lv
        sa += w * a
        sal += w * a * lv
        sb += w * d * d
        sbl += w * d * d * lv

    p_lt = sl / z
    dth_coeff = (sal / z - p_lt * (sa / z)) / kgap
    ds_coeff = p_lt - v * dth_coeff
    dk_term = p_lt * (sb / z) - sbl / z
    return dth_coeff, ds_coeff, dk_term
