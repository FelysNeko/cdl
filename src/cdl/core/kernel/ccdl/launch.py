import torch
import triton

from cdl.core.kernel.ccdl.kernel import (
    _ccdl_bwd,
    _ccdl_bwd_2d,
    _ccdl_fwd,
    _ccdl_fwd_2d,
)
from cdl.core.kernel.formats.spec import FormatSpec
from cdl.core.kernel.layout import (
    flatten_q,
    flatten_q_2d,
    flatten_theta,
    flatten_theta_2d,
)


def _ccdl_buckets(m: int, nb: int) -> tuple[int, int]:
    return (0 if m <= 512 else (1 if m <= 2048 else 2)), (
        0 if nb <= 1 else (1 if nb <= 4 else 2)
    )


def _launch_ccdl_fwd_2d(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, lead, m, nb_r, nb_c, tpb = flatten_theta_2d(theta)
    h = nb_r * tpb
    k = nb_c * tpb
    ntiles = m * nb_r * nb_c
    lq, qs = flatten_q_2d(lead, nb_r, nb_c, q)

    is_fp4 = fmt.tn <= 16
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_theta = torch.empty(
        (m, h, nb_c * packed_cols), dtype=torch.uint8, device=theta.device
    )
    packed_scale = torch.empty((m, nb_r, nb_c), dtype=torch.uint8, device=theta.device)
    if ntiles > 0:
        _ccdl_fwd_2d[(ntiles,)](
            flat,
            fmt.theta_level,
            fmt.theta_gap,
            theta_kappa,
            fmt.theta_ordinal,
            lq,
            qs,
            fmt.q_level,
            fmt.q_gap,
            q_kappa,
            fmt.q_ordinal,
            alpha,
            seed,
            packed_theta,
            packed_scale,
            m,
            h,
            k,
            nb_r,
            nb_c,
            TPB=tpb,
            TN=fmt.tn,
            TOPK=theta_topk,
            QN=fmt.qn,
            QT=q_topk,
            SCALE_BIAS=fmt.scale_bias,
            num_warps=4,
        )

    packed_theta = packed_theta.reshape(*lead, nb_r, tpb, nb_c, packed_cols)
    if not is_fp4:
        packed_theta = packed_theta.view(torch.float8_e4m3fn)
    if qs == 0:
        packed_scale = packed_scale[:1].reshape(q.shape)
    else:
        packed_scale = packed_scale.reshape(*lead, nb_r, 1, nb_c, 1)
    return packed_theta, packed_scale


def launch_ccdl_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    block_2d: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_2d:
        return _launch_ccdl_fwd_2d(
            theta,
            theta_kappa,
            theta_topk,
            q,
            q_kappa,
            q_topk,
            alpha,
            seed,
            fmt,
        )
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    lq, qs = flatten_q(lead, nb, q)

    is_fp4 = fmt.tn <= 16
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_theta = torch.empty(
        (m, nb * packed_cols), dtype=torch.uint8, device=theta.device
    )
    packed_scale = torch.empty((m, nb), dtype=torch.uint8, device=theta.device)
    if m > 0 and nb > 0:
        m_bucket, nb_bucket = _ccdl_buckets(m, nb)
        _ccdl_fwd[lambda meta: (m, triton.cdiv(nb, meta["BB"]))](
            flat,
            fmt.theta_level,
            fmt.theta_gap,
            theta_kappa,
            fmt.theta_ordinal,
            lq,
            qs,
            fmt.q_level,
            fmt.q_gap,
            q_kappa,
            fmt.q_ordinal,
            alpha,
            seed,
            packed_theta,
            packed_scale,
            m,
            k,
            nb,
            M_BUCKET=m_bucket,
            NB_BUCKET=nb_bucket,
            TPB=tpb,
            TN=fmt.tn,
            TOPK=theta_topk,
            QN=fmt.qn,
            QT=q_topk,
            SCALE_BIAS=fmt.scale_bias,
        )

    packed_theta = packed_theta.reshape(*lead, nb, packed_cols)
    if not is_fp4:
        packed_theta = packed_theta.view(torch.float8_e4m3fn)
    if qs == 0:
        packed_scale = packed_scale[:1].reshape(q.shape)
    else:
        packed_scale = packed_scale.reshape(*lead, nb, 1)
    return packed_theta, packed_scale


def _launch_ccdl_bwd_2d(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    packed_scale: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    need_dq: bool = True,
    need_dkt: bool = True,
    need_dkq: bool = True,
    need_dal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb_r, nb_c, tpb = flatten_theta_2d(theta)
    h = nb_r * tpb
    k = nb_c * tpb
    ntiles = m * nb_r * nb_c
    g = grad_proxy.reshape(m, h, k).contiguous()
    lq, qs = flatten_q_2d(lead, nb_r, nb_c, q)
    ps_flat = packed_scale.reshape(-1, nb_r, nb_c).contiguous()

    dtheta = theta.new_empty((m, h, k), dtype=theta.dtype)
    need_scalars = need_dkt or need_dkq or need_dal
    dq = (
        torch.empty((m, nb_r, nb_c), dtype=torch.float32, device=theta.device)
        if need_dq
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    partials = (
        torch.empty((ntiles, 3), dtype=torch.float32, device=theta.device)
        if need_scalars
        else torch.empty(1, dtype=torch.float32, device=theta.device)
    )
    if ntiles > 0:
        _ccdl_bwd_2d[(ntiles,)](
            g,
            flat,
            fmt.theta_level,
            fmt.theta_gap,
            theta_kappa,
            fmt.theta_ordinal,
            lq,
            qs,
            fmt.q_level,
            fmt.q_gap,
            q_kappa,
            fmt.q_ordinal,
            ps_flat,
            0 if qs == 0 else nb_r * nb_c,
            alpha,
            dtheta,
            dq if need_dq else dtheta,
            partials,
            h,
            k,
            nb_r,
            nb_c,
            TPB=tpb,
            TN=fmt.tn,
            TOPK=theta_topk,
            QN=fmt.qn,
            QT=q_topk,
            SCALE_BIAS=fmt.scale_bias,
            NEED_DQ=need_dq,
            NEED_DKT=need_dkt,
            NEED_DKQ=need_dkq,
            NEED_DAL=need_dal,
            num_warps=4,
        )
    scalars = (
        partials.sum(0)
        if need_scalars
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    return dtheta, dq, scalars


def launch_ccdl_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    packed_scale: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    need_dq: bool = True,
    need_dkt: bool = True,
    need_dkq: bool = True,
    need_dal: bool = True,
    *,
    block_2d: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_2d:
        return _launch_ccdl_bwd_2d(
            grad_proxy,
            theta,
            theta_kappa,
            theta_topk,
            q,
            q_kappa,
            q_topk,
            packed_scale,
            alpha,
            fmt,
            need_dq,
            need_dkt,
            need_dkq,
            need_dal,
        )
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    lq, qs = flatten_q(lead, nb, q)
    ps_flat = packed_scale.reshape(-1, nb)

    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    need_scalars = need_dkt or need_dkq or need_dal
    dq = (
        torch.empty((m, nb), dtype=torch.float32, device=theta.device)
        if need_dq
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    partials = (
        torch.zeros((256, 3), dtype=torch.float32, device=theta.device)
        if need_scalars
        else torch.empty(1, dtype=torch.float32, device=theta.device)
    )
    if m > 0 and nb > 0:
        m_bucket, nb_bucket = _ccdl_buckets(m, nb)
        _ccdl_bwd[lambda meta: (m, triton.cdiv(nb, meta["BB"]))](
            g,
            flat,
            fmt.theta_level,
            fmt.theta_gap,
            theta_kappa,
            fmt.theta_ordinal,
            lq,
            qs,
            fmt.q_level,
            fmt.q_gap,
            q_kappa,
            fmt.q_ordinal,
            ps_flat,
            0 if qs == 0 else nb,
            alpha,
            dtheta,
            dq if need_dq else dtheta,
            partials,
            k,
            nb,
            k,
            M_BUCKET=m_bucket,
            NB_BUCKET=nb_bucket,
            TPB=tpb,
            TN=fmt.tn,
            TOPK=theta_topk,
            QN=fmt.qn,
            QT=q_topk,
            SCALE_BIAS=fmt.scale_bias,
            NEED_DQ=need_dq,
            NEED_DKT=need_dkt,
            NEED_DKQ=need_dkq,
            NEED_DAL=need_dal,
        )
    scalars = (
        partials.sum(0)
        if need_scalars
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    return dtheta, dq, scalars
