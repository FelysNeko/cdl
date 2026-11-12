import torch
import triton

from cdl.core.kernel.ccdl.kernel import _ccdl_bwd, _ccdl_fwd
from cdl.core.kernel.config import DEFAULT_CONFIG, KernelConfig
from cdl.core.kernel.formats.spec import FormatSpec
from cdl.core.kernel.layout import flatten_q, flatten_theta


def launch_ccdl_fwd(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    q_topk: int,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    lq, qs = flatten_q(lead, nb, q)

    theta_indices = theta.new_empty((m, k), dtype=torch.int16)
    q_indices = torch.empty((m, nb), dtype=torch.uint8, device=theta.device)
    s = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    _ccdl_fwd[(m, triton.cdiv(nb, config.block_chunk))](
        flat,
        lq,
        qs,
        fmt.theta_level,
        fmt.theta_gap,
        fmt.q_level,
        fmt.q_gap,
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
        fmt.q_ordinal,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        QT=q_topk,
        TPB=tpb,
        TN=fmt.tn,
        QN=fmt.qn,
        BB=config.block_chunk,
        num_warps=config.num_warps,
    )

    theta_indices = theta_indices.reshape(*lead, nb, tpb)
    if qs == 0:
        q_indices = q_indices[:1].reshape(q.shape)
    else:
        q_indices = q_indices.reshape(*lead, nb, 1)
    s = s.reshape(*lead, nb)
    return theta_indices, q_indices, s


def launch_ccdl_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    q_topk: int,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    s_flat = s.reshape(m, nb)
    lq, qs = flatten_q(lead, nb, q)

    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    dq = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    ngb = triton.cdiv(nb, config.block_chunk)
    partials = torch.empty((m * ngb, 3), dtype=torch.float32, device=theta.device)
    _ccdl_bwd[(m, ngb)](
        g,
        flat,
        lq,
        qs,
        s_flat,
        fmt.theta_level,
        fmt.theta_gap,
        fmt.q_level,
        fmt.q_gap,
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
        fmt.q_ordinal,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        QT=q_topk,
        TPB=tpb,
        TN=fmt.tn,
        QN=fmt.qn,
        BB=config.block_chunk,
        num_warps=config.num_warps,
    )
    return dtheta, dq, partials.sum(0)
