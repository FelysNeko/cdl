import torch
import triton

from cdl.core.kernel.config import DEFAULT_CONFIG, KernelConfig
from cdl.core.kernel.ecdl.kernel import _ecdl_absmax, _ecdl_bwd, _ecdl_fwd
from cdl.core.kernel.formats.spec import FormatSpec
from cdl.core.kernel.layout import flatten_theta


def launch_ecdl_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    beta: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    ngb = triton.cdiv(nb, config.block_chunk)

    bamax = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    _ecdl_absmax[(m, ngb)](
        flat,
        bamax,
        k,
        nb,
        TPB=tpb,
        BB=config.block_chunk,
        num_warps=config.num_warps,
    )
    global_absmax = bamax.max()
    alpha_eff = global_absmax / fmt.f_max * beta

    theta_indices = theta.new_empty((m, k), dtype=torch.int16)
    q_indices = torch.empty((m, nb), dtype=torch.uint8, device=theta.device)
    s = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
    _ecdl_fwd[(m, ngb)](
        flat,
        bamax,
        global_absmax,
        fmt.q_max,
        fmt.theta_level,
        fmt.theta_gap,
        fmt.q_level,
        theta_kappa,
        alpha_eff,
        seed,
        theta_indices,
        q_indices,
        s,
        k,
        nb,
        fmt.q_ordinal,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        TPB=tpb,
        TN=fmt.tn,
        QN=fmt.qn,
        BB=config.block_chunk,
        num_warps=config.num_warps,
    )

    theta_indices = theta_indices.reshape(*lead, nb, tpb)
    q_indices = q_indices.reshape(*lead, nb, 1)
    s = s.reshape(*lead, nb)
    return theta_indices, q_indices, s, alpha_eff


def launch_ecdl_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    s: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, _lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    s_flat = s.reshape(m, nb)

    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    ngb = triton.cdiv(nb, config.block_chunk)
    partials = torch.empty((m * ngb, 2), dtype=torch.float32, device=theta.device)
    _ecdl_bwd[(m, ngb)](
        g,
        flat,
        s_flat,
        fmt.theta_level,
        fmt.theta_gap,
        kappa_backward,
        alpha,
        dtheta,
        partials,
        k,
        nb,
        k,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        TPB=tpb,
        TN=fmt.tn,
        BB=config.block_chunk,
        num_warps=config.num_warps,
    )
    return dtheta, partials.sum(0)
