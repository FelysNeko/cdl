import torch

from cdl.api.format import Format
from cdl.core import formulae


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor,
    topk: int,
    companded: bool,
    num_quantized_params: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    topk_pmf, topk_indices = formulae.topk_cpmf(
        theta, f, q, kappa_forward, topk, companded
    )

    entropies = None
    if num_quantized_params is not None:
        entropies = formulae.topk_mpmf_entropy(
            topk_pmf,
            topk_indices,
            f.n,
            num_quantized_params,
        )

    if kappa_backward == kappa_forward:
        proxy_pmf, proxy_indices = topk_pmf, topk_indices
    else:
        proxy_pmf, proxy_indices = formulae.topk_cpmf(
            theta, f, q, kappa_backward, topk, companded
        )

    proxy = formulae.topk_q_d(f.level, q, proxy_pmf, proxy_indices)
    chosen = formulae.topk_q_p_indices(topk_pmf, topk_indices)

    return chosen, proxy.to(theta.dtype), entropies


def compute_global_scale(
    theta: torch.Tensor,
    theta_format: Format,
    q_format: Format,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_b = (
        theta.detach()
        .float()
        .abs()
        .amax()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    f_max = float(theta_format.level.amax() * q_format.level.amax())
    return a_b, a_b / f_max * beta


@torch.no_grad()
def compute_block_scale(
    theta: torch.Tensor,
    q_max: torch.Tensor,
    global_abs_max: torch.Tensor,
) -> torch.Tensor:
    theta_f = theta.float()
    block_abs_max = theta_f.abs().amax(dim=-1, keepdim=True)
    q = block_abs_max * q_max.float() / global_abs_max
    return q


def extended_cdl_topk_forward(
    theta: torch.Tensor,
    theta_format: Format,
    theta_topk: int,
    q_format: Format,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor,
    beta: torch.Tensor,
    companded: bool,
    num_quantized_params: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    a_b, global_scale = compute_global_scale(theta, theta_format, q_format, beta)
    q = compute_block_scale(theta, q_format.level.amax(), a_b)
    q_indices = q_format.ordinal(q)
    q_hat = q_format.level[q_indices.long()]

    theta_indices, grad_proxy, entropies = cdl_topk_forward(
        theta,
        theta_format,
        q_hat * global_scale,
        kappa_forward,
        kappa_backward,
        theta_topk,
        companded,
        num_quantized_params,
    )

    return theta_indices, q_indices, grad_proxy, global_scale, entropies


@torch.no_grad()
def extended_nearest_round(
    theta: torch.Tensor,
    theta_format: Format,
    q_format: Format,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_b, global_scale = compute_global_scale(theta, theta_format, q_format, beta)
    q = compute_block_scale(theta, q_format.level.amax(), a_b)
    q_indices = q_format.ordinal(q)
    q_hat = q_format.level[q_indices.long()]
    scale = q_hat * global_scale
    theta_indices = theta_format.ordinal(theta / scale)
    return theta_indices, q_indices, global_scale
