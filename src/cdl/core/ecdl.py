import torch

from cdl.api.format import Format
from cdl.core import formulae


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
    companded: bool,
    num_quantized_params: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    topk_pmf, topk_indices = formulae.topk_cpmf(theta, f, q, kappa, topk, companded)

    entropies = None
    if num_quantized_params is not None:
        entropies = formulae.topk_mpmf_entropy(
            topk_pmf,
            topk_indices,
            f.n,
            num_quantized_params,
        )

    proxy = formulae.topk_q_d(f.level, q, topk_pmf, topk_indices)
    chosen = formulae.topk_q_p_indices(topk_pmf, topk_indices)

    return chosen, proxy.to(theta.dtype), entropies


def compute_block_scale(theta: torch.Tensor, q_max: torch.Tensor) -> torch.Tensor:
    global_abs_max = theta.abs().amax().clamp_min(torch.finfo(theta.dtype).tiny)
    block_abs_max = theta.amax(dim=-1, keepdim=True)
    q = block_abs_max * q_max / global_abs_max
    return q


def extended_cdl_topk_forward(
    theta: torch.Tensor,
    theta_format: Format,
    theta_topk: int,
    q_format: Format,
    kappa: torch.Tensor,
    alpha: torch.Tensor,
    companded: bool,
    num_quantized_params: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    q = compute_block_scale(theta, q_format.level.amax())
    q_indices = q_format.ordinal(q)
    q_hat = q_format.level[q_indices]

    theta_indices, grad_proxy, entropies = cdl_topk_forward(
        theta,
        theta_format,
        q_hat * alpha,
        kappa,
        theta_topk,
        companded,
        num_quantized_params,
    )

    return theta_indices, q_indices, grad_proxy, entropies


@torch.no_grad()
def extended_nearest_round(
    theta: torch.Tensor,
    theta_format: Format,
    q_format: Format,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = compute_block_scale(theta, q_format.level.amax())
    q_indices = q_format.ordinal(q)
    q_hat = q_format.level[q_indices]
    scale = q_hat * alpha
    theta_indices = theta_format.ordinal(theta / scale)
    return theta_indices, q_indices
