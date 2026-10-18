import torch

from cdl.core import formulae
from cdl.core.format.format import Format


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    num_quantized_params: int,
    topk: int,
    companded: bool,
    requires_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    topk_pmf, topk_indices = formulae.topk_cpmf(theta, f, q, kappa, topk, companded)

    entropies = None
    if requires_entropy:
        entropies = formulae.topk_mpmf_entropy(
            topk_pmf,
            topk_indices,
            f.n,
            num_quantized_params,
        )

    proxy = formulae.topk_q_d(f, q, topk_pmf, topk_indices)
    chosen = formulae.topk_q_p_indices(topk_pmf, topk_indices)

    return chosen, proxy, entropies


def cascaded_cdl_topk_forward(
    theta: torch.Tensor,
    theta_format: Format,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    theta_num_quantized_params: int,
    theta_requires_entropy: bool,
    q: torch.Tensor,
    q_format: Format,
    q_kappa: torch.Tensor,
    q_topk: int,
    q_requires_entropy: bool,
    alpha: torch.Tensor,
    companded: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    q_indices, q_proxy, q_entropies = cdl_topk_forward(
        q,
        q_format,
        torch.ones(()),
        q_kappa,
        q.numel(),
        q_topk,
        companded,
        q_requires_entropy,
    )
    q_hat = q_format.level[q_indices]
    proxied_q_hat = q_proxy + (q_hat - q_proxy).detach()
    theta_indices, theta_proxy, theta_entropies = cdl_topk_forward(
        theta,
        theta_format,
        proxied_q_hat * alpha,
        theta_kappa,
        theta_num_quantized_params,
        theta_topk,
        companded,
        theta_requires_entropy,
    )

    return theta_indices, q_indices, theta_proxy, q_entropies, theta_entropies
