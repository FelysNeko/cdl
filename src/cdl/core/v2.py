import torch

from cdl.core import formulae
from cdl.api.format import Format


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

    return chosen, proxy, entropies


def cascaded_cdl_topk_forward(
    theta: torch.Tensor,
    theta_format: Format,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_format: Format,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
    companded: bool,
    q_num_quantized_params: int | None = None,
    theta_num_quantized_params: int | None = None,
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
        q_topk,
        companded,
        q_num_quantized_params,
    )
    q_hat = q_format.level[q_indices]
    proxied_q_hat = q_proxy + (q_hat - q_proxy).detach()
    theta_indices, theta_proxy, theta_entropies = cdl_topk_forward(
        theta,
        theta_format,
        proxied_q_hat * alpha,
        theta_kappa,
        theta_topk,
        companded,
        theta_num_quantized_params,
    )

    return theta_indices, q_indices, theta_proxy, q_entropies, theta_entropies


@torch.no_grad()
def cascaded_nearest_round(
    theta: torch.Tensor,
    theta_format: Format,
    q: torch.Tensor,
    q_format: Format,
    alpha: torch.Tensor,
) -> torch.Tensor:
    q_hat = q_format.quantize(q, q.new_ones(()))
    return theta_format.quantize(theta, q_hat * alpha)
