import torch

from cdl.core.v1.navie import (
    cdl_topk_cpmf,
    cdl_topk_mpmf_entropy,
    cdl_topk_q_d,
    cdl_topk_q_p_indices,
)


def cdl_topk_forward(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
    num_quantized_params: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_pmf, topk_indices = cdl_topk_cpmf(theta, a, q, kappa, topk)

    entropies = cdl_topk_mpmf_entropy(
        topk_pmf,
        topk_indices,
        a.numel(),
        num_quantized_params,
    )

    proxy = cdl_topk_q_d(a, q, topk_pmf, topk_indices)
    chosen = cdl_topk_q_p_indices(topk_pmf, topk_indices)

    return chosen, proxy, entropies


def cascaded_cdl_topk_forward(
    theta: torch.Tensor,
    theta_lut: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    theta_num_quantized_params: int,
    q: torch.Tensor,
    q_lut: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_indices, q_proxy, q_entropies = cdl_topk_forward(
        q,
        q_lut,
        torch.ones(()),
        q_kappa,
        q_topk,
        q.numel(),
    )
    q_hat = q_lut[q_indices]
    proxied_q_hat = q_proxy + (q_hat - q_proxy).detach()
    theta_indices, theta_proxy, theta_entropies = cdl_topk_forward(
        theta,
        theta_lut,
        proxied_q_hat * alpha,
        theta_kappa,
        theta_topk,
        theta_num_quantized_params,
    )

    return theta_indices, q_indices, theta_proxy, q_entropies, theta_entropies
