import torch


def cdl_topk_cpmf(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = -kappa * ((theta / q).unsqueeze(-1) - a) ** 2
    topk_values, topk_indices = logits.topk(topk)
    topk_pmf = torch.softmax(topk_values, dim=-1)
    return topk_pmf, topk_indices


def cdl_topk_mpmf_entropy(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
    num_slots: int,
    num_quantized_params: int,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    groups = topk_pmf.numel() // (num_quantized_params * topk)
    idx = topk_indices.reshape(groups, num_quantized_params, topk)

    mpmf = topk_pmf.new_zeros(groups, num_slots)
    mpmf.scatter_add_(1, idx.reshape(groups, -1), topk_pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)
    return num_quantized_params * entropies.mean()


@torch.no_grad()
def cdl_topk_q_p_indices(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    cdf = topk_pmf.cumsum(-1)
    quantile = torch.rand(topk_pmf.shape[:-1], dtype=cdf.dtype, device=cdf.device)
    sub_indices = (cdf <= quantile.unsqueeze(-1)).sum(-1, keepdim=True)
    sub_indices.clamp_(max=topk - 1)
    chosen = topk_indices.gather(dim=-1, index=sub_indices).squeeze(-1)
    return chosen

@torch.no_grad()
def cdl_topk_q_p(
    a: torch.Tensor,
    q: torch.Tensor,
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    chosen = cdl_topk_q_p_indices(topk_pmf, topk_indices)
    return a[chosen] * q


def cdl_topk_q_d(
    a: torch.Tensor,
    q: torch.Tensor,
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    return (topk_pmf * a[topk_indices]).sum(-1) * q



def cdl_topk_forward(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    relaxed: bool,
    num_quantized_params: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_pmf, topk_indices = cdl_topk_cpmf(theta, a, q, kappa, topk)

    entropies = cdl_topk_mpmf_entropy(
        topk_pmf,
        topk_indices,
        a.numel(),
        num_quantized_params,
    )

    q_d = cdl_topk_q_d(a, q, topk_pmf, topk_indices)

    if relaxed:
        return q_d, entropies

    q_p = cdl_topk_q_p(a, q, topk_pmf, topk_indices)

    return q_d + (q_p - q_d).detach(), entropies
