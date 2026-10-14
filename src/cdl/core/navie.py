import torch


def cdl_topk_forward(
    relaxed: bool,
    num_quantized_params: int,
    topk: int,
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = -kappa * (theta.unsqueeze(-1) / q - a) ** 2
    topk_values, topk_indices = logits.topk(topk)
    pmf = torch.softmax(topk_values, dim=-1)

    groups = pmf.numel() // (num_quantized_params * topk)
    idx = topk_indices.reshape(groups, num_quantized_params, topk)

    mpmf = pmf.new_zeros(groups, a.numel())
    mpmf.scatter_add_(1, idx.reshape(groups, -1), pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)
    entropies = num_quantized_params * entropies.mean()

    q_d = (pmf * a[topk_indices]).sum(-1) * q

    if relaxed:
        return q_d, entropies

    with torch.no_grad():
        cdf = pmf.cumsum(-1)
        quantile = torch.rand(pmf.shape[:-1], dtype=cdf.dtype, device=cdf.device)
        sub_indices = (cdf <= quantile.unsqueeze(-1)).sum(-1, keepdim=True)
        sub_indices.clamp_(max=topk - 1)
        chosen = topk_indices.gather(dim=-1, index=sub_indices).squeeze(-1)
        q_p = a[chosen] * q

    return q_d + (q_p - q_d).detach(), entropies
