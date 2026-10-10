import torch

from cdl.core.format.format import Format


def cdl_topk_cpmf(
    fmt: Format,
    topk: int,
    theta: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = theta / q
    cont_pos, topk_level, topk_pos = fmt.window(v, topk)
    logits = -kappa * (cont_pos.unsqueeze(-1) - topk_pos.float()) ** 2
    topk_pmf = torch.softmax(logits, dim=-1)
    return topk_pmf, topk_level, topk_pos


@torch.no_grad()
def cdl_topk_q_p(
    topk_pmf: torch.Tensor,
    topk_level: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    cdf = topk_pmf.cumsum(-1)
    quantile = torch.rand(topk_pmf.shape[:-1], dtype=cdf.dtype, device=cdf.device)
    sub_indices = (cdf <= quantile.unsqueeze(-1)).sum(-1, keepdim=True)
    sub_indices.clamp_(max=topk - 1)
    q_p = topk_level.gather(dim=-1, index=sub_indices).squeeze(-1)
    return q_p * q


def cdl_topk_q_d(
    topk_pmf: torch.Tensor,
    topk_level: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    q_d = (topk_pmf * topk_level).sum(-1)
    return q_d * q


def cdl_topk_mpmf_entropy(
    topk_pmf: torch.Tensor,
    topk_pos: torch.Tensor,
    num_slots: int,
    num_quantized_params: int,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    groups = topk_pmf.numel() // (num_quantized_params * topk)
    pmf = topk_pmf.reshape(groups, num_quantized_params, topk)
    idx = topk_pos.reshape(groups, num_quantized_params, topk)

    mpmf = pmf.new_zeros(groups, num_slots)
    mpmf.scatter_add_(1, idx.reshape(groups, -1), pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)

    return num_quantized_params * entropies.mean()


def cdl_topk_forward(
    fmt: Format,
    relaxed: bool,
    num_quantized_params: int,
    topk: int,
    theta: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_pmf, topk_level, topk_pos = cdl_topk_cpmf(fmt, topk, theta, q, kappa)
    entropy = cdl_topk_mpmf_entropy(topk_pmf, topk_pos, fmt.n, num_quantized_params)
    q_d = cdl_topk_q_d(topk_pmf, topk_level, q)

    if relaxed:
        return q_d, entropy

    q_p = cdl_topk_q_p(topk_pmf, topk_level, q)

    return q_d + (q_p - q_d).detach(), entropy


@torch.no_grad()
def quant_nearest(
    fmt: Format,
    theta: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    return fmt.level[fmt.ordinal(theta / q)] * q
