import torch

from cdl.core.format.format import Format


def topk_cpmf_companded(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = theta / q
    cont_pos, start = f.window(v, topk)
    offset = torch.arange(topk, device=v.device, dtype=start.dtype)
    topk_indices = start.unsqueeze(-1) + offset
    logits = -kappa * (cont_pos.unsqueeze(-1) - topk_indices.float()) ** 2
    topk_pmf = torch.softmax(logits, dim=-1)
    return topk_pmf, topk_indices


def topk_cpmf_naive(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = -kappa * ((theta / q).unsqueeze(-1) - a) ** 2
    topk_values, topk_indices = logits.topk(topk)
    topk_pmf = torch.softmax(topk_values, dim=-1)
    return topk_pmf, topk_indices


def topk_cpmf(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
    companded: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if companded:
        return topk_cpmf_companded(theta, f, q, kappa, topk)
    else:
        return topk_cpmf_naive(theta, f.level, q, kappa, topk)


@torch.no_grad()
def topk_q_p_indices(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    cdf = topk_pmf.cumsum(-1)
    quantile = torch.rand(topk_pmf.shape[:-1], dtype=cdf.dtype, device=cdf.device)
    sub_indices = (cdf <= quantile.unsqueeze(-1)).sum(-1, keepdim=True)
    sub_indices.clamp_(max=topk - 1)
    chosen = topk_indices.gather(dim=-1, index=sub_indices).squeeze(-1)
    return chosen.long()


@torch.no_grad()
def topk_q_p(
    f: Format,
    q: torch.Tensor,
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    chosen = topk_q_p_indices(topk_pmf, topk_indices)
    return f.level[chosen] * q


def topk_q_d(
    f: Format,
    q: torch.Tensor,
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    return f.expectation(topk_pmf, topk_indices) * q


def topk_mpmf_entropy(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
    num_slots: int,
    num_quantized_params: int,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    groups = topk_pmf.numel() // (num_quantized_params * topk)

    mpmf = topk_pmf.new_zeros(groups, num_slots)
    mpmf.scatter_add_(
        1,
        topk_indices.reshape(groups, -1).long(),
        topk_pmf.reshape(groups, -1),
    )
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1).mean()
    return num_quantized_params * entropies
