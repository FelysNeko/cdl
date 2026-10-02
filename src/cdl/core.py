import torch
from torch.distributions import Categorical


def cdl_topk_cpmf(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_numel = a.numel()
    with torch.no_grad():
        center = (theta / q - a[0]).round_().clamp_(0, a_numel - 1)
        start = (center - topk // 2).clamp_(0, a_numel - topk)
    topk_indices = start.long().unsqueeze(-1) + torch.arange(topk, device=theta.device)

    topk_a_hat = q * a[topk_indices]
    logits = -alpha * (theta.unsqueeze(-1) - topk_a_hat) ** 2
    topk_pmf = torch.softmax(logits, dim=-1)
    return topk_pmf, topk_a_hat, topk_indices


def cdl_topk_q_p(
    topk_pmf: torch.Tensor,
    topk_a_hat: torch.Tensor,
) -> torch.Tensor:
    sub_indices = Categorical(probs=topk_pmf).sample().unsqueeze(-1)
    return topk_a_hat.gather(dim=-1, index=sub_indices).squeeze(-1)


def cdl_topk_q_d(
    topk_pmf: torch.Tensor,
    topk_a_hat: torch.Tensor,
) -> torch.Tensor:
    return (topk_pmf * topk_a_hat).sum(-1)


def cdl_topk_mpmf_entropy(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
    num_slots: int,
    num_quantized_params: int,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    groups = topk_pmf.numel() // (num_quantized_params * topk)
    pmf = topk_pmf.reshape(groups, num_quantized_params, topk)
    idx = topk_indices.reshape(groups, num_quantized_params, topk)

    mpmf = pmf.new_zeros(groups, num_slots)
    mpmf.scatter_add_(1, idx.reshape(groups, -1), pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)

    return num_quantized_params * entropies.mean()


def cdl_topk_train_forward(
    relaxed: bool,
    num_quantized_params: int,
    topk: int,
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_pmf, topk_a_hat, topk_indices = cdl_topk_cpmf(theta, a, q, alpha, topk)
    entropy = cdl_topk_mpmf_entropy(
        topk_pmf, topk_indices, a.numel(), num_quantized_params
    )
    q_d = cdl_topk_q_d(topk_pmf, topk_a_hat)

    if relaxed:
        return q_d, entropy

    with torch.no_grad():
        q_p = cdl_topk_q_p(topk_pmf, topk_a_hat)

    return q_d + (q_p - q_d).detach(), entropy


@torch.no_grad()
def cdl_topk_infer_sample(
    topk: int,
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    topk_pmf, topk_a_hat, _ = cdl_topk_cpmf(theta, a, q, alpha, topk)
    return cdl_topk_q_p(topk_pmf, topk_a_hat)
