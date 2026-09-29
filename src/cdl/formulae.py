import torch


def nearest_window(theta: torch.Tensor, a_hat: torch.Tensor, topk: int) -> torch.Tensor:
    size = a_hat.numel()
    position = torch.searchsorted(a_hat, theta)
    offset = torch.arange(-topk, topk, device=theta.device)
    return (position[..., None] + offset) % size


def cpmf(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = a.numel()
    a_hat = q * a

    if size <= 2 * topk:
        logits = -alpha * (theta[..., None] - a_hat) ** 2
        logits, idx = logits.topk(topk, dim=-1)
        return torch.softmax(logits, -1), a_hat[idx], idx

    idx = nearest_window(theta, a_hat, topk)
    cand = q * a[idx]
    logits = -alpha * (theta[..., None] - cand) ** 2
    logits, chosen = logits.topk(topk, dim=-1)
    return torch.softmax(logits, -1), cand.gather(-1, chosen), idx.gather(-1, chosen)


def q_d(pmf: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    return (pmf * vals).sum(-1)


def q_p(pmf: torch.Tensor, vals: torch.Tensor, topk: int) -> torch.Tensor:
    u = torch.rand_like(pmf[..., :1])
    idx = torch.searchsorted(pmf.cumsum(-1), u).clamp_(max=topk - 1)
    return vals.gather(-1, idx).squeeze(-1)


def quant_from_cpmf(pmf: torch.Tensor, vals: torch.Tensor, topk: int) -> torch.Tensor:
    forward = q_d(pmf, vals)

    with torch.no_grad():
        backward = q_p(pmf, vals, topk)

    return forward + (backward - forward).detach()


def mpmf_entropy(
    pmf: torch.Tensor, idx: torch.Tensor, size: int, group_size: int
) -> torch.Tensor:
    topk = pmf.shape[-1]
    groups = pmf.numel() // (group_size * topk)

    pmf = pmf.reshape(groups, group_size, topk)
    idx = idx.reshape(groups, group_size, topk)

    mpmf = pmf.new_zeros(groups, size)
    mpmf.scatter_add_(1, idx.reshape(groups, -1), pmf.reshape(groups, -1))
    mpmf = mpmf / group_size

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)
    return group_size * entropies.mean()
