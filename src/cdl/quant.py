import torch


def nearest_window(theta: torch.Tensor, a_hat: torch.Tensor, topk: int) -> torch.Tensor:
    size = a_hat.numel()
    width = 2 * topk
    position = torch.searchsorted(a_hat, theta)
    start = (position - topk).clamp(0, size - width)
    return start[..., None] + torch.arange(width, device=theta.device)


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


def mpmf_entropy(pmf: torch.Tensor, idx: torch.Tensor, size: int) -> torch.Tensor:
    numel = pmf.numel() // pmf.shape[-1]
    mpmf = pmf.new_zeros(size).index_add(0, idx.reshape(-1), pmf.reshape(-1))
    mpmf = mpmf / numel

    tiny = torch.finfo(mpmf.dtype).tiny
    return -(mpmf * mpmf.clamp_min(tiny).log2()).sum()
