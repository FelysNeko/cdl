import torch


def cpmf(
    theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = -alpha * (theta[..., None] - a_hat) ** 2
    logits, idx = logits.topk(topk, dim=-1)
    return torch.softmax(logits, -1), a_hat[idx], idx


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


def weight_mpmf_entropy(
    theta: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    theta = theta.reshape(-1)
    logits = -alpha * (theta[:, None] - q * a) ** 2
    mpmf = torch.softmax(logits, -1).mean(0)

    tiny = torch.finfo(mpmf.dtype).tiny
    return -(mpmf * mpmf.clamp_min(tiny).log2()).sum()


def uniform_quant(theta: torch.Tensor) -> torch.Tensor:
    forward = torch.round(theta)
    backward = theta
    return backward + (forward - backward).detach()


def probabilistic_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    a: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    assert not a.requires_grad

    pmf, vals, _ = cpmf(theta, alpha, q * a, topk)
    return quant_from_cpmf(pmf, vals, topk)


def soft_deterministic_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    a: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    assert not a.requires_grad

    pmf, vals, _ = cpmf(theta, alpha, q * a, topk)
    return q_d(pmf, vals)
