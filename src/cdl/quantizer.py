import torch


def cpmf(
    theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = -alpha * (theta[..., None] - a_hat) ** 2
    logits, idx = logits.topk(topk, dim=-1)
    return torch.softmax(logits, -1), a_hat[idx]


def q_d(
    theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor, topk: int
) -> torch.Tensor:
    pmf, vals = cpmf(theta, alpha, a_hat, topk)
    return (pmf * vals).sum(-1)


def q_p(
    theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor, topk: int
) -> torch.Tensor:
    pmf, vals = cpmf(theta, alpha, a_hat, topk)
    u = torch.rand_like(pmf[..., :1])
    idx = torch.searchsorted(pmf.cumsum(-1), u).clamp_(max=topk - 1)
    return vals.gather(-1, idx).squeeze(-1)


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

    a_hat = q * a
    forward = q_d(theta, alpha, a_hat, topk)

    with torch.no_grad():
        backward = q_p(theta, alpha, a_hat, topk)

    return forward + (backward - forward).detach()


def soft_deterministic_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    a: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    assert not a.requires_grad
    return q_d(theta, alpha, q * a, topk)
