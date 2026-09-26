import torch
from torch.distributions import Categorical


def cpmf(theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
    d = (theta[..., None] - a_hat) ** 2  # [..., 1] - [..., K] = [..., K]
    logits = -alpha * d
    logits = logits - logits.amax(-1, keepdim=True)  # guards against underflow
    e = torch.exp(logits)
    return e / e.sum(-1, keepdim=True)


def q_d(theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
    return cpmf(theta, alpha, a_hat) @ a_hat


def q_p(theta: torch.Tensor, alpha: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
    pmf = cpmf(theta, alpha, a_hat)
    idx = Categorical(probs=pmf).sample()
    return a_hat[idx]


def uniform_quant(theta: torch.Tensor) -> torch.Tensor:
    forward = torch.round(theta)
    backward = theta
    return backward + (forward - backward).detach()


def probabilistic_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    assert not a.requires_grad

    a_hat = q * a
    forward = q_d(theta, alpha, a_hat)

    with torch.no_grad():
        backward = q_p(theta, alpha, a_hat)

    return forward + (backward - forward).detach()


def soft_deterministic_quant(
    theta: torch.Tensor, q: torch.Tensor, alpha: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    assert not a.requires_grad
    return q_d(theta, alpha, q * a)
