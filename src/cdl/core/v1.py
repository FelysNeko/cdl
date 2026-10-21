import torch

from cdl.core import formulae
from cdl.core.format.format import Format


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
    companded: bool,
    relaxed: bool,
    num_quantized_params: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    topk_pmf, topk_indices = formulae.topk_cpmf(theta, f, q, kappa, topk, companded)

    entropies = None
    if num_quantized_params is not None:
        entropies = formulae.topk_mpmf_entropy(
            topk_pmf,
            topk_indices,
            f.n,
            num_quantized_params,
        )

    q_d = formulae.topk_q_d(f.level, q, topk_pmf, topk_indices)

    if relaxed:
        return q_d, entropies

    q_p = formulae.topk_q_p(f, q, topk_pmf, topk_indices)

    return q_d + (q_p - q_d).detach(), entropies


@torch.no_grad()
def nearest_round(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
) -> torch.Tensor:
    return f.quantize(theta, q)
