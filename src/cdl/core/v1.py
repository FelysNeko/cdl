import torch

from cdl.core import formulae
from cdl.core.format.format import Format


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa: torch.Tensor,
    num_quantized_params: int,
    topk: int,
    companded: bool,
    relaxed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_pmf, topk_indices = formulae.topk_cpmf(theta, f, q, kappa, topk, companded)

    entropy = formulae.topk_mpmf_entropy(
        topk_pmf,
        topk_indices,
        f.n,
        num_quantized_params,
    )

    q_d = formulae.topk_q_d(f, q, topk_pmf, topk_indices)

    if relaxed:
        return q_d, entropy

    q_p = formulae.topk_q_p(f, q, topk_pmf, topk_indices)

    return q_d + (q_p - q_d).detach(), entropy


@torch.no_grad()
def quant_nearest(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
) -> torch.Tensor:
    return f.level[f.ordinal(theta / q)] * q
