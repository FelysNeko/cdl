import math
from typing import Literal

import torch

from cdl.api.format import Format
from cdl.core import formulae


def cdl_topk_forward(
    theta: torch.Tensor,
    f: Format,
    q: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    topk: int,
    companded: bool,
    num_quantized_params: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    topk_pmf, topk_indices = formulae.topk_cpmf(
        theta, f, q, kappa_forward, topk, companded
    )

    entropies = None
    if num_quantized_params is not None:
        entropies = formulae.topk_mpmf_entropy(
            topk_pmf,
            topk_indices,
            f.n,
            num_quantized_params,
        )

    if kappa_backward is None:
        proxy_pmf, proxy_indices = topk_pmf, topk_indices
    else:
        proxy_pmf, proxy_indices = formulae.topk_cpmf(
            theta, f, q, kappa_backward, topk, companded
        )

    proxy = formulae.topk_q_d(f.level, q, proxy_pmf, proxy_indices)
    chosen = formulae.topk_q_p_indices(topk_pmf, topk_indices)

    return chosen, proxy.to(theta.dtype), entropies


@torch.no_grad()
def compute_nvidia_scale(
    theta: torch.Tensor,
    theta_format: Format,
    q_format: Format,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_b = theta.detach().float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    f_max = float(theta_format.level.amax() * q_format.level.amax())
    block_abs_max = theta.float().abs().amax(dim=-1, keepdim=True)
    q = block_abs_max * q_format.level.amax().float() / a_b
    return q_format.ordinal(q), a_b / f_max


@torch.no_grad()
def compute_ocp_scale(
    theta: torch.Tensor,
    theta_format: Format,
) -> tuple[torch.Tensor, torch.Tensor]:
    emax = math.floor(math.log2(float(theta_format.level.amax())))
    block_abs_max = (
        theta.float()
        .abs()
        .amax(dim=-1, keepdim=True)
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    byte = (block_abs_max.log2().floor() - emax + 127).clamp(0, 254)
    ones = torch.ones((), device=theta.device, dtype=torch.float32)
    return byte.to(torch.uint8), ones


def extended_cdl_topk_forward(
    theta: torch.Tensor,
    theta_format: Format,
    theta_topk: int,
    q_format: Format,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    companded: bool,
    mode: Literal["ocp", "nvidia"],
    num_quantized_params: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    if mode == "ocp":
        q_indices, global_scale = compute_ocp_scale(theta, theta_format)
    elif mode == "nvidia":
        q_indices, global_scale = compute_nvidia_scale(theta, theta_format, q_format)
    else:
        raise ValueError(f"unknown scale mode {mode!r}")
    q_hat = q_format.level[q_indices.long()]

    theta_indices, grad_proxy, entropies = cdl_topk_forward(
        theta,
        theta_format,
        q_hat * global_scale,
        kappa_forward,
        kappa_backward,
        theta_topk,
        companded,
        num_quantized_params,
    )

    return theta_indices, q_indices, grad_proxy, global_scale, entropies


@torch.no_grad()
def extended_nearest_round(
    theta: torch.Tensor,
    theta_format: Format,
    q_format: Format,
    mode: Literal["ocp", "nvidia"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "ocp":
        q_indices, global_scale = compute_ocp_scale(theta, theta_format)
    elif mode == "nvidia":
        q_indices, global_scale = compute_nvidia_scale(theta, theta_format, q_format)
    else:
        raise ValueError(f"unknown scale mode {mode!r}")
    q_hat = q_format.level[q_indices.long()]
    scale = q_hat * global_scale
    theta_indices = theta_format.ordinal(theta / scale)
    return theta_indices, q_indices, global_scale
