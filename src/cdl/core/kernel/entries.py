"""Format-specialised entry points for the quantizer kernels."""

import torch

from cdl.core.kernel.ccdl.autograd import ccdl_quant
from cdl.core.kernel.ecdl.autograd import ecdl_quant
from cdl.core.kernel.formats.spec import FormatId


def nvfp4_ccdl_quant(
    theta: torch.Tensor,
    theta_kappa_forward: torch.Tensor,
    theta_kappa_backward: torch.Tensor | None,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa_forward: torch.Tensor,
    q_kappa_backward: torch.Tensor | None,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        theta_kappa_forward,
        theta_kappa_backward,
        theta_topk,
        q,
        q_kappa_forward,
        q_kappa_backward,
        q_topk,
        alpha,
        seed,
        FormatId.NVFP4,
    )


def mxfp4_ccdl_quant(
    theta: torch.Tensor,
    theta_kappa_forward: torch.Tensor,
    theta_kappa_backward: torch.Tensor | None,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa_forward: torch.Tensor,
    q_kappa_backward: torch.Tensor | None,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        theta_kappa_forward,
        theta_kappa_backward,
        theta_topk,
        q,
        q_kappa_forward,
        q_kappa_backward,
        q_topk,
        alpha,
        seed,
        FormatId.MXFP4,
    )


def mxfp8_ccdl_quant(
    theta: torch.Tensor,
    theta_kappa_forward: torch.Tensor,
    theta_kappa_backward: torch.Tensor | None,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa_forward: torch.Tensor,
    q_kappa_backward: torch.Tensor | None,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        theta_kappa_forward,
        theta_kappa_backward,
        theta_topk,
        q,
        q_kappa_forward,
        q_kappa_backward,
        q_topk,
        alpha,
        seed,
        FormatId.MXFP8,
    )


def nvfp4_ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    seed: torch.Tensor,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return ecdl_quant(
        theta,
        kappa_forward,
        kappa_backward,
        seed,
        FormatId.NVFP4,
        theta_topk,
        "nvidia",
    )


def mxfp4_ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    seed: torch.Tensor,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return ecdl_quant(
        theta,
        kappa_forward,
        kappa_backward,
        seed,
        FormatId.MXFP4,
        theta_topk,
        "ocp",
    )


def mxfp8_ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    seed: torch.Tensor,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return ecdl_quant(
        theta,
        kappa_forward,
        kappa_backward,
        seed,
        FormatId.MXFP8,
        theta_topk,
        "ocp",
    )
