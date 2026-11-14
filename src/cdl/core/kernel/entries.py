"""Format-specialised entry points for the quantizer kernels."""

import torch

from cdl.core.kernel.ccdl.autograd import ccdl_quant
from cdl.core.kernel.ecdl.autograd import ecdl_quant
from cdl.core.kernel.formats.spec import FormatId


def nvfp4_ccdl_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        FormatId.NVFP4,
        theta_topk,
        q_topk,
    )


def mxfp4_ccdl_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        FormatId.MXFP4,
        theta_topk,
        q_topk,
    )


def mxfp8_ccdl_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ccdl_quant(
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        FormatId.MXFP8,
        theta_topk,
        q_topk,
    )


def nvfp4_ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor,
    beta: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return ecdl_quant(
        theta,
        kappa_forward,
        kappa_backward,
        beta,
        seed,
        FormatId.NVFP4,
        theta_topk,
    )
