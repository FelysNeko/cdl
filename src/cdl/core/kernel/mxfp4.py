import torch

from cdl.core.kernel.backbone.ccdl import CdlQuantFn, quant_bwd, quant_fwd
from cdl.core.kernel.backbone.format import (
    e2m1_triton_ordinal,
    e8m0_triton_ordinal,
)
from cdl.core.tables import e2m1_tables, e8m0_tables

TN = 16
QN = 255


@torch.library.custom_op("cdl::mxfp4_quant_fwd", mutates_args=())
def mxfp4_quant_fwd(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_level, theta_gap = e2m1_tables(torch.float32, theta.device)
    q_level, q_gap = e8m0_tables(torch.float32, theta.device)
    return quant_fwd(
        theta,
        q,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        theta_topk,
        TN,
        QN,
        q_topk,
        e8m0_triton_ordinal,
        e2m1_triton_ordinal,
    )


@mxfp4_quant_fwd.register_fake
def _(theta, q, theta_kappa, q_kappa, alpha, seed, theta_topk, q_topk):
    m, nb, tpb = theta.shape
    rq = q.shape[0]
    return (
        theta.new_empty((m, nb * tpb), dtype=torch.int16),
        theta.new_empty((1 if rq == 1 else m, nb), dtype=torch.uint8),
        theta.new_empty((m, nb), dtype=torch.float32),
    )


@torch.library.custom_op("cdl::mxfp4_quant_bwd", mutates_args=())
def mxfp4_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_level, theta_gap = e2m1_tables(torch.float32, theta.device)
    q_level, q_gap = e8m0_tables(torch.float32, theta.device)
    return quant_bwd(
        grad_proxy,
        theta,
        q,
        s,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        theta_kappa,
        q_kappa,
        alpha,
        theta_topk,
        TN,
        QN,
        q_topk,
        e8m0_triton_ordinal,
        e2m1_triton_ordinal,
    )


@mxfp4_quant_bwd.register_fake
def _(grad_proxy, theta, q, s, theta_kappa, q_kappa, alpha, theta_topk, q_topk):
    m, nb, tpb = theta.shape
    return (
        theta.new_empty((m, nb * tpb), dtype=theta.dtype),
        theta.new_empty((m, nb), dtype=torch.float32),
        theta.new_empty((3,), dtype=torch.float32),
    )


def mxfp4_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return CdlQuantFn.apply(
        mxfp4_quant_fwd,
        mxfp4_quant_bwd,
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        theta_topk,
        q_topk,
    )
