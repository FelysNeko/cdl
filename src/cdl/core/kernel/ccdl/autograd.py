import math

import torch

from cdl.core.kernel.ccdl.launch import launch_ccdl_bwd, launch_ccdl_fwd
from cdl.core.kernel.formats.spec import get_format
from cdl.core.kernel.layout import q_indices_shape, reduce_q_grad


@torch.library.custom_op("cdl::ccdl_quant_fwd", mutates_args=())
def ccdl_quant_fwd(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ccdl_fwd(
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        fmt,
        theta_topk,
        q_topk,
    )


@ccdl_quant_fwd.register_fake
def _ccdl_quant_fwd_fake(
    theta, q, theta_kappa, q_kappa, alpha, seed, format_id, theta_topk, q_topk
):
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((*lead, nb, tpb), dtype=torch.int16),
        theta.new_empty(q_indices_shape(lead, nb, q), dtype=torch.uint8),
        theta.new_empty((*lead, nb), dtype=torch.float32),
    )


@torch.library.custom_op("cdl::ccdl_quant_bwd", mutates_args=())
def ccdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ccdl_bwd(
        grad_proxy,
        theta,
        q,
        s,
        theta_kappa,
        q_kappa,
        alpha,
        fmt,
        theta_topk,
        q_topk,
    )


@ccdl_quant_bwd.register_fake
def _ccdl_quant_bwd_fake(
    grad_proxy,
    theta,
    q,
    s,
    theta_kappa,
    q_kappa,
    alpha,
    format_id,
    theta_topk,
    q_topk,
):
    m = math.prod(theta.shape[:-2])
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((m, nb * tpb), dtype=theta.dtype),
        theta.new_empty((m, nb), dtype=torch.float32),
        theta.new_empty((3,), dtype=torch.float32),
    )


class CcdlQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        theta: torch.Tensor,
        q: torch.Tensor,
        theta_kappa: torch.Tensor,
        q_kappa: torch.Tensor,
        alpha: torch.Tensor,
        seed: torch.Tensor,
        format_id: int,
        theta_topk: int,
        q_topk: int,
    ):
        seed.add_(1)
        theta_indices, q_indices, s = ccdl_quant_fwd(
            theta,
            q,
            theta_kappa,
            q_kappa,
            alpha,
            seed,
            format_id,
            theta_topk,
            q_topk,
        )
        ctx.save_for_backward(theta, q, s, theta_kappa, q_kappa, alpha)
        ctx.format_id = format_id
        ctx.topk = (theta_topk, q_topk)
        carrier = theta.new_zeros(()).expand(theta.shape)
        return theta_indices, q_indices, carrier

    @staticmethod
    def backward(ctx, g_theta_indices, g_q_indices, g_proxy):
        theta, q, s, theta_kappa, q_kappa, alpha = ctx.saved_tensors
        theta_topk, q_topk = ctx.topk
        dtheta, dq, scalars = ccdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            q,
            s,
            theta_kappa,
            q_kappa,
            alpha,
            ctx.format_id,
            theta_topk,
            q_topk,
        )
        return (
            dtheta.reshape(theta.shape),
            reduce_q_grad(dq, theta, q),
            scalars[0],
            scalars[1],
            scalars[2],
            None,
            None,
            None,
            None,
        )


def ccdl_quant(
    theta: torch.Tensor,
    q: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
    q_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return CcdlQuantFn.apply(
        theta,
        q,
        theta_kappa,
        q_kappa,
        alpha,
        seed,
        int(format_id),
        theta_topk,
        q_topk,
    )
