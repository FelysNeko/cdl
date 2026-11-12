import math

import torch

from cdl.core.kernel.ecdl.launch import launch_ecdl_bwd, launch_ecdl_fwd
from cdl.core.kernel.formats.spec import get_format


@torch.library.custom_op("cdl::ecdl_quant_fwd", mutates_args=())
def ecdl_quant_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_fwd(
        theta,
        theta_kappa,
        alpha,
        seed,
        fmt,
        theta_topk,
    )


@ecdl_quant_fwd.register_fake
def _ecdl_quant_fwd_fake(theta, theta_kappa, alpha, seed, format_id, theta_topk):
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((*lead, nb, tpb), dtype=torch.int16),
        theta.new_empty((*lead, nb, 1), dtype=torch.uint8),
        theta.new_empty((*lead, nb), dtype=torch.float32),
    )


@torch.library.custom_op("cdl::ecdl_quant_bwd", mutates_args=())
def ecdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    s: torch.Tensor,
    theta_kappa: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_bwd(
        grad_proxy,
        theta,
        s,
        theta_kappa,
        alpha,
        fmt,
        theta_topk,
    )


@ecdl_quant_bwd.register_fake
def _ecdl_quant_bwd_fake(
    grad_proxy, theta, s, theta_kappa, alpha, format_id, theta_topk
):
    m = math.prod(theta.shape[:-2])
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((m, nb * tpb), dtype=theta.dtype),
        theta.new_empty((2,), dtype=torch.float32),
    )


class EcdlQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        theta: torch.Tensor,
        theta_kappa: torch.Tensor,
        alpha: torch.Tensor,
        seed: torch.Tensor,
        format_id: int,
        theta_topk: int,
    ):
        seed.add_(1)
        theta_indices, q_indices, s = ecdl_quant_fwd(
            theta,
            theta_kappa,
            alpha,
            seed,
            format_id,
            theta_topk,
        )
        ctx.save_for_backward(theta, s, theta_kappa, alpha)
        ctx.format_id = format_id
        ctx.theta_topk = theta_topk
        carrier = theta.new_zeros(()).expand(theta.shape)
        return theta_indices, q_indices, carrier

    @staticmethod
    def backward(ctx, g_theta_indices, g_q_indices, g_proxy):
        theta, s, theta_kappa, alpha = ctx.saved_tensors
        dtheta, scalars = ecdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            s,
            theta_kappa,
            alpha,
            ctx.format_id,
            ctx.theta_topk,
        )
        return (
            dtheta.reshape(theta.shape),
            scalars[0],
            scalars[1],
            None,
            None,
            None,
        )


def ecdl_quant(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return EcdlQuantFn.apply(
        theta,
        theta_kappa,
        alpha,
        seed,
        int(format_id),
        theta_topk,
    )
