import math

import torch

from cdl.core.kernel.ecdl.launch import launch_ecdl_bwd, launch_ecdl_fwd
from cdl.core.kernel.formats.spec import get_format


@torch.library.custom_op("cdl::ecdl_quant_fwd", mutates_args=())
def ecdl_quant_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    beta: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_fwd(
        theta,
        theta_kappa,
        beta,
        seed,
        fmt,
        theta_topk,
    )


@ecdl_quant_fwd.register_fake
def _ecdl_quant_fwd_fake(theta, theta_kappa, beta, seed, format_id, theta_topk):
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((*lead, nb, tpb), dtype=torch.int16),
        theta.new_empty((*lead, nb, 1), dtype=torch.uint8),
        theta.new_empty((*lead, nb), dtype=torch.float32),
        beta.new_empty(()),
    )


@torch.library.custom_op("cdl::ecdl_quant_bwd", mutates_args=())
def ecdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    s: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_bwd(
        grad_proxy,
        theta,
        s,
        kappa_backward,
        alpha,
        fmt,
        theta_topk,
    )


@ecdl_quant_bwd.register_fake
def _ecdl_quant_bwd_fake(
    grad_proxy, theta, s, kappa_backward, alpha, format_id, theta_topk
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
        kappa_forward: torch.Tensor,
        kappa_backward: torch.Tensor,
        beta: torch.Tensor,
        seed: torch.Tensor,
        format_id: int,
        theta_topk: int,
    ):
        seed.add_(1)
        theta_indices, q_indices, s, alpha_eff = ecdl_quant_fwd(
            theta,
            kappa_forward,
            beta,
            seed,
            format_id,
            theta_topk,
        )
        ctx.save_for_backward(
            theta, s, kappa_forward, kappa_backward, beta, alpha_eff
        )
        ctx.format_id = format_id
        ctx.theta_topk = theta_topk
        carrier = theta.new_zeros(()).expand(theta.shape)
        ctx.mark_non_differentiable(alpha_eff)
        return theta_indices, q_indices, carrier, alpha_eff

    @staticmethod
    def backward(ctx, g_theta_indices, g_q_indices, g_proxy, g_alpha_eff):
        theta, s, kappa_forward, kappa_backward, beta, alpha_eff = (
            ctx.saved_tensors
        )
        # The kernel returns d(alpha_eff); alpha_eff = A_b / f_max * beta, so
        # d(beta) = d(alpha_eff) * alpha_eff / beta.
        dtheta, scalars = ecdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            s,
            kappa_backward,
            alpha_eff,
            ctx.format_id,
            ctx.theta_topk,
        )
        dbeta = scalars[1] * alpha_eff / beta
        # The kernel differentiates with respect to the backward kappa.  If it
        # equals the forward kappa, the same value is the forward gradient, so
        # route it back to log_kappa and keep the buffer gradient empty.
        if kappa_backward == kappa_forward:
            dkappa_forward = scalars[0]
            dkappa_backward = None
        else:
            dkappa_forward = None
            dkappa_backward = scalars[0]
        return (
            dtheta.reshape(theta.shape),
            dkappa_forward,
            dkappa_backward,
            dbeta,
            None,
            None,
            None,
        )


def ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor,
    beta: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return EcdlQuantFn.apply(
        theta,
        kappa_forward,
        kappa_backward,
        beta,
        seed,
        int(format_id),
        theta_topk,
    )
