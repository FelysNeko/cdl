import math

import torch

from cdl.core.kernel.ecdl.launch import launch_ecdl_bwd, launch_ecdl_fwd
from cdl.core.kernel.formats.spec import get_format


@torch.library.custom_op("cdl::ecdl_quant_fwd", mutates_args=())
def ecdl_quant_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
    scale_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_fwd(
        theta,
        theta_kappa,
        seed,
        fmt,
        theta_topk,
        scale_mode,
    )


@ecdl_quant_fwd.register_fake
def _ecdl_quant_fwd_fake(theta, theta_kappa, seed, format_id, theta_topk, scale_mode):
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((*lead, nb, tpb), dtype=torch.int16),
        theta.new_empty((*lead, nb, 1), dtype=torch.uint8),
        theta.new_empty((*lead, nb), dtype=torch.float32),
        theta.new_empty(()),
    )


@torch.library.custom_op("cdl::ecdl_quant_bwd", mutates_args=())
def ecdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q_indices: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_bwd(
        grad_proxy,
        theta,
        q_indices,
        kappa_backward,
        alpha,
        fmt,
        theta_topk,
    )


@ecdl_quant_bwd.register_fake
def _ecdl_quant_bwd_fake(
    grad_proxy, theta, q_indices, kappa_backward, alpha, format_id, theta_topk
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
        kappa_backward: torch.Tensor | None,
        seed: torch.Tensor,
        format_id: int,
        theta_topk: int,
        scale_mode: str,
    ):
        seed.add_(1)
        theta_indices, q_indices, _s, alpha_eff = ecdl_quant_fwd(
            theta,
            kappa_forward,
            seed,
            format_id,
            theta_topk,
            scale_mode,
        )
        ctx.save_for_backward(
            theta, q_indices, kappa_forward, kappa_backward, alpha_eff
        )
        ctx.format_id = format_id
        ctx.theta_topk = theta_topk
        carrier = theta.new_zeros(()).expand(theta.shape)
        ctx.mark_non_differentiable(alpha_eff)
        return theta_indices, q_indices, carrier, alpha_eff

    @staticmethod
    def backward(ctx, g_theta_indices, g_q_indices, g_proxy, g_alpha_eff):
        theta, q_indices, kappa_forward, kappa_backward, alpha_eff = ctx.saved_tensors
        # None means coupled mode: reuse kappa_forward and route d(kappa) back to it.
        coupled = kappa_backward is None
        # scalars[1] is d(alpha_eff); alpha_eff carries no learnable parameter
        # in either scale mode, so it is dropped.
        dtheta, _scalars = ecdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            q_indices,
            kappa_forward if coupled else kappa_backward,
            alpha_eff,
            ctx.format_id,
            ctx.theta_topk,
        )
        if coupled:
            dkappa_forward = _scalars[0] if ctx.needs_input_grad[1] else None
            dkappa_backward = None
        else:
            dkappa_forward = None
            dkappa_backward = _scalars[0] if ctx.needs_input_grad[2] else None
        return (
            dtheta.reshape(theta.shape),
            dkappa_forward,
            dkappa_backward,
            None,
            None,
            None,
            None,
        )


def ecdl_quant(
    theta: torch.Tensor,
    kappa_forward: torch.Tensor,
    kappa_backward: torch.Tensor | None,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
    scale_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return EcdlQuantFn.apply(
        theta,
        kappa_forward,
        kappa_backward,
        seed,
        int(format_id),
        theta_topk,
        scale_mode,
    )
