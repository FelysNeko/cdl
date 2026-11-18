import math

import torch

from cdl.core.kernel.ecdl.launch import launch_ecdl_bwd, launch_ecdl_fwd
from cdl.core.kernel.formats.spec import FormatId, get_format


@torch.library.custom_op("cdl::ecdl_quant_fwd", mutates_args=())
def ecdl_quant_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
    theta_topk: int,
    scale_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    is_fp4 = format_id in (FormatId.NVFP4, FormatId.MXFP4)
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn
    return (
        theta.new_empty((*lead, nb, packed_cols), dtype=packed_dtype),
        theta.new_empty((*lead, nb, 1), dtype=torch.uint8),
        theta.new_empty((), dtype=torch.float32),
    )


@torch.library.custom_op("cdl::ecdl_quant_bwd", mutates_args=())
def ecdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    packed_scale: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
    scale_mode: str,
    need_dkappa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ecdl_bwd(
        grad_proxy,
        theta,
        packed_scale,
        kappa_backward,
        alpha,
        fmt,
        theta_topk,
        scale_mode,
        need_dkappa,
    )


@ecdl_quant_bwd.register_fake
def _ecdl_quant_bwd_fake(
    grad_proxy,
    theta,
    packed_scale,
    kappa_backward,
    alpha,
    format_id,
    theta_topk,
    scale_mode,
    need_dkappa,
):
    m = math.prod(theta.shape[:-2])
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((m, nb * tpb), dtype=theta.dtype),
        theta.new_empty((2 if need_dkappa else 0,), dtype=torch.float32),
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
        packed_theta, packed_scale, alpha_eff = ecdl_quant_fwd(
            theta,
            kappa_forward,
            seed,
            format_id,
            theta_topk,
            scale_mode,
        )
        ctx.save_for_backward(
            theta, packed_scale, kappa_forward, kappa_backward, alpha_eff
        )
        ctx.format_id = format_id
        ctx.theta_topk = theta_topk
        ctx.scale_mode = scale_mode
        carrier = theta.new_zeros(()).expand(theta.shape)
        ctx.mark_non_differentiable(alpha_eff)
        return packed_theta, packed_scale, carrier, alpha_eff

    @staticmethod
    def backward(ctx, g_packed_theta, g_packed_scale, g_proxy, g_alpha_eff):
        theta, packed_scale, kappa_forward, kappa_backward, alpha_eff = (
            ctx.saved_tensors
        )
        # None means coupled mode: reuse kappa_forward and route d(kappa) back to it.
        coupled = kappa_backward is None
        kappa_bwd = kappa_forward if coupled else kappa_backward
        need_dkappa = ctx.needs_input_grad[1] if coupled else ctx.needs_input_grad[2]
        # scalars[1] is d(alpha_eff); alpha_eff carries no learnable parameter
        # in either scale mode, so it is dropped.
        dtheta, scalars = ecdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            packed_scale,
            kappa_bwd,
            alpha_eff,
            ctx.format_id,
            ctx.theta_topk,
            ctx.scale_mode,
            need_dkappa,
        )
        dkappa_forward = scalars[0] if (coupled and need_dkappa) else None
        dkappa_backward = scalars[0] if ((not coupled) and need_dkappa) else None
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
