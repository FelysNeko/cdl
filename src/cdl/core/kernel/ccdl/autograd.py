import math

import torch

from cdl.core.kernel.ccdl.launch import launch_ccdl_bwd, launch_ccdl_fwd
from cdl.core.kernel.formats.spec import FormatId, get_format
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    is_fp4 = format_id in (FormatId.NVFP4, FormatId.MXFP4)
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn
    return (
        theta.new_empty((*lead, nb, packed_cols), dtype=packed_dtype),
        theta.new_empty(q_indices_shape(lead, nb, q), dtype=torch.uint8),
    )


@torch.library.custom_op("cdl::ccdl_quant_bwd", mutates_args=())
def ccdl_quant_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    packed_scale: torch.Tensor,
    theta_kappa: torch.Tensor,
    q_kappa: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    theta_topk: int,
    q_topk: int,
    need_dq: bool,
    need_dkt: bool,
    need_dkq: bool,
    need_dal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ccdl_bwd(
        grad_proxy,
        theta,
        q,
        packed_scale,
        theta_kappa,
        q_kappa,
        alpha,
        fmt,
        theta_topk,
        q_topk,
        need_dq,
        need_dkt,
        need_dkq,
        need_dal,
    )


@ccdl_quant_bwd.register_fake
def _ccdl_quant_bwd_fake(
    grad_proxy,
    theta,
    q,
    packed_scale,
    theta_kappa,
    q_kappa,
    alpha,
    format_id,
    theta_topk,
    q_topk,
    need_dq,
    need_dkt,
    need_dkq,
    need_dal,
):
    m = math.prod(theta.shape[:-2])
    nb, tpb = theta.shape[-2:]
    return (
        theta.new_empty((m, nb * tpb), dtype=theta.dtype),
        theta.new_empty((m, nb) if need_dq else (0,), dtype=torch.float32),
        theta.new_empty(
            (3 if (need_dkt or need_dkq or need_dal) else 0,), dtype=torch.float32
        ),
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
        packed_theta, packed_scale = ccdl_quant_fwd(
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
        # Only the u8 scale byte is kept for backward; the kernel rebuilds the
        # block scale from it (no fp32 ``s`` tensor).
        ctx.save_for_backward(theta, q, packed_scale, theta_kappa, q_kappa, alpha)
        ctx.format_id = format_id
        ctx.topk = (theta_topk, q_topk)
        carrier = theta.new_zeros(()).expand(theta.shape)
        return packed_theta, packed_scale, carrier

    @staticmethod
    def backward(ctx, g_packed_theta, g_packed_scale, g_proxy):
        theta, q, packed_scale, theta_kappa, q_kappa, alpha = ctx.saved_tensors
        theta_topk, q_topk = ctx.topk
        need_q, need_kt, need_kq, need_al = ctx.needs_input_grad[1:5]
        dtheta, dq, scalars = ccdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            q,
            packed_scale,
            theta_kappa,
            q_kappa,
            alpha,
            ctx.format_id,
            theta_topk,
            q_topk,
            need_q,
            need_kt,
            need_kq,
            need_al,
        )
        return (
            dtheta.reshape(theta.shape),
            reduce_q_grad(dq, theta, q) if need_q else None,
            scalars[0] if need_kt else None,
            scalars[1] if need_kq else None,
            scalars[2] if need_al else None,
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
