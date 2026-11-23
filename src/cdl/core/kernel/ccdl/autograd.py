import math

import torch

from cdl.core.kernel.ccdl.launch import launch_ccdl_bwd, launch_ccdl_fwd
from cdl.core.kernel.formats.spec import FormatId, get_format
from cdl.core.kernel.layout import q_indices_shape, reduce_q_grad


@torch.library.custom_op("cdl::ccdl_quant_fwd", mutates_args=())
def ccdl_quant_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
    seed: torch.Tensor,
    format_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ccdl_fwd(
        theta,
        theta_kappa,
        theta_topk,
        q,
        q_kappa,
        q_topk,
        alpha,
        seed,
        fmt,
    )


@ccdl_quant_fwd.register_fake
def _ccdl_quant_fwd_fake(
    theta, theta_kappa, theta_topk, q, q_kappa, q_topk, alpha, seed, format_id
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
    theta_kappa: torch.Tensor,
    theta_topk: int,
    q: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    packed_scale: torch.Tensor,
    alpha: torch.Tensor,
    format_id: int,
    need_dq: bool,
    need_dkt: bool,
    need_dkq: bool,
    need_dal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fmt = get_format(format_id, theta.device)
    return launch_ccdl_bwd(
        grad_proxy,
        theta,
        theta_kappa,
        theta_topk,
        q,
        q_kappa,
        q_topk,
        packed_scale,
        alpha,
        fmt,
        need_dq,
        need_dkt,
        need_dkq,
        need_dal,
    )


@ccdl_quant_bwd.register_fake
def _ccdl_quant_bwd_fake(
    grad_proxy,
    theta,
    theta_kappa,
    theta_topk,
    q,
    q_kappa,
    q_topk,
    packed_scale,
    alpha,
    format_id,
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
        theta_kappa_forward: torch.Tensor,
        theta_kappa_backward: torch.Tensor | None,
        theta_topk: int,
        q: torch.Tensor,
        q_kappa_forward: torch.Tensor,
        q_kappa_backward: torch.Tensor | None,
        q_topk: int,
        alpha: torch.Tensor,
        seed: torch.Tensor,
        format_id: int,
    ):
        packed_theta, packed_scale = ccdl_quant_fwd(
            theta,
            theta_kappa_forward,
            theta_topk,
            q,
            q_kappa_forward,
            q_topk,
            alpha,
            seed,
            format_id,
        )
        # Only the u8 scale byte is kept for backward; the kernel rebuilds the
        # block scale from it (no fp32 ``s`` tensor).
        ctx.save_for_backward(
            theta,
            theta_kappa_forward,
            theta_kappa_backward,
            q,
            q_kappa_forward,
            q_kappa_backward,
            packed_scale,
            alpha,
        )
        ctx.format_id = format_id
        ctx.topk = (theta_topk, q_topk)
        carrier = theta.new_zeros(()).expand(theta.shape)
        return packed_theta, packed_scale, carrier

    @staticmethod
    def backward(ctx, g_packed_theta, g_packed_scale, g_proxy):
        (
            theta,
            theta_kappa_forward,
            theta_kappa_backward,
            q,
            q_kappa_forward,
            q_kappa_backward,
            packed_scale,
            alpha,
        ) = ctx.saved_tensors
        theta_topk, q_topk = ctx.topk
        # None means coupled mode: reuse forward kappa and route its gradient back.
        coupled_kt = theta_kappa_backward is None
        coupled_kq = q_kappa_backward is None
        kt_bwd = theta_kappa_forward if coupled_kt else theta_kappa_backward
        kq_bwd = q_kappa_forward if coupled_kq else q_kappa_backward
        need_q = ctx.needs_input_grad[4]
        need_dkt = ctx.needs_input_grad[1] if coupled_kt else ctx.needs_input_grad[2]
        need_dkq = ctx.needs_input_grad[5] if coupled_kq else ctx.needs_input_grad[6]
        need_dal = ctx.needs_input_grad[8]
        dtheta, dq, scalars = ccdl_quant_bwd(
            g_proxy.contiguous(),
            theta,
            kt_bwd,
            theta_topk,
            q,
            kq_bwd,
            q_topk,
            packed_scale,
            alpha,
            ctx.format_id,
            need_q,
            need_dkt,
            need_dkq,
            need_dal,
        )
        dkt_fwd = scalars[0] if (coupled_kt and need_dkt) else None
        dkt_bwd = scalars[0] if ((not coupled_kt) and need_dkt) else None
        dkq_fwd = scalars[1] if (coupled_kq and need_dkq) else None
        dkq_bwd = scalars[1] if ((not coupled_kq) and need_dkq) else None
        return (
            dtheta.reshape(theta.shape),
            dkt_fwd,
            dkt_bwd,
            None,
            reduce_q_grad(dq, theta, q) if need_q else None,
            dkq_fwd,
            dkq_bwd,
            None,
            scalars[2] if need_dal else None,
            None,
            None,
        )


def ccdl_quant(
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
    format_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return CcdlQuantFn.apply(
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
        int(format_id),
    )
