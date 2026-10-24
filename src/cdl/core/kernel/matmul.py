"""Fused cascaded-CDL quantization + NVFP4 matmul with analytic backward.

`FusedCdlMatmul` computes y = Q_cdl(x) @ Q_cdl(w).T via the flashinfer
mm_fp4 kernel in one autograd.Function: the forward never materializes the
soft proxy, and the backward maps dL/dy to the fused quantization backward
plus (optionally) fp4 tensor-core GEMMs.
"""

import os

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.model.wrapper.flashinfer import mm_fp4, nvfp4_dequantize, nvfp4_quantize


def swz(sf: torch.Tensor) -> torch.Tensor:
    from cdl.core.matmul import swizzle_nvfp4_scale_factors

    return swizzle_nvfp4_scale_factors(sf)


def fusable_quant(quant) -> bool:
    """Whether `quant` is a cascaded CDL quantizer the fused kernel supports.

    The Triton kernels hard-code the E2M1 theta alphabet and the
    UE4M3ExcludeZero scale alphabet with NVFP4 block size 16.
    """
    return (
        hasattr(quant, "log_q")
        and isinstance(getattr(quant, "theta_format", None), E2M1)
        and isinstance(getattr(quant, "q_format", None), UE4M3ExcludeZero)
        and getattr(quant, "block_size", 0) == 16
    )


def _quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Same global-scale convention as nvfp4_pack in cdl.model.components.
    k = x.shape[-1]
    r = -k % 32
    if r:
        x = torch.nn.functional.pad(x, (0, r))
    amax = x.float().abs().amax()
    global_scale = torch.where(
        amax > 0,
        amax.new_full((), 448.0 * 6.0) / amax,
        amax.new_full((), torch.finfo(torch.float32).max),
    ).reshape(1)
    packed, scale = nvfp4_quantize(x, global_scale)
    return packed, scale, (1.0 / global_scale).float()


def _dq_grad(dq: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # dq is per-row [M, NB]; q may be per-row or a single broadcast row.
    if q.shape[0] == dq.shape[0]:
        return dq.reshape(q.shape) * q
    return (dq * q.reshape(1, -1)).sum(0, keepdim=True).reshape(q.shape)


class FusedCdlMatmul(torch.autograd.Function):
    """y = Q_cdl(x) @ Q_cdl(w).T via mm_fp4, analytic backward to all params."""

    @staticmethod
    def forward(
        ctx,
        theta_x,
        log_q_x,
        log_kt_x,
        log_kq_x,
        log_al_x,
        theta_w,
        log_q_w,
        log_kt_w,
        log_kq_w,
        log_al_w,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        seed,
        topk_x: int,
        qtopk_x: int,
        topk_w: int,
        qtopk_w: int,
        fp4_bwd: bool,
    ):
        q_x, kt_x, kq_x, al_x = torch._foreach_exp(
            [log_q_x, log_kt_x, log_kq_x, log_al_x]
        )
        q_w, kt_w, kq_w, al_w = torch._foreach_exp(
            [log_q_w, log_kt_w, log_kq_w, log_al_w]
        )

        seed.add_(1)
        packed_x, scale_x, x_sw, s_x = torch.ops.cdl.cdl_quant_fwd(
            theta_x,
            q_x,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            kt_x,
            kq_x,
            al_x,
            seed,
            topk_x,
            qtopk_x,
        )
        seed.add_(1)
        packed_w, scale_w, w_sw, s_w = torch.ops.cdl.cdl_quant_fwd(
            theta_w,
            q_w,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            kt_w,
            kq_w,
            al_w,
            seed,
            topk_w,
            qtopk_w,
        )

        out = mm_fp4(packed_x, packed_w.T, x_sw, w_sw.T, (al_x * al_w).reshape(1))
        if out.shape != (theta_x.shape[0], theta_w.shape[0]):
            out = out[: theta_x.shape[0], : theta_w.shape[0]]

        ctx.save_for_backward(
            theta_x,
            packed_x,
            scale_x,
            s_x,
            q_x,
            kt_x,
            kq_x,
            al_x,
            theta_w,
            packed_w,
            scale_w,
            s_w,
            q_w,
            kt_w,
            kq_w,
            al_w,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
        )
        ctx.topk_x = topk_x
        ctx.qtopk_x = qtopk_x
        ctx.topk_w = topk_w
        ctx.qtopk_w = qtopk_w
        if fp4_bwd:
            use_fp4 = True
        else:
            env = os.environ.get("CDL_FP4_BWD", "auto")
            use_fp4 = (
                float(theta_x.shape[0]) * theta_w.shape[0] * theta_x.shape[1] >= 5e9
                if env == "auto"
                else env == "1"
            )
        ctx.fp4_bwd = use_fp4
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            theta_x,
            packed_x,
            scale_x,
            s_x,
            q_x,
            kt_x,
            kq_x,
            al_x,
            theta_w,
            packed_w,
            scale_w,
            s_w,
            q_w,
            kt_w,
            kq_w,
            al_w,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
        ) = ctx.saved_tensors
        g = grad_output.contiguous()
        k_x = theta_x.shape[1]
        m_x, n = theta_x.shape[0], theta_w.shape[0]

        if ctx.fp4_bwd:
            # dx = Q(g) @ Q(W^T)^T and dw = Q(g^T) @ Q(X^T)^T via fp4 tensor
            # cores; W/X are requantized along the transposed orientation and
            # the gradients themselves are nvfp4-quantized (approximate bwd).
            gx, gx_sc, gx_al = _quantize_nvfp4(g)
            w_dq = nvfp4_dequantize(packed_w, scale_w, al_w.reshape(1))[:n]
            pwt, swt, alwt = _quantize_nvfp4(w_dq[:, :k_x].T.contiguous())
            dx = mm_fp4(gx, pwt.T, swz(gx_sc), swz(swt).T, (gx_al * alwt).reshape(1))
            gtq, gtq_sc, gt_al = _quantize_nvfp4(g.T.contiguous())
            x_dq = nvfp4_dequantize(packed_x, scale_x, al_x.reshape(1))[:m_x]
            pxt, sxt, alxt = _quantize_nvfp4(x_dq[:, :k_x].T.contiguous())
            dw = mm_fp4(gtq, pxt.T, swz(gtq_sc), swz(sxt).T, (gt_al * alxt).reshape(1))
        else:
            w_dq = nvfp4_dequantize(packed_w, scale_w, al_w.reshape(1))[:n]
            dx = g @ w_dq[:, :k_x]
            x_dq = nvfp4_dequantize(packed_x, scale_x, al_x.reshape(1))[:m_x]
            dw = g.T @ x_dq[:, :k_x]

        dtheta_x, dq_x, sums_x = torch.ops.cdl.cdl_quant_bwd(
            dx,
            theta_x,
            q_x,
            s_x,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            kt_x,
            kq_x,
            al_x,
            ctx.topk_x,
            ctx.qtopk_x,
        )
        dtheta_w, dq_w, sums_w = torch.ops.cdl.cdl_quant_bwd(
            dw,
            theta_w,
            q_w,
            s_w,
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            kt_w,
            kq_w,
            al_w,
            ctx.topk_w,
            ctx.qtopk_w,
        )
        dkt_x, dkq_x, dal_x = sums_x[0], sums_x[1], sums_x[2]
        dkt_w, dkq_w, dal_w = sums_w[0], sums_w[1], sums_w[2]

        return (
            dtheta_x,
            _dq_grad(dq_x, q_x),
            dkt_x * kt_x,
            dkq_x * kq_x,
            dal_x * al_x,
            dtheta_w,
            _dq_grad(dq_w, q_w),
            dkt_w * kt_w,
            dkq_w * kq_w,
            dal_w * al_w,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_cdl_matmul(
    theta_x: torch.Tensor,
    quant_x,
    theta_w: torch.Tensor,
    quant_w,
    seed: torch.Tensor | None = None,
    fp4_bwd: bool = False,
) -> torch.Tensor:
    """Quantize both operands with cascaded CDL and matmul them on fp4 cores.

    theta_x: [M, K] contiguous (K % 16 == 0); theta_w: [N, K].
    quant_x / quant_w: cascaded CDL quantizer modules exposing log_q,
    log_theta_kappa, log_q_kappa, log_alpha, theta_topk, q_topk and
    theta_format / q_format (E2M1 / UE4M3ExcludeZero; see fusable_quant).
    seed: int64 device scalar advanced inside the graph; defaults to
    quant_x.kernel_seed.

    In eval mode the quantizers would normally use the deterministic
    nearest-round path instead — callers handle that before invoking this.
    """
    if seed is None:
        seed = quant_x.kernel_seed
    theta_level, theta_gap, q_level, q_gap = _resolve_tables(theta_x.device)
    return FusedCdlMatmul.apply(
        theta_x,
        quant_x.log_q,
        quant_x.log_theta_kappa,
        quant_x.log_q_kappa,
        quant_x.log_alpha,
        theta_w,
        quant_w.log_q,
        quant_w.log_theta_kappa,
        quant_w.log_q_kappa,
        quant_w.log_alpha,
        theta_level,
        theta_gap,
        q_level,
        q_gap,
        seed,
        quant_x.theta_topk,
        quant_x.q_topk,
        quant_w.theta_topk,
        quant_w.q_topk,
        fp4_bwd,
    )


def _resolve_tables(device):
    from cdl.core.kernel.tables import e2m1_tables, ue4m3_tables

    return (*e2m1_tables(device), *ue4m3_tables(device))
