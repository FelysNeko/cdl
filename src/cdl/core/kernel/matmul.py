"""Fused cascaded-CDL quantization + block-scaled matmul with analytic backward.

`FusedCdlMatmul` computes y = Q_cdl(x) @ Q_cdl(w).T via the flashinfer
b12x block-scaled GEMMs in one autograd.Function: the forward never
materializes the soft proxy, and the backward maps dL/dy to the fused
quantization backward plus (optionally) low-precision tensor-core GEMMs.

Supported modes (see `gemm_mode`):
- nvfp4: E2M1 theta / UE4M3ExcludeZero scales, block 16, mm_fp4.
- mxfp4: E2M1 theta / E8M0 scales, block 32, mm_fp4(use_nvfp4=False).
- mxfp8: E4M3 theta / E8M0 scales, block 32, mm_mxfp8 (no global alpha).
"""

import os

import torch

from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import E4M3, E8M0, UE4M3ExcludeZero
from cdl.core.kernel.quant import cdl_quant_bwd, cdl_quant_fwd
from cdl.core.matmul import mxfp4_dequant, mxfp8_dequant
from cdl.core.matmul import swizzle_nvfp4_scale_factors as swz
from cdl.model.wrapper.flashinfer import (
    mm_fp4,
    mm_mxfp8,
    mxfp4_quantize,
    mxfp8_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
)


def gemm_mode(quant) -> str | None:
    """Which fused GEMM mode `quant` supports, or None if unsupported."""
    theta_format = getattr(quant, "theta_format", None)
    q_format = getattr(quant, "q_format", None)
    block = getattr(quant, "block_size", 0)
    if not hasattr(quant, "log_q"):
        return None
    if (
        isinstance(theta_format, E2M1)
        and isinstance(q_format, UE4M3ExcludeZero)
        and block == 16
    ):
        return "nvfp4"
    if isinstance(theta_format, E2M1) and isinstance(q_format, E8M0) and block == 32:
        return "mxfp4"
    if isinstance(theta_format, E4M3) and isinstance(q_format, E8M0) and block == 32:
        return "mxfp8"
    return None


def fusable_quant(quant) -> bool:
    """Whether `quant` is a cascaded CDL quantizer the fused kernel supports."""
    return gemm_mode(quant) is not None


def pad_quantum(mode: str) -> int:
    """Contraction-dim alignment the mode's GEMM requires before quantization."""
    if mode == "nvfp4":
        return 16
    if mode == "mxfp4":
        return 32
    if mode == "mxfp8":
        return 128
    raise ValueError(f"unknown fused GEMM mode {mode!r}")


def layer_pads(quant, kdim: int, rows: int) -> tuple[int, int]:
    """(kpad, npad) the GEMM layout requires for a layer of this shape.

    `kdim` is the contraction width of the (im2col) weight and `rows` its
    output width; the pads are applied by the calling layer before
    quantization. Non-fused quantizers only pad for the MX layouts.
    """
    mode = gemm_mode(quant)
    if mode is not None:
        kpad = -kdim % pad_quantum(mode)
        npad = max(0, 128 - rows) if mode == "mxfp8" else 0
        return kpad, npad
    block = getattr(quant, "block_size", 0)
    if not block or block == 16:
        return 0, 0
    fp8 = (
        getattr(quant, "theta_format", None) is not None
        and quant.theta_format.bits == 8
    )
    kpad = -kdim % (128 if fp8 else block)
    npad = max(0, 128 - rows) if fp8 else 0
    return kpad, npad


def _requant_mxfp4(x: torch.Tensor):
    """STE-quantize x [rows, k] for the MXFP4 backward GEMMs.

    Pads k to a multiple of 32 and returns (packed, swizzled scales, alpha).
    """
    k = x.shape[-1]
    r = -k % 32
    if r:
        x = torch.nn.functional.pad(x, (0, r))
    packed, scale = mxfp4_quantize(x)
    return packed, scale, torch.ones(1, device=x.device, dtype=torch.float32)


def _requant_mxfp8(x: torch.Tensor):
    """STE-quantize x [rows, k] for the MXFP8 backward GEMMs (k padded to 128).

    Returns (e4m3 bytes, 1D flattened swizzled scales).
    """
    k = x.shape[-1]
    r = -k % 128
    if r:
        x = torch.nn.functional.pad(x, (0, r))
    packed, scale = mxfp8_quantize(x)
    return packed, swz(scale).flatten()


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
    """y = Q_cdl(x) @ Q_cdl(w).T via block-scaled tensor cores, analytic backward."""

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
        mode: str,
    ):
        q_x, kt_x, kq_x, al_x = torch._foreach_exp(
            [log_q_x, log_kt_x, log_kq_x, log_al_x]
        )
        q_w, kt_w, kq_w, al_w = torch._foreach_exp(
            [log_q_w, log_kt_w, log_kq_w, log_al_w]
        )

        seed.add_(1)
        block = 16 if mode == "nvfp4" else 32
        theta_fp8 = mode == "mxfp8"
        packed_x, scale_x, x_sw, s_x = cdl_quant_fwd(
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
            block,
            theta_fp8,
            topk_x,
            qtopk_x,
        )
        seed.add_(1)
        packed_w, scale_w, w_sw, s_w = cdl_quant_fwd(
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
            block,
            theta_fp8,
            topk_w,
            qtopk_w,
        )

        if mode == "nvfp4":
            out = mm_fp4(packed_x, packed_w.T, x_sw, w_sw.T, (al_x * al_w).reshape(1))
        elif mode == "mxfp4":
            out = mm_fp4(
                packed_x,
                packed_w.T,
                x_sw,
                w_sw.T,
                (al_x * al_w).reshape(1),
                block_size=32,
                use_nvfp4=False,
            )
        else:
            # mm_mxfp8 has no global-scale epilogue; apply the CDL alpha
            # product as a scalar epilogue so gradients stay exact.
            out = mm_mxfp8(
                packed_x.view(torch.float8_e4m3fn),
                packed_w.view(torch.float8_e4m3fn).T,
                x_sw.flatten(),
                w_sw.flatten(),
            ) * (al_x * al_w)
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
        ctx.mode = mode
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
        mode = ctx.mode
        block = 16 if mode == "nvfp4" else 32
        theta_fp8 = mode == "mxfp8"

        if ctx.fp4_bwd:
            # dx = Q(g) @ Q(W^T)^T and dw = Q(g^T) @ Q(X^T)^T on tensor cores;
            # W/X are requantized along the transposed orientation and the
            # gradients themselves are STE-quantized (approximate bwd).
            if mode == "nvfp4":
                gx, gx_sc, gx_al = _quantize_nvfp4(g)
                w_dq = nvfp4_dequantize(packed_w, scale_w, al_w.reshape(1))[:n]
                pwt, swt, alwt = _quantize_nvfp4(w_dq[:, :k_x].T.contiguous())
                dx = mm_fp4(
                    gx, pwt.T, swz(gx_sc), swz(swt).T, (gx_al * alwt).reshape(1)
                )
                gtq, gtq_sc, gt_al = _quantize_nvfp4(g.T.contiguous())
                x_dq = nvfp4_dequantize(packed_x, scale_x, al_x.reshape(1))[:m_x]
                pxt, sxt, alxt = _quantize_nvfp4(x_dq[:, :k_x].T.contiguous())
                dw = mm_fp4(
                    gtq, pxt.T, swz(gtq_sc), swz(sxt).T, (gt_al * alxt).reshape(1)
                )
            elif mode == "mxfp4":
                gx, gx_sc, gx_al = _requant_mxfp4(g)
                w_dq = mxfp4_dequant(packed_w, scale_w, al_w.reshape(1))[:n]
                pwt, swt, alwt = _requant_mxfp4(w_dq[:, :k_x].T.contiguous())
                dx = mm_fp4(
                    gx,
                    pwt.T,
                    gx_sc,
                    swt.T,
                    (gx_al * alwt).reshape(1),
                    block_size=32,
                    use_nvfp4=False,
                )
                gtq, gtq_sc, gt_al = _requant_mxfp4(g.T.contiguous())
                x_dq = mxfp4_dequant(packed_x, scale_x, al_x.reshape(1))[:m_x]
                pxt, sxt, alxt = _requant_mxfp4(x_dq[:, :k_x].T.contiguous())
                dw = mm_fp4(
                    gtq,
                    pxt.T,
                    gtq_sc,
                    sxt.T,
                    (gt_al * alxt).reshape(1),
                    block_size=32,
                    use_nvfp4=False,
                )
            else:
                gx, gx_sc = _requant_mxfp8(g)
                w_dq = mxfp8_dequant(packed_w, scale_w)[:n]
                pwt, swt = _requant_mxfp8(w_dq[:, :k_x].T.contiguous())
                dx = mm_mxfp8(
                    gx.view(torch.float8_e4m3fn),
                    pwt.view(torch.float8_e4m3fn).T,
                    gx_sc,
                    swt,
                )
                gtq, gtq_sc = _requant_mxfp8(g.T.contiguous())
                x_dq = mxfp8_dequant(packed_x, scale_x)[:m_x]
                pxt, sxt = _requant_mxfp8(x_dq[:, :k_x].T.contiguous())
                dw = mm_mxfp8(
                    gtq.view(torch.float8_e4m3fn),
                    pxt.view(torch.float8_e4m3fn).T,
                    gtq_sc,
                    sxt,
                )
        else:
            if mode == "nvfp4":
                w_dq = nvfp4_dequantize(packed_w, scale_w, al_w.reshape(1))[:n]
                x_dq = nvfp4_dequantize(packed_x, scale_x, al_x.reshape(1))[:m_x]
            elif mode == "mxfp4":
                w_dq = mxfp4_dequant(packed_w, scale_w, al_w.reshape(1))[:n]
                x_dq = mxfp4_dequant(packed_x, scale_x, al_x.reshape(1))[:m_x]
            else:
                w_dq = mxfp8_dequant(packed_w, scale_w)[:n]
                x_dq = mxfp8_dequant(packed_x, scale_x)[:m_x]
            dx = g @ w_dq[:, :k_x]
            dw = g.T @ x_dq[:, :k_x]

        dtheta_x, dq_x, sums_x = cdl_quant_bwd(
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
            block,
            theta_fp8,
            ctx.topk_x,
            ctx.qtopk_x,
        )
        dtheta_w, dq_w, sums_w = cdl_quant_bwd(
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
            block,
            theta_fp8,
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
    """Quantize both operands with cascaded CDL and matmul them on tensor cores.

    theta_x: [M, K] contiguous (K % block == 0); theta_w: [N, K].
    quant_x / quant_w: cascaded CDL quantizer modules exposing log_q,
    log_theta_kappa, log_q_kappa, log_alpha, theta_topk, q_topk and
    theta_format / q_format (see gemm_mode for the supported combinations).
    seed: int64 device scalar advanced inside the graph; defaults to
    quant_x.kernel_seed.

    In eval mode the quantizers would normally use the deterministic
    nearest-round path instead — callers handle that before invoking this.
    """
    mode = gemm_mode(quant_x)
    if mode is None or gemm_mode(quant_w) != mode:
        raise NotImplementedError(
            "the fused backend requires both quantizers on the same supported "
            "format pair: nvfp4 (E2M1/UE4M3ExcludeZero, block 16), mxfp4 "
            "(E2M1/E8M0, block 32) or mxfp8 (E4M3/E8M0, block 32)"
        )
    if seed is None:
        seed = quant_x.kernel_seed
    if mode == "nvfp4":
        theta_level, theta_gap, q_level, q_gap = _nvfp4_tables(theta_x.device)
    else:
        theta_level, theta_gap, q_level, q_gap = _mx_tables(
            theta_x.device, mode == "mxfp8"
        )
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
        mode,
    )


def _nvfp4_tables(device):
    from cdl.core.kernel.tables import e2m1_tables, ue4m3_tables

    return (*e2m1_tables(device), *ue4m3_tables(device))


def _mx_tables(device, theta_fp8: bool):
    from cdl.core.kernel.tables import e2m1_tables, e4m3_tables, e8m0_tables

    theta = e4m3_tables(device) if theta_fp8 else e2m1_tables(device)
    return (*theta, *e8m0_tables(device))
