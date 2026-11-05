import torch

from cdl.api.quant import QTensor
from cdl.core.mm.swizzle import expand_rows, pad_k, pad_n, swizzle_scale_factors
from cdl.model.wrapper.flashinfer import mm_fp4, nvfp4_dequantize


def matmul(x: QTensor, w: QTensor) -> torch.Tensor:
    return MatMul.apply(x, x.proxy(), w, w.proxy())


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: QTensor,
        x_proxy: torch.Tensor | None,
        w: QTensor,
        w_proxy: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.x_k = x_proxy.shape[-1] if x_proxy is not None else None
        ctx.w_k = w_proxy.shape[-1] if w_proxy is not None else None

        x_theta, x_q, x_alpha = x.nvfp4()
        w_theta, w_q, w_alpha = w.nvfp4()

        x_q = expand_rows(x_q, x_theta.shape[0])
        x_theta, x_q = pad_k(x_theta, x_q, 2, 16, 32)
        w_theta, w_q = pad_k(w_theta, w_q, 2, 16, 32)
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        n = w_theta.shape[0]
        w_theta, w_q = pad_n(w_theta, w_q, 8, 0)

        return mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_scale_factors(x_q),
            swizzle_scale_factors(w_q).T,
            x_alpha * w_alpha,
            16,
            True,
        )[:, :n]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor]:
        grad_output = grad_output.contiguous()
        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = ctx.saved_tensors
        dx = dw = None

        if ctx.x_k is not None:
            w = nvfp4_dequantize(w_theta, w_q, w_alpha)
            dx = grad_output @ w[:, : ctx.x_k]

        if ctx.w_k is not None:
            x = nvfp4_dequantize(x_theta, x_q, x_alpha)
            dw = grad_output.T @ x[:, : ctx.w_k]

        return None, dx, None, dw
