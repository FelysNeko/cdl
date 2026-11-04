import torch

from cdl.api.quant import QTensor
from cdl.core.mm.swizzle import expand_rows, pad_k, pad_n, swizzle_scale_factors
from cdl.model.wrapper.flashinfer import mm_mxfp8


def mxfp8_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    rows, k = packed.shape
    vals = packed.float().reshape(rows, k // 32, 32)
    scales = torch.exp2(scale.float() - 127.0)
    out = (vals * scales[..., None]).flatten(1)
    return (out * alpha.float()).to(torch.bfloat16)


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

        x_theta, x_q, x_alpha = x.mxfp8()
        w_theta, w_q, w_alpha = w.mxfp8()

        x_q = expand_rows(x_q, x_theta.shape[0])
        x_theta, x_q = pad_k(x_theta, x_q, 1, 32, 128)
        w_theta, w_q = pad_k(w_theta, w_q, 1, 32, 128)
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        n = w_theta.shape[0]
        w_theta, w_q = pad_n(w_theta, w_q, 8, 128)

        out = mm_mxfp8(
            x_theta,
            w_theta.T,
            swizzle_scale_factors(x_q).flatten(),
            swizzle_scale_factors(w_q).flatten(),
        )
        out = out[:, :n]
        return (out.float() * (x_alpha * w_alpha)).to(torch.bfloat16)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor]:
        grad_output = grad_output.contiguous()
        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = ctx.saved_tensors
        dx = dw = None

        if ctx.x_k is not None:
            w = mxfp8_dequant(w_theta, w_q, w_alpha)
            dx = grad_output @ w[:, : ctx.x_k]

        if ctx.w_k is not None:
            x = mxfp8_dequant(x_theta, x_q, x_alpha)
            dw = grad_output.T @ x[:, : ctx.w_k]

        return None, dx, None, dw
