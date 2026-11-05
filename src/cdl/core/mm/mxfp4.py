import functools

import torch

from cdl.api.quant import QTensor
from cdl.core.mm.swizzle import expand_rows, pad_k, pad_n, swizzle_scale_factors
from cdl.model.wrapper.flashinfer import mm_fp4


@functools.cache
def get_e2m1_lut(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ),
        dtype=dtype,
        device=device,
    )


def mxfp4_dequant(
    packed: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    rows, half_k = packed.shape
    nib = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, half_k * 2)
    code = nib.long()
    ordinal = torch.where(code < 8, code + 8, 15 - code)
    lut = get_e2m1_lut(torch.bfloat16, packed.device)
    vals = lut[ordinal].reshape(rows, half_k // 16, 32)
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

        x_theta, x_q, x_alpha = x.mxfp4()
        w_theta, w_q, w_alpha = w.mxfp4()

        x_q = expand_rows(x_q, x_theta.shape[0])
        x_theta, x_q = pad_k(x_theta, x_q, 2, 32, 32)
        w_theta, w_q = pad_k(w_theta, w_q, 2, 32, 32)
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        n = w_theta.shape[0]
        w_theta, w_q = pad_n(w_theta, w_q, 8, 0)

        return mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_scale_factors(x_q),
            swizzle_scale_factors(w_q).T,
            x_alpha * w_alpha,
            32,
            False,
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
            w = mxfp4_dequant(w_theta, w_q, w_alpha)
            dx = grad_output @ w[:, : ctx.x_k]

        if ctx.w_k is not None:
            x = mxfp4_dequant(x_theta, x_q, x_alpha)
            dw = grad_output.T @ x[:, : ctx.w_k]

        return None, dx, None, dw
