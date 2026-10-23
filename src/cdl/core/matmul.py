import torch

from cdl.api.tensor import QuantizedTensor
from cdl.model.wrapper.flashinfer import mm_fp4, nvfp4_dequantize


def swizzle_nvfp4_scale_factors(sf: torch.Tensor) -> torch.Tensor:
    m, b = sf.shape
    rows, cols = (m + 127) // 128 * 128, (b + 3) // 4 * 4
    out = torch.zeros(rows, cols, dtype=torch.uint8, device=sf.device)
    out[:m, :b] = sf
    return (
        out.view(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


def matmul(x: QuantizedTensor, w: QuantizedTensor) -> torch.Tensor:
    return MatMul.apply(x, x.proxy(), w, w.proxy())


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: QuantizedTensor,
        x_proxy: torch.Tensor | None,
        w: QuantizedTensor,
        w_proxy: torch.Tensor | None,
    ) -> torch.Tensor:
        x_theta, x_q, x_alpha = x.fp4()
        w_theta, w_q, w_alpha = w.fp4()
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)
        ctx.x_k = x_proxy.shape[-1] if x_proxy is not None else None
        ctx.w_k = w_proxy.shape[-1] if w_proxy is not None else None

        out = mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_nvfp4_scale_factors(x_q),
            swizzle_nvfp4_scale_factors(w_q).T,
            x_alpha * w_alpha,
        )
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor]:
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = saved_tensors
        grad_output = grad_output.contiguous()

        w = nvfp4_dequantize(w_theta, w_q, w_alpha)
        dx = grad_output @ w[:, : ctx.x_k]

        x = nvfp4_dequantize(x_theta, x_q, x_alpha)
        dw = grad_output.T @ x[:, : ctx.w_k]

        return None, dx, None, dw
