import flashinfer
import torch

from cdl.api.tensor import QuantizedTensor


def swizzle_e4m3(sf: torch.Tensor) -> torch.Tensor:
    m, b = sf.shape
    rows, cols = (m + 127) // 128 * 128, (b + 3) // 4 * 4
    out = torch.zeros(rows, cols, dtype=torch.uint8, device=sf.device)
    out[:m, :b] = sf
    return (
        out.view(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: QuantizedTensor,
        x_proxy: torch.Tensor,
        w: QuantizedTensor,
        w_proxy: torch.Tensor,
    ) -> torch.Tensor:
        x_theta, x_q, x_alpha = x.fp4()
        w_theta, w_q, w_alpha = w.fp4()
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        out = flashinfer.mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_e4m3(x_q),
            swizzle_e4m3(w_q).T,
            x_alpha * w_alpha,
            backend="b12x",
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

        w = flashinfer.nvfp4_kv_dequantize(w_theta, w_q, w_alpha)
        dx = grad_output @ w

        x = flashinfer.nvfp4_kv_dequantize(x_theta, x_q, x_alpha)
        dw = grad_output.T @ x

        return None, dx, None, dw
