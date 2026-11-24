import torch

from cdl.api.quant import QTensor
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.core.kernel.quant import nvfp4_quant
from cdl.core.kernel.transpose import fp4_repack_transpose, rht16_transpose
from cdl.core.mm.swizzle import expand_rows, pad_k, pad_n, swizzle_scale_factors
from cdl.model.wrapper.flashinfer import mm_fp4


def rand_seed(device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2**31 - 1, (), dtype=torch.int32, device=device)


def mm_nvfp4(
    a_theta: torch.Tensor,
    a_q: torch.Tensor,
    a_alpha: torch.Tensor,
    b_theta: torch.Tensor,
    b_q: torch.Tensor,
    b_alpha: torch.Tensor,
    m: int,
    n: int,
) -> torch.Tensor:
    a_q = expand_rows(a_q, a_theta.shape[0])
    b_q = expand_rows(b_q, b_theta.shape[0])
    b_theta, b_q = pad_n(b_theta, b_q, 8, 0)
    return mm_fp4(
        a_theta,
        b_theta.T,
        swizzle_scale_factors(a_q),
        swizzle_scale_factors(b_q).T,
        a_alpha * b_alpha,
        16,
        True,
    )[:m, :n]


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

        m = x.shape[0]
        n = w.shape[0]
        ctx.m = m
        ctx.n = n

        x_theta, x_q = pad_k(x_theta, x_q, 2, 16, 32)
        w_theta, w_q = pad_k(w_theta, w_q, 2, 16, 32)
        ctx.weight_2d = w_q.shape[0] != w_theta.shape[0]
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        return mm_nvfp4(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha, m, n)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor | None, None, torch.Tensor | None]:
        grad_output = grad_output.contiguous()
        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = ctx.saved_tensors
        dx = dw = None

        if ctx.x_k is not None:
            if ctx.weight_2d:
                w_rep = fp4_repack_transpose(w_theta)
                w_q_rep = expand_rows(w_q.T.contiguous(), w_rep.shape[0])
                w_rep, w_q_rep = pad_k(w_rep, w_q_rep, 2, 16, 32)
                dy_theta, dy_q, dy_alpha = nvfp4_quant(
                    grad_output, rand_seed(grad_output.device), False
                )
                dy_theta, dy_q = pad_k(dy_theta, dy_q, 2, 16, 32)
                dx = mm_nvfp4(
                    dy_theta, dy_q, dy_alpha, w_rep, w_q_rep, w_alpha, ctx.m, ctx.x_k
                )
            else:
                w = nvfp4_dequant(w_theta, w_q, w_alpha)
                dx = grad_output @ w[: ctx.n, : ctx.x_k]

        if ctx.w_k is not None:
            x = nvfp4_dequant(x_theta, expand_rows(x_q, x_theta.shape[0]), x_alpha)[
                : ctx.m, : ctx.w_k
            ]
            signs = torch.where(
                torch.randint(0, 2, (16,), device=grad_output.device, dtype=torch.bool),
                1.0,
                -1.0,
            )
            dy_t = rht16_transpose(grad_output, signs)
            x_t = rht16_transpose(x, signs)
            dy_t_theta, dy_t_q, dy_t_alpha = nvfp4_quant(
                dy_t, rand_seed(grad_output.device), False
            )
            x_t_theta, x_t_q, x_t_alpha = nvfp4_quant(
                x_t, rand_seed(grad_output.device), False
            )
            dw = mm_nvfp4(
                dy_t_theta,
                dy_t_q,
                dy_t_alpha,
                x_t_theta,
                x_t_q,
                x_t_alpha,
                ctx.n,
                ctx.w_k,
            )

        return None, dx, None, dw
