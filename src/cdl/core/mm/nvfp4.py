import torch

from cdl.api.quant import QTensor
from cdl.core.kernel.dequant import nvfp4_dequant
from cdl.core.kernel.quant import nvfp4_quant
from cdl.core.kernel.transpose import fp4_repack_transpose, rht16_transpose
from cdl.core.mm.swizzle import expand_rows, pad_k, pad_n, swizzle_scale_factors
from cdl.model.wrapper.flashinfer import mm_fp4


@torch.library.custom_op(
    "cdl::rand_seed",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_seeded,),
)
def rand_seed(ref: torch.Tensor) -> torch.Tensor:
    """Sample a 0-d int32 seed; prevents AOTAutograd/Inductor from rematerializing different seeds across fwd/bwd."""
    return torch.randint(0, 2**31, (), dtype=torch.int32, device=ref.device)


@rand_seed.register_fake
def _(ref: torch.Tensor) -> torch.Tensor:
    return torch.empty((), dtype=torch.int32, device=ref.device)


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


def matmul(x: QTensor, w: QTensor, low_precision_backward: bool) -> torch.Tensor:
    return MatMul.apply(x, x.proxy(), w, w.proxy(), low_precision_backward)


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: QTensor,
        x_proxy: torch.Tensor | None,
        w: QTensor,
        w_proxy: torch.Tensor | None,
        low_precision_backward: bool,
    ) -> torch.Tensor:
        need_dgrad = x_proxy is not None and ctx.needs_input_grad[1]
        need_wgrad = w_proxy is not None and ctx.needs_input_grad[3]
        ctx.x_k = x_proxy.shape[-1] if need_dgrad else None
        ctx.w_k = w_proxy.shape[-1] if need_wgrad else None
        ctx.low_precision_backward = low_precision_backward

        x_theta, x_q, x_alpha = x.nvfp4()
        w_theta, w_q, w_alpha = w.nvfp4()

        m = x.shape[0]
        n = w.shape[0]
        ctx.m = m
        ctx.n = n

        x_theta, x_q = pad_k(x_theta, x_q, 2, 16, 32)
        w_theta, w_q = pad_k(w_theta, w_q, 2, 16, 32)
        ctx.weight_2d = w_q.shape[0] != w_theta.shape[0]

        save_x: tuple[torch.Tensor | None, ...] = (None, None, None)
        save_xt: tuple[torch.Tensor | None, ...] = (None, None, None, None)
        if need_wgrad:
            if low_precision_backward:
                x_raw = x.source()
                seed = rand_seed(x_raw)
                x_t = rht16_transpose(x_raw, seed)
                x_t_theta, x_t_q, x_t_alpha = nvfp4_quant(x_t, None, False)
                save_xt = (x_t_theta, x_t_q, x_t_alpha, seed)
            else:
                save_x = (x_theta, x_q, x_alpha)

        save_w: tuple[torch.Tensor | None, ...] = (
            (w_theta, w_q, w_alpha) if need_dgrad else (None, None, None)
        )

        ctx.save_for_backward(*save_x, *save_w, *save_xt)

        return mm_nvfp4(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha, m, n)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor | None, None, torch.Tensor | None, None]:
        grad_output = grad_output.contiguous()
        (
            x_theta,
            x_q,
            x_alpha,
            w_theta,
            w_q,
            w_alpha,
            x_t_theta,
            x_t_q,
            x_t_alpha,
            rht_seed,
        ) = ctx.saved_tensors
        dx = dw = None

        if ctx.x_k is not None:
            if ctx.low_precision_backward and ctx.weight_2d:
                w_rep = fp4_repack_transpose(w_theta)
                w_q_rep = expand_rows(w_q.T.contiguous(), w_rep.shape[0])
                w_rep, w_q_rep = pad_k(w_rep, w_q_rep, 2, 16, 32)
                dy_theta, dy_q, dy_alpha = nvfp4_quant(
                    grad_output, rand_seed(grad_output), False
                )
                dy_theta, dy_q = pad_k(dy_theta, dy_q, 2, 16, 32)
                dx = mm_nvfp4(
                    dy_theta, dy_q, dy_alpha, w_rep, w_q_rep, w_alpha, ctx.m, ctx.x_k
                )
            else:
                w = nvfp4_dequant(w_theta, expand_rows(w_q, w_theta.shape[0]), w_alpha)
                dx = grad_output @ w[: ctx.n, : ctx.x_k]

        if ctx.w_k is not None:
            if ctx.low_precision_backward:
                dy_t = rht16_transpose(grad_output, rht_seed)
                dy_t_theta, dy_t_q, dy_t_alpha = nvfp4_quant(
                    dy_t, rand_seed(grad_output), False
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
            else:
                x = nvfp4_dequant(x_theta, expand_rows(x_q, x_theta.shape[0]), x_alpha)[
                    : ctx.m, : ctx.w_k
                ]
                dw = grad_output.T @ x

        return None, dx, None, dw, None
