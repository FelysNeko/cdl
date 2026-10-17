import functools

import flashinfer
import torch


@functools.cache
def get_indices_to_e2m1_mapping(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=device,
    )


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    mapping = get_indices_to_e2m1_mapping(x.device)
    codes = mapping[x].reshape(x.shape[0], -1)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def pack_e4m3(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(x.shape[0], -1)
        .float()
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


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


class NVFP4:
    def __init__(
        self,
        theta: torch.Tensor,
        q: torch.Tensor,
        alpha: torch.Tensor,
    ) -> None:
        self.theta = theta
        self.q = q
        self.alpha = alpha

    @torch.no_grad()
    def pack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = pack_e2m1(self.theta)
        q = pack_e4m3(self.q)
        alpha = self.alpha.float().reshape(1)
        return theta, q, alpha


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: NVFP4,
        x_proxy: torch.Tensor,
        w: NVFP4,
        w_proxy: torch.Tensor,
    ) -> torch.Tensor:
        x_theta, x_q, x_alpha = x.pack()
        w_theta, w_q, w_alpha = w.pack()
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
