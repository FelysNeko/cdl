import torch

from cdl.api.tensor import QuantizedTensor
from cdl.core.format.fp8 import E4M3, E8M0
from cdl.model.wrapper.flashinfer import mm_fp4, mm_mxfp8, nvfp4_dequantize

E2M1_DEQ_LUT = (
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
)


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


def unswizzle_nvfp4_scale_factors(sf: torch.Tensor) -> torch.Tensor:
    """Inverse of `swizzle_nvfp4_scale_factors` on padded inputs."""
    rows, cols = sf.shape
    return (
        sf.view(rows // 128, 32, 4, cols // 4, 4)
        .permute(0, 2, 1, 3, 4)
        .reshape(rows, cols)
    )


def mxfp4_dequant(
    packed: torch.Tensor, scale: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Dequant packed MXFP4 bytes with linear-layout E8M0 scales to bfloat16.

    packed: [rows, k/2] uint8 (two E2M1 codes per byte, low nibble first);
    scale: [rows, k/32] uint8 E8M0 codes; alpha: scalar global scale tensor.
    """
    rows, half_k = packed.shape
    nib = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, half_k * 2)
    chosen = torch.where(nib < 8, nib + 8, 15 - nib).long()
    lut = torch.tensor(E2M1_DEQ_LUT, dtype=torch.float32, device=packed.device)
    vals = lut[chosen].reshape(rows, (half_k * 2) // 32, 32)
    scales = torch.exp2(scale.float() - 127.0)
    out = (vals * scales[..., None]).flatten(1)
    return (out * alpha.float()).to(torch.bfloat16)


def mxfp8_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequant packed MXFP8 (raw E4M3) bytes with linear E8M0 scales to bfloat16."""
    rows, k = packed.shape
    vals = packed.view(torch.float8_e4m3fn).float().reshape(rows, k // 32, 32)
    scales = torch.exp2(scale.float() - 127.0)
    return (vals * scales[..., None]).flatten(1).to(torch.bfloat16)


def matmul(x: QuantizedTensor, w: QuantizedTensor) -> torch.Tensor:
    return MatMul.apply(x, x.proxy(), w, w.proxy())


def _is_mx(x: QuantizedTensor) -> bool:
    return isinstance(getattr(x, "q_format", None), E8M0)


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: QuantizedTensor,
        x_proxy: torch.Tensor | None,
        w: QuantizedTensor,
        w_proxy: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.x_k = x_proxy.shape[-1] if x_proxy is not None else getattr(x, "k", None)
        ctx.w_k = w_proxy.shape[-1] if w_proxy is not None else getattr(w, "k", None)
        if _is_mx(x):
            ctx.mx = True
            return _mx_forward(ctx, x, w)
        ctx.mx = False
        x_theta, x_q, x_alpha = x.fp4()
        w_theta, w_q, w_alpha = w.fp4()
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        return mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_nvfp4_scale_factors(x_q),
            swizzle_nvfp4_scale_factors(w_q).T,
            x_alpha * w_alpha,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor]:
        grad_output = grad_output.contiguous()
        if ctx.mx:
            dense_x, dense_w = ctx.saved_tensors
            dx = grad_output @ dense_w[:, : ctx.x_k]
            dw = grad_output.T @ dense_x[:, : ctx.w_k]
            return None, dx, None, dw

        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = ctx.saved_tensors
        w = nvfp4_dequantize(w_theta, w_q, w_alpha)
        dx = grad_output @ w[:, : ctx.x_k]
        x = nvfp4_dequantize(x_theta, x_q, x_alpha)
        dw = grad_output.T @ x[:, : ctx.w_k]
        return None, dx, None, dw


def _mx_forward(ctx, x: QuantizedTensor, w: QuantizedTensor) -> torch.Tensor:
    theta_format = getattr(x, "theta_format", None)
    if isinstance(theta_format, E4M3):
        x_packed, x_scale = x.fp8()
        w_packed, w_scale = w.fp8()
        dense_x = mxfp8_dequant(x_packed, x_scale)
        dense_w = mxfp8_dequant(w_packed, w_scale)
        out = mm_mxfp8(
            x_packed.view(torch.float8_e4m3fn),
            w_packed.view(torch.float8_e4m3fn).T,
            swizzle_nvfp4_scale_factors(x_scale).flatten(),
            swizzle_nvfp4_scale_factors(w_scale).flatten(),
        )
    else:
        x_theta, x_q, x_alpha = x.fp4()
        w_theta, w_q, w_alpha = w.fp4()
        dense_x = mxfp4_dequant(x_theta, x_q, x_alpha)
        dense_w = mxfp4_dequant(w_theta, w_q, w_alpha)
        out = mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_nvfp4_scale_factors(x_q),
            swizzle_nvfp4_scale_factors(w_q).T,
            x_alpha * w_alpha,
            block_size=32,
            use_nvfp4=False,
        )
    ctx.save_for_backward(dense_x.detach(), dense_w.detach())
    return out
