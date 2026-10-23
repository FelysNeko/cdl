# All flashinfer interop lives here: the custom ops registered for
# torch.compile tracing plus the scale layout helpers the kernels require.
import flashinfer
import torch


@torch.library.custom_op("cdl::mm_fp4", mutates_args=())
def mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return flashinfer.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        torch.bfloat16,
        block_size=16,
        backend="b12x",
    )


@mm_fp4.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=torch.bfloat16, device=a.device)


@torch.library.custom_op("cdl::nvfp4_quantize", mutates_args=())
def nvfp4_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, k] with k % 32 == 0 (mm_fp4 TMA alignment), bfloat16, row-major.
    # Returns (packed fp4 [rows, k / 2] uint8, block scales [rows, k / 16] uint8)
    # in the linear (non-swizzled) scale layout.
    packed, scale = flashinfer.fp4_quantize(
        x, global_scale, is_sf_swizzled_layout=False
    )
    return packed, scale


@nvfp4_quantize.register_fake
def _(x: torch.Tensor, global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = x.shape
    return (
        torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device),
        torch.empty(rows, k // 16, dtype=torch.uint8, device=x.device),
    )


@torch.library.custom_op("cdl::nvfp4_dequantize", mutates_args=())
def nvfp4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> torch.Tensor:
    # packed: [rows, k / 2] uint8, scale: [rows, k / 16] uint8 (linear layout).
    # Returns the dequantized bfloat16 tensor of shape [rows, k].
    return flashinfer.nvfp4_kv_dequantize(
        packed, scale, global_scale_inv, output_dtype=torch.bfloat16
    )


@nvfp4_dequantize.register_fake
def _(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
) -> torch.Tensor:
    rows = packed.shape[0]
    k = packed.shape[-1] * 2
    return torch.empty(rows, k, dtype=torch.bfloat16, device=packed.device)
