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
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    if use_nvfp4:
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
    return flashinfer.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        torch.bfloat16,
        block_size=32,
        use_nvfp4=False,
        backend="b12x",
    )


@mm_fp4.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    block_size: int = 16,
    use_nvfp4: bool = True,
) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=torch.bfloat16, device=a.device)


@torch.library.custom_op("cdl::mm_mxfp8", mutates_args=())
def mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    # a: [m, k] mxfp8 e4m3 row-major; b: [k, n] e4m3 column-major; scales are
    # the 1D flattened 128x4 swizzled E8M0 layouts (block size 32).
    return flashinfer.mm_mxfp8(
        a, b, a_scale, b_scale, out_dtype=torch.bfloat16, backend="b12x"
    )


@mm_mxfp8.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=torch.bfloat16, device=a.device)


@torch.library.custom_op("cdl::mxfp4_quantize", mutates_args=())
def mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, k] with k % 32 == 0, bfloat16, row-major. Returns (packed fp4
    # [rows, k / 2] uint8, 128x4 swizzled E8M0 scales [rows_pad, k / 32 pad4]
    # uint8) as consumed by mm_fp4 with use_nvfp4=False.
    packed, scale = flashinfer.mxfp4_quantize(x)
    return packed, scale


@mxfp4_quantize.register_fake
def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = x.shape
    return (
        torch.empty(rows, k // 2, dtype=torch.uint8, device=x.device),
        torch.empty(
            (rows + 127) // 128 * 128,
            (k // 32 + 3) // 4 * 4,
            dtype=torch.uint8,
            device=x.device,
        ),
    )


@torch.library.custom_op("cdl::mxfp8_quantize", mutates_args=())
def mxfp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, k] with k % 32 == 0, bfloat16, row-major. Returns (e4m3 bytes
    # [rows, k] as uint8, linear E8M0 scales [rows, k / 32] uint8).
    rows, k = x.shape
    packed, scale = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
    return packed.view(torch.uint8), scale.reshape(rows, k // 32)


@mxfp8_quantize.register_fake
def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, k = x.shape
    return (
        torch.empty(rows, k, dtype=torch.uint8, device=x.device),
        torch.empty(rows, k // 32, dtype=torch.uint8, device=x.device),
    )


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
