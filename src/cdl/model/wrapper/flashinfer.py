import flashinfer
import torch


@torch.library.custom_op("cdl::mm_fp4", mutates_args=())
def mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    block_size: int,
    use_nvfp4: bool,
) -> torch.Tensor:
    return flashinfer.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        torch.bfloat16,
        block_size=block_size,
        use_nvfp4=use_nvfp4,
        backend="b12x",
    )


@mm_fp4.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    block_size: int,
    use_nvfp4: bool,
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
