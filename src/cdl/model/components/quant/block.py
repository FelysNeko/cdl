from collections.abc import Sequence

import torch
import torch.nn.functional as F


def group_blocks_by_dims(
    x: torch.Tensor,
    block_dims: Sequence[bool],
    block_size: int,
) -> tuple[torch.Tensor, tuple[bool, ...], tuple[int, ...]]:
    orig_shape = tuple(x.shape)
    pads = tuple(
        (-s % block_size) if is_block else 0
        for s, is_block in zip(orig_shape, block_dims, strict=True)
    )
    if any(pads):
        pad_args: list[int] = []
        for p in reversed(pads):
            pad_args.extend((0, p))
        x = F.pad(x, pad_args)

    blocked_shape: list[int] = []
    block_mask: list[bool] = []
    for s, is_block in zip(x.shape, block_dims, strict=True):
        if is_block:
            blocked_shape.extend((s // block_size, block_size))
            block_mask.extend((False, True))
        else:
            blocked_shape.append(s)
            block_mask.append(False)

    return x.reshape(blocked_shape), tuple(block_mask), orig_shape


def ungroup_blocks(
    tensor: torch.Tensor,
    block_mask: Sequence[bool],
    orig_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    merged_shape: list[int] = []
    for size, is_block in zip(tensor.shape, block_mask, strict=True):
        if is_block:
            merged_shape[-1] *= size
        else:
            merged_shape.append(size)
    out = tensor.reshape(merged_shape)
    if orig_shape is not None and tuple(out.shape) != tuple(orig_shape):
        out = out[tuple(slice(None, d) for d in orig_shape)]
    return out


def squeeze_block_scale(
    scale: torch.Tensor,
    block_mask: Sequence[bool],
) -> torch.Tensor:
    dims = tuple(i for i, is_block in enumerate(block_mask) if is_block)
    return scale.squeeze(dims) if dims else scale


def check_triton_block_dims(block_dims: Sequence[bool]) -> bool:
    if not any(block_dims[:-1]) and block_dims[-1]:
        return False
    if (
        len(block_dims) >= 2
        and not any(block_dims[:-2])
        and block_dims[-2]
        and block_dims[-1]
    ):
        return True
    raise NotImplementedError(
        f"triton kernel only supports 1D or 2D trailing block scaling, got block_dims={block_dims}"
    )
