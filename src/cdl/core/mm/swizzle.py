import torch
import torch.nn.functional as F


def swizzle_scale_factors(sf: torch.Tensor) -> torch.Tensor:
    m, b = sf.shape
    rows, cols = (m + 127) // 128 * 128, (b + 3) // 4 * 4
    out = torch.zeros(rows, cols, dtype=torch.uint8, device=sf.device)
    out[:m, :b] = sf
    return (
        out.view(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


def expand_rows(scale: torch.Tensor, rows: int) -> torch.Tensor:
    if scale.shape[0] != rows:
        scale = scale.expand(rows, -1).contiguous()
    return scale


def pad_k(
    packed: torch.Tensor,
    scale: torch.Tensor,
    elements_per_byte: int,
    block: int,
    multiple: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = packed.shape[-1] * elements_per_byte
    pad = -k % multiple
    if pad:
        packed = F.pad(packed, (0, pad // elements_per_byte))
        scale = F.pad(scale, (0, pad // block))
    return packed, scale


def pad_n(
    packed: torch.Tensor,
    scale: torch.Tensor,
    multiple: int,
    minimum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = max(packed.shape[0], minimum)
    target += -target % multiple
    n_pad = target - packed.shape[0]
    if n_pad:
        packed = F.pad(packed, (0, 0, 0, n_pad))
        scale = F.pad(scale, (0, 0, 0, n_pad))
    return packed, scale
