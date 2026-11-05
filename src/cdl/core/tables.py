import functools

import torch

from cdl.api.format import gaps


@functools.cache
def e2m1_tables(
    dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    key = torch.arange(16, device=device)
    m = torch.where(key >= 8, key - 8, 7 - key)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=dtype, device=device
    )
    level = torch.where(key >= 8, magnitude[m], -magnitude[m])
    return level, gaps(level)


@functools.cache
def ue4m3_exclude_zero_tables(
    dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    level = (
        torch.arange(1, 127, dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .to(dtype)
    )
    return level, gaps(level)


@functools.cache
def e8m0_tables(
    dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    level = (
        torch.arange(255, dtype=torch.uint8, device=device)
        .view(torch.float8_e8m0fnu)
        .to(dtype)
    )
    return level, gaps(level)


@functools.cache
def e4m3_tables(
    dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cpu") if device is None else device
    key = torch.arange(254, device=device)
    u = torch.where(key >= 127, key - 127, 254 - key).to(torch.uint8)
    level = u.view(torch.float8_e4m3fn).to(dtype)
    return level, gaps(level)
