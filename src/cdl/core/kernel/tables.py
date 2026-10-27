"""Lookup tables for the formats supported by the fused CDL kernel.

The Triton kernels consume (level, gap) tables instead of hard-coded
alphabets; these tables must be built from the same definitions as the
reference formats in cdl.core.format (bit-exact level and gap values,
including the zero-gap roll fix from cdl.api.format).
"""

import functools

import torch

from cdl.core.format.fp8 import E4M3


def _gap_from_level(level: torch.Tensor) -> torch.Tensor:
    gap = level.diff()
    gap = torch.cat([gap, gap[-1:]])
    return torch.where(gap == 0, gap.roll(1), gap).contiguous()


def _format_tables(format_: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return format_.level.contiguous(), _gap_from_level(format_.level)


@functools.cache
def _e2m1_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    level = torch.tensor(
        [
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
        ],
        dtype=torch.float32,
        device=device,
    )
    return level, _gap_from_level(level)


@functools.cache
def _ue4m3_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(1, 127, dtype=torch.uint8, device=device)
    level = key.view(torch.float8_e4m3fn).float().contiguous()
    return level, _gap_from_level(level)


def e2m1_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(level, gap) tables for the E2M1 theta alphabet on `device`."""
    return _e2m1_tables(torch.device(device))


def ue4m3_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(level, gap) tables for the UE4M3ExcludeZero scale alphabet."""
    return _ue4m3_tables(torch.device(device))


@functools.cache
def _e8m0_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Built via exp2 (values bit-identical to the E8M0 view) so that Dynamo
    # tracing never materializes a float8_e8m0fnu buffer, which the Inductor
    # CPU codegen cannot handle.
    level = torch.exp2(torch.arange(255, dtype=torch.float32) - 127.0).to(device)
    return level, _gap_from_level(level)


@functools.cache
def _e4m3_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    level, gap = _format_tables(E4M3(torch.float32))
    return level.to(device), gap.to(device)


def e8m0_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(level, gap) tables for the E8M0 scale alphabet (255 power-of-two levels)."""
    return _e8m0_tables(torch.device(device))


def e4m3_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(level, gap) tables for the signed E4M3 theta alphabet (254 levels)."""
    return _e4m3_tables(torch.device(device))
