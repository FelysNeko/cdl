"""Format bundles shared by the quantizer kernels.

A :class:`FormatSpec` couples the host-side tables of a quantization format
(theta levels/gaps, scale levels/gaps) with the Triton ordinal functions used
inside the kernels.  All tensors are cached per ``(format, device)``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import torch

from cdl.core.kernel.formats.ordinals import (
    e2m1_single_zero_triton_ordinal,
    e4m3_triton_ordinal,
    e8m0_triton_ordinal,
    ue4m3_triton_ordinal,
)
from cdl.core.tables import (
    e2m1_single_zero_tables,
    e4m3_tables,
    e8m0_tables,
    ue4m3_exclude_zero_tables,
)


class FormatId(IntEnum):
    NVFP4 = 0
    MXFP4 = 1
    MXFP8 = 2


@dataclass(frozen=True)
class FormatSpec:
    name: str
    tn: int
    qn: int
    q_max: float
    f_max: float
    theta_level: torch.Tensor
    theta_gap: torch.Tensor
    q_level: torch.Tensor
    q_gap: torch.Tensor
    theta_ordinal: Callable
    q_ordinal: Callable
    theta_emax: int
    # Added to the scale-level index to obtain the stored scale byte
    # (UE4M3 excludes zero, so its byte is index + 1; E8M0 byte == index).
    scale_bias: int


def _theta_emax(theta_level: torch.Tensor) -> int:
    return math.floor(math.log2(float(theta_level.abs().amax())))


def _build(fmt_id: FormatId, device: torch.device) -> FormatSpec:
    if fmt_id is FormatId.NVFP4:
        theta_level, theta_gap = e2m1_single_zero_tables(torch.float32, device)
        q_level, q_gap = ue4m3_exclude_zero_tables(torch.float32, device)
        return FormatSpec(
            "nvfp4",
            theta_level.numel(),
            q_level.numel(),
            float(q_level[-1]),
            float(theta_level.amax().item()) * float(q_level.amax().item()),
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            e2m1_single_zero_triton_ordinal,
            ue4m3_triton_ordinal,
            _theta_emax(theta_level),
            1,
        )
    if fmt_id is FormatId.MXFP4:
        theta_level, theta_gap = e2m1_single_zero_tables(torch.float32, device)
        q_level, q_gap = e8m0_tables(torch.float32, device)
        return FormatSpec(
            "mxfp4",
            theta_level.numel(),
            q_level.numel(),
            float(q_level[-1]),
            float(theta_level.amax().item()) * float(q_level.amax().item()),
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            e2m1_single_zero_triton_ordinal,
            e8m0_triton_ordinal,
            _theta_emax(theta_level),
            0,
        )
    if fmt_id is FormatId.MXFP8:
        theta_level, theta_gap = e4m3_tables(torch.float32, device)
        q_level, q_gap = e8m0_tables(torch.float32, device)
        return FormatSpec(
            "mxfp8",
            theta_level.numel(),
            q_level.numel(),
            float(q_level[-1]),
            float(theta_level.amax().item()) * float(q_level.amax().item()),
            theta_level,
            theta_gap,
            q_level,
            q_gap,
            e4m3_triton_ordinal,
            e8m0_triton_ordinal,
            _theta_emax(theta_level),
            0,
        )
    raise ValueError(f"unknown format id {fmt_id!r}")


_CACHE: dict[tuple[FormatId, torch.device], FormatSpec] = {}


def get_format(format_id: int, device: torch.device) -> FormatSpec:
    fmt_id = FormatId(format_id)
    key = (fmt_id, device)
    spec = _CACHE.get(key)
    if spec is None:
        spec = _build(fmt_id, device)
        _CACHE[key] = spec
    return spec
