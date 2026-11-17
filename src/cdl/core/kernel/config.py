import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelConfig:
    """Launch configuration shared by all quantizer kernels."""

    block_chunk: int = 8
    num_warps: int = 4


def from_env() -> KernelConfig:
    return KernelConfig(
        block_chunk=int(os.environ.get("CDL_BB", "8")),
        num_warps=int(os.environ.get("CDL_NUM_WARPS", "4")),
    )


DEFAULT_CONFIG = from_env()
