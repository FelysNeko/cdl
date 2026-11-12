import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelConfig:
    """Launch configuration shared by all quantizer kernels."""

    block_chunk: int = 32
    num_warps: int = 1


def from_env() -> KernelConfig:
    return KernelConfig(
        block_chunk=int(os.environ.get("CDL_BB", "32")),
        num_warps=int(os.environ.get("CDL_NUM_WARPS", "1")),
    )


DEFAULT_CONFIG = from_env()
