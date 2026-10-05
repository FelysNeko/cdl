from abc import ABC, abstractmethod

import torch
from torch import nn


class Format(nn.Module, ABC):
    def __init__(self, n: int, bits: int, level: torch.Tensor) -> None:
        super().__init__()
        self.n = n
        self.bits = bits
        self.register_buffer("level", level.float(), persistent=False)
        gap = self.level.diff()
        gap = torch.cat([gap, gap[-1:]])
        gap = torch.where(gap == 0, gap.roll(1), gap)
        self.register_buffer("gap", gap, persistent=False)

    @abstractmethod
    def ordinal(self, v: torch.Tensor) -> torch.Tensor: ...
