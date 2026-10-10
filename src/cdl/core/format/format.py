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

    def window(
        self, v: torch.Tensor, topk: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.ordinal(v)
        with torch.no_grad():
            start = (key - topk // 2).clamp_(0, self.n - topk)
        topk_pos = start.unsqueeze(-1) + torch.arange(topk, device=v.device)
        topk_level = self.level[topk_pos]

        key_level = self.level[key]
        gap = torch.where(
            v >= key_level, self.gap[key], self.gap[(key - 1).clamp(min=0)]
        )
        cont_pos = key.float() + (v - key_level) / gap
        return cont_pos, topk_level, topk_pos
