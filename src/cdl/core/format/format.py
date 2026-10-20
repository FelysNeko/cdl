import abc

import torch
from torch import nn


class Format(nn.Module, abc.ABC):
    def __init__(self, level: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("level", level, persistent=False)
        gap = self.level.diff()
        gap = torch.cat([gap, gap[-1:]])
        gap = torch.where(gap == 0, gap.roll(1), gap)
        self.register_buffer("gap", gap, persistent=False)

    @property
    def n(self) -> int:
        return self.level.numel()

    @property
    def bits(self) -> int:
        return (self.n - 1).bit_length()

    @abc.abstractmethod
    def ordinal(self, v: torch.Tensor) -> torch.Tensor: ...

    def window(self, v: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 < topk <= self.n

        key = self.ordinal(v)
        assert self.n - 1 <= torch.iinfo(key.dtype).max

        with torch.no_grad():
            start = torch.where(key < topk // 2, 0, key - topk // 2).clamp_(
                max=self.n - topk
            )

        index = key.long()
        key_level = self.level[index]
        left = torch.where(key == 0, 0, key - 1).long()
        gap = torch.where(v >= key_level, self.gap[index], self.gap[left])
        cont_pos = index.float() + (v - key_level) / gap
        return cont_pos, start
