import torch

from cdl.core.format.format import Format


class IntX(Format):
    def __init__(self, signed: bool, bits: int) -> None:
        start = 0
        if signed:
            start = -(2 ** (bits - 1))
        n = 2**bits
        end = start + n
        level = torch.arange(start, end)
        super().__init__(n, bits, level)
        self.start = start

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        return (v.round().long() - self.start).clamp_(0, self.n - 1)
