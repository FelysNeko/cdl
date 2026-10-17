import torch

from cdl.core.format.format import Format


class IntX(Format):
    def __init__(self, signed: bool, bits: int) -> None:
        start = 0
        if signed:
            start = -(2 ** (bits - 1))
        level = torch.arange(start, start + 2**bits)
        super().__init__(level)
        self.start = start

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        return (v.round().long() - self.start).clamp_(0, self.n - 1)
