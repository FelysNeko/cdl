import torch

from cdl.api.format import Format


class IntX(Format):
    def __init__(self, signed: bool, bits: int, dtype: torch.dtype) -> None:
        start = 0
        if signed:
            start = -(2 ** (bits - 1))
        level = torch.arange(start, start + 2**bits, dtype=dtype)
        super().__init__(level)
        self.start = start

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        r = (v.round() - self.start).clamp(0, self.n - 1)
        return r.to(torch.uint8)

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        return self.level[i.long()]
