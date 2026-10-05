import torch

from cdl.format.format import Format


class FP4E2M1(Format):
    def __init__(self) -> None:
        key = torch.arange(16)
        m = torch.where(key >= 8, key - 8, 7 - key)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        level = torch.where(key >= 8, magnitude[m], -magnitude[m])
        super().__init__(16, 4, level)
        mid = (magnitude[:-1] + magnitude[1:]) / 2
        self.register_buffer("mid", mid, persistent=False)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        m = (v.abs().clamp(max=6.0).unsqueeze(-1) > self.mid).sum(-1)
        return torch.where(torch.signbit(v), 7 - m, m + 8).long()
