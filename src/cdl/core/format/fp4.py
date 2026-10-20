import torch

from cdl.core.format.format import Format


class E2M1(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        key = torch.arange(16)
        m = torch.where(key >= 8, key - 8, 7 - key)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=dtype)
        level = torch.where(key >= 8, magnitude[m], -magnitude[m])
        super().__init__(level)
        mid = (magnitude[:-1] + magnitude[1:]) / 2
        self.register_buffer("mid", mid, persistent=False)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        m = (v.abs().clamp(max=6.0).unsqueeze(-1) > self.mid).sum(-1, dtype=torch.uint8)
        return torch.where(torch.signbit(v), 7 - m, m + 8)

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        code = torch.where(i < 8, 15 - i, i - 8).to(torch.uint8)
        return code[..., 0::2] | (code[..., 1::2] << 4)


class UE2M1(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        level = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=dtype)
        super().__init__(level)
        mid = (level[:-1] + level[1:]) / 2
        self.register_buffer("mid", mid, persistent=False)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clamp(0.0, 6.0)
        m = (v.unsqueeze(-1) > self.mid).sum(-1, dtype=torch.uint8)
        return m

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        code = i.to(torch.uint8)
        return code[..., 0::2] | (code[..., 1::2] << 4)
