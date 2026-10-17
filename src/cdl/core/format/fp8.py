import torch

from cdl.core.format.format import Format


class FP8E4M3(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        key = torch.arange(254)
        u = torch.where(key >= 127, key - 127, 254 - key).to(torch.uint8)
        level = u.view(torch.float8_e4m3fn).to(dtype)
        super().__init__(level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.where(u < 128, u + 127, 254 - u).clamp_(0, 253)


class FP8E8M0(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        key = torch.arange(255)
        level = key.to(torch.uint8).view(torch.float8_e8m0fnu).to(dtype)
        super().__init__(level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(2.0**-127, 2.0**127).to(torch.float8_e8m0fnu).view(torch.uint8)
        return u.clamp_(0, 254)
