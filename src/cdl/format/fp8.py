import torch

from cdl.format.format import Format


class FP8E4M3(Format):
    def __init__(self) -> None:
        key = torch.arange(254)
        u = torch.where(key >= 127, key - 127, 254 - key).to(torch.uint8)
        level = u.view(torch.float8_e4m3fn).to(torch.float32)
        super().__init__(254, 8, level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).long()
        return torch.where(u < 128, u + 127, 254 - u).clamp_(0, 253)
