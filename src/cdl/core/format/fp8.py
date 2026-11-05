import torch

from cdl.api.format import Format
from cdl.core.tables import e4m3_tables, e8m0_tables, ue4m3_exclude_zero_tables


class E4M3(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        level, _ = e4m3_tables(dtype)
        super().__init__(level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.where(u < 128, u + 127, 254 - u).clamp_(0, 253)

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        u = torch.where(i < 127, 254 - i, i - 127)
        return u.to(torch.uint8).view(torch.float8_e4m3fn)


class UE4M3ExcludeZero(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        level, _ = ue4m3_exclude_zero_tables(dtype)
        super().__init__(level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(2.0**-9, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        return (u - 1).clamp_(0, 125)

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        return (i + 1).to(torch.uint8)


class E8M0(Format):
    def __init__(self, dtype: torch.dtype) -> None:
        level, _ = e8m0_tables(dtype)
        super().__init__(level)

    def ordinal(self, v: torch.Tensor) -> torch.Tensor:
        u = v.clamp(2.0**-127, 2.0**127).to(torch.float8_e8m0fnu).view(torch.uint8)
        return u.clamp_(0, 254)

    def pack(self, i: torch.Tensor) -> torch.Tensor:
        return i.to(torch.uint8)
