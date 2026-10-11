import torch
import torch.nn.functional as F
from torch import nn


class PadShortcut(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        if (planes - in_planes) % 2 != 0:
            raise ValueError("shortcut requires an even channel difference")
        self.stride = stride
        self.pad = (planes - in_planes) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]

        return F.pad(x, (0, 0, 0, 0, self.pad, self.pad))
