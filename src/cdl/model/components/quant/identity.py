import torch
from torch import nn


class IdentityQuant(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
