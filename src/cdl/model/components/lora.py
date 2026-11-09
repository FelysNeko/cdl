import math

import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        device = base.weight.device
        dtype = base.weight.dtype

        self.base = base
        self.r = r
        self.scaling = alpha / r

        self.a = nn.Parameter(
            torch.empty(r, base.in_features, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.b = nn.Parameter(
            torch.zeros(base.out_features, r, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.a.T @ self.b.T
        return base_out + self.scaling * lora_out


def recursively_replace_linear_with_lora(
    module: nn.Module,
    target_names: set[str],
    r: int,
    alpha: int,
    dropout: float,
) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            new_layer = LoRALayer(child, r, alpha, dropout)
            setattr(module, name, new_layer)
        else:
            recursively_replace_linear_with_lora(child, target_names, r, alpha, dropout)


class LoRALayerMixin(nn.Module):
    def replace_linear_with_lora(
        self,
        target_names: set[str],
        r: int,
        alpha: int,
        dropout: float,
    ) -> None:
        recursively_replace_linear_with_lora(self, target_names, r, alpha, dropout)
