from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, eq=False)
class BatchOutput:
    ce_loss: torch.Tensor
    weight_entropy: torch.Tensor
    activation_entropy: torch.Tensor
    error_rate: torch.Tensor

    def compute_joint_loss(self, lam: float, gam: float) -> torch.Tensor:
        return self.ce_loss + lam * self.weight_entropy + gam * self.activation_entropy


class Trainer:
    def __init__(self, resnet: nn.Module, device: torch.device) -> None:
        self.resnet = resnet.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def compile(self) -> None:
        torch._dynamo.config.cache_size_limit = 128
        self.resnet = torch.compile(self.resnet, dynamic=False, mode="reduce-overhead")

    def forward_one_batch(self, x: torch.Tensor, y: torch.Tensor) -> BatchOutput:
        x, y = x.to(self.device), y.to(self.device)
        out: torch.Tensor = self.resnet(x)

        cross_entropy_loss = self.criterion(out, y)
        weight_entropy_loss, activation_entropy_loss = self.resnet.drain_entropies()

        is_error = out.argmax(dim=-1) != y
        error_rate = is_error.float().mean()

        return BatchOutput(
            ce_loss=cross_entropy_loss,
            weight_entropy=weight_entropy_loss,
            activation_entropy=activation_entropy_loss,
            error_rate=error_rate,
        )

    def recalibrate_bn(self, loader: DataLoader, num_batches: int) -> None:
        self.resnet.recalibrate_bn(loader, num_batches, self.device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.resnet.eval()
            correct = total = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.resnet(x)
                correct += (out.argmax(-1) == y).sum().item()
                total += y.numel()
            return correct / total
