from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cdl.model.quant import CdlQuant, CdlQuantForActivation, CdlQuantForWeight
from cdl.model.resnet import QResNet


@dataclass(frozen=True, eq=False)
class BatchOutput:
    ce_loss: torch.Tensor
    weight_entropy: torch.Tensor
    activation_entropy: torch.Tensor
    error_rate: float

    def compute_joint_loss(self, lam: float, gam: float) -> torch.Tensor:
        return self.ce_loss + lam * self.weight_entropy + gam * self.activation_entropy


class Trainer:
    def __init__(self, resnet: QResNet, device: torch.device) -> None:
        self.resnet = resnet.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def compile(self) -> None:
        torch._dynamo.config.cache_size_limit = 128
        self.resnet = torch.compile(self.resnet, dynamic=True)

    @contextmanager
    def batchnorm_batch_stats(self):
        states: list[
            tuple[nn.BatchNorm2d, bool, float | None, torch.Tensor | None]
        ] = []

        for m in self.resnet.modules():
            if not isinstance(m, nn.BatchNorm2d):
                continue
            states.append(
                (
                    m,
                    m.training,
                    m.momentum,
                    None
                    if m.num_batches_tracked is None
                    else m.num_batches_tracked.clone(),
                )
            )
            m.train(True)
            m.momentum = 0.0

        try:
            yield
        finally:
            for m, training, momentum, num_batches_tracked in states:
                m.train(training)
                m.momentum = momentum
                if num_batches_tracked is not None:
                    m.num_batches_tracked.copy_(num_batches_tracked)

    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m
            for m in self.resnet.modules()
            if isinstance(m, CdlQuantForActivation) and m.q.isnan()
        ]
        if not modules:
            return

        stats = {m: [0.0, 0, 0] for m in modules}

        def hook(mod, args):
            x: torch.Tensor = args[0]
            s = stats[mod]
            s[0] += x.abs().sum().item()
            s[1] += x.numel()
            s[2] = x.shape[1:].numel()

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.resnet.training
        self.resnet.eval()
        for m in modules:
            m.bypassing = True
        try:
            with self.batchnorm_batch_stats():
                for x in samples:
                    x = x.to(self.device)
                    self.resnet(x)
        finally:
            for h in handles:
                h.remove()
            for m in modules:
                m.bypassing = False
            self.resnet.train(was_training)

        for m in modules:
            total, n, dim = stats[m]
            if n == 0:
                raise RuntimeError("some quantizers were never reached by samples")
            m.init_q_and_numel(total / n, dim)

    def forward_one_batch(self, x: torch.Tensor, y: torch.Tensor) -> BatchOutput:
        x, y = x.to(self.device), y.to(self.device)
        out: torch.Tensor = self.resnet(x)

        cross_entropy_loss = self.criterion(out, y)
        weight_entropy_loss = torch.zeros(())
        activation_entropy_loss = torch.zeros(())

        for m in self.resnet.modules():
            if isinstance(m, CdlQuantForWeight):
                e = m.drain_entropy()
                weight_entropy_loss = weight_entropy_loss + e

            if isinstance(m, CdlQuantForActivation):
                e = m.drain_entropy()
                activation_entropy_loss = activation_entropy_loss + e

        is_error = out.argmax(dim=-1) != y
        error_rate = is_error.float().mean().item()

        return BatchOutput(
            ce_loss=cross_entropy_loss,
            weight_entropy=weight_entropy_loss,
            activation_entropy=activation_entropy_loss,
            error_rate=error_rate,
        )

    def get_param_groups(self, learning_rate: float, weight_decay: float) -> list[dict]:
        customized_param_set = set()
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}

        def add(param: nn.Parameter, lr: float, wd: float) -> None:
            customized_param_set.add(param)
            buckets.setdefault((lr, wd), []).append(param)

        for m in self.resnet.modules():
            if not isinstance(m, CdlQuant):
                continue

            if m.num_quantized_params == 0:
                raise RuntimeError("quantizer not initialized")

            add(m.q, m.scale_q_lr(learning_rate), 0.0)
            add(m.log_kappa, m.scale_kappa_lr(learning_rate), 0.0)

        decay = []
        no_decay = []
        for param in self.resnet.parameters():
            if param in customized_param_set:
                continue
            if param.ndim <= 1:
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in buckets.items()
        ]
        if decay:
            groups.append(
                {"params": decay, "lr": learning_rate, "weight_decay": weight_decay}
            )
        if no_decay:
            groups.append(
                {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0}
            )
        return groups

    @torch.no_grad()
    def floor_quant(self, min_q: float) -> None:
        for m in self.resnet.modules():
            if not isinstance(m, CdlQuant):
                continue
            m.q.clamp_(min=min_q)
            m.log_kappa.clamp_(min=-20.0, max=20.0)

    @torch.no_grad()
    def kappa_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for tag, cls in (("w", CdlQuantForWeight), ("a", CdlQuantForActivation)):
            vals = [
                m.kappa.item()
                for m in self.resnet.modules()
                if isinstance(m, cls) and m.num_quantized_params > 0
            ]
            if not vals:
                continue
            t = torch.tensor(vals)
            out[f"kappa_{tag}_mean"] = t.mean().item()
            out[f"kappa_{tag}_min"] = t.min().item()
            out[f"kappa_{tag}_max"] = t.max().item()
        return out

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

    def get_quant_counts(self) -> tuple[int, int]:
        n_weight = sum(
            m.num_quantized_params
            for m in self.resnet.modules()
            if isinstance(m, CdlQuantForWeight)
        )
        n_activation = sum(
            m.num_quantized_params
            for m in self.resnet.modules()
            if isinstance(m, CdlQuantForActivation)
        )
        return n_weight, n_activation
