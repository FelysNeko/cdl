from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from cdl.model.model import CdlQuant, CdlQuantForActivation, CdlQuantForWeight
from cdl.model.resnet import QResNet


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

    def compute_one_batch_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = x.to(self.device), y.to(self.device)
        out = self.resnet(x)

        cross_entropy_loss = self.criterion(out, y)
        weight_entropy_loss = 0.0
        activation_entropy_loss = 0.0

        for m in self.resnet.modules():
            if isinstance(m, CdlQuantForWeight):
                e = m.compute_entropy_and_reset()
                weight_entropy_loss = weight_entropy_loss + e

            if isinstance(m, CdlQuantForActivation):
                e = m.compute_entropy_and_reset()
                activation_entropy_loss = activation_entropy_loss + e

        return cross_entropy_loss, weight_entropy_loss, activation_entropy_loss

    def get_param_groups(self, learning_rate: float, weight_decay: float) -> list[dict]:
        parameters = []
        customized_param_set = set()

        def register_parameter(ps: Any, lr: float, wd: float):
            parameters.append({"params": ps, "lr": lr, "weight_decay": wd})

        for m in self.resnet.modules():
            if not isinstance(m, CdlQuant):
                continue

            if m.numel == 0:
                raise RuntimeError("quantizer not initialized")

            q_lr = m.scale_q_lr(learning_rate)
            register_parameter(m.q, q_lr, 0.0)
            customized_param_set.add(m.q)

            alpha_lr = m.scale_alpha_lr(learning_rate)
            register_parameter(m.alpha, alpha_lr, 0.0)
            customized_param_set.add(m.alpha)

        decay = []
        no_decay = []
        for param in self.resnet.parameters():
            if param in customized_param_set:
                continue
            if param.ndim <= 1:
                no_decay.append(param)
            else:
                decay.append(param)

        if decay:
            register_parameter(decay, learning_rate, weight_decay)

        if no_decay:
            register_parameter(no_decay, learning_rate, 0.0)

        return parameters

    @torch.no_grad()
    def floor_quant(self, min_q: float, min_alpha: float) -> None:
        for m in self.resnet.modules():
            if not isinstance(m, CdlQuant):
                continue
            m.q.clamp_(min=min_q)
            m.alpha.clamp_(min=min_alpha)

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
