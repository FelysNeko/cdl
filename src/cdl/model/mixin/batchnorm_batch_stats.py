import contextlib
from collections.abc import Iterable

import torch
from torch import nn


class BatchNormBatchStatsMixin(nn.Module):
    @contextlib.contextmanager
    def batchnorm_batch_stats(self):
        states: list[
            tuple[nn.BatchNorm2d, bool, float | None, torch.Tensor | None]
        ] = []

        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                continue

            num_batches_tracked = None
            if m.num_batches_tracked is not None:
                num_batches_tracked = m.num_batches_tracked.clone()

            states.append(
                (
                    m,
                    m.training,
                    m.momentum,
                    num_batches_tracked,
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
    def recalibrate_bn(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        num_batches: int,
        device: torch.device,
    ) -> None:
        if num_batches <= 0:
            return

        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self._recalibrate_bn(loader, num_batches, device)

    @torch.no_grad()
    def _recalibrate_bn(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        num_batches: int,
        device: torch.device,
    ) -> None:
        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        if not bns:
            return

        was_training = self.training
        moments = [m.momentum for m in bns]
        # Keep stochastic quantizers (and every other non-BN module) in eval
        # mode while batch norm statistics are re-estimated.
        self.eval()
        for m in bns:
            m.train()
        try:
            for m in bns:
                m.reset_running_stats()
                m.momentum = None
            for i, (x, _) in enumerate(loader):
                if i >= num_batches:
                    break
                self(x.to(device))
        finally:
            for m, momentum in zip(bns, moments):
                m.momentum = momentum
            self.train(was_training)
