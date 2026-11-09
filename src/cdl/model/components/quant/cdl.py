import abc
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.cdl import cdl_topk_forward, nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


@dataclass
class CdlTensor(QTensor):
    theta: torch.Tensor
    grad_proxy: torch.Tensor | None

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return self.grad_proxy


class CdlQuant(Quant):
    def __init__(
        self,
        f: Format,
        topk: int,
        kappa: float,
        relaxed: bool,
    ):
        super().__init__()
        self.f = f
        self.topk = topk
        self.relaxed = relaxed
        self.companded = True
        self.num_quantized_params = 0

        self.log_q = nn.Parameter(torch.full((), math.log(1.0)))
        self.log_kappa = nn.Parameter(torch.full((), math.log(kappa)))

        self.initialized = False
        self.bypassing = False

    @property
    def q(self) -> torch.Tensor:
        return self.log_q.exp()

    @property
    def kappa(self) -> torch.Tensor:
        return self.log_kappa.exp()

    @abc.abstractmethod
    def get_q_lr(self, eta: float) -> float: ...

    def get_kappa_lr(self, eta: float) -> float:
        assert self.num_quantized_params > 0
        return eta / self.num_quantized_params**0.5

    @torch.no_grad()
    def init_q_from_mean(self, act_abs_mean: torch.Tensor, numel: int) -> None:
        q = 2 * act_abs_mean / 2 ** ((self.f.bits - 1) / 2)
        self.log_q.fill_(q.clamp_min(torch.finfo(torch.float32).tiny).log().item())
        self.num_quantized_params = numel
        self.initialized = True

    def forward(self, x: torch.Tensor) -> QTensor:
        if self.bypassing:
            return CdlTensor(x, None)

        assert self.initialized

        if not self.training:
            theta = nearest_round(x, self.f, self.q)
            return CdlTensor(theta, None)

        quantized, _ = cdl_topk_forward(
            x,
            self.f,
            self.q,
            self.kappa,
            self.topk,
            self.companded,
            self.relaxed,
        )
        return CdlTensor(quantized.detach(), quantized)


class CdlQuantForWeight(CdlQuant):
    def __init__(
        self,
        weight: torch.Tensor,
        f: Format,
        kappa: float,
        relaxed: bool,
    ):
        super().__init__(f, f.n, kappa, relaxed)
        self.init_q_from_mean(weight.detach().abs().mean(), weight.numel())

    def get_q_lr(self, eta: float) -> float:
        n = self.num_quantized_params
        bits = self.f.bits
        return eta / math.sqrt(n * (2 ** (bits - 1)))


class CdlQuantForActivation(CdlQuant):
    def __init__(
        self,
        f: Format,
        topk: int,
        kappa: float,
        relaxed: bool,
    ):
        super().__init__(f, topk, kappa, relaxed)

    def get_q_lr(self, eta: float) -> float:
        n = self.num_quantized_params
        bits = self.f.bits
        return eta / math.sqrt(n * (2**bits))


class CdlQuantMixin(BatchNormBatchStatsMixin):
    def get_weight_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, CdlQuantForWeight)
        )

    def get_activation_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, CdlQuantForActivation)
        )

    @torch.no_grad()
    def quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, CdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.q"] = m.q.item()
            out[f"{name}.kappa"] = m.kappa.item()
        return out

    @torch.no_grad()
    def get_param_groups(
        self,
        learning_rate: float,
        train_kappa: bool,
    ) -> tuple[list[dict], set[nn.Parameter]]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[float, list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CdlQuant):
                continue
            assert m.initialized

            customized_param_set.update(m.parameters())

            q_lr = m.get_q_lr(learning_rate)
            buckets.setdefault(q_lr, []).append(m.log_q)

            if train_kappa:
                kappa_lr = m.get_kappa_lr(learning_rate)
                buckets.setdefault(kappa_lr, []).append(m.log_kappa)
            else:
                m.log_kappa.requires_grad_(False)

        groups = [
            {"params": params, "lr": lr, "weight_decay": 0.0}
            for lr, params in buckets.items()
        ]
        return groups, customized_param_set

    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [m for m in self.modules() if isinstance(m, CdlQuantForActivation)]
        if not modules:
            return

        stats: dict[CdlQuantForActivation, tuple[torch.Tensor, int, int]] = {}

        def hook(m: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            assert x.ndim == 2
            block_sum = x.abs().sum()
            previous = stats.get(m)
            if previous is None:
                stats[m] = (block_sum, x.numel(), x.shape[1])
                return
            total, n, width = previous
            assert width == x.shape[1]
            stats[m] = (total + block_sum, n + x.numel(), width)

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.training
        self.eval()
        for m in modules:
            m.bypassing = True
        try:
            with self.batchnorm_batch_stats():
                for x in samples:
                    self(x)
        finally:
            for h in handles:
                h.remove()
            for m in modules:
                m.bypassing = False
            self.train(was_training)

        for m in modules:
            assert m in stats
            total, n, width = stats[m]
            m.init_q_from_mean(total / n, width)
