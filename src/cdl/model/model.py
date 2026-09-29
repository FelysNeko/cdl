import abc

import torch
import torch.nn.functional as F
from torch import nn

from cdl.formulae import (
    cpmf,
    mpmf_entropy,
    q_d,
    quant_from_cpmf,
)


class CdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        a: torch.Tensor,
        q: float,
        numel: int,
        relaxed: bool,
        topk: int,
    ):
        super().__init__()
        self.numel = numel
        self.relaxed = relaxed
        self.topk = min(topk, a.numel())
        self.register_buffer("a", a)
        self.q = nn.Parameter(torch.full((), q))
        self.alpha = nn.Parameter(torch.full((), 500.0))
        self.entropy_sum = 0.0
        self.forward_count = 0

    @property
    def bits(self) -> int:
        return self.a.numel().bit_length() - 1

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_alpha_lr(self, eta: float) -> float:
        return eta / self.numel**0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pmf, vals, idx = cpmf(input, self.alpha, self.q, self.a, self.topk)

        if self.training:
            entropy = mpmf_entropy(pmf, idx, self.a.numel(), self.numel)
            self.entropy_sum = self.entropy_sum + entropy
            self.forward_count += 1

        if self.relaxed:
            return q_d(pmf, vals)
        return quant_from_cpmf(pmf, vals, self.topk)

    def compute_entropy_and_reset(self) -> torch.Tensor:
        """
        Must be called exactly once per ``loss.backward()`` (before it).
        Gradient accumulation does not change this cadence: drain follows
        backward, not optimizer.step.
        """
        if self.forward_count == 0:
            raise RuntimeError("entropy read before any training forward")
        entropy = self.entropy_sum / self.forward_count
        self.entropy_sum = 0.0
        self.forward_count = 0
        return entropy


class CdlQuantForWeight(CdlQuant):
    def __init__(self, bits: int, weight: torch.Tensor, relaxed: bool):
        start = -(2 ** (bits - 1))
        a = torch.arange(start, start + 2**bits, dtype=torch.float32)
        q = 2 * weight.detach().abs().mean().item() / 2 ** ((bits - 1) / 2)
        numel = weight.numel()
        super().__init__(a, q, numel, relaxed, a.numel())

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.numel * 2 ** (self.bits - 1)) ** 0.5


class CdlQuantForActivation(CdlQuant):
    def __init__(self, bits: int, relaxed: bool, topk: int):
        start = 0
        a = torch.arange(start, start + 2**bits, dtype=torch.float32)
        super().__init__(a, torch.nan, 0, relaxed, topk)
        self.bypassing = False
        self.initialized = False

    def scale_q_lr(self, eta: float) -> float:
        assert self.initialized
        return eta / (self.numel * 2**self.bits) ** 0.5

    def scale_alpha_lr(self, eta: float) -> float:
        assert self.initialized
        return super().scale_alpha_lr(eta)

    @torch.no_grad()
    def init_q_and_numel(self, act_abs_mean: float, numel: int) -> None:
        self.q.fill_(2 * act_abs_mean / 2 ** ((self.bits - 1) / 2))
        self.numel = numel
        self.initialized = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bypassing:
            return input

        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        return super().forward(input)


class QConv2d(nn.Conv2d):
    def __init__(self, *args, w_bits: int, relaxed: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quant = CdlQuantForWeight(w_bits, self.weight, relaxed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QLinear(nn.Linear):
    def __init__(self, *args, w_bits: int, relaxed: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quant = CdlQuantForWeight(w_bits, self.weight, relaxed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight_quant(self.weight), self.bias)
