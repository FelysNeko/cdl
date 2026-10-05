import abc

import torch
import torch.nn.functional as F
from torch import nn

from cdl.core import (
    cdl_topk_infer_sample,
    cdl_topk_train_forward,
)
from cdl.format.format import Format
from cdl.format.intx import IntX


class CdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        fmt: Format,
        q: float,
        alpha: float,
        numel: int,
        relaxed: bool,
        topk: int,
    ):
        super().__init__()
        self.numel = numel
        self.relaxed = relaxed
        self.topk = min(topk, fmt.n)
        self.fmt = fmt
        self.q = nn.Parameter(torch.full((), q))
        self.alpha = nn.Parameter(torch.full((), alpha))
        self.entropy_sum = 0.0
        self.forward_count = 0

    @property
    def bits(self) -> int:
        return self.fmt.bits

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_alpha_lr(self, eta: float) -> float:
        return eta / self.numel**0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return cdl_topk_infer_sample(self.fmt, self.topk, input, self.q, self.alpha)

        quantized, entropy = cdl_topk_train_forward(
            self.fmt,
            self.relaxed,
            self.numel,
            self.topk,
            input,
            self.q,
            self.alpha,
        )
        self.entropy_sum = self.entropy_sum + entropy
        self.forward_count += 1
        return quantized

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
        fmt = IntX(True, bits)
        q = 2 * weight.detach().abs().mean().item() / 2 ** ((bits - 1) / 2)
        numel = weight.numel()
        super().__init__(fmt, q, 500.0, numel, relaxed, fmt.n)

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.numel * 2 ** (self.bits - 1)) ** 0.5


class CdlQuantForActivation(CdlQuant):
    def __init__(self, bits: int, relaxed: bool, topk: int):
        fmt = IntX(False, bits)
        super().__init__(fmt, torch.nan, 50.0, 0, relaxed, topk)
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
