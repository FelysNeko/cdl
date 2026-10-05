import abc
import math

import torch
import torch.nn.functional as F
from torch import nn

from cdl.core import cdl_topk_forward, quant_nearest
from cdl.format.format import Format
from cdl.format.intx import IntX


def rho_from_kappa(kappa: float, kappa_max: float) -> float:
    p = min(max(kappa / kappa_max, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class CdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        fmt: Format,
        q: float,
        numel: int,
        relaxed: bool,
        topk: int,
        kappa_max: float,
        kappa_init: float,
    ):
        super().__init__()
        self.numel = numel
        self.relaxed = relaxed
        self.topk = min(topk, fmt.n)
        self.fmt = fmt
        self.q = nn.Parameter(torch.full((), q))
        self.kappa_max = float(kappa_max)
        rho = rho_from_kappa(kappa_init, kappa_max)
        self.rho = nn.Parameter(torch.full((), rho))
        self.entropy_sum = 0.0
        self.forward_count = 0

    @property
    def bits(self) -> int:
        return self.fmt.bits

    @property
    def kappa(self) -> torch.Tensor:
        return self.kappa_max * torch.sigmoid(self.rho)

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_kappa_lr(self, eta: float) -> float:
        return eta / self.numel**0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return quant_nearest(self.fmt, input, self.q)

        quantized, entropy = cdl_topk_forward(
            self.fmt,
            self.relaxed,
            self.numel,
            self.topk,
            input,
            self.q,
            self.kappa,
        )
        self.entropy_sum = self.entropy_sum + entropy
        self.forward_count += 1
        return quantized

    def compute_entropy_and_reset(self) -> torch.Tensor:
        if self.forward_count == 0:
            raise RuntimeError("entropy read before any training forward")
        entropy = self.entropy_sum / self.forward_count
        self.entropy_sum = 0.0
        self.forward_count = 0
        return entropy


class CdlQuantForWeight(CdlQuant):
    def __init__(
        self,
        bits: int,
        weight: torch.Tensor,
        relaxed: bool,
        kappa_max: float,
        kappa_init: float,
    ):
        fmt = IntX(True, bits)
        q = 2 * weight.detach().abs().mean().item() / 2 ** ((bits - 1) / 2)
        numel = weight.numel()
        super().__init__(fmt, q, numel, relaxed, fmt.n, kappa_max, kappa_init)

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.numel * 2 ** (self.bits - 1)) ** 0.5


class CdlQuantForActivation(CdlQuant):
    def __init__(
        self,
        bits: int,
        relaxed: bool,
        topk: int,
        kappa_max: float,
        kappa_init: float,
    ):
        fmt = IntX(False, bits)
        super().__init__(fmt, torch.nan, 0, relaxed, topk, kappa_max, kappa_init)
        self.bypassing = False
        self.initialized = False

    def scale_q_lr(self, eta: float) -> float:
        assert self.initialized
        return eta / (self.numel * 2**self.bits) ** 0.5

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
    def __init__(
        self,
        *args,
        w_bits: int,
        relaxed: bool,
        kappa_max: float,
        kappa_init: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_quant = CdlQuantForWeight(
            w_bits, self.weight, relaxed, kappa_max, kappa_init
        )

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
    def __init__(
        self,
        *args,
        w_bits: int,
        relaxed: bool,
        kappa_max: float,
        kappa_init: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_quant = CdlQuantForWeight(
            w_bits, self.weight, relaxed, kappa_max, kappa_init
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight_quant(self.weight), self.bias)
