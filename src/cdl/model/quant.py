import abc
import math

import torch
from torch import nn

from cdl.core.companded import cdl_topk_forward, quant_nearest
from cdl.core.format.format import Format
from cdl.core.format.intx import IntX


class CdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        fmt: Format,
        q: float,
        num_quantized_params: int,
        relaxed: bool,
        topk: int,
        kappa: float,
    ):
        super().__init__()
        if topk > fmt.n:
            raise ValueError(f"top-{topk} exceeds the {fmt.n} quantization levels")
        self.num_quantized_params = num_quantized_params
        self.relaxed = relaxed
        self.topk = topk
        self.fmt = fmt
        self.q = nn.Parameter(torch.full((), q))
        self.log_kappa = nn.Parameter(torch.full((), math.log(kappa)))
        self.__entropy: torch.Tensor | None = None

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa)

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_kappa_lr(self, eta: float) -> float:
        return eta / self.num_quantized_params**0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return quant_nearest(self.fmt, input, self.q)

        quantized, entropy = cdl_topk_forward(
            self.fmt,
            self.relaxed,
            self.num_quantized_params,
            self.topk,
            input,
            self.q,
            self.kappa,
        )

        if self.__entropy is not None:
            raise RuntimeError("entropy is overwritten by a second forward pass")
        self.__entropy = entropy

        return quantized

    def drain_entropy(self) -> torch.Tensor:
        if self.__entropy is None:
            raise RuntimeError("no entropy term to drain")
        entropy = self.__entropy
        self.__entropy = None
        return entropy


class CdlQuantForWeight(CdlQuant):
    def __init__(
        self,
        bits: int,
        weight: torch.Tensor,
        relaxed: bool,
        kappa: float,
    ):
        fmt = IntX(True, bits)
        q = 2 * weight.detach().abs().mean().item() / 2 ** ((bits - 1) / 2)
        num_quantized_params = weight.numel()
        super().__init__(fmt, q, num_quantized_params, relaxed, fmt.n, kappa)

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.num_quantized_params * 2 ** (self.fmt.bits - 1)) ** 0.5


class CdlQuantForActivation(CdlQuant):
    def __init__(
        self,
        bits: int,
        relaxed: bool,
        topk: int,
        kappa: float,
    ):
        fmt = IntX(False, bits)
        super().__init__(fmt, torch.nan, 0, relaxed, topk, kappa)
        self.initialized = False
        self.bypassing = False

    def scale_q_lr(self, eta: float) -> float:
        assert self.initialized
        return eta / (self.num_quantized_params * 2**self.fmt.bits) ** 0.5

    @torch.no_grad()
    def init_q_and_numel(self, act_abs_mean: float, num_quantized_params: int) -> None:
        self.q.fill_(2 * act_abs_mean / 2 ** ((self.fmt.bits - 1) / 2))
        self.num_quantized_params = num_quantized_params
        self.initialized = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bypassing:
            return input

        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        return super().forward(input)
