import abc
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from cdl.quant import mpmf_entropy, probabilistic_quant, soft_deterministic_quant


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

    @property
    def bits(self) -> int:
        return self.a.numel().bit_length() - 1

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_alpha_lr(self, eta: float) -> float:
        return eta / self.numel**0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.relaxed:
            return soft_deterministic_quant(
                input, self.q, self.alpha, self.a, self.topk
            )
        return probabilistic_quant(input, self.q, self.alpha, self.a, self.topk)

    def mpmf_entropy(self, theta: torch.Tensor) -> torch.Tensor:
        return mpmf_entropy(theta, self.q, self.alpha, self.a, self.topk)


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
        self.entropy_sum = 0.0
        self.forward_count = 0
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

        if self.training:
            self.entropy_sum = self.entropy_sum + self.mpmf_entropy(input)
            self.forward_count += 1

        return super().forward(input)

    def compute_entropy_and_reset(self) -> torch.Tensor:
        if self.forward_count == 0:
            raise RuntimeError("activation entropy read before any training forward")
        entropy = self.entropy_sum / self.forward_count
        self.entropy_sum = 0.0
        self.forward_count = 0
        return entropy


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

    def compute_entropy(self) -> torch.Tensor:
        """MPMF entropy H(W_hat_l) in bits of this layer's weight quantizer."""
        return self.weight_quant.mpmf_entropy(self.weight)


class QLinear(nn.Linear):
    def __init__(self, *args, w_bits: int, relaxed: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_quant = CdlQuantForWeight(w_bits, self.weight, relaxed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight_quant(self.weight), self.bias)

    def compute_entropy(self) -> torch.Tensor:
        """MPMF entropy H(W_hat_l) in bits of this layer's weight quantizer."""
        return self.weight_quant.mpmf_entropy(self.weight)


@torch.no_grad()
def init_act_q_pass(net: nn.Module, samples: Iterable[torch.Tensor]) -> None:
    modules = [
        m for m in net.modules() if isinstance(m, CdlQuantForActivation) and m.q.isnan()
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

    was_training = net.training
    net.eval()
    for m in modules:
        m.bypassing = True
    try:
        for x in samples:
            net(x)
    finally:
        for h in handles:
            h.remove()
        for m in modules:
            m.bypassing = False
        net.train(was_training)

    for m in modules:
        total, n, dim = stats[m]
        if n == 0:
            raise RuntimeError("some quantizers were never reached by samples")
        m.init_q_and_numel(total / n, dim)
