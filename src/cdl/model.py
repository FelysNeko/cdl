from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from cdl.quantizer import probabilistic_quant, soft_deterministic_quant


class CdlQuant(nn.Module):
    def __init__(
        self,
        bits: int,
        symmetric: bool,
        relaxed: bool,
    ):
        super().__init__()
        start = -(2 ** (bits - 1)) if symmetric else 0
        a = torch.arange(start, start + 2**bits, dtype=torch.float32)

        self.bits = bits
        self.relaxed = relaxed
        self.register_buffer("a", a)
        self.q = nn.Parameter(torch.full((), torch.nan))
        self.alpha = nn.Parameter(torch.full((), 500.0))
        self.bypass = False

    @torch.no_grad()
    def init_q(self, abs_mean: float) -> None:
        self.q.fill_(2 * abs_mean / 2 ** ((self.bits - 1) / 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bypass:
            return input

        if self.q.isnan():
            raise RuntimeError("Quantizer not initialized, run `init_q_pass`.")

        if self.relaxed:
            return soft_deterministic_quant(input, self.q, self.alpha, self.a)
        return probabilistic_quant(input, self.q, self.alpha, self.a)


class QConv2d(nn.Conv2d):
    def __init__(self, *args, bits: int, relaxed: bool, quantize_act: bool, **kwargs):
        super().__init__(*args, **kwargs)
        weight_abs_mean = self.weight.detach().abs().mean().item()
        self.weight_quant = CdlQuant(bits, True, relaxed)
        self.weight_quant.init_q(weight_abs_mean)
        self.activation_quant = CdlQuant(bits, False, relaxed) if quantize_act else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.activation_quant is not None:
            input = self.activation_quant(input)
        weight = self.weight_quant(self.weight)
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


@torch.no_grad
def init_q_pass(nets: nn.Module, samples: Iterable[torch.Tensor]) -> None:
    modules = [m for m in nets.modules() if isinstance(m, CdlQuant) and m.q.isnan()]
    if not modules:
        return

    mod_to_data_list = {m: [0.0, 0] for m in modules}

    def hook(mod, args):
        x: torch.Tensor = args[0]
        mod_to_data_list[mod][0] += x.abs().sum().item()
        mod_to_data_list[mod][1] += x.numel()

    handles = [m.register_forward_pre_hook(hook) for m in modules]

    was_training = nets.training
    nets.eval()
    for m in modules:
        m.bypass = True
    try:
        for x in samples:
            nets(x)
    finally:
        for h in handles:
            h.remove()
        for m in modules:
            m.bypass = False
        nets.train(was_training)

    for m in modules:
        total, n = mod_to_data_list[m]
        m.init_q(total / n)
