import abc
import math
from collections.abc import Iterable

import torch
from torch import nn

from cdl.core.format.format import Format
from cdl.core.v1 import cdl_topk_forward, quant_nearest
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


class CdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        f: Format,
        q: float,
        relaxed: bool,
        topk: int,
        kappa: float,
        companded: bool,
        num_quantized_params: int,
    ):
        super().__init__()
        if topk > f.n:
            raise ValueError(f"top-{topk} exceeds the {f.n} quantization levels")
        self.f = f
        q_tensor = torch.as_tensor(q).clamp_min(torch.finfo(torch.float32).tiny)
        self.log_q = nn.Parameter(q_tensor.log())
        self.relaxed = relaxed
        self.topk = topk
        self.companded = companded
        self.num_quantized_params = num_quantized_params
        self.log_kappa = nn.Parameter(torch.full((), math.log(kappa)))
        self.__entropy_enabled = False
        self.__entropy: torch.Tensor | None = None

    @property
    def q(self) -> torch.Tensor:
        return torch.exp(self.log_q)

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa)

    @property
    def entropy_enabled(self) -> bool:
        return self.__entropy_enabled

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_kappa_lr(self, eta: float) -> float:
        return eta / self.num_quantized_params**0.5

    def enable_entropy_computation(self) -> None:
        self.__entropy_enabled = True

    def disable_entropy_computation(self) -> None:
        self.__entropy_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return quant_nearest(x, self.f, self.q).to(x.dtype)

        quantized, entropy = cdl_topk_forward(
            x,
            self.f,
            self.q,
            self.kappa,
            self.topk,
            self.companded,
            self.relaxed,
            self.num_quantized_params if self.__entropy_enabled else None,
        )

        if entropy is not None:
            if self.__entropy is not None:
                raise RuntimeError("entropy is overwritten by a second forward pass")
            self.__entropy = entropy

        return quantized.to(x.dtype)

    def drain_entropy(self) -> torch.Tensor:
        if self.__entropy is None:
            raise RuntimeError("no entropy term to drain")
        entropy = self.__entropy
        self.__entropy = None
        return entropy


class CdlQuantForWeight(CdlQuant):
    def __init__(
        self,
        f: Format,
        weight: torch.Tensor,
        relaxed: bool,
        kappa: float,
    ):
        q = 2 * weight.detach().abs().mean().item() / 2 ** ((f.bits - 1) / 2)
        num_quantized_params = weight.numel()
        super().__init__(
            f,
            q,
            relaxed,
            f.n,
            kappa,
            True,
            num_quantized_params,
        )

    def scale_q_lr(self, eta: float) -> float:
        return eta / (self.num_quantized_params * 2 ** (self.f.bits - 1)) ** 0.5


class CdlQuantForActivation(CdlQuant):
    def __init__(
        self,
        f: Format,
        relaxed: bool,
        topk: int,
        kappa: float,
    ):
        super().__init__(f, torch.nan, relaxed, topk, kappa, True, 0)
        self.initialized = False
        self.bypassing = False

    def scale_q_lr(self, eta: float) -> float:
        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        return eta / (self.num_quantized_params * 2**self.f.bits) ** 0.5

    @torch.no_grad()
    def init_q_and_numel(self, act_abs_mean: float, num_quantized_params: int) -> None:
        q = 2 * act_abs_mean / 2 ** ((self.f.bits - 1) / 2)
        q = max(q, torch.finfo(torch.float32).tiny)
        self.log_q.fill_(math.log(q))
        self.num_quantized_params = num_quantized_params
        self.initialized = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bypassing:
            return input

        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        return super().forward(input)


class CdlQuantMixin(BatchNormBatchStatsMixin):
    def enable_entropy_computation(self) -> None:
        for m in self.modules():
            if isinstance(m, CdlQuant):
                m.enable_entropy_computation()

    def disable_entropy_computation(self) -> None:
        for m in self.modules():
            if isinstance(m, CdlQuant):
                m.disable_entropy_computation()

    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m
            for m in self.modules()
            if isinstance(m, CdlQuantForActivation) and not m.initialized
        ]
        if not modules:
            return

        stats = {m: [0.0, 0, 0] for m in modules}

        def hook(mod: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            s = stats[mod]
            s[0] += x.abs().sum().item()
            s[1] += x.numel()
            s[2] = x.shape[1:].numel()

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.training
        device = next(self.parameters()).device
        self.eval()
        for m in modules:
            m.bypassing = True
        try:
            with self.batchnorm_batch_stats():
                for x in samples:
                    self(x.to(device))
        finally:
            for h in handles:
                h.remove()
            for m in modules:
                m.bypassing = False
            self.train(was_training)

        for m in modules:
            total, n, dim = stats[m]
            if n == 0:
                raise RuntimeError("some quantizers were never reached by samples")
            m.init_q_and_numel(total / n, dim)

    def drain_entropies(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight_entropies: list[torch.Tensor] = []
        activation_entropies: list[torch.Tensor] = []

        for m in self.modules():
            if not isinstance(m, CdlQuant) or not m.entropy_enabled:
                continue
            if isinstance(m, CdlQuantForWeight):
                weight_entropies.append(m.drain_entropy())
            elif isinstance(m, CdlQuantForActivation):
                activation_entropies.append(m.drain_entropy())

        parameter = next(self.parameters(), None)
        zero = (
            parameter.new_zeros(())
            if parameter is not None
            else torch.zeros((), dtype=torch.float32)
        )
        weight_entropy_loss = (
            torch.stack(weight_entropies).sum() if weight_entropies else zero
        )
        activation_entropy_loss = (
            torch.stack(activation_entropies).sum() if activation_entropies else zero
        )
        return weight_entropy_loss, activation_entropy_loss

    def get_param_groups(
        self, learning_rate: float, weight_decay: float, freeze_kappa: bool
    ) -> list[dict]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CdlQuant):
                continue

            if m.num_quantized_params == 0:
                raise RuntimeError("quantizer not initialized")

            customized_param_set.add(m.log_q)
            q_lr = m.scale_q_lr(learning_rate)
            buckets.setdefault((q_lr, 0.0), []).append(m.log_q)

            customized_param_set.add(m.log_kappa)
            if freeze_kappa:
                m.log_kappa.requires_grad_(False)
            else:
                kappa_lr = m.scale_kappa_lr(learning_rate)
                buckets.setdefault((kappa_lr, 0.0), []).append(m.log_kappa)

        decay = []
        no_decay = []
        for param in self.parameters():
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
        for m in self.modules():
            if not isinstance(m, CdlQuant):
                continue
            m.log_q.clamp_(min=math.log(min_q))
            m.log_kappa.clamp_(min=-20.0, max=20.0)

    @torch.no_grad()
    def quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, CdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.kappa"] = m.kappa.item()
            out[f"{name}.q"] = m.q.item()
        return out

    def get_quant_counts(self) -> tuple[int, int]:
        n_weight = sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, CdlQuantForWeight)
        )
        n_activation = sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, CdlQuantForActivation)
        )
        return n_weight, n_activation
