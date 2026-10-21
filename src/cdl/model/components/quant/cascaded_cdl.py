import abc
import math
from collections.abc import Iterable

import torch
from torch import nn

from cdl.api.format import Format
from cdl.core.v2 import cascaded_cdl_topk_forward, cascaded_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


class CascadedCdlQuant(nn.Module, abc.ABC):
    def __init__(
        self,
        theta_format: Format,
        theta_kappa: float,
        theta_topk: int,
        theta_num_quantized_params: int,
        q: torch.Tensor,
        q_format: Format,
        q_kappa: float,
        q_topk: int,
        alpha: float,
    ):
        super().__init__()
        if theta_topk > theta_format.n:
            raise ValueError(
                f"top-{theta_topk} exceeds the {theta_format.n} quantization levels"
            )
        if q_topk > q_format.n:
            raise ValueError(
                f"top-{q_topk} exceeds the {q_format.n} quantization levels"
            )

        q_tensor = torch.as_tensor(q).detach().clone()
        if not torch.isfinite(q_tensor).all():
            raise ValueError("the initial q must contain only finite values")
        q_tensor = q_tensor.clamp_min(torch.finfo(torch.float32).tiny)

        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.theta_num_quantized_params = theta_num_quantized_params
        self.q_format = q_format
        self.q_topk = q_topk
        self.log_theta_kappa = nn.Parameter(torch.full((), math.log(theta_kappa)))
        self.log_q = nn.Parameter(q_tensor.log())
        self.log_q_kappa = nn.Parameter(torch.full((), math.log(q_kappa)))
        self.alpha = nn.Parameter(torch.full((), alpha, dtype=q_tensor.dtype))

    @property
    def theta_kappa(self) -> torch.Tensor:
        return self.log_theta_kappa.exp()

    @property
    def q(self) -> torch.Tensor:
        return self.log_q.exp()

    @property
    def q_kappa(self) -> torch.Tensor:
        return self.log_q_kappa.exp()

    @property
    @abc.abstractmethod
    def input_shape(self) -> tuple[int, ...]: ...

    @abc.abstractmethod
    def scale_q_lr(self, eta: float) -> float: ...

    def scale_theta_kappa_lr(self, eta: float) -> float:
        if self.theta_num_quantized_params == 0:
            raise RuntimeError("quantizer not initialized")
        return eta / self.theta_num_quantized_params**0.5

    def scale_q_kappa_lr(self, eta: float) -> float:
        return eta / self.q.numel() ** 0.5

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return cascaded_nearest_round(
                x,
                self.theta_format,
                self.q,
                self.q_format,
                self.alpha,
            ).to(x.dtype)

        theta_indices, q_indices, theta_proxy, _, _ = cascaded_cdl_topk_forward(
            x,
            self.theta_format,
            self.theta_kappa,
            self.theta_topk,
            self.q,
            self.q_format,
            self.q_kappa,
            self.q_topk,
            self.alpha,
            True,
        )
        return theta_indices, q_indices, self.alpha.detach(), theta_proxy


class CascadedCdlQuantForWeight(CascadedCdlQuant):
    def __init__(
        self,
        theta: torch.Tensor,
        theta_format: Format,
        theta_kappa: float,
        q_format: Format,
        q_kappa: float,
        alpha: float,
    ) -> None:
        if theta.ndim < 2:
            raise ValueError("theta must be in shape [out_feat, n, block_size]")

        q_tensor = (
            2
            * theta.detach().abs().mean(-1, keepdim=True)
            / 2 ** ((theta_format.bits - 1) / 2)
        ).clamp_min(torch.finfo(torch.float32).tiny)

        super().__init__(
            theta_format,
            theta_kappa,
            theta_format.n,
            theta.numel(),
            q_tensor,
            q_format,
            q_kappa,
            q_format.n,
            alpha,
        )
        self.__input_shape = tuple(theta.shape)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.__input_shape

    def scale_q_lr(self, eta: float) -> float:
        block_size = self.input_shape[-1]
        return eta / (block_size * 2 ** (self.theta_format.bits - 1)) ** 0.5


class CascadedCdlQuantForActivation(CascadedCdlQuant):
    def __init__(
        self,
        theta_format: Format,
        theta_kappa: float,
        theta_topk: int,
        q_format: Format,
        q_kappa: float,
        q_topk: int,
        alpha: float,
    ):
        super().__init__(
            theta_format,
            theta_kappa,
            theta_topk,
            0,
            torch.ones(()),
            q_format,
            q_kappa,
            q_topk,
            alpha,
        )
        self.initialized = False
        self.bypassing = False
        self.__input_shape: tuple[int, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, ...]:
        if self.__input_shape is None:
            raise RuntimeError("quantizer not initialized")
        return self.__input_shape

    def scale_q_lr(self, eta: float) -> float:
        if not self.initialized:
            raise RuntimeError("quantizer not initialized")
        block_size = self.input_shape[-1]
        return eta / (block_size * 2**self.theta_format.bits) ** 0.5

    @torch.no_grad()
    def init_q_and_numel(self, theta: torch.Tensor) -> None:
        if theta.ndim < 2:
            raise ValueError("theta must be in shape [out_feat, n, block_size]")

        reduce_dims = tuple(range(theta.ndim - 2)) + (theta.ndim - 1,)
        block_means = theta.detach().abs().mean(dim=reduce_dims)
        input_shape = tuple(theta.shape)

        n_blocks = block_means.numel()
        block_size = input_shape[-1]
        q = (
            2
            * block_means.to(device=self.log_q.device, dtype=self.log_q.dtype)
            / 2 ** ((self.theta_format.bits - 1) / 2)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        log_q = q.reshape(1, n_blocks, 1).log()

        if tuple(self.log_q.shape) != tuple(log_q.shape):
            self.log_q = nn.Parameter(log_q)
        else:
            self.log_q.copy_(log_q)

        self.theta_num_quantized_params = n_blocks * block_size
        self.__input_shape = tuple(input_shape)
        self.initialized = True

    def forward(
        self, input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bypassing:
            return input

        if not self.initialized:
            raise RuntimeError("quantizer not initialized")

        if input.shape[-2:] != self.input_shape[-2:]:
            raise ValueError(
                "activation grouping must match calibration: expected trailing "
                f"shape {tuple(self.input_shape[-2:])}, got {tuple(input.shape[-2:])}"
            )
        return super().forward(input)


class CdlQuantMixin(BatchNormBatchStatsMixin):
    @torch.no_grad()
    def init_act_q_pass(self, samples: Iterable[torch.Tensor]) -> None:
        modules = [
            m
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation) and not m.initialized
        ]
        if not modules:
            return

        # Per-block absolute sums and element counts accumulated over all
        # calibration batches.
        stats: dict[
            CascadedCdlQuantForActivation,
            tuple[torch.Tensor | None, int, tuple[int, ...] | None],
        ] = {m: (None, 0, None) for m in modules}

        def hook(mod: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            x = args[0]
            if x.ndim < 3:
                raise ValueError(
                    "activation inputs must be grouped as [..., n_blocks, block_size]"
                )

            block_sums, count, input_shape = stats[mod]
            n_blocks, block_size = x.shape[-2:]
            if input_shape is not None and input_shape[-2:] != (n_blocks, block_size):
                raise ValueError(
                    "all calibration samples for an activation quantizer must "
                    "have the same grouped block layout"
                )

            flat = x.reshape(-1, n_blocks, block_size)
            current_sums = flat.detach().abs().sum(dim=(0, 2))
            if block_sums is None:
                block_sums = current_sums
            else:
                block_sums = block_sums + current_sums
            stats[mod] = (
                block_sums,
                count + flat.shape[0] * block_size,
                tuple(x.shape) if input_shape is None else input_shape,
            )

        handles = [m.register_forward_pre_hook(hook) for m in modules]

        was_training = self.training
        parameter = next(self.parameters(), None)
        if parameter is None:
            raise RuntimeError("the model has no parameters")
        device = parameter.device
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
            block_sums, count, input_shape = stats[m]
            if block_sums is None or count == 0 or input_shape is None:
                raise RuntimeError("some quantizers were never reached by samples")
            block_means = block_sums / count
            block_size = input_shape[-1]
            # A compact theta whose per-block means equal the accumulated
            # calibration means, so initialization goes through the same path.
            theta_for_init = block_means.reshape(1, -1, 1).expand(1, -1, block_size)
            m.init_q_and_numel(theta_for_init)

    @torch.no_grad()
    def get_param_groups(
        self, learning_rate: float, weight_decay: float, freeze_kappa: bool
    ) -> list[dict]:
        customized_param_set: set[nn.Parameter] = set()
        buckets: dict[tuple[float, float], list[nn.Parameter]] = {}

        for m in self.modules():
            if not isinstance(m, CascadedCdlQuant):
                continue

            if m.theta_num_quantized_params == 0:
                raise RuntimeError("quantizer not initialized")

            customized_param_set.add(m.log_q)
            q_lr = m.scale_q_lr(learning_rate)
            buckets.setdefault((q_lr, 0.0), []).append(m.log_q)

            customized_param_set.add(m.log_theta_kappa)
            customized_param_set.add(m.log_q_kappa)
            if freeze_kappa:
                m.log_theta_kappa.requires_grad_(False)
                m.log_q_kappa.requires_grad_(False)
            else:
                theta_kappa_lr = m.scale_theta_kappa_lr(learning_rate)
                q_kappa_lr = m.scale_q_kappa_lr(learning_rate)
                buckets.setdefault((theta_kappa_lr, 0.0), []).append(m.log_theta_kappa)
                buckets.setdefault((q_kappa_lr, 0.0), []).append(m.log_q_kappa)

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
            if not isinstance(m, CascadedCdlQuant):
                continue
            m.log_q.clamp_(min=math.log(min_q))
            m.log_theta_kappa.clamp_(min=-20.0, max=20.0)
            m.log_q_kappa.clamp_(min=-20.0, max=20.0)

    @torch.no_grad()
    def quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, CascadedCdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.theta_kappa"] = m.theta_kappa.item()
            out[f"{name}.q"] = m.q.detach().float().mean().item()
            out[f"{name}.q_kappa"] = m.q_kappa.item()
            out[f"{name}.alpha"] = m.alpha.item()
        return out

    def get_quant_counts(self) -> tuple[int, int]:
        n_weight = sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForWeight)
        )
        n_activation = sum(
            m.theta_num_quantized_params
            for m in self.modules()
            if isinstance(m, CascadedCdlQuantForActivation)
        )
        return n_weight, n_activation
