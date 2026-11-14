import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cdl.api.format import Format
from cdl.api.quant import QTensor, Quant
from cdl.core.ecdl import extended_cdl_topk_forward, extended_nearest_round
from cdl.model.mixin.batchnorm_batch_stats import BatchNormBatchStatsMixin


def group_tensor_into_blocks(
    theta: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    k = theta.shape[-1]

    extra = k % block_size
    pad_size = 0
    if extra:
        pad_size = block_size - extra
        theta = F.pad(theta, (0, pad_size))

    num_blocks = (k + pad_size) // block_size
    cdl_style_theta = theta.reshape(*theta.shape[:-1], num_blocks, block_size)
    return cdl_style_theta, pad_size


@dataclass
class ExtendedCdlTensor(QTensor):
    theta_format: Format
    theta_indices: torch.Tensor
    q_format: Format
    q_indices: torch.Tensor
    alpha: torch.Tensor
    grad_proxy: torch.Tensor | None
    k_dim: int

    def dequantize(self) -> torch.Tensor:
        theta = self.theta_format.level[self.theta_indices.long()]
        q = self.q_format.level[self.q_indices.long()]
        alpha = self.alpha.to(theta.dtype)
        x = (theta * q * alpha).flatten(-2)[..., : self.k_dim]
        return x

    def proxy(self) -> torch.Tensor | None:
        if self.grad_proxy is None:
            return None
        return self.grad_proxy.flatten(-2)[..., : self.k_dim]


@dataclass
class ExtendedCdlBypassTensor(QTensor):
    theta: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.theta

    def proxy(self) -> torch.Tensor | None:
        return None


class ExtendedCdlQuant(Quant):
    tensor: type[ExtendedCdlTensor] = ExtendedCdlTensor

    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        kappa_backward: float,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.k_dim = input_shape[-1]
        self.theta_format = theta_format
        self.theta_topk = theta_topk
        self.q_format = q_format
        self.block_size = block_size
        self.num_quantized_params = math.prod(input_shape)

        self.log_kappa = nn.Parameter(torch.full((), math.log(kappa_forward)))
        self.log_beta = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "kappa_backward",
            torch.tensor(kappa_backward, dtype=torch.float32),
            persistent=False,
        )

        self.bypassing = False

    @property
    def kappa(self) -> torch.Tensor:
        return self.log_kappa.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def training_forward(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta_indices, q_indices, grad_proxy, global_scale, _ = (
            extended_cdl_topk_forward(
                theta,
                self.theta_format,
                self.theta_topk,
                self.q_format,
                self.kappa,
                self.kappa_backward,
                self.beta,
                True,
            )
        )
        return theta_indices, q_indices, grad_proxy, global_scale

    def forward(self, x: torch.Tensor) -> QTensor:
        if self.bypassing:
            return ExtendedCdlBypassTensor(x)

        if x.ndim != len(self.input_shape):
            raise ValueError(
                f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
            )
        for declared, actual in zip(self.input_shape, x.shape):
            if declared != 1 and declared != actual:
                raise ValueError(
                    f"expected input shaped like {self.input_shape}, got {tuple(x.shape)}"
                )

        theta, _ = group_tensor_into_blocks(x, self.block_size)

        if not self.training:
            theta_indices, q_indices, global_scale = extended_nearest_round(
                theta,
                self.theta_format,
                self.q_format,
                self.beta,
            )
            grad_proxy = None
        else:
            theta_indices, q_indices, grad_proxy, global_scale = self.training_forward(theta)

        return self.tensor(
            self.theta_format,
            theta_indices,
            self.q_format,
            q_indices,
            global_scale.detach(),
            grad_proxy,
            self.k_dim,
        )


class ExtendedCdlQuantForActivation(ExtendedCdlQuant):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        theta_format: Format,
        theta_topk: int,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
        kappa_backward: float,
    ):
        super().__init__(
            input_shape,
            theta_format,
            theta_topk,
            q_format,
            block_size,
            kappa_forward,
            kappa_backward,
        )


class ExtendedCdlQuantForWeight(ExtendedCdlQuant):
    def __init__(
        self,
        weight: torch.Tensor,
        theta_format: Format,
        q_format: Format,
        block_size: int,
        kappa_forward: float,
    ):
        super().__init__(
            tuple(weight.shape),
            theta_format,
            theta_format.n,
            q_format,
            block_size,
            kappa_forward,
            kappa_forward,
        )


class ExtendedCdlQuantMixin(BatchNormBatchStatsMixin):
    def extended_cdl_get_weight_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, ExtendedCdlQuantForWeight)
        )

    def extended_cdl_get_activation_quant_counts(self) -> int:
        return sum(
            m.num_quantized_params
            for m in self.modules()
            if isinstance(m, ExtendedCdlQuantForActivation)
        )

    @torch.no_grad()
    def extended_cdl_quant_stats(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, m in self.named_modules():
            if not isinstance(m, ExtendedCdlQuant):
                continue
            name = name.removeprefix("_orig_mod.")
            out[f"{name}.kappa"] = m.kappa.item()
            out[f"{name}.beta"] = m.beta.item()
        return out

    @torch.no_grad()
    def extended_cdl_get_param_groups(
        self,
        learning_rate: float,
        train_beta: bool,
        train_kappa: bool,
    ) -> tuple[list[dict], set[nn.Parameter]]:
        customized_param_set: set[nn.Parameter] = set()
        params: list[nn.Parameter] = []

        for m in self.modules():
            if not isinstance(m, ExtendedCdlQuant):
                continue

            customized_param_set.update(m.parameters())

            if train_beta:
                params.append(m.log_beta)
            else:
                m.log_beta.requires_grad_(False)

            if train_kappa:
                params.append(m.log_kappa)
            else:
                m.log_kappa.requires_grad_(False)

        groups = (
            [{"params": params, "lr": learning_rate, "weight_decay": 0.0}]
            if params
            else []
        )
        return groups, customized_param_set
