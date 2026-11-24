"""SmolLM2 quantized QAT/LoRA experiment runner.

Supports STE, CCDL, and ECDL quantizers across NVFP4, MXFP4, and MXFP8 formats,
LoRA or full-parameter training with optional FP32 master weights, resumable
checkpoints, and TensorBoard logging.
"""

import argparse
import json
import logging
import math
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Literal

import torch
import tqdm
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention

from cdl.api.quant import Quant
from cdl.checkpoint import load_checkpoint, save_checkpoint
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import E4M3, E8M0, UE4M3ExcludeZero
from cdl.dataset.token_cache import CacheSpec, TokenCache
from cdl.model.components.linear import QLinear, QLinearMixin
from cdl.model.components.quant.ccdl.ccdl import CascadedCdlQuantMixin
from cdl.model.components.quant.ccdl.mxfp4 import (
    Mxfp4CascadedCdlQuantForActivation,
    Mxfp4CascadedCdlQuantForWeight,
)
from cdl.model.components.quant.ccdl.mxfp8 import (
    Mxfp8CascadedCdlQuantForActivation,
    Mxfp8CascadedCdlQuantForWeight,
)
from cdl.model.components.quant.ccdl.nvfp4 import (
    Nvfp4CascadedCdlQuantForActivation,
    Nvfp4CascadedCdlQuantForWeight,
)
from cdl.model.components.quant.ecdl.ecdl import ExtendedCdlQuantMixin
from cdl.model.components.quant.ecdl.mxfp4 import (
    Mxfp4ExtendedCdlQuantForActivation,
    Mxfp4ExtendedCdlQuantForWeight,
)
from cdl.model.components.quant.ecdl.mxfp8 import (
    Mxfp8ExtendedCdlQuantForActivation,
    Mxfp8ExtendedCdlQuantForWeight,
)
from cdl.model.components.quant.ecdl.nvfp4 import (
    Nvfp4ExtendedCdlQuantForActivation,
    Nvfp4ExtendedCdlQuantForWeight,
)
from cdl.model.components.quant.identity import IdentityQuant
from cdl.model.components.quant.ste import SteQuant

logger = logging.getLogger(__name__)

MODEL_DTYPE = torch.bfloat16
KERNEL_VARIANTS = ("torch", "triton")
BACKEND_VARIANTS = ("native", "nvfp4", "mxfp4", "mxfp8")
LR_DECAY_VARIANTS = ("constant", "cosine")
KV_QUANT_ATTN_NAME = "cdl_kv_quant_sdpa"


@dataclass(frozen=True)
class _QuantVariantSpec:
    family: Literal["ccdl", "ecdl"]
    block_size: int
    theta_format: type
    q_format: type | None
    act_cls: type
    weight_cls: type


_QUANT_VARIANTS: dict[str, _QuantVariantSpec] = {
    "nvfp4_ccdl": _QuantVariantSpec(
        "ccdl",
        16,
        E2M1,
        UE4M3ExcludeZero,
        Nvfp4CascadedCdlQuantForActivation,
        Nvfp4CascadedCdlQuantForWeight,
    ),
    "mxfp4_ccdl": _QuantVariantSpec(
        "ccdl",
        32,
        E2M1,
        E8M0,
        Mxfp4CascadedCdlQuantForActivation,
        Mxfp4CascadedCdlQuantForWeight,
    ),
    "mxfp8_ccdl": _QuantVariantSpec(
        "ccdl",
        32,
        E4M3,
        E8M0,
        Mxfp8CascadedCdlQuantForActivation,
        Mxfp8CascadedCdlQuantForWeight,
    ),
    "nvfp4_ecdl": _QuantVariantSpec(
        "ecdl",
        16,
        E2M1,
        None,
        Nvfp4ExtendedCdlQuantForActivation,
        Nvfp4ExtendedCdlQuantForWeight,
    ),
    "mxfp4_ecdl": _QuantVariantSpec(
        "ecdl",
        32,
        E2M1,
        None,
        Mxfp4ExtendedCdlQuantForActivation,
        Mxfp4ExtendedCdlQuantForWeight,
    ),
    "mxfp8_ecdl": _QuantVariantSpec(
        "ecdl",
        32,
        E4M3,
        None,
        Mxfp8ExtendedCdlQuantForActivation,
        Mxfp8ExtendedCdlQuantForWeight,
    ),
}

QUANT_VARIANTS = ("none", "ste", *_QUANT_VARIANTS)
ACTIVATION_VARIANTS = QUANT_VARIANTS
WEIGHT_VARIANTS = QUANT_VARIANTS
KV_VARIANTS = QUANT_VARIANTS

type MasterPairs = list[tuple[nn.Parameter, nn.Parameter]]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    quant_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    activation: str = "ste"  # "none", "ste", "*_ccdl", or "*_ecdl"
    weight: str = "nvfp4_ccdl"  # "none", "ste", "*_ccdl", or "*_ecdl"
    kv: str = "none"  # "none", "ste", "*_ccdl", or "*_ecdl"
    weight_kappa: float = 2.0
    weight_kappa_bwd: float | None = None
    weight_q_kappa_bwd: float | None = None
    activation_kappa: float = 2.0
    activation_kappa_bwd: float | None = None
    activation_q_kappa_bwd: float | None = None
    kv_kappa: float = 2.0
    kv_kappa_bwd: float | None = None
    kv_q_kappa_bwd: float | None = None
    theta_topk: int = 5
    q_topk: int = 5
    block_size: int = 16  # 16 for nvfp4, 32 for mxfp4/mxfp8
    kernel: str = "triton"
    backend: str = "nvfp4"
    use_lora: bool = True
    freeze_base: bool = False  # with use_lora=false, keep base weights frozen
    train_head: bool = True  # only applies when use_lora=false
    lora_r: int = 8
    lora_alpha: int = 16

    def validate(self, dtype: torch.dtype = MODEL_DTYPE) -> None:
        if self.kernel not in KERNEL_VARIANTS:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.backend not in BACKEND_VARIANTS:
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.activation not in ACTIVATION_VARIANTS:
            raise ValueError(f"unknown activation variant {self.activation!r}")
        if self.weight not in WEIGHT_VARIANTS:
            raise ValueError(f"unknown weight variant {self.weight!r}")
        if self.kv not in KV_VARIANTS:
            raise ValueError(f"unknown kv variant {self.kv!r}")

        uses_mx = self.backend.startswith("mxfp") or any(
            v in _QUANT_VARIANTS and _QUANT_VARIANTS[v].block_size == 32
            for v in (self.activation, self.weight, self.kv)
        )
        required_block = 32 if uses_mx else 16
        if self.block_size != required_block:
            raise ValueError(
                f"block_size must be {required_block} for this variant/backend combo"
            )

        for variant in (self.activation, self.kv):
            spec = _QUANT_VARIANTS.get(variant)
            if spec is None:
                continue
            theta_n = spec.theta_format(dtype).n
            if not 1 <= self.theta_topk <= theta_n:
                raise ValueError(f"theta_topk {self.theta_topk} out of range")
            if (
                spec.q_format is not None
                and not 1 <= self.q_topk <= spec.q_format(dtype).n
            ):
                raise ValueError(f"q_topk {self.q_topk} out of range")

        for role, variant, k_bwd, qk_bwd in (
            ("weight", self.weight, self.weight_kappa_bwd, self.weight_q_kappa_bwd),
            (
                "activation",
                self.activation,
                None,
                self.activation_q_kappa_bwd,
            ),
            ("kv", self.kv, self.kv_kappa_bwd, self.kv_q_kappa_bwd),
        ):
            if k_bwd is not None and not (
                variant.endswith("_ccdl") or variant.endswith("_ecdl")
            ):
                raise ValueError(
                    f"{role}_kappa_bwd is only supported for *_ccdl or *_ecdl {role} variants, got {variant!r}"
                )
            if qk_bwd is not None and not variant.endswith("_ccdl"):
                raise ValueError(
                    f"{role}_q_kappa_bwd is only supported for *_ccdl {role} variants, got {variant!r}"
                )


@dataclass
class DataConfig:
    ratios: dict[str, float] = field(
        default_factory=lambda: {"web": 0.58, "math": 0.14, "cosmopedia": 0.04}
    )
    num_docs: int = 20_000
    seed: int = 0
    seq_len: int = 1024
    batch_size: int = 16
    num_test_blocks: int = 64
    num_calib_blocks: int = 8
    cache_dir: str | None = None


@dataclass
class TrainingConfig:
    epochs: int = 1
    max_steps: int = 0  # 0 = full epoch
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    train_alpha: bool = True
    train_kappa: bool = False
    quant_adamw_lr: float = 3e-4
    gradient_checkpointing: bool = False
    fp32_master_weights: bool = False
    lr_decay: str = "constant"
    min_lr_ratio: float = 0.1
    save_every_iters: int = 500

    def validate(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.lr_decay not in LR_DECAY_VARIANTS:
            raise ValueError(f"unknown lr_decay {self.lr_decay!r}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")


@dataclass
class LoggingConfig:
    log_every_steps: int = 10
    eval_every_iters: int = 500


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: Path | str) -> Config:
    """Load and validate a YAML experiment config against the structured schema."""
    raw = OmegaConf.load(yaml_path)
    schema = OmegaConf.structured(Config)
    return OmegaConf.to_object(OmegaConf.merge(schema, raw))


# ---------------------------------------------------------------------------
# cli / runtime setup
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    logger.info(f"using {device} for training device")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device


def resolve_settings(args: argparse.Namespace) -> tuple[Path, Config]:
    if args.resume:
        resume_dir: Path = args.resume.expanduser()
        yaml_path = resume_dir / "config.yaml"
        logger.info(f"using {yaml_path} as config file")
        cfg = load_config(yaml_path)
        if args.cache_dir is not None:
            cfg.data.cache_dir = str(args.cache_dir)
        return resume_dir, cfg

    output_dir = (args.output_dir or Path("output")).expanduser()
    run_name = (
        args.run_name or f"{time.strftime('%Y-%m-%dT%H-%M-%S')}-{uuid.uuid4().hex[:6]}"
    )
    output_dir = output_dir / run_name

    cfg = load_config(args.config)
    if args.cache_dir is not None:
        cfg.data.cache_dir = str(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"setting output directory to {output_dir}")
    OmegaConf.save(OmegaConf.structured(cfg), output_dir / "config.yaml")
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args).copy(), f, indent=2, default=str)
    return output_dir, cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# model / quant setup
# ---------------------------------------------------------------------------


def _ste_simulation(cfg: ModelConfig) -> str:
    return cfg.backend if cfg.backend.startswith("mxfp") else "nvfp4"


def _build_input_quant(
    variant: str,
    input_shape: Sequence[int],
    kappa_forward: float,
    kappa_backward: float | None,
    q_kappa_backward: float | None,
    cfg: ModelConfig,
    dtype: torch.dtype,
) -> Quant:
    if variant == "none":
        return IdentityQuant()
    if variant == "ste":
        return SteQuant(_ste_simulation(cfg), False)
    spec = _QUANT_VARIANTS[variant]
    if spec.family == "ccdl":
        return spec.act_cls(
            tuple(input_shape),
            cfg.theta_topk,
            cfg.q_topk,
            dtype,
            kappa_forward=kappa_forward,
            theta_kappa_backward=kappa_backward,
            q_kappa_backward=q_kappa_backward,
            kernel=cfg.kernel,
        )
    block_dims = (False,) * (len(input_shape) - 1) + (True,)
    return spec.act_cls(
        input_shape,
        block_dims,
        cfg.theta_topk,
        dtype,
        kappa_forward=kappa_forward,
        kappa_backward=kappa_backward,
        kernel=cfg.kernel,
    )


def _fake_quant(x: torch.Tensor, quant: Quant) -> torch.Tensor:
    q = quant(x.contiguous())
    fp_x = q.dequantize()
    proxy = q.proxy()
    if proxy is not None:
        return proxy + (fp_x - proxy).detach()
    return fp_x


def _kv_quant_sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    k_quant = getattr(module, "k_quant", None)
    if k_quant is not None:
        key = _fake_quant(key, k_quant)
    v_quant = getattr(module, "v_quant", None)
    if v_quant is not None:
        value = _fake_quant(value, v_quant)
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )


def _build_weight_quant(cfg: ModelConfig, weight: torch.Tensor, dtype: torch.dtype):
    if cfg.weight == "none":
        return IdentityQuant()
    if cfg.weight == "ste":
        return SteQuant(_ste_simulation(cfg), False)
    spec = _QUANT_VARIANTS[cfg.weight]
    if spec.family == "ccdl":
        return spec.weight_cls(
            weight,
            dtype=dtype,
            kappa_forward=cfg.weight_kappa,
            theta_kappa_backward=cfg.weight_kappa_bwd,
            q_kappa_backward=cfg.weight_q_kappa_bwd,
            kernel=cfg.kernel,
        )
    block_dims = (False,) * (weight.ndim - 1) + (True,)
    return spec.weight_cls(
        weight,
        block_dims,
        dtype=dtype,
        kappa_forward=cfg.weight_kappa,
        kappa_backward=cfg.weight_kappa_bwd,
        kernel=cfg.kernel,
    )


class ModelController(QLinearMixin, CascadedCdlQuantMixin, ExtendedCdlQuantMixin):
    """Wraps a causal LM so QLinear/CCDL/ECDL mixins can traverse it."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def apply_quant(self, cfg: ModelConfig, dtype: torch.dtype = MODEL_DTYPE) -> None:
        targets = set(cfg.quant_targets)
        for name, module in list(self.named_modules()):
            parent_name, _, leaf = name.rpartition(".")
            if leaf not in targets or not isinstance(module, nn.Linear):
                continue
            parent = self.get_submodule(parent_name) if parent_name else self
            qlinear = QLinear.from_linear(module)
            if cfg.use_lora:
                qlinear.set_lora(cfg.lora_r, cfg.lora_alpha)
            qlinear.set_activation_quant(
                _build_input_quant(
                    cfg.activation,
                    (1, qlinear.in_features),
                    cfg.activation_kappa,
                    cfg.activation_kappa_bwd,
                    cfg.activation_q_kappa_bwd,
                    cfg,
                    dtype,
                )
            )
            qlinear.set_weight_quant(_build_weight_quant(cfg, qlinear.weight, dtype))
            qlinear.set_backend("native")
            setattr(parent, leaf, qlinear)

        if cfg.kv != "none":
            ALL_ATTENTION_FUNCTIONS.register(
                KV_QUANT_ATTN_NAME, _kv_quant_sdpa_attention_forward
            )
            self.model.config._attn_implementation = KV_QUANT_ATTN_NAME
            for module in self.modules():
                if not isinstance(module, LlamaAttention):
                    continue
                kv_shape = (1, module.config.num_key_value_heads, 1, module.head_dim)
                module.k_quant = _build_input_quant(
                    cfg.kv,
                    kv_shape,
                    cfg.kv_kappa,
                    cfg.kv_kappa_bwd,
                    cfg.kv_q_kappa_bwd,
                    cfg,
                    dtype,
                )
                module.v_quant = _build_input_quant(
                    cfg.kv,
                    kv_shape,
                    cfg.kv_kappa,
                    cfg.kv_kappa_bwd,
                    cfg.kv_q_kappa_bwd,
                    cfg,
                    dtype,
                )
                module.config._attn_implementation = KV_QUANT_ATTN_NAME

    def quant_stats(self) -> dict[str, float]:
        return {
            **self.cascaded_cdl_quant_stats(),
            **self.extended_cdl_quant_stats(),
        }

    def weight_quant_counts(self) -> int:
        return (
            self.cascaded_cdl_get_weight_quant_counts()
            + self.extended_cdl_get_weight_quant_counts()
        )

    def activation_quant_counts(self) -> int:
        return (
            self.cascaded_cdl_get_activation_quant_counts()
            + self.extended_cdl_get_activation_quant_counts()
        )

    def quant_param_groups(self, cfg: TrainingConfig) -> tuple[list[dict], set[int]]:
        cascaded_groups, cascaded_params = self.cascaded_cdl_get_param_groups(
            cfg.quant_adamw_lr, cfg.train_alpha, cfg.train_kappa
        )
        extended_groups, extended_params = self.extended_cdl_get_param_groups(
            cfg.quant_adamw_lr, cfg.train_kappa
        )
        groups = cascaded_groups + extended_groups
        param_ids = {id(p) for p in (cascaded_params | extended_params)}
        return groups, param_ids


def untie_and_freeze(
    model: LlamaForCausalLM, *, freeze_all: bool = True, train_head: bool = True
) -> None:
    """Untie the LM head, then apply the requested freeze policy."""
    embed = model.get_input_embeddings()
    head = model.lm_head
    if head.weight is embed.weight:
        head.weight = nn.Parameter(embed.weight.detach().clone())
        model.config.tie_word_embeddings = False

    if not freeze_all:
        embed.weight.requires_grad_(False)
        if not train_head:
            head.weight.requires_grad_(False)
        return

    for p in model.parameters():
        p.requires_grad_(False)
    if train_head:
        head.weight.requires_grad_(True)


def init_activation_q(
    controller: ModelController,
    calib_blocks: torch.Tensor,
    device: torch.device,
    backend: str = "nvfp4",
) -> None:
    """Initialize CCDL activation scales from calibration blocks, then set backend."""
    controller.q_linear_switch_backend("native")
    samples = [block.unsqueeze(0).to(device) for block in calib_blocks]
    controller.cascaded_cdl_init_act_q_pass(iter(samples))
    controller.q_linear_switch_backend(backend)

    for key, value in controller.quant_stats().items():
        logger.info(f"{key} = {value}")


def log_trainable_census(controller: ModelController, model: nn.Module) -> None:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable {n_trainable / 1e6:.3f}M params")
    logger.info(f"quantized weight elements {controller.weight_quant_counts():,}")
    logger.info(
        f"quantized activation elements {controller.activation_quant_counts():,}"
    )


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def build_cache(
    cfg: Config,
    tokenizer: PreTrainedTokenizerBase,
    cache_dir: str | Path | None,
    rebuild: bool,
) -> TokenCache:
    spec = CacheSpec(
        model_id=cfg.model.model_id,
        tokenizer=tokenizer,
        ratios=dict(cfg.data.ratios),
        seed=cfg.data.seed,
        seq_len=cfg.data.seq_len,
        num_docs=cfg.data.num_docs,
        num_test_blocks=cfg.data.num_test_blocks,
        num_calib_blocks=cfg.data.num_calib_blocks,
    )
    target_blocks = (
        cfg.train.max_steps * cfg.data.batch_size * cfg.train.grad_accum_steps
        if cfg.train.max_steps > 0
        else 0
    )
    return TokenCache.open(
        spec,
        directory=cache_dir,
        rebuild=rebuild,
        target_blocks=target_blocks,
    )


def build_loaders(
    cache: TokenCache, cfg: DataConfig, workers: int
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    train_loader = DataLoader(
        cache.train_dataset,
        cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
    )
    test_loader = DataLoader(cache.test_dataset, batch_size=cfg.batch_size)
    return train_loader, test_loader, cache.calib_blocks(cfg.num_calib_blocks)


# ---------------------------------------------------------------------------
# optimizer / master weights
# ---------------------------------------------------------------------------


def build_optimizer(
    controller: ModelController, model: nn.Module, cfg: TrainingConfig
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR,
    MasterPairs,
]:
    """Single AdamW optimizer across base/LoRA and quantizer parameters, paired
    with one warmup + constant/cosine LambdaLR scheduler."""
    quant_groups, quant_param_ids = controller.quant_param_groups(cfg)
    quant_params = [p for group in quant_groups for p in group["params"]]

    master_pairs: MasterPairs = []

    def _opt_param(p: nn.Parameter) -> nn.Parameter:
        if cfg.fp32_master_weights and p.dtype != torch.float32:
            master = nn.Parameter(p.detach().float().clone())
            master_pairs.append((p, master))
            return master
        return p

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in quant_param_ids:
            continue
        target = no_decay if p.ndim <= 1 else decay
        target.append(_opt_param(p))

    param_groups = [
        {"params": decay, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.learning_rate, "weight_decay": 0.0},
        {
            "params": [_opt_param(p) for p in quant_params],
            "lr": cfg.quant_adamw_lr,
            "weight_decay": 0.0,
        },
    ]
    param_groups = [g for g in param_groups if g["params"]]
    if not param_groups:
        raise ValueError("no trainable parameters")

    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg.betas))

    def _lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.lr_decay == "cosine" and cfg.max_steps > cfg.warmup_steps:
            progress = min(
                1.0, (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
            )
            return cfg.min_lr_ratio + 0.5 * (1.0 - cfg.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    return optimizer, lr_scheduler, master_pairs


@torch.no_grad()
def _sync_master_from_model(master_pairs: MasterPairs) -> None:
    for model_p, master_p in master_pairs:
        master_p.data.copy_(model_p.data)


def _zero_grads(
    trainable: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    has_master: bool,
    set_to_none: bool,
) -> None:
    if not has_master:
        optimizer.zero_grad(set_to_none=set_to_none)
        return
    for p in trainable:
        if set_to_none:
            p.grad = None
        elif p.grad is not None:
            p.grad.zero_()


def _step_optimizer(
    trainable: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    master_pairs: MasterPairs,
    grad_clip: float,
) -> torch.Tensor:
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    if not master_pairs:
        optimizer.step()
    else:
        with torch.no_grad():
            for model_p, master_p in master_pairs:
                if model_p.grad is not None:
                    if master_p.grad is None:
                        master_p.grad = torch.empty_like(master_p)
                    master_p.grad.copy_(model_p.grad)
            optimizer.step()
            for model_p, master_p in master_pairs:
                model_p.data.copy_(master_p.data)
    lr_scheduler.step()
    return grad_norm


# ---------------------------------------------------------------------------
# loss / evaluation / compile
# ---------------------------------------------------------------------------


def lm_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused lm_head + cross entropy (Liger), shifted for causal LM."""
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
    targets = input_ids[:, 1:]
    return liger_fused_linear_cross_entropy(
        hidden.reshape(-1, hidden.shape[-1]),
        model.lm_head.weight,
        targets.reshape(-1),
        bias=model.lm_head.bias,
        reduction="mean",
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.compiler.set_stance("force_eager"):
        for input_ids in loader:
            input_ids = input_ids.to(device)
            loss = lm_loss(model, input_ids)
            tokens = input_ids.numel() - input_ids.shape[0]
            total_loss += loss.item() * tokens
            total_tokens += tokens
    model.train(was_training)
    return math.exp(total_loss / total_tokens)


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    tag: str,
) -> None:
    ppl = evaluate(model, loader, device)
    logger.info(f"{tag}: test ppl {ppl:.2f}")
    writer.add_scalar("eval/ppl", ppl, step)
    writer.flush()


def compile_model(model: nn.Module) -> nn.Module:
    torch._dynamo.config.cache_size_limit = 128

    # Small eager warmup (fwd+bwd) populates third-party workspaces and format
    # caches before cudagraph capture, without matching training batch/seqlen.
    device = next(model.parameters()).device
    if device.type == "cuda":
        was_training = model.training
        model.train()
        loss = lm_loss(model, torch.zeros((1, 128), dtype=torch.long, device=device))
        if loss.requires_grad:
            loss.backward()
        model.zero_grad(set_to_none=True)
        for name, buf in model.named_buffers():
            if name.endswith("kernel_seed"):
                buf.zero_()
        model.train(was_training)
        torch.cuda.synchronize(device)

    layers = model.model.layers
    use_ckpt = getattr(model, "is_gradient_checkpointing", False)
    mode = "default" if use_ckpt else "reduce-overhead"
    for layer in layers:
        layer.compile(dynamic=False, mode=mode)
    logger.info(f"torch.compile enabled on {len(layers)} decoder layers (mode={mode})")
    return model


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def _log_train_step(
    writer: SummaryWriter,
    controller: ModelController,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    step: int,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
) -> None:
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
    writer.add_scalar("train/grad_norm", grad_norm.item(), step)
    for key, value in controller.quant_stats().items():
        writer.add_scalar(f"quant/{key}", value, step)
    if device.type == "cuda":
        writer.add_scalar(
            "cuda/mem_peak_gb",
            torch.cuda.max_memory_allocated() / 2**30,
            step,
        )
        torch.cuda.reset_peak_memory_stats()


def _init_grad_buffers_if_accum(trainable: list[nn.Parameter], grad_accum: int) -> None:
    if grad_accum <= 1:
        return
    # Keep persistent .grad buffers so AccumulateGrad never aliases a
    # cudagraph_trees pool tensor across micro-batches.
    for p in trainable:
        if p.grad is None:
            p.grad = torch.zeros_like(p)


def run_training(
    controller: ModelController,
    run_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    writer: SummaryWriter,
    device: torch.device,
    cfg: Config,
    output_dir: Path,
    start_step: int,
    master_pairs: MasterPairs | None = None,
) -> int:
    trainable = [p for p in controller.model.parameters() if p.requires_grad]
    pairs = master_pairs or []
    grad_accum = cfg.train.grad_accum_steps
    _init_grad_buffers_if_accum(trainable, grad_accum)

    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    start_epoch = min(start_step // steps_per_epoch, cfg.train.epochs - 1)
    resume_skip = max(start_step - start_epoch * steps_per_epoch, 0) * grad_accum
    logger.info(
        f"training started from epoch {start_epoch} "
        f"(batch_size={cfg.data.batch_size}, grad_accum_steps={grad_accum})"
    )

    global_step = start_step
    last_saved = -1
    epoch = start_epoch
    stopped_early = False

    for epoch in range(start_epoch, cfg.train.epochs):
        controller.model.train()
        start = time.perf_counter()
        batches = (
            islice(train_loader, resume_skip, None) if resume_skip else train_loader
        )
        accum_count = 0
        window_steps = grad_accum
        accum_loss: torch.Tensor | None = None

        for batch_idx, input_ids in enumerate(
            tqdm.tqdm(
                batches,
                desc=f"epoch {epoch}",
                initial=resume_skip,
                total=batches_per_epoch,
            ),
            start=resume_skip,
        ):
            if accum_count == 0:
                _zero_grads(
                    trainable,
                    optimizer,
                    has_master=bool(pairs),
                    set_to_none=grad_accum == 1,
                )
                window_steps = min(grad_accum, batches_per_epoch - batch_idx)
                accum_loss = None

            torch.compiler.cudagraph_mark_step_begin()
            loss = lm_loss(run_model, input_ids.to(device))
            step_loss = (
                loss.detach() / window_steps if window_steps > 1 else loss.detach()
            )
            accum_loss = step_loss if accum_loss is None else accum_loss + step_loss
            (loss / window_steps if window_steps > 1 else loss).backward()

            accum_count += 1
            if accum_count < window_steps:
                continue
            accum_count = 0

            grad_norm = _step_optimizer(
                trainable, optimizer, lr_scheduler, pairs, cfg.train.grad_clip
            )
            global_step += 1

            if (
                cfg.logging.log_every_steps > 0
                and global_step % cfg.logging.log_every_steps == 0
            ):
                _log_train_step(
                    writer,
                    controller,
                    optimizer,
                    device,
                    step=global_step,
                    loss=accum_loss,
                    grad_norm=grad_norm,
                )

            if (
                cfg.train.save_every_iters > 0
                and global_step % cfg.train.save_every_iters == 0
            ):
                save_checkpoint(
                    output_dir / "last.ckpt",
                    controller,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                )
                last_saved = global_step

            if (
                cfg.logging.eval_every_iters > 0
                and global_step % cfg.logging.eval_every_iters == 0
            ):
                run_eval(
                    run_model,
                    test_loader,
                    device,
                    writer,
                    global_step,
                    tag=f"step {global_step}",
                )

            if cfg.train.max_steps and global_step >= cfg.train.max_steps:
                stopped_early = True
                break

        resume_skip = 0
        logger.info(f"epoch {epoch} completed in {time.perf_counter() - start:.1f}s")
        if stopped_early:
            logger.info(f"stopping early at step {global_step}")
            break

    if global_step != last_saved:
        save_checkpoint(
            output_dir / "last.ckpt",
            controller,
            optimizer,
            lr_scheduler,
            min(epoch, cfg.train.epochs - 1),
            global_step,
        )

    if (
        cfg.logging.eval_every_iters <= 0
        or global_step % cfg.logging.eval_every_iters != 0
    ):
        run_eval(
            run_model,
            test_loader,
            device,
            writer,
            global_step,
            tag=f"final (step {global_step})",
        )

    return global_step


def training_pipeline() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    device = resolve_device(args.device)
    output_dir, cfg = resolve_settings(args)
    cfg.model.validate()
    cfg.train.validate()
    seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, dtype=MODEL_DTYPE)
    untie_and_freeze(
        model,
        freeze_all=cfg.model.use_lora or cfg.model.freeze_base,
        train_head=not cfg.model.use_lora and cfg.model.train_head,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    cache = build_cache(cfg, tokenizer, cfg.data.cache_dir, args.rebuild_cache)
    logger.info(f"token cache {cache.directory}: train {cache.n_train_blocks:,} blocks")
    train_loader, test_loader, calib_blocks = build_loaders(
        cache, cfg.data, args.workers
    )

    controller = ModelController(model)
    controller.apply_quant(cfg.model)
    controller.to(device)

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    init_activation_q(controller, calib_blocks, device, cfg.model.backend)
    optimizer, lr_scheduler, master_pairs = build_optimizer(
        controller, model, cfg.train
    )
    log_trainable_census(controller, model)

    start_step = 0
    if args.resume:
        _, start_step = load_checkpoint(
            args.resume, controller, optimizer, lr_scheduler, device
        )
        if master_pairs:
            _sync_master_from_model(master_pairs)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    run_model = compile_model(controller.model) if args.compile else controller.model
    run_training(
        controller,
        run_model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        writer,
        device,
        cfg,
        output_dir,
        start_step,
        master_pairs=master_pairs,
    )

    logger.info("training finished")
    writer.close()


if __name__ == "__main__":
    training_pipeline()
