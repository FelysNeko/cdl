"""SmolLM2 NVFP4/CDL experiment runner.

Supports STE, CCDL, and ECDL quantizers for weights and activations, LoRA,
resumable checkpoints, and TensorBoard logging.
"""

import argparse
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

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

from cdl.checkpoint import load_checkpoint, save_checkpoint
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.dataset.token_cache import CacheSpec, TokenCache
from cdl.model.components.linear import QLinear, QLinearMixin
from cdl.model.components.quant.ccdl.ccdl import CascadedCdlQuantMixin
from cdl.model.components.quant.ccdl.nvfp4 import (
    Nvfp4CascadedCdlQuantForActivation,
    Nvfp4CascadedCdlQuantForWeight,
)
from cdl.model.components.quant.cdl import CdlQuantMixin
from cdl.model.components.quant.ecdl.ecdl import ExtendedCdlQuantMixin
from cdl.model.components.quant.ecdl.nvfp4 import (
    Nvfp4ExtendedCdlQuantForActivation,
    Nvfp4ExtendedCdlQuantForWeight,
)
from cdl.model.components.quant.ste import SteQuant

logger = logging.getLogger(__name__)

MODEL_DTYPE = torch.bfloat16

ACTIVATION_VARIANTS = ("ste", "nvfp4_ccdl", "nvfp4_ecdl")
WEIGHT_VARIANTS = ("ste", "nvfp4_ccdl", "nvfp4_ecdl")
KERNEL_VARIANTS = ("torch", "triton")

# Removed flags that may still appear in old run configs.
REMOVED_MODEL_FLAGS = ("activation_theta_ste", "activation_theta_norm")
# Weight q init is always amax now; old run configs still carry the key.
IGNORED_MODEL_KEYS = ("weight_q_init", "activation_absolute_alpha")


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
    activation: str = "ste"  # "ste", "nvfp4_ccdl", or "nvfp4_ecdl"
    weight: str = "nvfp4_ccdl"  # "ste", "nvfp4_ccdl", or "nvfp4_ecdl"
    weight_kappa: float = 2.0  # CDL inverse temperature for weight quant
    activation_kappa: float = 2.0  # CDL inverse temperature for activation quant
    # Decoupled backward temperature for activation ECDL; None reuses the
    # forward kappa.
    activation_kappa_bwd: float | None = None
    theta_topk: int = 5  # CDL window over E2M1 levels (activation quant)
    q_topk: int = 5  # CDL window over UE4M3 levels (activation quant)
    block_size: int = 16  # nvfp4 requires 16
    kernel: str = "triton"  # "torch" or "triton"
    use_lora: bool = True
    freeze_base: bool = False  # with use_lora=false, keep the base frozen
    # With use_lora=false and freeze_base=true, train lm_head (head-only
    # ablations) or keep it frozen too (scalers-only run).
    train_head: bool = True
    lora_r: int = 8
    lora_alpha: int = 16

    def validate(self, dtype: torch.dtype = MODEL_DTYPE) -> None:
        if self.block_size != 16:
            raise ValueError("nvfp4 requires block_size=16")
        if self.kernel not in KERNEL_VARIANTS:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.activation not in ACTIVATION_VARIANTS:
            raise ValueError(f"unknown activation variant {self.activation!r}")
        if self.weight not in WEIGHT_VARIANTS:
            raise ValueError(f"unknown weight variant {self.weight!r}")
        if self.activation == "nvfp4_ccdl":
            if not 1 <= self.theta_topk <= E2M1(dtype).n:
                raise ValueError(f"theta_topk {self.theta_topk} out of range")
            if not 1 <= self.q_topk <= UE4M3ExcludeZero(dtype).n:
                raise ValueError(f"q_topk {self.q_topk} out of range")
        elif self.activation == "nvfp4_ecdl":
            if not 1 <= self.theta_topk <= E2M1(dtype).n:
                raise ValueError(f"theta_topk {self.theta_topk} out of range")


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
    cache_dir: str | None = None  # None = auto-derived append-only token cache


@dataclass
class TrainingConfig:
    epochs: int = 1
    max_steps: int = 0  # 0 = full epoch; otherwise stop after this many steps
    learning_rate: float = 1e-4  # AdamW eta, base model (decay/no_decay groups)
    warmup_steps: int = 200
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    # Learned quant scales: alpha = CCDL weight global scale (log_alpha),
    # beta = ECDL global scale (log_beta).
    train_alpha: bool = True
    train_beta: bool = True
    train_kappa: bool = False
    quant_adamw_lr: float = 3e-4  # uniform lr for all quant params (AdamW)
    # Liger fused CE keeps the LM-head logits out of the graph, so bs16 fits
    # without activation checkpointing; leave the escape hatch for larger models.
    gradient_checkpointing: bool = False
    save_every_iters: int = 500


@dataclass
class LoggingConfig:
    log_every_steps: int = 10
    eval_every_iters: int = 500  # test ppl only; full benchmarks run separately
    # Kept for old configs / external eval scripts. Training no longer runs
    # HellaSwag; the final benchmark suite is executed on the checkpoint.
    hellaswag_limit: int = 0  # 0 = full validation split (10,042 docs)
    hellaswag_batch_size: int = 64  # sequences per forward (4 per doc)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: Path) -> Config:
    raw = OmegaConf.load(yaml_path)
    model = raw.get("model")
    if model is not None:
        for key in REMOVED_MODEL_FLAGS:
            if model.get(key):
                raise ValueError(
                    f"{key} was removed; use the decoupled-kappa activation path"
                )
            model.pop(key, None)
        for key in IGNORED_MODEL_KEYS:
            model.pop(key, None)
    train = raw.get("train")
    if train is not None:
        # legacy keys no longer in the schema (old run dirs still carry them)
        for key in ("quant_optimizer", "quant_learning_rate",
                    "activation_train_alpha", "weight_train_alpha"):
            train.pop(key, None)
    schema = OmegaConf.structured(Config)
    return OmegaConf.to_object(OmegaConf.merge(schema, raw))


# ---------------------------------------------------------------------------
# cli / run settings
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
    run_name = args.run_name
    if run_name is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        run_name = f"{timestamp}-{uuid.uuid4().hex[:6]}"
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


class ModelController(
    QLinearMixin,
    CdlQuantMixin,
    CascadedCdlQuantMixin,
    ExtendedCdlQuantMixin,
):
    """Wraps the causal LM so the quant/LoRA mixins can traverse it."""

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
                _build_activation_quant(cfg, qlinear.in_features, dtype)
            )
            qlinear.set_weight_quant(_build_weight_quant(cfg, qlinear.weight, dtype))
            qlinear.set_backend("native")
            setattr(parent, leaf, qlinear)


def _build_activation_quant(cfg: ModelConfig, in_features: int, dtype: torch.dtype):
    if cfg.activation == "ste":
        return SteQuant("nvfp4")
    if cfg.activation == "nvfp4_ccdl":
        return Nvfp4CascadedCdlQuantForActivation(
            (1, in_features),
            cfg.theta_topk,
            cfg.q_topk,
            dtype,
            kappa=cfg.activation_kappa,
            kernel=cfg.kernel,
        )
    if cfg.activation == "nvfp4_ecdl":
        kappa_backward = (
            cfg.activation_kappa
            if cfg.activation_kappa_bwd is None
            else cfg.activation_kappa_bwd
        )
        return Nvfp4ExtendedCdlQuantForActivation(
            (1, in_features),
            cfg.theta_topk,
            dtype,
            kappa_forward=cfg.activation_kappa,
            kappa_backward=kappa_backward,
            kernel=cfg.kernel,
        )
    raise ValueError(f"unknown activation variant {cfg.activation!r}")


def _build_weight_quant(cfg: ModelConfig, weight: torch.Tensor, dtype: torch.dtype):
    if cfg.weight == "ste":
        return SteQuant("nvfp4")
    if cfg.weight == "nvfp4_ccdl":
        return Nvfp4CascadedCdlQuantForWeight(
            weight,
            dtype=dtype,
            kernel=cfg.kernel,
            kappa=cfg.weight_kappa,
        )
    if cfg.weight == "nvfp4_ecdl":
        return Nvfp4ExtendedCdlQuantForWeight(
            weight,
            dtype=dtype,
            kappa_forward=cfg.weight_kappa,
            kernel=cfg.kernel,
        )
    raise ValueError(f"unknown weight variant {cfg.weight!r}")


def untie_and_freeze(
    model: LlamaForCausalLM, *, freeze_all: bool = True, train_head: bool = True
) -> None:
    """Untie the LM head, then apply the requested freeze policy.

    With ``freeze_all`` the whole model is frozen and ``train_head`` optionally
    re-enables the LM head (head-only ablation). Without it only the input
    embedding is frozen, so the base model and head stay trainable.
    """
    embed = model.get_input_embeddings()
    head = model.lm_head
    if head.weight is embed.weight:
        head.weight = nn.Parameter(embed.weight.detach().clone())
        model.config.tie_word_embeddings = False

    if not freeze_all:
        embed.weight.requires_grad_(False)
        return

    for p in model.parameters():
        p.requires_grad_(False)
    if train_head:
        head.weight.requires_grad_(True)


def collect_quant_stats(controller: ModelController) -> dict[str, float]:
    stats: dict[str, float] = {}
    stats.update(controller.cdl_quant_stats())
    stats.update(controller.cascaded_cdl_quant_stats())
    stats.update(controller.extended_cdl_quant_stats())
    return stats


def total_weight_quant_counts(controller: ModelController) -> int:
    return (
        controller.cdl_get_weight_quant_counts()
        + controller.cascaded_cdl_get_weight_quant_counts()
        + controller.extended_cdl_get_weight_quant_counts()
    )


def total_activation_quant_counts(controller: ModelController) -> int:
    return (
        controller.cdl_get_activation_quant_counts()
        + controller.cascaded_cdl_get_activation_quant_counts()
        + controller.extended_cdl_get_activation_quant_counts()
    )


def init_activation_q(
    controller: ModelController, calib_blocks: torch.Tensor, device: torch.device
) -> None:
    """Initialize activation CDL scales from calibration blocks (no-op for STE)."""
    controller.q_linear_switch_backend("native")
    samples = [block.unsqueeze(0).to(device) for block in calib_blocks]
    controller.cdl_init_act_q_pass(iter(samples))
    controller.cascaded_cdl_init_act_q_pass(iter(samples))
    controller.q_linear_switch_backend("nvfp4")

    for key, value in collect_quant_stats(controller).items():
        logger.info(f"{key} = {value}")


def log_trainable_census(controller: ModelController, model: nn.Module) -> None:
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable {n_trainable / 1e6:.3f}M params")
    logger.info(f"quantized weight elements {total_weight_quant_counts(controller):,}")
    logger.info(
        f"quantized activation elements {total_activation_quant_counts(controller):,}"
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
        cfg.train.max_steps * cfg.data.batch_size if cfg.train.max_steps > 0 else 0
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
# optimizers
# ---------------------------------------------------------------------------


def _quant_param_groups(
    controller: ModelController, cfg: TrainingConfig
) -> tuple[list[dict], set[int]]:
    cdl_groups, cdl_params = controller.cdl_get_param_groups(
        cfg.quant_adamw_lr, cfg.train_kappa
    )
    cascaded_groups, cascaded_params = controller.cascaded_cdl_get_param_groups(
        cfg.quant_adamw_lr, cfg.train_alpha, cfg.train_kappa
    )
    extended_groups, extended_params = controller.extended_cdl_get_param_groups(
        cfg.quant_adamw_lr,
        cfg.train_beta,
        cfg.train_kappa,
    )
    groups = cdl_groups + cascaded_groups + extended_groups
    param_ids = {id(p) for p in (cdl_params | cascaded_params | extended_params)}
    return groups, param_ids


def _build_quant_optimizer(
    groups: list[dict], cfg: TrainingConfig
) -> torch.optim.Optimizer | None:
    if not groups:  # STE-only variants have no CDL params at all
        return None
    flattened = [
        {
            "params": group["params"],
            "lr": cfg.quant_adamw_lr,
            "weight_decay": 0.0,
        }
        for group in groups
    ]
    return torch.optim.AdamW(flattened, betas=tuple(cfg.betas))


def build_optimizers(
    controller: ModelController, model: nn.Module, cfg: TrainingConfig
) -> tuple[
    dict[str, torch.optim.Optimizer],
    dict[str, torch.optim.lr_scheduler.LambdaLR],
]:
    """Quantizer scales (q/alpha/kappa) take SGD with momentum at their
    per-bucket scaled lrs; all remaining parameters stay on AdamW. Both get
    the same linear warmup, stepped once per training step."""
    quant_groups, quant_param_ids = _quant_param_groups(controller, cfg)
    quant_optimizer = _build_quant_optimizer(quant_groups, cfg)

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad or id(p) in quant_param_ids:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)

    adamw_groups = [
        {
            "params": decay,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        {"params": no_decay, "lr": cfg.learning_rate, "weight_decay": 0.0},
    ]
    adamw_groups = [g for g in adamw_groups if g["params"]]
    adamw_optimizer = (
        torch.optim.AdamW(adamw_groups, betas=tuple(cfg.betas))
        if adamw_groups
        else None
    )

    def warmup(step: int) -> float:
        return min(1.0, (step + 1) / cfg.warmup_steps)

    optimizers: dict[str, torch.optim.Optimizer] = {}
    if quant_optimizer is not None:
        optimizers["quant"] = quant_optimizer
    if adamw_optimizer is not None:
        optimizers["adamw"] = adamw_optimizer
    lr_schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(opt, warmup)
        for name, opt in optimizers.items()
    }
    return optimizers, lr_schedulers


def optimizer_params(optimizer: torch.optim.Optimizer | None) -> list[nn.Parameter]:
    if optimizer is None:
        return []
    return [p for group in optimizer.param_groups for p in group["params"]]


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return max(group["lr"] for group in optimizer.param_groups)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def hellaswag_accuracy(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    limit: int,
    batch_size: int,
) -> tuple[float, float]:
    """Zero-shot HellaSwag loglikelihood scoring (acc / acc_norm), following
    the lm-eval-harness preprocessing.

    Standalone utility: training only logs test perplexity.
    """
    import re

    from datasets import load_dataset

    def preprocess(text: str) -> str:
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    docs = load_dataset("Rowan/hellaswag", split="validation")
    if limit > 0:
        docs = docs.select(range(min(limit, len(docs))))

    items = []
    for doc in docs:
        query = preprocess(doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        endings = [preprocess(e) for e in doc["endings"]]
        items.append((query, endings, int(doc["label"])))

    was_training = model.training
    model.eval()
    correct = norm_correct = total = 0
    group_size = max(1, batch_size // 4)
    for start in range(0, len(items), group_size):
        seqs, boundaries, char_lens = [], [], []
        for query, endings, _ in items[start : start + group_size]:
            ctx_len = len(tokenizer(query, add_special_tokens=False)["input_ids"])
            for ending in endings:
                full = tokenizer(query + " " + ending, add_special_tokens=False)[
                    "input_ids"
                ]
                seqs.append(full)
                boundaries.append(ctx_len)
                char_lens.append(len(ending) + 1)

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full(
            (len(seqs), max_len), tokenizer.eos_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention[row, : len(seq)] = 1

        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention.to(device)
        ).logits
        token_lp = (
            torch.log_softmax(logits[:, :-1].float(), dim=-1)
            .gather(-1, input_ids[:, 1:].to(device).unsqueeze(-1))
            .squeeze(-1)
        )

        for d, (_, _, gold) in enumerate(items[start : start + group_size]):
            sums, norms = [], []
            for row in range(d * 4, (d + 1) * 4):
                lp = (
                    token_lp[row, boundaries[row] - 1 : len(seqs[row]) - 1].sum().item()
                )
                sums.append(lp)
                norms.append(lp / char_lens[row])
            total += 1
            correct += int(max(range(4), key=lambda k: sums[k]) == gold)
            norm_correct += int(max(range(4), key=lambda k: norms[k]) == gold)
    model.train(was_training)
    if total == 0:
        raise ValueError("hellaswag produced no items")
    return correct / total, norm_correct / total


def lm_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused lm_head + cross entropy (Liger), shifted for causal LM.

    Liger computes the CE over vocabulary chunks and recomputes logits in
    backward, so the fp32 logits/log_softmax of the full sequence never stay
    resident. The reduction is the mean over shifted target tokens, matching
    the previous sum/targets.numel() loss.
    """
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
    """Test-set perplexity, logged to console and TensorBoard.

    HellaSwag is intentionally not evaluated during training: the full
    benchmark suite is run once on the final checkpoint, outside the training
    loop. The 1024-doc eval used to leave ~4.8 GB of logits in the caching
    allocator and roughly halved the step rate after the first eval.
    """
    ppl = evaluate(model, loader, device)
    logger.info(f"{tag}: test ppl {ppl:.2f}")
    writer.add_scalar("eval/ppl", ppl, step)
    writer.flush()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def compile_model(model: nn.Module) -> nn.Module:
    logger.info("torch.compile enabled")
    torch._dynamo.config.cache_size_limit = 128
    torch._inductor.config.triton.cudagraph_trees = False
    return torch.compile(model, dynamic=False, mode="reduce-overhead")


def log_train_step(
    writer: SummaryWriter,
    controller: ModelController,
    optimizers: dict[str, torch.optim.Optimizer],
    device: torch.device,
    *,
    step: int,
    loss: torch.Tensor,
    base_gnorm: torch.Tensor,
    quant_gnorm: torch.Tensor | None,
) -> None:
    writer.add_scalar("train/loss", loss.item(), step)
    if "adamw" in optimizers:
        writer.add_scalar("train/lr", current_lr(optimizers["adamw"]), step)
    writer.add_scalar("train/grad_norm", base_gnorm.item(), step)
    if quant_gnorm is not None:
        writer.add_scalar("train/quant_grad_norm", quant_gnorm.item(), step)
    for key, value in collect_quant_stats(controller).items():
        writer.add_scalar(f"quant/{key}", value, step)
    if device.type == "cuda":
        writer.add_scalar(
            "cuda/mem_peak_gb",
            torch.cuda.max_memory_allocated() / 2**30,
            step,
        )
        torch.cuda.reset_peak_memory_stats()


def run_training(
    controller: ModelController,
    model: nn.Module,
    run_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    lr_schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    writer: SummaryWriter,
    device: torch.device,
    cfg: Config,
    output_dir: Path,
    start_step: int,
) -> int:
    quant_trainable = optimizer_params(optimizers.get("quant"))
    quant_param_ids = {id(p) for p in quant_trainable}
    base_trainable = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in quant_param_ids
    ]

    global_step = start_step
    steps_per_epoch = len(train_loader)
    start_epoch = min(global_step // steps_per_epoch, cfg.train.epochs - 1)
    resume_skip = max(global_step - start_epoch * steps_per_epoch, 0)
    logger.info(f"training started from epoch {start_epoch}")

    last_saved = -1
    epoch = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, cfg.train.epochs):
        controller.model.train()
        start = time.perf_counter()
        batches = (
            islice(train_loader, resume_skip, None) if resume_skip else train_loader
        )
        for input_ids in tqdm.tqdm(
            batches,
            desc=f"epoch {epoch}",
            initial=resume_skip,
            total=steps_per_epoch,
        ):
            input_ids = input_ids.to(device)
            loss = lm_loss(run_model, input_ids)

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()

            base_gnorm = (
                torch.nn.utils.clip_grad_norm_(base_trainable, cfg.train.grad_clip)
                if base_trainable
                else torch.zeros((), device=device)
            )
            quant_gnorm = (
                torch.nn.utils.clip_grad_norm_(quant_trainable, cfg.train.grad_clip)
                if quant_trainable
                else None
            )

            for opt in optimizers.values():
                opt.step()
            global_step += 1
            for scheduler in lr_schedulers.values():
                scheduler.step()

            if (
                cfg.logging.log_every_steps > 0
                and global_step % cfg.logging.log_every_steps == 0
            ):
                log_train_step(
                    writer,
                    controller,
                    optimizers,
                    device,
                    step=global_step,
                    loss=loss,
                    base_gnorm=base_gnorm,
                    quant_gnorm=quant_gnorm,
                )

            if (
                cfg.train.save_every_iters > 0
                and global_step % cfg.train.save_every_iters == 0
            ):
                save_checkpoint(
                    output_dir / "last.ckpt",
                    controller,
                    optimizers,
                    lr_schedulers,
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
            optimizers,
            lr_schedulers,
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

    init_activation_q(controller, calib_blocks, device)
    optimizers, lr_schedulers = build_optimizers(controller, model, cfg.train)
    log_trainable_census(controller, model)

    start_step = 0
    if args.resume:
        # The epoch is derived from global_step below; the stored epoch field
        # has epoch-boundary semantics (+1 on load) that mid-epoch saves break.
        _, start_step = load_checkpoint(
            args.resume, controller, optimizers, lr_schedulers, device
        )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    run_model = compile_model(controller.model) if args.compile else controller.model
    run_training(
        controller,
        model,
        run_model,
        train_loader,
        test_loader,
        optimizers,
        lr_schedulers,
        writer,
        device,
        cfg,
        output_dir,
        start_step,
    )

    logger.info("training finished")
    writer.close()


if __name__ == "__main__":
    training_pipeline()
