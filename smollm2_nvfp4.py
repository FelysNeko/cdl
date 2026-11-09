import argparse
import json
import logging
import math
import random
import time
import uuid
from array import array
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from cdl.checkpoint import load_checkpoint, save_checkpoint
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.dataset.smollm_corpus import get_dataset
from cdl.model.components.linear import QLinear, QLinearMixin
from cdl.model.components.lora import LoRALayerMixin
from cdl.model.components.quant.ccdl.ccdl import CascadedCdlQuantMixin
from cdl.model.components.quant.ccdl.nvfp4 import (
    Nvfp4CascadedCdlQuantForActivation,
    Nvfp4CascadedCdlQuantForWeight,
)
from cdl.model.components.quant.ste import SteQuant

logger = logging.getLogger(__name__)

MODEL_DTYPE = torch.bfloat16


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
    activation: str = "ste"  # "ste" (STE nvfp4) or "nvfp4_ccdl" (cascaded CDL nvfp4)
    theta_topk: int = 5  # CDL window over E2M1 levels (activation quant)
    q_topk: int = 5  # CDL window over UE4M3 levels (activation quant)
    block_size: int = 16  # nvfp4 requires 16
    kernel: str = "triton"  # "torch" or "triton" (nvfp4_quant kernel)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.01


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


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    grad_clip: float = 1.0
    train_alpha: bool = True
    train_kappa: bool = False
    ce_chunks: int = 8
    gradient_checkpointing: bool = True
    save_every_iters: int = 500


@dataclass
class LoggingConfig:
    log_every_steps: int = 10
    eval_every_iters: int = 500
    hellaswag_limit: int = 0  # 0 = full validation split (10,042 docs)
    hellaswag_batch_size: int = 64  # sequences per forward (4 per doc)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: Path) -> Config:
    schema = OmegaConf.structured(Config)
    return OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.load(yaml_path)))


class ModelController(LoRALayerMixin, QLinearMixin, CascadedCdlQuantMixin):
    """Wraps the causal LM so the quant/LoRA mixins can traverse it."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def apply_quant(
        self,
        targets: set[str],
        activation: str,
        theta_topk: int,
        q_topk: int,
        dtype: torch.dtype,
        kernel: str,
    ) -> None:
        if activation == "nvfp4_ccdl":
            if not 1 <= theta_topk <= E2M1(dtype).n:
                raise ValueError(f"theta_topk {theta_topk} out of range")
            if not 1 <= q_topk <= UE4M3ExcludeZero(dtype).n:
                raise ValueError(f"q_topk {q_topk} out of range")
        elif activation != "ste":
            raise ValueError(f"unknown activation variant {activation!r}")

        for name, module in list(self.named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if not isinstance(module, nn.Linear) or leaf not in targets:
                continue
            qlinear = QLinear.from_linear(module)
            if activation == "ste":
                qlinear.set_activation_quant(SteQuant("nvfp4"))
            else:
                qlinear.set_activation_quant(
                    Nvfp4CascadedCdlQuantForActivation(
                        (1, qlinear.in_features),
                        theta_topk,
                        q_topk,
                        dtype,
                        kernel,
                    )
                )
            qlinear.set_weight_quant(
                Nvfp4CascadedCdlQuantForWeight(
                    qlinear.weight, dtype=dtype, kernel=kernel
                )
            )
            qlinear.set_backend("native")

            parent = (
                self if "." not in name else self.get_submodule(name.rsplit(".", 1)[0])
            )
            setattr(parent, leaf, qlinear)


def untie_and_freeze(model: LlamaForCausalLM, freeze_all: bool = True) -> None:
    """Untie the output head and freeze the base model. With freeze_all
    (LoRA mode) everything freezes except the head; otherwise only the
    input embedding freezes and the full base trains."""
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
    head.weight.requires_grad_(True)


def build_corpus(
    cfg: DataConfig, tokenizer
) -> tuple[TensorDataset, TensorDataset, torch.Tensor]:
    """Tokenize the sampled pool, pack into fixed-length blocks, split
    train/test; calib blocks overlap with the head of train.

    Tokens accumulate in array('q') at 8 B/token; a list[int] would hold
    ~36 B/token and swap-thrash the box at 100k docs."""
    pool = get_dataset(cfg.ratios, cfg.num_docs, cfg.seed)
    eos = tokenizer.eos_token_id
    texts = pool["text"]
    n_docs = len(texts)
    ids = array("q")
    batch = 1024
    for start in tqdm.tqdm(range(0, n_docs, batch), desc="tokenize"):
        for seq in tokenizer(texts[start : start + batch], add_special_tokens=False)[
            "input_ids"
        ]:
            ids.extend(seq)
            ids.append(eos)
        if (start // batch) % 20 == 0:
            done = min(start + batch, n_docs)
            logger.info(f"tokenized {done:,}/{n_docs:,} docs ({len(ids):,} tokens)")
    del texts, pool

    n_blocks = len(ids) // cfg.seq_len
    if n_blocks <= cfg.num_test_blocks:
        raise ValueError(
            f"corpus produced only {n_blocks} blocks of seq_len={cfg.seq_len}; "
            f"need more than num_test_blocks={cfg.num_test_blocks}"
        )
    flat = np.asarray(ids)  # zero-copy view of the array buffer
    blocks = torch.from_numpy(flat[: n_blocks * cfg.seq_len]).view(n_blocks, -1)
    test_blocks = blocks[-cfg.num_test_blocks :]
    train_blocks = blocks[: -cfg.num_test_blocks]
    calib_blocks = train_blocks[: cfg.num_calib_blocks]
    logger.info(
        f"{len(ids):,} tokens -> {n_blocks} blocks of {cfg.seq_len} "
        f"(train {len(train_blocks)}, test {len(test_blocks)}, calib {len(calib_blocks)})"
    )
    return TensorDataset(train_blocks), TensorDataset(test_blocks), calib_blocks


def sanitize_device(device: str) -> torch.device:
    device = torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    logger.info(f"using {device} for training device")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device


def resolve_settings(args: argparse.Namespace) -> tuple[Path, Config]:
    if args.resume:
        resume_dir: Path = args.resume.expanduser()
        yaml_path = resume_dir / "config.yaml"
        logger.info(f"using {yaml_path} as config file")
        return resume_dir, load_config(yaml_path)

    output_dir = (args.output_dir or Path("output")).expanduser()
    run_name = args.run_name
    if run_name is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        run_name = f"{timestamp}-{uuid.uuid4().hex[:6]}"
    output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"setting output directory to {output_dir}")

    cfg = load_config(args.config)
    OmegaConf.save(OmegaConf.structured(cfg), output_dir / "config.yaml")
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args).copy(), f, indent=2, default=str)
    return output_dir, cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return max(group["lr"] for group in optimizer.param_groups)


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
    return parser.parse_args()


@torch.no_grad()
def hellaswag_accuracy(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    limit: int,
    batch_size: int,
) -> tuple[float, float]:
    """Zero-shot HellaSwag loglikelihood scoring (acc / acc_norm), following
    the lm-eval-harness preprocessing."""
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
        seqs, boundaries, char_lens, golds = [], [], [], []
        for query, endings, gold in items[start : start + group_size]:
            ctx_len = len(tokenizer(query, add_special_tokens=False)["input_ids"])
            for ending in endings:
                full = tokenizer(query + " " + ending, add_special_tokens=False)[
                    "input_ids"
                ]
                seqs.append(full)
                boundaries.append(ctx_len)
                char_lens.append(len(ending) + 1)
            golds.append(gold)

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full(
            (len(seqs), max_len), tokenizer.eos_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        for r, s in enumerate(seqs):
            input_ids[r, : len(s)] = torch.tensor(s)
            attention[r, : len(s)] = 1

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
            for r in range(d * 4, (d + 1) * 4):
                lp = token_lp[r, boundaries[r] - 1 : len(seqs[r]) - 1].sum().item()
                sums.append(lp)
                norms.append(lp / char_lens[r])
            total += 1
            correct += int(max(range(4), key=lambda k: sums[k]) == gold)
            norm_correct += int(max(range(4), key=lambda k: norms[k]) == gold)
    model.train(was_training)
    if total == 0:
        raise ValueError("hellaswag produced no items")
    return correct / total, norm_correct / total


def lm_loss(model: nn.Module, input_ids: torch.Tensor, ce_chunks: int) -> torch.Tensor:
    """Chunked lm_head + cross entropy, so fp32 logits never materialize for
    the full sequence. Shift follows the causal-LM convention."""
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, :-1]
    targets = input_ids[:, 1:]
    step = math.ceil(hidden.shape[1] / ce_chunks)
    loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for i in range(0, hidden.shape[1], step):
        logits = model.lm_head(hidden[:, i : i + step]).float()
        loss = loss + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets[:, i : i + step].reshape(-1),
            reduction="sum",
        )
    return loss / targets.numel()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, ce_chunks: int
) -> float:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for (input_ids,) in loader:
        input_ids = input_ids.to(device)
        loss = lm_loss(model, input_ids, ce_chunks)
        tokens = input_ids.numel() - input_ids.shape[0]
        total_loss += loss.item() * tokens
        total_tokens += tokens
    model.train(was_training)
    return math.exp(total_loss / total_tokens)


def training_pipeline() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    device = sanitize_device(args.device)
    output_dir, cfg = resolve_settings(args)
    if cfg.model.block_size != 16:
        raise ValueError("nvfp4 requires block_size=16")
    if cfg.model.kernel not in ("torch", "triton"):
        raise ValueError(f"unknown kernel {cfg.model.kernel!r}")
    if cfg.model.activation not in ("ste", "nvfp4_ccdl"):
        raise ValueError(f"unknown activation variant {cfg.model.activation!r}")
    seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_id, dtype=MODEL_DTYPE)
    untie_and_freeze(model, freeze_all=cfg.model.use_lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)

    train_set, test_set, calib_blocks = build_corpus(cfg.data, tokenizer)
    train_loader = DataLoader(
        train_set,
        cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(test_set, batch_size=cfg.data.batch_size)

    controller = ModelController(model)
    controller.apply_quant(
        set(cfg.model.quant_targets),
        cfg.model.activation,
        cfg.model.theta_topk,
        cfg.model.q_topk,
        MODEL_DTYPE,
        cfg.model.kernel,
    )
    if cfg.model.use_lora:
        controller.replace_linear_with_lora(
            set(cfg.model.quant_targets),
            cfg.model.lora_r,
            cfg.model.lora_alpha,
            cfg.model.lora_dropout,
        )
    controller.to(device)

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    controller.q_linear_switch_backend("native")
    controller.init_act_q_pass(
        calib_blocks[i].unsqueeze(0).to(device) for i in range(calib_blocks.shape[0])
    )  # initializes the activation CDL q; no-op for the STE variant
    controller.q_linear_switch_backend("nvfp4")

    for key, value in controller.quant_stats().items():
        logger.info(f"{key} = {value}")

    cdl_groups, cdl_params = controller.get_param_groups(
        cfg.train.learning_rate, cfg.train.train_alpha, cfg.train.train_kappa
    )
    cdl_param_ids = {id(p) for p in cdl_params}
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in cdl_param_ids:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = cdl_groups + [
        {
            "params": decay,
            "lr": cfg.train.learning_rate,
            "weight_decay": cfg.train.weight_decay,
        },
        {"params": no_decay, "lr": cfg.train.learning_rate, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg.train.betas))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / cfg.train.warmup_steps)
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable {n_trainable / 1e6:.3f}M params")
    logger.info(f"quantized weight elements {controller.get_weight_quant_counts():,}")
    logger.info(
        f"quantized activation elements {controller.get_activation_quant_counts():,}"
    )

    start_step = 0
    if args.resume:
        # epoch is derived from global_step below; the stored epoch field has
        # epoch-boundary semantics (+1 on load) that mid-epoch saves break.
        _, start_step = load_checkpoint(
            args.resume, controller, optimizer, lr_scheduler, device
        )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    run_model: nn.Module = controller.model
    if args.compile:
        logger.info("torch.compile enabled")
        torch._dynamo.config.cache_size_limit = 128
        torch._inductor.config.triton.cudagraph_trees = False
        run_model = torch.compile(
            controller.model, dynamic=False, mode="reduce-overhead"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    global_step = start_step
    steps_per_epoch = len(train_loader)
    start_epoch = min(global_step // steps_per_epoch, cfg.train.epochs - 1)
    resume_skip = global_step - start_epoch * steps_per_epoch
    resume_skip = max(resume_skip, 0)
    logger.info(f"training started from epoch {start_epoch}")
    last_saved = -1
    epoch = start_epoch
    for epoch in range(start_epoch, cfg.train.epochs):
        controller.model.train()
        start = time.perf_counter()
        batches = (
            islice(train_loader, resume_skip, None) if resume_skip else train_loader
        )
        for (input_ids,) in tqdm.tqdm(
            batches,
            desc=f"epoch {epoch}",
            initial=resume_skip,
            total=steps_per_epoch,
        ):
            input_ids = input_ids.to(device)
            loss = lm_loss(run_model, input_ids, cfg.train.ce_chunks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
            optimizer.step()
            global_step += 1

            if (
                cfg.logging.log_every_steps > 0
                and global_step % cfg.logging.log_every_steps == 0
            ):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr(optimizer), global_step)
                writer.add_scalar("train/grad_norm", gnorm.item(), global_step)
                for key, value in controller.quant_stats().items():
                    writer.add_scalar(f"quant/{key}", value, global_step)
                if device.type == "cuda":
                    writer.add_scalar(
                        "cuda/mem_peak_gb",
                        torch.cuda.max_memory_allocated() / 2**30,
                        global_step,
                    )
                    torch.cuda.reset_peak_memory_stats()

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
                ppl = evaluate(run_model, test_loader, device, cfg.train.ce_chunks)
                acc, acc_norm = hellaswag_accuracy(
                    run_model,
                    tokenizer,
                    device,
                    cfg.logging.hellaswag_limit,
                    cfg.logging.hellaswag_batch_size,
                )
                logger.info(
                    f"step {global_step}: test ppl {ppl:.2f} | "
                    f"hellaswag acc {acc * 100:.1f}% / acc_norm {acc_norm * 100:.1f}%"
                )
                writer.add_scalar("eval/ppl", ppl, global_step)
                writer.add_scalar("eval/hellaswag_acc", acc, global_step)
                writer.add_scalar("eval/hellaswag_acc_norm", acc_norm, global_step)
                writer.flush()

        resume_skip = 0
        lr_scheduler.step()
        logger.info(f"epoch {epoch} completed in {time.perf_counter() - start:.1f}s")

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
        ppl = evaluate(run_model, test_loader, device, cfg.train.ce_chunks)
        acc, acc_norm = hellaswag_accuracy(
            run_model,
            tokenizer,
            device,
            cfg.logging.hellaswag_limit,
            cfg.logging.hellaswag_batch_size,
        )
        logger.info(
            f"final (step {global_step}): test ppl {ppl:.2f} | "
            f"hellaswag acc {acc * 100:.1f}% / acc_norm {acc_norm * 100:.1f}%"
        )
        writer.add_scalar("eval/ppl", ppl, global_step)
        writer.add_scalar("eval/hellaswag_acc", acc, global_step)
        writer.add_scalar("eval/hellaswag_acc_norm", acc_norm, global_step)
        writer.flush()

    logger.info("training finished")
    writer.close()


if __name__ == "__main__":
    training_pipeline()
