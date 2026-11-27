"""Synthetic pretraining throughput benchmark for NVFP4 variants.

Simulates the full CE pretraining loop from ``smollm2.py`` (synthetic token
batches, automatic gradient accumulation from ``tokens_per_iter``, optional
``--compile``, plus peak CUDA memory and tokens/s reporting) without loading any
text corpus.
"""

import argparse
import json
import logging
import time
from typing import Literal

import torch
from transformers import AutoModelForCausalLM

from smollm2 import (
    MODEL_DTYPE,
    Config,
    ModelConfig,
    ModelController,
    TrainingConfig,
    _init_grad_buffers_if_accum,
    _step_optimizer,
    _zero_grads,
    build_optimizer,
    compile_model,
    init_activation_q,
    lm_loss,
    resolve_device,
    seed_all,
    untie_and_freeze,
)

VariantName = Literal["ste-ste", "ccdl-ste", "ccdl-ecdl"]

logger = logging.getLogger(__name__)


def build_variant_configs(
    model_id: str,
    grad_accum: int,
    *,
    gradient_checkpointing: bool = False,
) -> dict[VariantName, tuple[ModelConfig, TrainingConfig]]:
    """Return locked NVFP4 (2D+G4) pretraining ModelConfig/TrainingConfig presets."""

    def _train_cfg(quant_lr: float) -> TrainingConfig:
        return TrainingConfig(
            epochs=1,
            max_steps=4000,
            grad_accum_steps=grad_accum,
            learning_rate=2.0e-4,
            warmup_steps=200,
            weight_decay=0.01,
            betas=[0.9, 0.95],
            grad_clip=1.0,
            train_alpha=False,
            train_kappa=False,
            quant_adamw_lr=quant_lr,
            gradient_checkpointing=gradient_checkpointing,
            fp32_master_weights=True,
            lr_decay="cosine",
            min_lr_ratio=0.1,
            save_every_iters=0,
            kl_distill=False,
        )

    common_model = {
        "model_id": model_id,
        "weight_kappa": 64.0,
        "weight_kappa_end": None,
        "activation_kappa": 16.0,
        "kv_kappa": 16.0,
        "theta_topk": 5,
        "q_topk": 5,
        "block_size": 16,
        "weight_block_2d": True,
        "low_precision_backward": True,
        "skip_last_layers": 0,
        "kernel": "triton",
        "backend": "nvfp4",
        "use_lora": False,
        "freeze_base": False,
        "train_head": True,
    }

    return {
        "ste-ste": (
            ModelConfig(
                **common_model,
                weight="ste",
                activation="ste",
                kv="ste",
                weight_kappa_bwd=None,
                weight_q_kappa_bwd=None,
                activation_kappa_bwd=None,
                kv_kappa_bwd=None,
            ),
            _train_cfg(3.0e-4),
        ),
        "ccdl-ste": (
            ModelConfig(
                **common_model,
                weight="nvfp4_ccdl",
                activation="ste",
                kv="ste",
                weight_kappa_bwd=8.0,
                weight_q_kappa_bwd=2.0,
                activation_kappa_bwd=None,
                kv_kappa_bwd=None,
            ),
            _train_cfg(1.0e-3),
        ),
        "ccdl-ecdl": (
            ModelConfig(
                **common_model,
                weight="nvfp4_ccdl",
                activation="nvfp4_ecdl",
                kv="nvfp4_ecdl",
                weight_kappa_bwd=8.0,
                weight_q_kappa_bwd=2.0,
                activation_kappa_bwd=2.0,
                kv_kappa_bwd=2.0,
            ),
            _train_cfg(1.0e-3),
        ),
    }


def compute_grad_accum(
    batch_size: int, seq_length: int, tokens_per_iter: int
) -> int:
    """Infer gradient accumulation steps from batch_size, seq_length, and tokens/iter."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if seq_length < 2:
        raise ValueError(f"seq_length must be >= 2, got {seq_length}")
    tokens_per_micro = batch_size * seq_length
    if tokens_per_iter <= 0 or tokens_per_iter % tokens_per_micro != 0:
        raise ValueError(
            f"tokens_per_iter ({tokens_per_iter}) must be a positive multiple of "
            f"batch_size * seq_length ({tokens_per_micro})"
        )
    return tokens_per_iter // tokens_per_micro


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pretraining throughput")
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M",
        help="HF model id or local path",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["ste-ste", "ccdl-ste", "ccdl-ecdl"],
        default="ccdl-ste",
        help="Quantization variant preset",
    )
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    parser.add_argument(
        "--tokens-per-iter",
        type=int,
        default=16384,
        help="Tokens per optimizer step (used to infer grad_accum)",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Untimed warmup steps")
    parser.add_argument("--steps", type=int, default=10, help="Measured optimizer steps")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable activation checkpointing",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run_step(
    run_model: torch.nn.Module,
    trainable: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    master_pairs,
    batches: list[torch.Tensor],
    grad_accum: int,
    grad_clip: float,
) -> None:
    _zero_grads(
        trainable,
        optimizer,
        has_master=bool(master_pairs),
        set_to_none=grad_accum == 1,
    )
    for micro_idx in range(grad_accum):
        torch.compiler.cudagraph_mark_step_begin()
        loss = lm_loss(run_model, batches[micro_idx])
        (loss / grad_accum if grad_accum > 1 else loss).backward()
    _step_optimizer(trainable, optimizer, lr_scheduler, master_pairs, grad_clip)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    grad_accum = compute_grad_accum(
        args.batch_size, args.seq_length, args.tokens_per_iter
    )
    seed_all(args.seed)
    device = resolve_device(args.device)

    model_cfg, train_cfg = build_variant_configs(
        args.model,
        grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
    )[args.variant]
    cfg = Config(model=model_cfg, train=train_cfg)
    cfg.model.validate()
    cfg.train.validate()

    logger.info(
        f"pretrain bench: model={args.model} variant={args.variant} "
        f"bs={args.batch_size} seq={args.seq_length} tokens/iter={args.tokens_per_iter} "
        f"grad_accum={grad_accum} compile={args.compile}"
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=MODEL_DTYPE)
    vocab_size = model.config.vocab_size
    untie_and_freeze(model, freeze_all=False, train_head=True)

    controller = ModelController(model)
    controller.apply_quant(cfg.model)
    controller.to(device)

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    calib_blocks = torch.randint(0, vocab_size, (8, args.seq_length), dtype=torch.long)
    init_activation_q(
        controller,
        calib_blocks,
        device,
        cfg.model.backend,
        cfg.model.low_precision_backward,
    )
    optimizer, lr_scheduler, master_pairs = build_optimizer(
        controller, model, cfg.train
    )

    run_model = compile_model(controller.model) if args.compile else controller.model
    run_model.train()

    trainable = [p for p in controller.model.parameters() if p.requires_grad]
    _init_grad_buffers_if_accum(trainable, grad_accum)

    synthetic_batches = [
        torch.randint(
            0, vocab_size, (args.batch_size, args.seq_length), device=device, dtype=torch.long
        )
        for _ in range(grad_accum)
    ]

    for _ in range(args.warmup_steps):
        _run_step(
            run_model,
            trainable,
            optimizer,
            lr_scheduler,
            master_pairs,
            synthetic_batches,
            grad_accum,
            cfg.train.grad_clip,
        )

    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(args.steps):
        _run_step(
            run_model,
            trainable,
            optimizer,
            lr_scheduler,
            master_pairs,
            synthetic_batches,
            grad_accum,
            cfg.train.grad_clip,
        )
    _sync(device)
    elapsed = time.perf_counter() - t0

    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "model": args.model,
        "variant": args.variant,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "tokens_per_iter": args.tokens_per_iter,
        "grad_accum": grad_accum,
        "compile": args.compile,
        "gradient_checkpointing": args.gradient_checkpointing,
        "steps": args.steps,
        "elapsed_s": round(elapsed, 3),
        "step_time_ms": round(1000.0 * elapsed / args.steps, 2),
        "micro_step_ms": round(1000.0 * elapsed / (args.steps * grad_accum), 2),
        "tokens_per_sec": round(args.tokens_per_iter * args.steps / elapsed, 1),
        "peak_mem_gb": round(peak_mem_gb, 3),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
