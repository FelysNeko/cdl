import argparse
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from cdl.config import Config, ModelConfig, load_config
from cdl.model.resnet import QResNet
from cdl.trainer import Trainer

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

logger = logging.getLogger(__name__)


def get_loaders(
    batch_size: int,
    calibration_batches: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, Iterable[torch.Tensor]]:
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train_set = datasets.CIFAR100(
        root="data", train=True, transform=train_tf, download=True
    )
    test_set = datasets.CIFAR100(
        root="data", train=False, transform=test_tf, download=True
    )
    train_loader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=workers, drop_last=True
    )
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=workers)
    calibration_batches = (x for x, _ in islice(train_loader, calibration_batches))
    return train_loader, test_loader, calibration_batches


def get_q_resnet(
    cfg: ModelConfig,
    num_classes: int,
) -> QResNet:
    n = (cfg.layers - 2) // 6
    num_blocks = [n, n, n]
    q_resnet = QResNet(
        num_blocks,
        num_classes,
        cfg.relaxed,
        cfg.bits,
        cfg.bits_edge,
        cfg.topk_act,
        cfg.kappa,
    )
    num_params = sum(p.numel() for p in q_resnet.parameters())
    logger.info(
        f"built ResNet-{cfg.layers} with blocks {num_blocks} and {num_params} parameters"
    )
    return q_resnet


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
        logger.info("setting float32 matmul precision to high on cuda device")
        torch.set_float32_matmul_precision("high")

    return device


def resolve_settings(
    args: argparse.Namespace,
) -> tuple[Path, Config]:
    if args.resume:
        resume_dir: Path = args.resume.expanduser()
        yaml_path = resume_dir / "config.yaml"
        logger.info(f"using {yaml_path} as config file")

        return resume_dir, load_config(yaml_path)
    else:
        output_dir = args.output_dir or Path("output")
        output_dir = output_dir.expanduser()
        if args.run_name is None:
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            args.run_name = f"{timestamp}-{uuid.uuid4().hex[:6]}"

        output_dir = output_dir / args.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"setting output directory to {output_dir}")

        cfg = load_config(args.config)
        save_config(output_dir, cfg, args)

        return output_dir, cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"seeded with {seed}")


def save_config(output_dir: Path, cfg: Config, args: argparse.Namespace) -> None:
    cfg_payload = OmegaConf.structured(cfg)
    OmegaConf.save(cfg_payload, output_dir / "config.yaml")
    args_payload = vars(args).copy()
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_payload, f, indent=2, default=str)


def save_checkpoint(
    path: Path,
    trainer: Trainer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    resnet = trainer.resnet
    if hasattr(resnet, "_orig_mod"):
        resnet = resnet._orig_mod
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": resnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        },
        tmp,
    )
    os.replace(tmp, path)
    logger.info("checkpoint saved")


def load_checkpoint(
    resume_dir: Path,
    trainer: Trainer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int]:
    ckpt = torch.load(resume_dir / "last.ckpt", map_location=device, weights_only=False)
    trainer.resnet.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    random.setstate(ckpt["rng"]["python"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    logger.info("checkpoint loaded")
    return start_epoch, global_step


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return max(group["lr"] for group in optimizer.param_groups)


def total_grad_norm(optimizer: torch.optim.Optimizer) -> float:
    grads = [
        p.grad
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    return torch.nn.utils.get_total_norm(grads).item()


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


def training_pipeline() -> None:
    args = parse_args()
    device = sanitize_device(args.device)
    output_dir, cfg = resolve_settings(args)
    seed_all(args.seed)

    cifar_resnet = get_q_resnet(cfg.model, 100)
    train_loader, test_loader, calibration_batches = get_loaders(
        cfg.train.batch_size,
        cfg.train.calib_batches,
        args.workers,
    )

    trainer = Trainer(cifar_resnet, device)
    trainer.init_act_q_pass(calibration_batches)
    logger.info("activation quantizer calibrated")

    param_groups = trainer.get_param_groups(
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.freeze_kappa,
    )
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.train.momentum, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, list(cfg.train.milestones), cfg.train.sched_gamma
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, trainer, optimizer, lr_scheduler, device
        )

    n_weight, n_activation = trainer.get_quant_counts()
    logger.info(f"quantized {n_weight} weight elements")
    logger.info(f"quantized {n_activation} activation elements per sample")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    if args.compile:
        logger.info("torch.compile enabled")
        trainer.compile()

    logger.info(f"training started from epoch {start_epoch}")
    for epoch in range(start_epoch, cfg.train.epochs):
        trainer.resnet.train()

        start = time.perf_counter()
        for x, y in tqdm.tqdm(train_loader, desc=f"epoch {epoch}"):
            batch_output = trainer.forward_one_batch(x, y)
            loss = batch_output.compute_joint_loss(cfg.train.lam, cfg.train.gam)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if (
                cfg.logging.log_every_steps > 0
                and global_step % cfg.logging.log_every_steps == 0
            ):
                ce_loss = batch_output.ce_loss.item()
                weight_entropy = batch_output.weight_entropy.item()
                activation_entropy = batch_output.activation_entropy.item()

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                writer.add_scalar("train/ce_loss", ce_loss, global_step)
                writer.add_scalar(
                    "train/error_rate", batch_output.error_rate, global_step
                )
                writer.add_scalar("train/weight_entropy", weight_entropy, global_step)
                writer.add_scalar(
                    "train/activation_entropy", activation_entropy, global_step
                )
                writer.add_scalar(
                    "train/avg_weight_bits",
                    weight_entropy / n_weight if n_weight else 0.0,
                    global_step,
                )
                writer.add_scalar(
                    "train/avg_activation_bits",
                    activation_entropy / n_activation if n_activation else 0.0,
                    global_step,
                )
                writer.add_scalar("train/lr", current_lr(optimizer), global_step)
                writer.add_scalar(
                    "train/grad_norm", total_grad_norm(optimizer), global_step
                )
                for key, value in trainer.kappa_stats().items():
                    writer.add_scalar(f"train/{key}", value, global_step)

                if device.type == "cuda":
                    writer.add_scalar(
                        "cuda/mem_allocated_gb",
                        torch.cuda.memory_allocated() / 2**30,
                        global_step,
                    )
                    writer.add_scalar(
                        "cuda/mem_reserved_gb",
                        torch.cuda.memory_reserved() / 2**30,
                        global_step,
                    )
                    writer.add_scalar(
                        "cuda/mem_peak_gb",
                        torch.cuda.max_memory_allocated() / 2**30,
                        global_step,
                    )
                    torch.cuda.reset_peak_memory_stats()

            optimizer.step()
            trainer.floor_quant(1e-8)
            global_step += 1

        lr_scheduler.step()
        save_checkpoint(
            output_dir / "last.ckpt",
            trainer,
            optimizer,
            lr_scheduler,
            epoch,
            global_step,
        )

        duration = time.perf_counter() - start
        logger.info(f"epoch {epoch} completed in {duration:.1f}s")

        if epoch % cfg.logging.eval_every_epochs == 0 or epoch == cfg.train.epochs - 1:
            acc = trainer.evaluate(test_loader)
            acc_percent = acc * 100
            logger.info(f"epoch {epoch} evaluation accuracy is {acc_percent:.1f}%")
            writer.add_scalar("eval/acc", acc, epoch)
            writer.flush()

    logger.info("training finished")
    writer.close()
