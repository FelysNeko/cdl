import argparse
import json
import logging
import random
import sys
import time
import uuid
from itertools import islice
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from cdl.model.resnet import get_cifar_resnet
from cdl.trainer import Trainer

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

logger = logging.getLogger(__name__)


def get_loaders(batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
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
    return train_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--bits-edge", type=int, default=8)
    parser.add_argument("--topk-act", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--gam", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=int, nargs="+", default=[60, 120, 160])
    parser.add_argument("--sched-gamma", type=float, default=0.1)
    parser.add_argument("--calib-batches", type=int, default=1)
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def sanitize_device(device: str) -> torch.device:
    return torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )


def get_output_dir(output_dir: Path, run_name: str | None) -> Path:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{uuid.uuid4().hex[:6]}"

    output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_config(output_dir: Path, args: argparse.Namespace) -> Path:
    config = vars(args).copy()
    config["argv"] = sys.argv
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    return config_path


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    # q/alpha groups use scaled learning rates, so the maximum tracks the
    # base learning rate of the regular network parameters.
    return max(group["lr"] for group in optimizer.param_groups)


def training_pipeline() -> None:
    args = parse_args()
    output_dir = get_output_dir(args.output_dir, args.run_name)
    logger.info(f"output directory at {output_dir}")

    config_path = save_config(output_dir, args)
    logger.info(f"config saved to {config_path}")

    seed_all(args.seed)
    logger.info(f"seed with {args.seed}")

    device = sanitize_device(args.device)
    logger.info(f"training on {device}")

    cifar_resnet = get_cifar_resnet(
        args.layers, 100, args.relaxed, args.bits, args.bits_edge, args.topk_act
    )
    trainer = Trainer(cifar_resnet, device)
    logger.info("trainer initialized")

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    calibration_batches = (x for x, _ in islice(train_loader, args.calib_batches))
    trainer.init_act_q_pass(calibration_batches)
    logger.info("quantization steps calibrated")

    n_weight, n_activation = trainer.get_quant_counts()
    logger.info(f"weight quantization counts: {n_weight}")
    logger.info(f"activation quantization counts: {n_activation}")

    param_groups = trainer.get_param_groups(args.lr, args.wd)
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.milestones, args.sched_gamma
    )

    if args.compile:
        logger.info("torch compile enabled")
        trainer.compile()

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    global_step = 0

    for epoch in range(args.epochs):
        trainer.resnet.train()

        ce_sum = 0.0
        hw_sum = 0.0
        hx_sum = 0.0
        loss_sum = 0.0

        for x, y in tqdm.tqdm(train_loader):
            ce, hw, hx = trainer.compute_one_batch_loss(x, y)
            loss = ce + args.gam * hx + args.lam * hw

            ce_value = ce.item()
            hw_value = hw.item()
            hx_value = hx.item()
            loss_value = loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trainer.floor_quant(1e-8, 1e-8)

            ce_sum += ce_value
            hw_sum += hw_value
            hx_sum += hx_value
            loss_sum += loss_value

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                writer.add_scalar("train/ce", ce_value, global_step)
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/Hw_total", hw_value, global_step)
                writer.add_scalar("train/Hx_total", hx_value, global_step)
                writer.add_scalar(
                    "train/avg_weight_bits",
                    hw_value / n_weight if n_weight else 0.0,
                    global_step,
                )
                writer.add_scalar(
                    "train/avg_activation_bits",
                    hx_value / n_activation if n_activation else 0.0,
                    global_step,
                )
                writer.add_scalar("train/lr", current_lr(optimizer), global_step)

            global_step += 1

        lr_scheduler.step()

        batches = len(train_loader)
        epoch_loss = loss_sum / batches
        epoch_ce = ce_sum / batches
        epoch_hw = hw_sum / batches
        epoch_hx = hx_sum / batches
        epoch_avg_weight_bits = epoch_hw / n_weight if n_weight else 0.0
        epoch_avg_activation_bits = epoch_hx / n_activation if n_activation else 0.0
        epoch_lr = current_lr(optimizer)

        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
        writer.add_scalar("epoch/train_ce", epoch_ce, epoch)
        writer.add_scalar("epoch/train_Hw_total", epoch_hw, epoch)
        writer.add_scalar("epoch/train_Hx_total", epoch_hx, epoch)
        writer.add_scalar("epoch/train_avg_weight_bits", epoch_avg_weight_bits, epoch)
        writer.add_scalar(
            "epoch/train_avg_activation_bits", epoch_avg_activation_bits, epoch
        )
        writer.add_scalar("epoch/lr", epoch_lr, epoch)

        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs - 1:
            acc = trainer.evaluate(test_loader)

            writer.add_scalar("epoch/val_acc", acc, epoch)
            writer.flush()

    writer.close()
