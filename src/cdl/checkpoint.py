import logging
import os
import random
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    | dict[str, torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    global_step: int,
) -> None:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(optimizer, dict):
        optimizer_state = {k: v.state_dict() for k, v in optimizer.items()}
    else:
        optimizer_state = optimizer.state_dict()
    if isinstance(lr_scheduler, dict):
        scheduler_state = {k: v.state_dict() for k, v in lr_scheduler.items()}
    else:
        scheduler_state = lr_scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer_state,
            "lr_scheduler": scheduler_state,
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
    logger.info(f"checkpoint saved to {path}")


def load_checkpoint(
    resume_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    | dict[str, torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
) -> tuple[int, int]:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    ckpt = torch.load(resume_dir / "last.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if isinstance(optimizer, dict):
        for key, opt in optimizer.items():
            opt.load_state_dict(ckpt["optimizer"][key])
    else:
        optimizer.load_state_dict(ckpt["optimizer"])
    if isinstance(lr_scheduler, dict):
        for key, sched in lr_scheduler.items():
            sched.load_state_dict(ckpt["lr_scheduler"][key])
    else:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    random.setstate(ckpt["rng"]["python"])
    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["rng"]["cuda"]])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    logger.info(f"checkpoint loaded from {resume_dir}")
    return start_epoch, global_step
