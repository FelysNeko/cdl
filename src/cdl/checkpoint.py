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
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
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
    logger.info(f"checkpoint saved to {path}")


def load_checkpoint(
    resume_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int]:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    ckpt = torch.load(resume_dir / "last.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    random.setstate(ckpt["rng"]["python"])
    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["rng"]["cuda"]])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    logger.info(f"checkpoint loaded from {resume_dir}")
    return start_epoch, global_step
