import logging
import os
import random
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def into_orig_mod(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": into_orig_mod(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
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
    ckpt = torch.load(resume_dir / "last.ckpt", map_location=device, weights_only=False)
    into_orig_mod(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    random.setstate(ckpt["rng"]["python"])
    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
    logger.info(f"checkpoint loaded from {resume_dir}")
    return ckpt["epoch"] + 1, ckpt["global_step"]
