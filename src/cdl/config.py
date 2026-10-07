from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class Config:
    layers: int = 20
    relaxed: bool = False
    bits: int = 6
    bits_edge: int = 8
    topk_act: int = 5
    kappa: float = 1.0

    lam: float = 0.0
    gam: float = 0.0

    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    milestones: list[int] = field(default_factory=lambda: [60, 120, 160])
    sched_gamma: float = 0.1

    calib_batches: int = 1
    eval_every_epochs: int = 1
    log_every_steps: int = 50
    seed: int = 0


def load_config(yaml_path: Path) -> Config:
    schema = OmegaConf.structured(Config)
    yaml_cfg = OmegaConf.load(yaml_path)
    cfg = OmegaConf.merge(schema, yaml_cfg)
    return OmegaConf.to_object(cfg)
