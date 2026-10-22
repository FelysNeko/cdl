import argparse
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cdl.checkpoint import load_checkpoint, save_checkpoint
from cdl.core.format.fp4 import E2M1
from cdl.core.format.fp8 import UE4M3ExcludeZero
from cdl.dataset import get_loaders
from cdl.model.components.conv2d import NativeQuantizedConv2d
from cdl.model.components.linear import NativeQuantizedLinear
from cdl.model.components.quant.cascaded_cdl import (
    CascadedCdlQuantForActivation,
    CascadedCdlQuantForWeight,
    CdlQuantMixin,
)
from cdl.model.components.quant.nvfp4_ste import Nvfp4SteQuant
from cdl.model.resnet import ResNet

logger = logging.getLogger(__name__)

MODEL_DTYPE = torch.bfloat16


@dataclass
class ModelConfig:
    layers: int = 20
    weight_block_size: int = 32
    act_block_size: int = 32
    topk_act: int = 5
    kappa: float = 1.0
    alpha: float = 1.0
    activation_quant: str = "cdl"  # "cdl" or "nvfp4"


@dataclass
class TrainingConfig:
    epochs: int = 200
    freeze_kappa: bool = False
    freeze_alpha: bool = False
    batch_size: int = 64
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    milestones: list[int] = field(default_factory=lambda: [60, 120, 160])
    sched_gamma: float = 0.1
    calib_batches: int = 1


@dataclass
class LoggingConfig:
    eval_every_epochs: int = 1
    log_every_steps: int = 50
    recal_bn_batches: int = 100


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: Path) -> Config:
    schema = OmegaConf.structured(Config)
    yaml_cfg = OmegaConf.load(yaml_path)
    cfg = OmegaConf.merge(schema, yaml_cfg)
    return OmegaConf.to_object(cfg)


class CascadeResNet(ResNet, CdlQuantMixin):
    def __init__(
        self,
        num_blocks: list[int],
        num_classes: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__(num_blocks, num_classes)
        self._replace_with_cascade_components(cfg)

    @torch.no_grad()
    def _replace_with_cascade_components(self, cfg: ModelConfig) -> None:
        # The stem and the fc head stay unquantized.
        for name, module in list(self.named_modules()):
            if not name or name in ("stem", "fc"):
                continue

            parent = (
                self if "." not in name else self.get_submodule(name.rsplit(".", 1)[0])
            )
            child_name = name.rsplit(".", 1)[-1]

            if isinstance(module, nn.Conv2d):
                theta_format = E2M1(MODEL_DTYPE)
                q_format = UE4M3ExcludeZero(MODEL_DTYPE)
                weight_quant = CascadedCdlQuantForWeight(
                    module.weight.flatten(1),
                    theta_format,
                    cfg.kappa,
                    q_format,
                    cfg.kappa,
                    cfg.alpha,
                    cfg.weight_block_size,
                )
                activation_quant = self._make_activation_quant(cfg, q_format)
                replacement = NativeQuantizedConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=MODEL_DTYPE,
                    activation_quant=activation_quant,
                    weight_quant=weight_quant,
                    im2col=True,
                    backend="native",
                )
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    assert replacement.bias is not None
                    replacement.bias.copy_(module.bias)
                setattr(parent, child_name, replacement)

            elif isinstance(module, nn.Linear):
                theta_format = E2M1(MODEL_DTYPE)
                q_format = UE4M3ExcludeZero(MODEL_DTYPE)
                weight_quant = CascadedCdlQuantForWeight(
                    module.weight,
                    theta_format,
                    cfg.kappa,
                    q_format,
                    cfg.kappa,
                    cfg.alpha,
                    cfg.weight_block_size,
                )
                activation_quant = self._make_activation_quant(cfg, q_format)
                replacement = NativeQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=MODEL_DTYPE,
                    activation_quant=activation_quant,
                    weight_quant=weight_quant,
                    backend="native",
                )
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    assert replacement.bias is not None
                    replacement.bias.copy_(module.bias)
                setattr(parent, child_name, replacement)

    @staticmethod
    def _make_activation_quant(cfg: ModelConfig, q_format) -> nn.Module:
        if cfg.activation_quant == "nvfp4":
            return Nvfp4SteQuant()
        if cfg.activation_quant != "cdl":
            raise ValueError(
                f"unknown activation_quant {cfg.activation_quant!r}; "
                "expected 'cdl' or 'nvfp4'"
            )
        return CascadedCdlQuantForActivation(
            E2M1(MODEL_DTYPE),
            cfg.kappa,
            cfg.topk_act,
            q_format,
            cfg.kappa,
            q_format.n,
            cfg.alpha,
            cfg.act_block_size,
        )


def get_q_resnet(
    cfg: ModelConfig,
    num_classes: int,
) -> CascadeResNet:
    n = (cfg.layers - 2) // 6
    num_blocks = [n, n, n]
    q_resnet = CascadeResNet(num_blocks, num_classes, cfg)
    num_params = sum(p.numel() for p in q_resnet.parameters())
    logger.info(
        f"built ResNet-{cfg.layers} with blocks {num_blocks} and {num_params} parameters"
    )
    return q_resnet


def cast_vanilla_modules_to(model: nn.Module, dtype: torch.dtype) -> None:
    # Quantizer parameters stay in fp32 as master weights; only the vanilla
    # (unquantized) modules follow the model dtype.
    from cdl.api.quant import Quant

    for module in model.modules():
        if isinstance(module, Quant) or isinstance(
            module, (NativeQuantizedConv2d, NativeQuantizedLinear)
        ):
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            module.to(dtype)


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

        run_name = args.run_name
        if run_name is None:
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            run_name = f"{timestamp}-{uuid.uuid4().hex[:6]}"

        output_dir = output_dir / run_name
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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x.to(MODEL_DTYPE))
            correct += (out.argmax(-1) == y).sum().item()
            total += y.numel()
        return correct / total


def training_pipeline() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    device = sanitize_device(args.device)
    output_dir, cfg = resolve_settings(args)
    seed_all(args.seed)

    logger.info(f"activation quantizer: {cfg.model.activation_quant}")

    cifar_resnet = get_q_resnet(cfg.model, 100).to(device)
    cast_vanilla_modules_to(cifar_resnet, MODEL_DTYPE)
    train_loader, test_loader, calibration_batches = get_loaders(
        cfg.train.batch_size,
        cfg.train.calib_batches,
        args.workers,
    )
    criterion = nn.CrossEntropyLoss()

    cifar_resnet.init_act_q_pass(
        x.to(MODEL_DTYPE) for x in calibration_batches
    )
    for key, value in cifar_resnet.quant_stats().items():
        logger.info(f"{key} = {value}")

    param_groups = cifar_resnet.get_param_groups(
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.freeze_kappa,
        cfg.train.freeze_alpha,
    )
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.train.momentum, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, list(cfg.train.milestones), cfg.train.sched_gamma
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, cifar_resnet, optimizer, lr_scheduler, device
        )

    n_weight, n_activation = cifar_resnet.get_quant_counts()
    logger.info(f"quantized {n_weight} weight elements")
    logger.info(f"quantized {n_activation} activation elements per sample")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    model: nn.Module = cifar_resnet
    if args.compile:
        logger.info("torch.compile enabled")
        torch._dynamo.config.cache_size_limit = 128
        model = torch.compile(model, dynamic=False, mode="reduce-overhead")

    logger.info(f"training started from epoch {start_epoch}")
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()

        start = time.perf_counter()
        for x, y in tqdm.tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            out = model(x.to(MODEL_DTYPE))

            loss = criterion(out, y)
            error_rate = (out.argmax(dim=-1) != y).float().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if (
                cfg.logging.log_every_steps > 0
                and global_step % cfg.logging.log_every_steps == 0
            ):
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/epoch", epoch, global_step)
                writer.add_scalar("train/error_rate", error_rate.item(), global_step)
                writer.add_scalar("train/lr", current_lr(optimizer), global_step)
                writer.add_scalar("train/grad_norm", total_grad_norm(optimizer), global_step)

                for key, value in cifar_resnet.quant_stats().items():
                    writer.add_scalar(f"quant/{key}", value, global_step)

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
            cifar_resnet.floor_quant(1e-8)
            global_step += 1

        lr_scheduler.step()
        save_checkpoint(
            output_dir / "last.ckpt",
            model,
            optimizer,
            lr_scheduler,
            epoch,
            global_step,
        )

        duration = time.perf_counter() - start
        logger.info(f"epoch {epoch} completed in {duration:.1f}s")

        if epoch % cfg.logging.eval_every_epochs == 0 or epoch == cfg.train.epochs - 1:
            if cfg.logging.recal_bn_batches > 0:
                cifar_resnet.recalibrate_bn(
                    ((x.to(MODEL_DTYPE), y) for x, y in train_loader),
                    cfg.logging.recal_bn_batches,
                    device,
                )
            acc = evaluate(model, test_loader, device)
            acc_percent = acc * 100
            logger.info(f"epoch {epoch} evaluation accuracy is {acc_percent:.1f}%")
            writer.add_scalar("eval/acc", acc, epoch)
            writer.flush()

    logger.info("training finished")
    writer.close()


if __name__ == "__main__":
    training_pipeline()
