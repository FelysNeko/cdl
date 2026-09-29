import argparse
import time
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cdl.model import (
    CdlQuant,
    CdlQuantForActivation,
    CdlQuantForWeight,
    init_act_q_pass,
)
from cdl.resnet import get_cifar_resnet

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


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


def get_param_groups(
    net: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    parameters = []
    customized_param_set = set()

    def register_parameter(ps: nn.Parameter, lr: float, wd: float):
        customized_param_set.add(ps)
        parameters.append({"params": ps, "lr": lr, "weight_decay": wd})

    for m in net.modules():
        if not isinstance(m, CdlQuant):
            continue

        if m.numel == 0:
            raise RuntimeError("quantizer not initialized")

        q_lr = m.scale_q_lr(learning_rate)
        register_parameter(m.q, q_lr, 0.0)

        alpha_lr = m.scale_alpha_lr(learning_rate)
        register_parameter(m.alpha, alpha_lr, 0.0)

    parameters.append(
        {
            "params": (x for x in net.parameters() if x not in customized_param_set),
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    )

    return parameters


@torch.no_grad()
def floor_quant(net: nn.Module, min_q: float, min_alpha: float) -> None:
    for m in net.modules():
        if not isinstance(m, CdlQuant):
            continue
        m.q.clamp_(min=min_q)
        m.alpha.clamp_(min=min_alpha)


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # fork_rng: keep eval-time Q_p sampling off the global RNG stream, so the
    # training trajectory does not depend on eval frequency or test-set size.
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        net.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (net(x).argmax(-1) == y).sum().item()
            total += y.numel()
        return correct / total


def weight_entropy(net: nn.Module) -> torch.Tensor:
    total = None
    for m in net.modules():
        if isinstance(m, CdlQuantForWeight):
            e = m.compute_entropy_and_reset()
            total = e if total is None else total + e
    if total is None:
        raise RuntimeError("no quantized weight layers found")
    return total


def activation_entropy(net: nn.Module) -> torch.Tensor:
    total = None
    for m in net.modules():
        if isinstance(m, CdlQuantForActivation):
            e = m.compute_entropy_and_reset()
            total = e if total is None else total + e
    if total is None:
        raise RuntimeError("no activation quantizers found")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--bits-edge", type=int, default=8)
    parser.add_argument("--topk-act", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--gam", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=int, nargs="+", default=[60, 120, 160])
    parser.add_argument("--sched-gamma", type=float, default=0.1)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sanitize_device(device: str) -> torch.device:
    return torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )


def train() -> None:
    args = parse_args()
    device = sanitize_device(args.device)
    torch.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    net = get_cifar_resnet(
        args.layers, 100, args.relaxed, args.bits, args.bits_edge, args.topk_act
    ).to(device)

    init_act_q_pass(
        net, (x.to(device) for x, _ in islice(train_loader, args.calib_batches))
    )

    optimizer = torch.optim.SGD(
        get_param_groups(net, args.lr, args.wd),
        momentum=args.momentum,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.milestones, args.sched_gamma
    )
    criterion = nn.CrossEntropyLoss()

    if args.compile:
        torch._dynamo.config.cache_size_limit = 128
        net = torch.compile(net, dynamic=True)

    tick = time.perf_counter()
    for epoch in range(args.epochs):
        net.train()
        ce_sum = 0.0
        hw_sum = 0.0
        hx_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(net(x), y)
            ce_sum += loss.item()

            # Objective of (30): L + gamma * H(x) + lambda * H(w). The entropies
            # are always evaluated, so the activation accumulators are always
            # drained and --gam 0 / --lam 0 stay valid weights.
            hx = activation_entropy(net)
            hw = weight_entropy(net)
            loss = loss + args.gam * hx + args.lam * hw
            hx_sum += hx.item()
            hw_sum += hw.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            floor_quant(net, 1e-8, 1e-8)
        lr_scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            acc = evaluate(net, test_loader, device)
            batches = len(train_loader)
            print(
                f"epoch {epoch:3d}  loss {ce_sum / batches:.4f}  "
                f"Hw {hw_sum / batches:.3f}  Hx {hx_sum / batches:.3f}  "
                f"acc {acc * 100:.2f}  {time.perf_counter() - tick:.0f}s",
                flush=True,
            )
