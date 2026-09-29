import argparse
import time
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cdl.model.resnet import get_cifar_resnet
from cdl.trainer import Trainer

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


def training_pipeline() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cifar_resnet = get_cifar_resnet(
        args.layers, 100, args.relaxed, args.bits, args.bits_edge, args.topk_act
    )
    device = sanitize_device(args.device)
    trainer = Trainer(cifar_resnet, device)

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    calibration_batches = (x for x, _ in islice(train_loader, args.calib_batches))
    trainer.init_act_q_pass(calibration_batches)

    param_groups = trainer.get_param_groups(args.lr, args.wd)
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.milestones, args.sched_gamma
    )

    if args.compile:
        trainer.compile()

    tick = time.perf_counter()
    for epoch in range(args.epochs):
        trainer.resnet.train()
        ce_sum = 0.0
        hw_sum = 0.0
        hx_sum = 0.0
        for x, y in train_loader:
            ce, hw, hx = trainer.compute_one_batch_loss(x, y)
            loss = ce + args.gam * hx + args.lam * hw
            ce_sum += ce.item()
            hw_sum += hw.item()
            hx_sum += hx.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            trainer.floor_quant(1e-8, 1e-8)
        lr_scheduler.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            acc = trainer.evaluate(test_loader)
            batches = len(train_loader)
            print(
                f"epoch {epoch:3d}  loss {ce_sum / batches:.4f}  "
                f"Hw {hw_sum / batches:.6g}  Hx {hx_sum / batches:.6g}  "
                f"acc {acc * 100:.2f}  {time.perf_counter() - tick:.0f}s",
                flush=True,
            )
