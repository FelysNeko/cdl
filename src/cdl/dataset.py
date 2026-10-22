from collections.abc import Iterable
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def get_loaders(
    batch_size: int,
    num_calib_batches: int,
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
    num_calib_batches = (x for x, _ in islice(train_loader, num_calib_batches))
    return train_loader, test_loader, num_calib_batches
