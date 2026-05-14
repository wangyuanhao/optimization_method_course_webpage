from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    test_loader: DataLoader
    n_train: int
    input_dim: int
    num_classes: int


class InfiniteLoader:
    """Simple endless wrapper around a PyTorch DataLoader."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def get_dataset(
    dataset: str,
    data_dir: str = "data",
    batch_size: int = 256,
    eval_batch_size: int = 1024,
    num_workers: int = 2,
) -> DatasetBundle:
    dataset = dataset.lower()

    if dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        input_dim = 28 * 28
        num_classes = 10

    elif dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        input_dim = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected 'mnist' or 'cifar10'.")

    # Mini-batch loader for stochastic-gradient steps.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Deterministic loaders for full-gradient computation and metric evaluation.
    train_eval_loader = DataLoader(
        train_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return DatasetBundle(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        n_train=len(train_set),
        input_dim=input_dim,
        num_classes=num_classes,
    )
