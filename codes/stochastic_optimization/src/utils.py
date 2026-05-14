from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_param_vector(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector([p.detach() for p in model.parameters()]).detach().clone()


def set_param_vector(model: nn.Module, vector: torch.Tensor) -> None:
    vector_to_parameters(vector.detach(), model.parameters())


def zero_grad(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def grad_vector_on_batch(
    model: nn.Module,
    loss_fn: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> Tuple[float, torch.Tensor, int]:
    """Return mean loss, full parameter-gradient vector, and batch size."""
    model.train()
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    zero_grad(model)
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    grad_vec = torch.cat(grads).detach().clone()
    return float(loss.item()), grad_vec, int(data.size(0))


def grad_vector_at_params_on_batch(
    model: nn.Module,
    params: torch.Tensor,
    loss_fn: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> Tuple[float, torch.Tensor, int]:
    """Temporarily evaluate a batch gradient at a supplied parameter vector."""
    current = get_param_vector(model)
    set_param_vector(model, params)
    loss, grad, batch_size = grad_vector_on_batch(model, loss_fn, data, target, device)
    set_param_vector(model, current)
    return loss, grad, batch_size


def full_gradient(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Tuple[float, torch.Tensor, int]:
    """Compute the exact empirical full gradient over a loader.

    The returned gradient is the average gradient of the mean per-sample loss over
    the whole training set.
    """
    total_loss = 0.0
    total_n = 0
    grad_sum: Optional[torch.Tensor] = None

    for data, target in loader:
        loss, grad, batch_size = grad_vector_on_batch(model, loss_fn, data, target, device)
        if grad_sum is None:
            grad_sum = torch.zeros_like(grad)
        grad_sum.add_(grad, alpha=batch_size)
        total_loss += loss * batch_size
        total_n += batch_size

    assert grad_sum is not None
    return total_loss / total_n, grad_sum / total_n, total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(data)
        loss = loss_fn(logits, target)
        total_loss += float(loss.item()) * data.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == target).sum().item())
        total_n += int(data.size(0))
    return total_loss / total_n, total_correct / total_n


def default_hidden_dim(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset == "mnist":
        return 256
    if dataset == "cifar10":
        return 512
    raise ValueError(dataset)


def default_lr(dataset: str, algorithm: str) -> float:
    """Simple robust defaults for qualitative experiments.

    These are not paper-tuned hyperparameters. Adjust them if curves are unstable.
    """
    dataset = dataset.lower()
    algorithm = algorithm.lower()
    if dataset == "mnist":
        return {
            "sgd": 0.08,
            "storm": 0.04,
            "page": 0.04,
            "spider": 0.035,
            "sarah": 0.035,
        }[algorithm]
    if dataset == "cifar10":
        return {
            "sgd": 0.02,
            "storm": 0.01,
            "page": 0.01,
            "spider": 0.008,
            "sarah": 0.008,
        }[algorithm]
    raise ValueError(dataset)
