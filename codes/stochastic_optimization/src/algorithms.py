from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .data import InfiniteLoader
from .utils import (
    full_gradient,
    get_param_vector,
    grad_vector_at_params_on_batch,
    grad_vector_on_batch,
    set_param_vector,
)


@dataclass
class StepResult:
    loss_estimate: float
    samples_used: int
    step_type: str


class BaseAlgorithm:
    """Base class for manual stochastic-gradient algorithms."""

    name: str = "base"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader,
        full_loader,
        device: torch.device,
        lr: float,
        n_train: int,
        batch_size: int,
        refresh_period: int = 100,
        page_p: float = 0.05,
        storm_beta: float = 0.1,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_iter = InfiniteLoader(train_loader)
        self.full_loader = full_loader
        self.device = device
        self.lr = lr
        self.n_train = n_train
        self.batch_size = batch_size
        self.refresh_period = refresh_period
        self.page_p = page_p
        self.storm_beta = storm_beta
        self.t = 0
        self.v: Optional[torch.Tensor] = None
        self.prev_params: Optional[torch.Tensor] = None
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(12345)

    def _update_with_vector(self, direction: torch.Tensor) -> torch.Tensor:
        old_params = get_param_vector(self.model)
        new_params = old_params - self.lr * direction
        set_param_vector(self.model, new_params)
        return old_params

    def step(self) -> StepResult:
        raise NotImplementedError


class SGDAlgorithm(BaseAlgorithm):
    name = "SGD"

    def step(self) -> StepResult:
        data, target = self.train_iter.next()
        loss, grad, batch_size = grad_vector_on_batch(
            self.model, self.loss_fn, data, target, self.device
        )
        self._update_with_vector(grad)
        self.t += 1
        return StepResult(loss_estimate=loss, samples_used=batch_size, step_type="mini_batch")


class SARAHAlgorithm(BaseAlgorithm):
    """SARAH-style recursive estimator.

    At refresh steps, compute the empirical full gradient. Otherwise use
        v_t = v_{t-1} + grad_batch(x_t) - grad_batch(x_{t-1}).

    This implementation is intended for qualitative comparison and clear sample
    counting, not for maximum speed.
    """

    name = "SARAH"

    def step(self) -> StepResult:
        refresh = (self.v is None) or (self.t % self.refresh_period == 0)

        if refresh:
            loss, self.v, n_used = full_gradient(
                self.model, self.loss_fn, self.full_loader, self.device
            )
            self.prev_params = self._update_with_vector(self.v)
            self.t += 1
            return StepResult(loss_estimate=loss, samples_used=n_used, step_type="full_gradient")

        data, target = self.train_iter.next()
        current_params = get_param_vector(self.model)
        old_params = self.prev_params
        assert old_params is not None
        loss_cur, grad_cur, batch_size = grad_vector_on_batch(
            self.model, self.loss_fn, data, target, self.device
        )
        _, grad_old, _ = grad_vector_at_params_on_batch(
            self.model, old_params, self.loss_fn, data, target, self.device
        )
        self.v = self.v + grad_cur - grad_old
        self.prev_params = self._update_with_vector(self.v)
        self.t += 1
        return StepResult(
            loss_estimate=loss_cur,
            samples_used=2 * batch_size,
            step_type="recursive_difference",
        )


class SPIDERAlgorithm(BaseAlgorithm):
    """SPIDER-style recursive estimator.

    For this level-1 reproduction, SPIDER is implemented as a periodically
    refreshed recursive gradient estimator. It is close in code to SARAH, but is
    exposed separately so its refresh period, batch size, and learning rate can
    be tuned independently.
    """

    name = "SPIDER"

    def step(self) -> StepResult:
        refresh = (self.v is None) or (self.t % self.refresh_period == 0)

        if refresh:
            loss, self.v, n_used = full_gradient(
                self.model, self.loss_fn, self.full_loader, self.device
            )
            self.prev_params = self._update_with_vector(self.v)
            self.t += 1
            return StepResult(loss_estimate=loss, samples_used=n_used, step_type="full_gradient")

        data, target = self.train_iter.next()
        current_params = get_param_vector(self.model)
        old_params = self.prev_params
        assert old_params is not None
        loss_cur, grad_cur, batch_size = grad_vector_on_batch(
            self.model, self.loss_fn, data, target, self.device
        )
        _, grad_old, _ = grad_vector_at_params_on_batch(
            self.model, old_params, self.loss_fn, data, target, self.device
        )
        self.v = self.v + grad_cur - grad_old
        self.prev_params = self._update_with_vector(self.v)
        self.t += 1
        return StepResult(
            loss_estimate=loss_cur,
            samples_used=2 * batch_size,
            step_type="recursive_difference",
        )


class PAGEAlgorithm(BaseAlgorithm):
    """PAGE-style probabilistic gradient estimator.

    With probability p, refresh by computing the full gradient. Otherwise update
    recursively:
        v_t = v_{t-1} + grad_batch(x_t) - grad_batch(x_{t-1}).
    """

    name = "PAGE"

    def step(self) -> StepResult:
        if self.v is None:
            do_refresh = True
        else:
            u = torch.rand((), generator=self.rng).item()
            do_refresh = u < self.page_p

        if do_refresh:
            loss, self.v, n_used = full_gradient(
                self.model, self.loss_fn, self.full_loader, self.device
            )
            self.prev_params = self._update_with_vector(self.v)
            self.t += 1
            return StepResult(loss_estimate=loss, samples_used=n_used, step_type="full_gradient")

        data, target = self.train_iter.next()
        old_params = self.prev_params
        assert old_params is not None
        loss_cur, grad_cur, batch_size = grad_vector_on_batch(
            self.model, self.loss_fn, data, target, self.device
        )
        _, grad_old, _ = grad_vector_at_params_on_batch(
            self.model, old_params, self.loss_fn, data, target, self.device
        )
        self.v = self.v + grad_cur - grad_old
        self.prev_params = self._update_with_vector(self.v)
        self.t += 1
        return StepResult(
            loss_estimate=loss_cur,
            samples_used=2 * batch_size,
            step_type="recursive_difference",
        )


class STORMAlgorithm(BaseAlgorithm):
    """STORM-style recursive momentum estimator.

    First step:
        v_0 = grad_batch(x_0)

    Later steps use the same mini-batch at current and previous parameters:
        v_t = grad_batch(x_t) + (1 - beta) * (v_{t-1} - grad_batch(x_{t-1}))

    Then x_{t+1} = x_t - lr * v_t.
    """

    name = "STORM"

    def step(self) -> StepResult:
        data, target = self.train_iter.next()

        if self.v is None:
            loss, grad, batch_size = grad_vector_on_batch(
                self.model, self.loss_fn, data, target, self.device
            )
            self.v = grad
            self.prev_params = self._update_with_vector(self.v)
            self.t += 1
            return StepResult(loss_estimate=loss, samples_used=batch_size, step_type="init")

        old_params = self.prev_params
        assert old_params is not None
        loss_cur, grad_cur, batch_size = grad_vector_on_batch(
            self.model, self.loss_fn, data, target, self.device
        )
        _, grad_old, _ = grad_vector_at_params_on_batch(
            self.model, old_params, self.loss_fn, data, target, self.device
        )
        beta = self.storm_beta
        self.v = grad_cur + (1.0 - beta) * (self.v - grad_old)
        self.prev_params = self._update_with_vector(self.v)
        self.t += 1
        return StepResult(
            loss_estimate=loss_cur,
            samples_used=2 * batch_size,
            step_type="storm_recursive",
        )


ALGORITHMS = {
    "sgd": SGDAlgorithm,
    "sarah": SARAHAlgorithm,
    "spider": SPIDERAlgorithm,
    "page": PAGEAlgorithm,
    "storm": STORMAlgorithm,
}
