from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .algorithms import ALGORITHMS
from .data import get_dataset
from .model import ThreeLayerMLP
from .utils import default_hidden_dim, default_lr, ensure_dir, evaluate, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Level-1 qualitative reproduction of Figure 13.1 without New 2."
    )
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sgd", "spider", "sarah", "storm", "page"],
        required=True,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-passes", type=float, default=10.0)
    parser.add_argument("--eval-interval-passes", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    # Variance-reduced algorithm knobs.
    parser.add_argument(
        "--refresh-period",
        type=int,
        default=100,
        help="Full-gradient refresh period for SPIDER/SARAH, measured in algorithm steps.",
    )
    parser.add_argument(
        "--page-p",
        type=float,
        default=0.05,
        help="Full-gradient refresh probability for PAGE.",
    )
    parser.add_argument(
        "--storm-beta",
        type=float,
        default=0.10,
        help="STORM recursive estimator beta parameter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    hidden_dim = args.hidden_dim or default_hidden_dim(args.dataset)
    lr = args.lr if args.lr is not None else default_lr(args.dataset, args.algorithm)

    bundle = get_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    model = ThreeLayerMLP(
        input_dim=bundle.input_dim,
        hidden_dim=hidden_dim,
        num_classes=bundle.num_classes,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()

    algorithm_cls = ALGORITHMS[args.algorithm]
    algorithm = algorithm_cls(
        model=model,
        loss_fn=loss_fn,
        train_loader=bundle.train_loader,
        full_loader=bundle.train_eval_loader,
        device=device,
        lr=lr,
        n_train=bundle.n_train,
        batch_size=args.batch_size,
        refresh_period=args.refresh_period,
        page_p=args.page_p,
        storm_beta=args.storm_beta,
    )

    output_dir = ensure_dir(args.output_dir)
    out_file = output_dir / f"{args.dataset}_{args.algorithm}_seed{args.seed}.csv"

    records = []
    sample_count = 0
    step_count = 0
    max_samples = int(args.max_passes * bundle.n_train)
    next_eval_at = 0.0
    start_time = time.time()

    def log_metrics(step_type: str, loss_estimate: float | None = None) -> None:
        train_loss, train_acc = evaluate(model, loss_fn, bundle.train_eval_loader, device)
        test_loss, test_acc = evaluate(model, loss_fn, bundle.test_loader, device)
        records.append(
            {
                "dataset": args.dataset,
                "algorithm": algorithm_cls.name,
                "seed": args.seed,
                "step": step_count,
                "sample_count": sample_count,
                "samples_over_ntrain": sample_count / bundle.n_train,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "last_step_type": step_type,
                "last_loss_estimate": loss_estimate,
                "lr": lr,
                "batch_size": args.batch_size,
                "hidden_dim": hidden_dim,
                "refresh_period": args.refresh_period,
                "page_p": args.page_p,
                "storm_beta": args.storm_beta,
                "elapsed_sec": time.time() - start_time,
            }
        )
        pd.DataFrame(records).to_csv(out_file, index=False)

    print(
        f"Running dataset={args.dataset}, algorithm={algorithm_cls.name}, "
        f"lr={lr}, hidden_dim={hidden_dim}, max_passes={args.max_passes}, device={device}"
    )

    log_metrics(step_type="initial", loss_estimate=None)
    next_eval_at = args.eval_interval_passes

    pbar = tqdm(total=args.max_passes, desc=f"{args.dataset}/{algorithm_cls.name}")
    current_passes = 0.0
    while sample_count < max_samples:
        result = algorithm.step()
        step_count += 1
        sample_count += result.samples_used
        new_passes = min(sample_count / bundle.n_train, args.max_passes)
        pbar.update(max(0.0, new_passes - current_passes))
        current_passes = new_passes

        if sample_count / bundle.n_train >= next_eval_at or sample_count >= max_samples:
            log_metrics(step_type=result.step_type, loss_estimate=result.loss_estimate)
            while next_eval_at <= sample_count / bundle.n_train:
                next_eval_at += args.eval_interval_passes

    pbar.close()
    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
