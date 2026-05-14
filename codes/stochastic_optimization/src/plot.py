from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ALGORITHM_ORDER = ["SGD", "SPIDER", "SARAH", "STORM", "PAGE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 13.1 level-1 reproduction.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/figure13_level1_reproduction.pdf")
    parser.add_argument("--max-passes", type=float, default=None)
    return parser.parse_args()


def read_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            required = {"dataset", "algorithm", "samples_over_ntrain", "train_loss", "test_accuracy"}
            if required.issubset(df.columns):
                frames.append(df)
        except Exception as exc:
            print(f"Skipping {f}: {exc}")
    if not frames:
        raise RuntimeError("No valid result CSV files found.")
    return pd.concat(frames, ignore_index=True)


def plot_panel(ax, df: pd.DataFrame, dataset: str, y_col: str, ylabel: str, title: str) -> None:
    sub = df[df["dataset"] == dataset].copy()
    for alg in ALGORITHM_ORDER:
        alg_df = sub[sub["algorithm"] == alg].sort_values("samples_over_ntrain")
        if alg_df.empty:
            continue
        # Average over seeds if multiple runs exist at identical x values.
        alg_df = (
            alg_df.groupby("samples_over_ntrain", as_index=False)[y_col]
            .mean()
            .sort_values("samples_over_ntrain")
        )
        ax.plot(alg_df["samples_over_ntrain"], alg_df[y_col], label=alg, linewidth=1.7)
    ax.set_xlabel(r"#samples / $n_{\mathrm{train}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = read_results(results_dir)
    if args.max_passes is not None:
        df = df[df["samples_over_ntrain"] <= args.max_passes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    plot_panel(
        axes[0, 0], df, dataset="mnist", y_col="train_loss", ylabel="Training Loss", title="(a) MNIST"
    )
    plot_panel(
        axes[0, 1], df, dataset="cifar10", y_col="train_loss", ylabel="Training Loss", title="(b) CIFAR-10"
    )
    plot_panel(
        axes[1, 0], df, dataset="mnist", y_col="test_accuracy", ylabel="Test Accuracy", title="(c) MNIST"
    )
    plot_panel(
        axes[1, 1], df, dataset="cifar10", y_col="test_accuracy", ylabel="Test Accuracy", title="(d) CIFAR-10"
    )

    for ax in axes.ravel():
        ax.legend(frameon=True)

    fig.suptitle("Qualitative reproduction of Figure 13.1 without New 2", y=0.995, fontsize=14)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
