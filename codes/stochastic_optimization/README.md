# Qualitative reproduction of Figure 13.1 without New 2

This is a small PyTorch project for **level-1 / qualitative reproduction** of Figure 13.1-style experiments.
It compares

- SGD
- SPIDER
- SARAH
- STORM
- PAGE

on MNIST and CIFAR-10 using a three-layer MLP. The goal is not exact numerical replication of the paper, but to reproduce the main qualitative behavior under a unified sample-complexity axis.

The x-axis is

```text
# stochastic-gradient samples used / n_train
```

For variance-reduced methods, a full gradient costs `n_train`; a recursive mini-batch gradient difference costs `2 * batch_size`, because it evaluates gradients at both the current and previous parameter vectors on the same mini-batch.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke test

```bash
python -m src.train --dataset mnist --algorithm sgd --max-passes 0.2 --device cpu --output-dir results
```

## Run all algorithms

```bash
bash scripts/run_all.sh
```

By default the script runs both MNIST and CIFAR-10 for 10 effective passes. On CPU this can take a while. For a quick check, edit `MAX_PASSES` in `scripts/run_all.sh` to `1` or `2`.

## Plot

```bash
python -m src.plot --results-dir results --output results/figure13_level1_reproduction.pdf
```

This generates a four-panel plot:

1. MNIST training loss
2. CIFAR-10 training loss
3. MNIST test accuracy
4. CIFAR-10 test accuracy

## Notes

- `New 2` is deliberately omitted.
- Hyperparameters are simple defaults, not paper-tuned settings.
- The implementation prioritizes clarity and consistent sample counting over speed.
- The recursive estimators are implemented with parameter-vector swapping. This is easy to read but slower than highly optimized implementations.
