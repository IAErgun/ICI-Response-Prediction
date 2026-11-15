"""Summarize and plot the data-efficiency learning curve produced by
`python -m brainmets.training learning-curve`.

Reads learning_curve_results.csv (one row per fold x fraction x repeat) and
reports, per training fraction, the mean +/- SD of the best val metric across
folds/repeats, plus mean n_train. Saves a PNG of metric vs n_train.

Usage:
    python -m brainmets.evaluation.plot_learning_curve
    python -m brainmets.evaluation.plot_learning_curve --csv /path/to/learning_curve_results.csv
"""
import os
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

from ..training.config import LEARNING_CURVE


def main():
    """Plot validation metric against training-set size from the sweep's CSV."""
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None,
                   help="default: where the sweep writes, "
                        "$BM_WORK_ROOT/image_only_models_lc_t1post/learning_curve_results.csv")
    p.add_argument("--out", default=None, help="output PNG path (default: alongside the csv)")
    args = p.parse_args()
    csv_path = args.csv or os.path.join(LEARNING_CURVE.output_dir, "learning_curve_results.csv")

    df = pd.read_csv(csv_path)
    # Drop rows where no checkpoint was saved (best_val_metric == -1 sentinel).
    n_missing = int((df["best_val_metric"] < 0).sum())
    if n_missing:
        print(f"WARNING: {n_missing} run(s) saved no checkpoint (train never cleared the limit); excluded.")
    df = df[df["best_val_metric"] >= 0]
    if df.empty:
        raise SystemExit("No valid rows to summarize.")

    metric = df["metric"].iloc[0]
    # Aggregate across folds (and repeats) at each fraction.
    g = df.groupby("fraction")
    summary = g.agg(
        n_train_mean=("n_train", "mean"),
        metric_mean=("best_val_metric", "mean"),
        metric_sd=("best_val_metric", "std"),
        n_runs=("best_val_metric", "size"),
    ).reset_index().sort_values("fraction")
    summary["metric_sd"] = summary["metric_sd"].fillna(0.0)

    print(f"\nLearning curve — val {metric} (best epoch), aggregated across folds/repeats:\n")
    print(f"{'fraction':>9} {'n_train':>9} {'mean':>8} {'SD':>7} {'runs':>5}")
    print("-" * 42)
    for _, r in summary.iterrows():
        print(f"{r['fraction']:>9.2f} {r['n_train_mean']:>9.0f} "
              f"{r['metric_mean']:>8.4f} {r['metric_sd']:>7.4f} {int(r['n_runs']):>5}")

    # Plot metric vs n_train with SD error bars.
    out = args.out or os.path.join(os.path.dirname(csv_path), "learning_curve.png")
    plt.figure(figsize=(6, 4))
    plt.errorbar(summary["n_train_mean"], summary["metric_mean"], yerr=summary["metric_sd"],
                 marker="o", capsize=4)
    plt.xlabel("# training metastases")
    plt.ylabel(f"val {metric} (best epoch)")
    plt.title("Data-efficiency learning curve (generic CNN, T1Post)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nsaved plot -> {out}\n")


if __name__ == "__main__":
    main()
