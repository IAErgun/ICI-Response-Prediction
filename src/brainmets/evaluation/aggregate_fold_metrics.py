"""Aggregate per-fold train/val metrics from the TensorBoard logs written by
`python -m brainmets.training run` and print a per-fold table plus mean +/- SD.

The training script logs metrics only as TensorBoard scalars (one timestamped
event dir per fold). This reads them back and summarizes the cross-validation
result, reporting the value of the epoch that the trainer would actually have
SAVED -- replaying its exact checkpoint-selection rule rather than the naive
peak. If a fold was re-run, the most recent event dir (by mtime) is used.

Trainer rule (training/engine.py):
    if (key_metric_train > key_metric_train_limit) & (key_metric_val >= best_metric):
        best_metric = key_metric_val; best_metric_epoch = epoch; save()
    best_metric initialized to -1; ties update to the LATEST epoch (>=).
    key_metric_train_limit = 0.50 (binary) / 0.0 (multiclass).
    Binary key metric = ROC_AUC; multiclass key metric = Cohen_Kappa.

Usage:
    python -m brainmets.evaluation.aggregate_fold_metrics                 # the T1-post binary run
    python -m brainmets.evaluation.aggregate_fold_metrics --prefix Fold_MultiClass_
    python -m brainmets.evaluation.aggregate_fold_metrics \
        --logdir "$BM_WORK_ROOT/image_only_models_t1post_t2_flair"        # another preset
    python -m brainmets.evaluation.aggregate_fold_metrics --train-limit 0.5   # override the gate

``--logdir`` defaults to where the T1-post presets write,
``$BM_WORK_ROOT/image_only_models_t1post``.
"""
import os
import re
import glob
import argparse

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ..training.config import PRESETS

# train_key_metric_limit defaults, matching the two train() call sites.
DEFAULT_TRAIN_LIMIT = {"binary": 0.50, "multiclass": 0.0}

# Scalar tags written by train(). Note the val-kappa tag lives under "Validation/"
# while val acc/auc live under "Val/" (matches the inconsistency in the trainer).
TAGS = {
    "train_acc": "Train/ACC",
    "val_acc": "Val/ACC",
    "train_auc": "Train/ROC_AUC",      # binary only
    "val_auc": "Val/ROC_AUC",          # binary only
    "train_kappa": "Train/Cohen_Kappa",
    "val_kappa": "Validation/Cohen_Kappa",
}


def latest_dir_per_fold(logdir: str, prefix: str) -> dict[int, str]:
    """Map fold number -> most recent event dir for that fold."""
    pattern = os.path.join(logdir, f"network_image_{prefix}*")
    fold_re = re.compile(rf"network_image_{re.escape(prefix)}(\d+)_")
    by_fold = {}
    for d in glob.glob(pattern):
        if not os.path.isdir(d):
            continue
        m = fold_re.search(os.path.basename(d))
        if not m:
            continue
        fold = int(m.group(1))
        if fold not in by_fold or os.path.getmtime(d) > os.path.getmtime(by_fold[fold]):
            by_fold[fold] = d
    return dict(sorted(by_fold.items()))


def scalars(ea: EventAccumulator, tag: str) -> list[float]:
    """Return list of values for a tag, or [] if the tag is absent."""
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [s.value for s in ea.Scalars(tag)]


def summarize_fold(event_dir: str, train_limit_override: float | None = None) -> dict | None:
    """Read one fold's TensorBoard scalars and pick out its key metric.

    Binary runs are scored on ROC AUC and multiclass on linear-weighted Cohen's kappa;
    which one applies is inferred from whether a val_auc series exists. Reports the best
    validation value subject to the train-metric floor, so a fold that never cleared the
    floor is reported as having no usable checkpoint.
    """
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()
    vals = {k: scalars(ea, t) for k, t in TAGS.items()}

    # Binary uses ROC_AUC as the key metric, multiclass uses Cohen_Kappa.
    is_binary = len(vals["val_auc"]) > 0
    if is_binary:
        train_series, val_series = vals["train_auc"], vals["val_auc"]
        limit = DEFAULT_TRAIN_LIMIT["binary"]
    else:
        train_series, val_series = vals["train_kappa"], vals["val_kappa"]
        limit = DEFAULT_TRAIN_LIMIT["multiclass"]
    if train_limit_override is not None:
        limit = train_limit_override

    n = min(len(train_series), len(val_series))
    if n == 0:
        return None  # nothing logged yet

    # Replay the trainer's checkpoint-selection rule exactly:
    #   save iff (train_metric > limit) AND (val_metric >= best_metric_so_far)
    # best_metric starts at -1; '>=' makes ties resolve to the LATEST epoch.
    best_metric = -1.0
    best_metric_epoch = -1  # 1-indexed, -1 means "never saved"
    for epoch in range(n):
        if (train_series[epoch] > limit) and (val_series[epoch] >= best_metric):
            best_metric = val_series[epoch]
            best_metric_epoch = epoch + 1

    saved = best_metric_epoch != -1
    best_idx0 = best_metric_epoch - 1 if saved else None

    def at(series, idx0):
        return series[idx0] if (idx0 is not None and 0 <= idx0 < len(series)) else float("nan")

    return {
        "is_binary": is_binary,
        "epochs": n,
        "limit": limit,
        "saved": saved,
        "ckpt_epoch": best_metric_epoch if saved else None,
        "ckpt_val_metric": best_metric if saved else float("nan"),
        "val_auc_at_ckpt": at(vals["val_auc"], best_idx0),
        "val_acc_at_ckpt": at(vals["val_acc"], best_idx0),
        "val_kappa_at_ckpt": at(vals["val_kappa"], best_idx0),
        "last_val_metric": val_series[-1],
        "last_val_acc": vals["val_acc"][-1] if vals["val_acc"] else float("nan"),
        "last_train_metric": train_series[-1] if train_series else float("nan"),
    }


def main():
    """Pool the per-fold metrics into one cross-validation table."""
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=None,
                   help="a run's output directory (default: the T1-post presets', "
                        "$BM_WORK_ROOT/image_only_models_t1post)")
    p.add_argument("--prefix", default="Fold_Binary_",
                   help="fold name prefix, e.g. Fold_Binary_ or Fold_MultiClass_")
    p.add_argument("--train-limit", type=float, default=None,
                   help="override key_metric_train_limit (default: 0.50 binary / 0.0 multiclass)")
    args = p.parse_args()
    # Resolved here rather than at import, so --logdir works without $BM_WORK_ROOT set.
    logdir = args.logdir or PRESETS["t1post-binary"].output_dir

    folds = latest_dir_per_fold(logdir, args.prefix)
    if not folds:
        raise SystemExit(f"No event dirs matching network_image_{args.prefix}* under {logdir}")

    metric_name = None
    rows = []
    print(f"\n{args.prefix} — reading {len(folds)} fold(s) from {logdir}")
    print("Reporting the checkpoint-selected epoch (trainer's save rule replayed).\n")
    header = (f"{'fold':>4} {'epochs':>7} {'ckpt_ep':>8} {'val_auroc':>10} "
              f"{'val_acc':>8} {'ckpt_val':>9} {'last_val':>9} {'last_train':>11}")
    print(header)
    print("-" * len(header))
    for fold, d in folds.items():
        s = summarize_fold(d, train_limit_override=args.train_limit)
        if s is None:
            print(f"{fold:>4}   (no scalars logged yet)")
            continue
        metric_name = "ROC_AUC" if s["is_binary"] else "Cohen_Kappa"
        if not s["saved"]:
            print(f"{fold:>4} {s['epochs']:>7} {'NONE':>8} "
                  f"(no checkpoint: train {metric_name} never exceeded {s['limit']})")
            continue
        rows.append(s)
        print(f"{fold:>4} {s['epochs']:>7} {s['ckpt_epoch']:>8} {s['val_auc_at_ckpt']:>10.4f} "
              f"{s['val_acc_at_ckpt']:>8.4f} {s['ckpt_val_metric']:>9.4f} "
              f"{s['last_val_metric']:>9.4f} {s['last_train_metric']:>11.4f}")

    if rows:
        aurocs = np.array([r["val_auc_at_ckpt"] for r in rows])
        accs = np.array([r["val_acc_at_ckpt"] for r in rows])
        key = np.array([r["ckpt_val_metric"] for r in rows])
        print("-" * len(header))
        print(f"\nMetrics at the checkpoint-selected epoch (n={len(rows)} saved fold(s)):")
        if not np.all(np.isnan(aurocs)):
            print(f"  Val AUROC   : {np.nanmean(aurocs):.4f} +/- {np.nanstd(aurocs):.4f}   "
                  f"per-fold {[round(float(x), 4) for x in aurocs]}")
        print(f"  Val ACC     : {accs.mean():.4f} +/- {accs.std():.4f}   "
              f"per-fold {[round(float(x), 4) for x in accs]}")
        print(f"  Key ({metric_name}): {key.mean():.4f} +/- {key.std():.4f}")
        print("\nNote: matches the trainer's saved checkpoints (train>limit gate + latest-wins "
              "tie-break). This is mean-of-folds, not a pooled out-of-fold metric.\n")


if __name__ == "__main__":
    main()
