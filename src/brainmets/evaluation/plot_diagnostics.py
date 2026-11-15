"""Diagnostic plots for a binary model, from pooled out-of-fold predictions
(oof_predictions.csv, written by :mod:`brainmets.evaluation.export_val_predictions`).

Writes PNGs + prints summary stats:
  - roc.png         : per-fold ROC (faint) + pooled OOF ROC, AUC with bootstrap 95% CI
  - pr.png          : precision-recall curve + average precision
  - calibration.png : reliability diagram (10 bins) + ECE / Brier
  - confusion.png   : confusion matrices at threshold 0.5 and at Youden's J
  - score_hist.png  : predicted-probability histograms by true class

Usage:
    python -m brainmets.evaluation.plot_diagnostics
    python -m brainmets.evaluation.plot_diagnostics --csv <oof_predictions.csv> --outdir <dir> \
        --label "T1+T2+FLAIR CNN"
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

from ..training.config import PRESETS

POS_LABEL = 1  # binary: 1 = not-progression (Stable/Response), 0 = Progression


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                     seed: int = 1) -> tuple[float, float]:
    """Bootstrap percentile interval for AUROC.

    Resamples that end up single-class are skipped, since AUROC is undefined there.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def ece_and_brier(y: np.ndarray, p: np.ndarray, n_bins: int = 10
                  ) -> tuple[float, float, list[tuple[float, float, int]]]:
    """Expected calibration error and Brier score, plus the reliability-curve points.

    ECE is the bin-count-weighted gap between mean predicted probability and observed
    frequency; the returned points are (mean prediction, observed fraction, n) for each
    non-empty bin, which is what the reliability plot draws.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    pts = []  # (mean_pred, frac_pos, n) per non-empty bin
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.sum() / len(p)
        ece += abs(acc - conf) * w
        pts.append((conf, acc, m.sum()))
    brier = np.mean((p - y) ** 2)
    return ece, brier, pts


def main():
    """Draw ROC and calibration diagnostics from the pooled predictions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="pooled out-of-fold predictions (default: the T1-post binary run's, "
                         "$BM_WORK_ROOT/image_only_models_t1post/oof_predictions.csv)")
    ap.add_argument("--outdir", default=None, help="default: same dir as the csv")
    ap.add_argument("--label", default="T1Post CNN", help="model name used in plot titles")
    args = ap.parse_args()
    csv_path = args.csv or os.path.join(PRESETS["t1post-binary"].output_dir, "oof_predictions.csv")
    outdir = args.outdir or os.path.dirname(csv_path) or "."
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    y = df["y_true"].to_numpy().astype(int)
    p = df["y_prob"].to_numpy().astype(float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"pooled OOF: {len(y)} mets  (pos/not-progression={n_pos}, neg/progression={n_neg})")

    pooled_auc = roc_auc_score(y, p)
    lo, hi = bootstrap_auc_ci(y, p)
    ap_score = average_precision_score(y, p)
    ece, brier, cal_pts = ece_and_brier(y, p)
    print(f"AUROC (pooled OOF): {pooled_auc:.4f}  (95% CI {lo:.4f}-{hi:.4f}, met-level bootstrap)")
    print(f"Average precision : {ap_score:.4f}")
    print(f"ECE: {ece:.4f}   Brier: {brier:.4f}")

    # --- ROC: per-fold faint + pooled bold ---
    plt.figure(figsize=(5, 5))
    for fold, g in df.groupby("fold"):
        yf, pf = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
        if len(np.unique(yf)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yf, pf)
        plt.plot(fpr, tpr, color="gray", alpha=0.35, lw=1,
                 label="per-fold" if fold == df["fold"].min() else None)
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr, color="C0", lw=2.2, label=f"pooled OOF (AUC {pooled_auc:.3f}, CI {lo:.3f}-{hi:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("1 - specificity")
    plt.ylabel("sensitivity")
    plt.title(f"ROC - {args.label} (binary, OOF)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc.png"), dpi=150)
    plt.close()

    # --- Precision-Recall ---
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, color="C1", lw=2, label=f"AP {ap_score:.3f}")
    plt.axhline(n_pos / len(y), ls="--", color="k", alpha=0.5, label=f"prevalence {n_pos/len(y):.3f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.ylim(0, 1.02)
    plt.title(f"Precision-Recall - {args.label} (OOF)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr.png"), dpi=150)
    plt.close()

    # --- Calibration / reliability ---
    plt.figure(figsize=(5, 5))
    if cal_pts:
        xs, ys, _ = zip(*cal_pts)
        plt.plot(xs, ys, "o-", color="C2", label=f"CNN (ECE {ece:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed fraction positive")
    plt.title(f"Calibration - {args.label} (Brier {brier:.3f})")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration.png"), dpi=150)
    plt.close()

    # --- Confusion matrices: 0.5 and Youden ---
    fpr_, tpr_, thr_ = roc_curve(y, p)
    youden = thr_[np.argmax(tpr_ - fpr_)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, thr, name in [(axes[0], 0.5, "threshold 0.5"),
                          (axes[1], youden, f"Youden thr={youden:.3f}*")]:
        cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prog", "notProg"])
        ax.set_yticklabels(["Prog", "notProg"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name, fontsize=9)
    fig.suptitle(f"Confusion - {args.label} (OOF). *Youden picked on val = optimistic; tune on train for reporting.", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion.png"), dpi=150)
    plt.close()

    # --- Score histogram by true class ---
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 26)
    plt.hist(p[y == 0], bins=bins, alpha=0.6, label="Progression (0)", color="C3", density=True)
    plt.hist(p[y == 1], bins=bins, alpha=0.6, label="not-Progression (1)", color="C0", density=True)
    plt.xlabel("predicted P(not-progression)")
    plt.ylabel("density")
    plt.title(f"Score distribution by class - {args.label} (OOF)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_hist.png"), dpi=150)
    plt.close()

    print(f"\nsaved roc.png, pr.png, calibration.png, confusion.png, score_hist.png -> {outdir}")


if __name__ == "__main__":
    main()
