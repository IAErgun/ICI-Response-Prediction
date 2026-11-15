"""Bootstrap 95% CIs for the T1-post vs T1-post+T2+FLAIR comparison.

The comparison is a grid of feature sets x 2 inputs (T1 / ALL=T1+T2+FLAIR) x 2 splits
(cv / test), on the multisequence subset; in the paper it is Table 6. Each cell is a
PATIENT-LEVEL binary AUROC that needs a bootstrap CI. Point this at a manifest
listing the per-cell prediction files, so the exporter's naming does not matter.

INPUT: a manifest CSV with columns:  feature_set, input, split, path
  - feature_set : free text label (e.g. "Clinical + shape + first-order intensity")
  - input       : "T1"  or  "ALL"
  - split       : "cv"  or  "test"
  - path        : path to a PATIENT-LEVEL prediction CSV with columns
                  patient_id, y_true, y_prob_ensemble   (one row per patient-course)

For each file: point AUROC + percentile bootstrap 95% CI (resamples rows; since the file is
patient-level, that is a patient-level bootstrap). Prints the four columns per feature set,
each formatted "0.xxx (lo–hi)", ready to paste into the table.

Usage:  python -m brainmets.evaluation.bootstrap_sequence_comparison_ci --manifest manifest.csv [--n-boot 2000]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EN = "–"
COLS = [("cv", "T1"), ("cv", "ALL"), ("test", "T1"), ("test", "ALL")]
COLNAMES = ["CV T1-post", "CV T1+T2+FLAIR", "Test T1-post", "Test T1+T2+FLAIR"]


def auroc_ci(path: str, n_boot: int, seed: int) -> tuple[str, int] | None:
    """Point AUROC and its bootstrap percentile interval for one predictions file.

    Returns None when the file holds only one class, where AUROC is undefined.
    """
    df = pd.read_csv(path)
    y = df["y_true"].to_numpy().astype(int)
    p = df["y_prob_ensemble"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    point = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{point:.3f} ({lo:.3f}{EN}{hi:.3f})", len(y)


def main():
    """Bootstrap confidence intervals for the sequence-comparison grid."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    need = {"feature_set", "input", "split", "path"}
    if not need.issubset(man.columns):
        raise SystemExit(f"manifest must have columns {need}; got {list(man.columns)}")

    print(f"\n{args.n_boot}x patient-level bootstrap AUROC (95% CI):\n")
    header = f"{'Feature set':52s} " + " ".join(f"{c:22s}" for c in COLNAMES)
    print(header + "\n" + "-" * len(header))
    for feat in man["feature_set"].drop_duplicates():
        cells = []
        for split, inp in COLS:
            row = man[(man.feature_set == feat) & (man.split == split) & (man["input"] == inp)]
            if row.empty:
                cells.append("(missing)")
                continue
            res = auroc_ci(row.path.iloc[0], args.n_boot, args.seed)
            cells.append(res[0] if res else "(one class)")
        print(f"{feat[:52]:52s} " + " ".join(f"{c:22s}" for c in cells))
    print()


if __name__ == "__main__":
    main()
