"""Bootstrap 95% confidence intervals for the CNN held-out-TEST metrics.

Reads the ensemble prediction CSVs written by :mod:`brainmets.evaluation.export_test_predictions`
(test_predictions_<seqs>_<task>.csv: binary -> y_prob_ensemble; multiclass -> p0,p1,p2)
and reports the point estimate + percentile-bootstrap 95% CI:
  binary     -> AUROC
  multiclass -> linear-weighted Cohen's κ  and  one-vs-rest macro AUROC

Default resamples METS (matches the met-level reporting convention). Use --cluster to
resample PATIENTS instead (patient-level bootstrap -- more correct for clustered mets).

Usage:
    python -m brainmets.evaluation.bootstrap_test_ci                          # all test_predictions_*.csv under the model dirs
    python -m brainmets.evaluation.bootstrap_test_ci --csv <one.csv>
    python -m brainmets.evaluation.bootstrap_test_ci --cluster --n-boot 5000
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from ..cohort import config as cohort_cfg


def compute(y: np.ndarray, prob: np.ndarray, task: str) -> dict[str, float]:
    """The task's headline metrics: AUROC for binary, kappa and one-vs-rest AUROC for
    multiclass.
    """
    if task == "binary":
        return {"AUROC": roc_auc_score(y, prob)}
    return {"kappa": cohen_kappa_score(y, prob.argmax(1), weights="linear"),
            "AUROC_ovr": roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=[0, 1, 2])}


def bootstrap(df: pd.DataFrame, task: str, n_boot: int, cluster: bool, seed: int
              ) -> tuple[dict[str, float], dict[str, np.ndarray], int]:
    """Bootstrap percentile intervals for the test metrics.

    With `cluster` set, resampling is by PATIENT rather than by metastasis. That matters
    here: one patient contributes several lesions whose outcomes are correlated, so
    resampling lesions independently treats them as independent evidence and reports an
    interval that is too narrow.
    """
    rng = np.random.default_rng(seed)
    y = df["y_true"].to_numpy().astype(int)
    prob = df["y_prob_ensemble"].to_numpy() if task == "binary" else df[["p0", "p1", "p2"]].to_numpy()
    n_classes = 2 if task == "binary" else 3

    point = compute(y, prob, task)
    if cluster:
        groups = df["patient_id"].astype(str).to_numpy()
        pats = np.unique(groups)
        pat_idx = {p: np.where(groups == p)[0] for p in pats}

    boots = {k: [] for k in point}
    for _ in range(n_boot):
        if cluster:
            sampled = rng.choice(pats, size=len(pats), replace=True)
            idx = np.concatenate([pat_idx[p] for p in sampled])
        else:
            idx = rng.choice(len(y), size=len(y), replace=True)
        ys, ps = y[idx], prob[idx]
        if len(np.unique(ys)) < n_classes:      # need every class present to score
            continue
        try:
            for k, v in compute(ys, ps, task).items():
                boots[k].append(v)
        except ValueError:
            continue
    ci = {k: np.percentile(boots[k], [2.5, 97.5]) for k in boots}
    n_valid = len(next(iter(boots.values())))
    return point, ci, n_valid


def main():
    """Print point estimates and bootstrap intervals for the held-out test set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="one CSV; default = all test_predictions_*.csv under model dirs")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cluster", action="store_true", help="resample patients (patient-level bootstrap)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    files = [args.csv] if args.csv else sorted(glob.glob(
        os.path.join(cohort_cfg.work_root(), "image_only_models*", "test_predictions_*.csv")))
    if not files:
        raise SystemExit("No test_predictions_*.csv found -- run "
                         "brainmets.evaluation.export_test_predictions first.")

    level = "patient-level" if args.cluster else "met-level"
    print(f"\n{args.n_boot}x {level} bootstrap, 95% CI:\n")
    for f in files:
        df = pd.read_csv(f)
        task = "binary" if "binary" in os.path.basename(f) else "multiclass"
        point, ci, nvalid = bootstrap(df, task, args.n_boot, args.cluster, args.seed)
        name = os.path.basename(f)
        parts = [f"{k} {point[k]:.4f} [{ci[k][0]:.4f}, {ci[k][1]:.4f}]" for k in point]
        print(f"{name:44s} n={len(df):4d}  " + " | ".join(parts))
    print()


if __name__ == "__main__":
    main()
