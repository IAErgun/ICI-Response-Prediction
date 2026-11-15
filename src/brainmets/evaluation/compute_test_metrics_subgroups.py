"""Test-set metrics, overall and within clinical subgroups (the paper's Tables 4 and 5).

INPUT CONTRACT (same schema as export_test_predictions.py):
  binary     CSV: columns  met, patient_id, y_true, y_prob_ensemble
  multiclass CSV: columns  met, patient_id, y_true, p0, p1, p2[, pred]
Point this at the selected model's held-out-test predictions. Works for any model whose
predictions are exported in this schema; the level (lesion vs patient) is whatever the
prediction file contains.

Subgroups are merged from the folds CSV (met size) + clinical CSV, joined to the predictions
on (real patient id, met). That join is clean only while each (patient_id, met) pair is
unique in the held-out test set, i.e. while every test patient contributes one course.

PHI-SAFE: prints ONLY aggregate metrics per subgroup; never individual rows/identifiers.

Metric conventions match bootstrap_test_ci.py: linear-weighted Cohen's κ; one-vs-rest macro
AUROC over labels [0,1,2]. Binary positive class = 1 (stable/response) per the trainer's
class_dict (Progression=0). Continuous subgroups (age, met size, #BM) split at TERTILES.

Usage:
  python -m brainmets.evaluation.compute_test_metrics_subgroups --pred <test_predictions_*.csv> --task binary
  python -m brainmets.evaluation.compute_test_metrics_subgroups --pred <...multiclass.csv>      --task multiclass
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, cohen_kappa_score,
                             accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score)

from ..cohort import config as cfg
from ..cohort import data

MERGE_KEY = cfg.MERGE_KEY
POS = cfg.POS


def binary_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Binary test metrics at a 0.5 threshold, plus the class balance.

    A subgroup can contain only one class once the test set is split finely, and then
    several metrics are undefined rather than zero: AUROC, AUPRC, kappa and balanced
    accuracy need both classes, sensitivity and F1 need a positive, specificity needs a
    negative. Those come back as NaN (printed "n/a"). scikit-learn would otherwise return
    confident-looking values (AUPRC 1.0 or 0.0, specificity 0.0, kappa 0.0) instead of
    raising. For a subgroup containing both classes every value is computed exactly as
    before.
    """
    pred = (prob >= 0.5).astype(int)
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    both = has_pos and has_neg
    out = {"n": len(y), "Progression%": float(np.mean(y == 0))}
    out["AUROC"] = roc_auc_score(y, prob) if both else np.nan
    out["AUPRC"] = average_precision_score(y, prob) if both else np.nan
    out["Accuracy"] = accuracy_score(y, pred)
    out["BalancedAcc"] = balanced_accuracy_score(y, pred) if both else np.nan
    out["Kappa"] = cohen_kappa_score(y, pred) if both else np.nan
    out["Sensitivity"] = recall_score(y, pred, pos_label=1, zero_division=0) if has_pos else np.nan
    out["Specificity"] = recall_score(y, pred, pos_label=0, zero_division=0) if has_neg else np.nan
    out["Precision"] = precision_score(y, pred, pos_label=1, zero_division=0)
    out["F1"] = f1_score(y, pred, pos_label=1, zero_division=0) if has_pos else np.nan
    return out


def multiclass_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Three-class test metrics: per-class prevalence, accuracy, weighted kappa and both
    one-vs-rest and one-vs-one AUROC.

    Kappa and balanced accuracy are NaN for a subgroup containing a single class, where
    they are undefined; AUROC is NaN unless all three classes are present.
    """
    pred = prob.argmax(1)
    several = len(np.unique(y)) > 1
    out = {"n": len(y)}
    for c, name in [(0, "Progression%"), (1, "Stable%"), (2, "Response%")]:
        out[name] = float(np.mean(y == c))
    out["Accuracy"] = accuracy_score(y, pred)
    out["BalancedAcc"] = balanced_accuracy_score(y, pred) if several else np.nan
    out["Kappa_lw"] = cohen_kappa_score(y, pred, weights="linear") if several else np.nan
    for tag, mc in [("AUROC_ovr", "ovr"), ("AUROC_ovo", "ovo")]:
        try:
            out[tag] = roc_auc_score(y, prob, multi_class=mc, average="macro", labels=[0, 1, 2])
        except ValueError:
            out[tag] = np.nan
    return out


def fmt(m: dict[str, float]) -> str:
    """Format a metrics dict as one line: counts as integers, the rest to 3dp, and
    undefined values as "n/a" so they cannot be mistaken for a score.
    """
    parts = []
    for k, v in m.items():
        if k == "n":
            parts.append(f"{k}={v}")
        elif pd.isna(v):
            parts.append(f"{k}=n/a")
        else:
            parts.append(f"{k}={v:.3f}")
    return "  ".join(parts)


def tertile_label(s: pd.Series) -> pd.Series:
    """Cut a numeric column into Low/Medium/High tertiles, with NaN as "Unknown"."""
    s = pd.to_numeric(s, errors="coerce")
    q1, q2 = s.quantile([1/3, 2/3])
    def lab(x):
        if pd.isna(x):
            return "Unknown"
        return "Low" if x <= q1 else ("Medium" if x <= q2 else "High")
    return s.map(lab)


def ici_regimen(row: pd.Series | dict) -> str:
    """Subgroup label for a course's ICI regimen.

    Built from the full agent list in :data:`brainmets.cohort.config.ICI`, shared with
    the cohort package.

    This differs from the paper, whose subgroup table used IPI/NIVO/PEMBRO only:
    atezolizumab and durvalumab were too sparse in that cohort to form subgroups, so
    those courses were reported under "Other/none". Here they get their own labels, so
    on the paper's data the regimen subgroups will not match the published table
    exactly. The paper's three agents are kept as ``cohort.config.BASE3``.
    """
    got = [d for d in cfg.ICI if row.get(d) in POS]
    return {"IPI+NIVO": "Ipilimumab + nivolumab"}.get("+".join(got), "+".join(got)) or "Other/none"


def build_subgroup_frame() -> pd.DataFrame:
    # Exclusions and the fold split come from the cohort package, so this scores exactly
    # the rows the trainer held out.
    """The held-out test metastases joined to the clinical variables used for subgrouping."""
    df = data.prune(data.load_folds())
    _, df = data.split_cv_test(df)
    df = data.add_patient_round(df).rename(columns={"_pid": "patient_id"})
    df["met"] = df["Metastasis_ID"].astype(int) if "Metastasis_ID" in df.columns else np.nan
    clin = data.load_clinical(map_gender=False)
    if "Gender" in clin.columns:
        clin["Gender"] = clin["Gender"].map(lambda g: {0: "Male", 1: "Female"}.get(g, g))
    df = df.merge(clin, on=MERGE_KEY, how="left", suffixes=("", "_clin"))
    df["Sex"] = df.get("Gender")
    df["Histology"] = df.get("Histology_New")
    df["AgeGroup"] = tertile_label(df.get("Age_at_Treatment"))
    df["VolumeGroup"] = tertile_label(df.get("PreTreatment_Met"))       # lesion size
    df["NumMetsGroup"] = tertile_label(df.get("Number_of_BM"))
    df["ICIregimen"] = df.apply(ici_regimen, axis=1)
    return df[["patient_id", "met", "Sex", "Histology", "AgeGroup",
               "VolumeGroup", "NumMetsGroup", "ICIregimen"]]


def main():
    """Report test metrics overall and within each clinical subgroup."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred)
    pred["patient_id"] = pred["patient_id"].astype(str)
    metric = binary_metrics if args.task == "binary" else multiclass_metrics

    def probs(d):
        return d["y_prob_ensemble"].to_numpy() if args.task == "binary" \
            else d[["p0", "p1", "p2"]].to_numpy()

    print(f"\n### {args.task} — {os.path.basename(args.pred)}  (n={len(pred)}) ###")
    print("Full Dataset:", fmt(metric(pred["y_true"].to_numpy().astype(int), probs(pred))))

    sg = build_subgroup_frame()
    if "met" in pred.columns and sg["met"].notna().any():
        merged = pred.merge(sg, on=["patient_id", "met"], how="left")
    else:
        merged = pred.merge(sg.drop_duplicates("patient_id"), on="patient_id", how="left")
    unmatched = merged["Sex"].isna().mean()
    if unmatched > 0.02:
        print(f"  !! {unmatched:.0%} of prediction rows did not match a subgroup row — check the join keys.")

    for col, title in [("Sex", "Sex"), ("Histology", "Primary tumor histology"),
                       ("AgeGroup", "Age (tertiles)"), ("ICIregimen", "ICI regimen"),
                       ("VolumeGroup", "Lesion size (tertiles)"),
                       ("NumMetsGroup", "No. of brain mets (tertiles)")]:
        print(f"\n-- {title} --")
        for val, g in merged.groupby(col):
            y = g["y_true"].to_numpy().astype(int)
            print(f"  {str(val):24s} {fmt(metric(y, probs(g)))}")


if __name__ == "__main__":
    main()
