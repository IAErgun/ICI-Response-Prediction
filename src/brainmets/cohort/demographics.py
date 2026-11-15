"""Patient-level demographic aggregation for Table 1.

PHI-SAFE: every function here emits ONLY aggregates (N, median [IQR], n (%),
category counts). None of them print or return an individual patient row or
identifier.

Cohort rule (matches the trainer): a met is DEVELOPMENT if any ``Fold_k`` is in
{0, 1}; otherwise it is held-out test. The unit of Table 1 is the treatment course
(one baseline scan), deduplicated by :func:`cohort.data.one_row_per_course`; see
:mod:`cohort.table1` for the one-course-per-patient rebuild.
"""
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config as cfg
from . import data


def med_iqr(s: pd.Series) -> str:
    """``median [Q1–Q3]`` of a numeric column, or 'n/a' if nothing is numeric."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "n/a"
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f"{q2:.1f} [{q1:.1f}–{q3:.1f}]"


def cat_counts(s: pd.Series) -> tuple[str, int]:
    """Indented ``value: n (%)`` lines for a categorical column, plus its N."""
    s = s.dropna()
    n = len(s)
    body = "\n".join(f"    {val}: {c} ({100 * c / n:.1f}%)" for val, c in s.value_counts().items())
    return body, n


def regimen(row: Mapping, cols: Iterable[str] | None = None,
            empty: str = "None/Other") -> str:
    """Join the agent columns flagged positive for one row into a regimen label."""
    cols = cfg.ICI if cols is None else cols
    got = [c for c in cols if row.get(c) in cfg.POS]
    return " + ".join(got) if got else empty


def agent_columns(clin: pd.DataFrame) -> list[str]:
    """Agent-like columns present in the clinical CSV, by name token."""
    return [c for c in clin.columns
            if any(t in str(c).upper() for t in cfg.AGENT_TOKENS)]


def summarize(df: pd.DataFrame, label: str) -> None:
    """Print the Table 1 block for one cohort."""
    print(f"\n{'=' * 60}\n{label}   (N patients = {len(df)})\n{'=' * 60}")
    print("-- Continuous: median [IQR] --")
    for c in cfg.CONT:
        if c in df.columns:
            miss = int(pd.to_numeric(df[c], errors="coerce").isna().sum())
            print(f"  {c:28s} {med_iqr(df[c])}   (missing {miss})")
        else:
            print(f"  {c:28s} COLUMN NOT FOUND")
    print("-- Categorical: n (%) --")
    for c in cfg.CAT:
        if c in df.columns:
            body, n = cat_counts(df[c])
            print(f"  {c} (n={n}):\n{body}")
        else:
            print(f"  {c}: COLUMN NOT FOUND")
    print("-- ICI regimen (from the ICI agent flags): n (%) --")
    if all(d in df.columns for d in cfg.ICI):
        reg = df.apply(regimen, axis=1)
        n = len(reg)
        for val, c in reg.value_counts().items():
            print(f"    {val}: {c} ({100 * c / n:.1f}%)")
    else:
        print("    ICI flag column(s) missing:", [d for d in cfg.ICI if d not in df.columns])
    print("-- Individual ICI agent positivity: n (%) --")
    for d in cfg.ICI:
        if d in df.columns:
            pos = int(df[d].isin(cfg.POS).sum())
            print(f"    {d}: {pos} ({100 * pos / len(df):.1f}%)")


def table1_cells(kept: pd.DataFrame) -> dict:
    """Compact ``{(cohort, variable, level): value}`` view of the Table 1 aggregates.

    Used to diff two cohort selections cell by cell — see
    :func:`cohort.checks.sensitivity_earliest_course_key`.
    """
    out = {}
    for cohort in cfg.COHORTS:
        sub = kept[kept["_cohort"] == cohort]
        if not len(sub):
            continue
        for v in cfg.CONT:
            if v in sub.columns:
                s = pd.to_numeric(sub[v], errors="coerce").dropna()
                if len(s):
                    out[(cohort, v, "median [IQR]")] = (
                        f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]")
        for v in cfg.CAT:
            if v in sub.columns:
                for lvl, k in sub[v].astype(str).value_counts().items():
                    out[(cohort, v, lvl)] = f"{k} ({100 * k / len(sub):.1f}%)"
        for lvl, k in sub.apply(lambda r: regimen(r, empty="NONE"), axis=1).value_counts().items():
            out[(cohort, "ICI regimen", lvl)] = f"{k} ({100 * k / len(sub):.1f}%)"
        for a in cfg.ICI:
            if a in sub.columns:
                k = int(sub[a].isin(cfg.POS).sum())
                out[(cohort, "agent positivity", a)] = f"{k} ({100 * k / len(sub):.1f}%)"
    return out


def report() -> None:
    """Print the Table 1 demographics block for both internal cohorts.

    The unit is the treatment course, matching the paper's patient-level N. Some
    patients contributed more than one course; :mod:`cohort.table1` rebuilds this
    on a strict one-course-per-patient basis.
    """
    courses = data.load_courses()
    for cohort in cfg.COHORTS:
        sub = data.one_row_per_course(courses[courses["_cohort"] == cohort])
        summarize(sub, cfg.COHORT_LABELS[cohort])
    print(cfg.EXTERNAL_NOTE)
