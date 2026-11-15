"""Cohort loading primitives — the single definition of "which rows count".

The training pipeline's exclusion criteria, fold split and patient/round parsing were
previously re-implemented in every analysis script. They live here once, as small
composable steps, so a check can either take the whole assembled frame
(:func:`load_courses`) or drive the steps itself when it needs to measure something
about the assembly (as :func:`cohort.checks.verify_course_identity` does with the
clinical merge).
"""
import os
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config as cfg


def load_folds(path: str | None = None) -> pd.DataFrame:
    """Read the raw cohort/folds table (one row per metastasis)."""
    return pd.read_csv(path or cfg.folds_csv(), index_col=0)


def prune(df: pd.DataFrame, lower: int | None = None,
          verbose: bool = False) -> pd.DataFrame:
    """Apply the trainer's exclusion criteria, in the trainer's order.

    Mirrors what the trainer applies (see :mod:`training.cohort_input`): drop the wrong ICI treatment
    types, then require both the pre- and post-treatment metastasis to exceed the
    size threshold.
    """
    lower = cfg.LOWER if lower is None else lower
    n0 = len(df)
    df = df[df["rows_to_remove"] == 0]
    df = df[df["PreTreatment_Met"] > 0]
    df = df[df["PreTreatment_Met"] > lower]
    df = df[df["PostTreatment_Met"] > lower]
    if verbose:
        print(f"[prune] {n0} rows -> {len(df)} mets")
    return df


def require_sequences(df: pd.DataFrame, seqs: Iterable[str] | None,
                      verbose: bool = False) -> pd.DataFrame:
    """Keep only mets whose named sequences are present (the multisequence cohort).

    `seqs` accepts the short names in :data:`brainmets.cohort.config.PRESENCE` ('t2',
    'flair') or a literal presence-column name. Sequences in
    :data:`brainmets.cohort.config.ALWAYS_PRESENT` are skipped: T1-post exists for every
    metastasis, so there is no column to filter on and requesting it is a no-op rather
    than an error.
    """
    seqs = list(seqs or ())
    for s in seqs:
        if s in cfg.ALWAYS_PRESENT:
            continue
        col = cfg.PRESENCE.get(s, s)
        if col not in df.columns:
            raise SystemExit(f"!! presence column {col!r} not found. Columns: {list(df.columns)}")
        df = df[df[col] == True]  # noqa: E712 — the column holds True/False, not NaN-safe truthiness
        if verbose:
            print(f"after {col} == True: {len(df)}")
    return df


def parse_require_seqs(value: str | None) -> list[str]:
    """Parse a ``--require-seqs t2,flair`` option into a list."""
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def seq_tag(seqs: Iterable[str]) -> str:
    """Human-readable suffix for a sequence requirement, e.g. ' (t2+flair)'."""
    return f" ({'+'.join(seqs)})" if seqs else ""


def in_cv_mask(df: pd.DataFrame, num_folds: int | None = None) -> pd.Series:
    """True where a met is in train or validation of at least one fold.

    The complement is the held-out test set: rows that are never 0 or 1 in any
    ``Fold_k`` column, so the model never saw them.
    """
    num_folds = cfg.NUM_FOLDS if num_folds is None else num_folds
    fold_cols = [f"Fold_{k}" for k in range(num_folds)]
    return df[fold_cols].isin([0, 1]).any(axis=1)


def split_cv_test(df: pd.DataFrame,
                  num_folds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (development, held-out test) frames."""
    m = in_cv_mask(df, num_folds)
    return df[m], df[~m]


def assign_cohort(df: pd.DataFrame, num_folds: int | None = None) -> pd.DataFrame:
    """Add a ``_cohort`` column of 'dev' / 'test'."""
    df = df.copy()
    df["_cohort"] = np.where(in_cv_mask(df, num_folds), "dev", "test")
    return df


def add_patient_round(df: pd.DataFrame, key: str | None = None) -> pd.DataFrame:
    """Add ``_pid`` and ``_round`` parsed from the image path.

    Parsed EXACTLY as the trainer does: ``os.path.dirname`` strips the filename
    first, THEN ``split('/')`` gives [-3] = patient, [-2] = treatment round. The
    folds CSV only carries a re-indexed ``Patient_Number``; the real patient id
    lives in this path.
    """
    key = key or cfg.MERGE_KEY
    df = df.copy()
    parts = df[key].astype(str).map(os.path.dirname).str.split("/")
    df["_pid"] = parts.map(lambda s: s[-3])
    df["_round"] = parts.map(lambda s: s[-2])
    return df


def round_number(value: object) -> float:
    """Numeric part of a round label ('Round_10' -> 10), so 2 sorts before 10."""
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else np.nan


def load_clinical(path: str | None = None,
                  map_gender: bool = True) -> pd.DataFrame:
    """Read the clinical table, decoding Gender to Male/Female."""
    clin = pd.read_csv(path or cfg.clin_csv())
    if map_gender and "Gender" in clin.columns:
        clin["Gender"] = clin["Gender"].map(lambda g: {0: "Male", 1: "Female"}.get(g, g))
    return clin


def merge_clinical(df: pd.DataFrame, clin: pd.DataFrame,
                   key: str | None = None) -> pd.DataFrame:
    """Left-join the clinical table, suffixing its copies of shared columns.

    The folds CSV carries its own copy of most clinical variables, so the join adds
    a redundant ``<var>_clin`` set. Aggregation reads the UNSUFFIXED column, i.e.
    the folds copy whenever both exist — ``verify_course_identity`` reports which
    file actually supplies each Table 1 variable.
    """
    return df.merge(clin, on=key or cfg.MERGE_KEY, how="left", suffixes=("", "_clin"))


def load_courses(verbose: bool = True, lower: int | None = None,
                 num_folds: int | None = None,
                 require_seqs: Iterable[str] = ()) -> pd.DataFrame:
    """Prune, assign cohort, parse patient/round ids and merge the clinical table.

    Returns the MET-level merged frame sorted by (``_pid``, ``_round``), with helper
    columns ``_cohort``, ``_pid`` and ``_round``. Pass it through :func:`one_row_per_course`
    for one row per treatment course.
    """
    df = prune(load_folds(), lower=lower, verbose=verbose)
    df = require_sequences(df, require_seqs)
    df = add_patient_round(assign_cohort(df, num_folds))
    if verbose:
        for key in cfg.COHORTS:
            sub = df[df["_cohort"] == key]
            print(f"[cohort check] {cfg.COHORT_SHORT[key]}: {sub['_pid'].nunique()} patients, "
                  f"{sub[['_pid', '_round']].drop_duplicates().shape[0]} patient-courses, "
                  f"{len(sub)} mets")
    return merge_clinical(df, load_clinical()).sort_values(["_pid", "_round"])


def one_row_per_course(merged: pd.DataFrame, key: str | None = None) -> pd.DataFrame:
    """Collapse the met-level frame to one row per treatment course."""
    return merged.drop_duplicates(subset=key or cfg.MERGE_KEY, keep="first").copy()
