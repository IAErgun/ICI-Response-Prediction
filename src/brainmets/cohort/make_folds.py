"""Generate cross-validation fold columns for a cohort table that has none.

The rest of the pipeline expects the cohort table to carry `Fold_0` … `Fold_{k-1}`
columns, which the original dataset arrived with. Anyone bringing their own data has to
produce them, and producing them naively leaks.

Two properties matter more than the split being even:

**Group by patient.** A patient contributes several metastases whose outcomes are
correlated. If one lesion trains and another validates, the model has effectively seen
the answer, and the validation score is optimistic. Every split here is on whole
patients.

**Stratify by label.** Response is imbalanced, and the minority class is small enough
that an unstratified split can leave a fold with almost none of it, making that fold's
metric meaningless.

Both are handled by scikit-learn's `StratifiedGroupKFold`. What this module adds is the
encoding the pipeline expects, a held-out test split carved out before cross-validation,
and assertions that the leakage properties actually hold in the result.

The encoding, which is what the trainer and `cohort.data.in_cv_mask` read:

====================  ==================================================
`Fold_k` value        meaning
====================  ==================================================
0                     train, for fold k
1                     validation, for fold k
2                     held out from every fold (the test set)
====================  ==================================================

So a development metastasis is `1` in exactly one fold and `0` in the others, and a
held-out metastasis is `2` in all of them.

    python -m brainmets.cohort make-folds --in cohort.csv --out cohort_with_folds.csv
"""
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

from . import config as cfg
from . import data

TRAIN, VAL, HELD_OUT = 0, 1, 2
VALID_LABELS = ("Progression", "Stable", "Response")


def resolve_groups(df, group_col=None, from_image_path=False):
    """The patient id each row belongs to, as a Series.

    Defaults to the `Patient_Number` column. `from_image_path` instead parses the real
    patient id out of the image path, exactly as the trainer does — use that when the
    table has no patient column, or when you want the id the rest of the pipeline uses.
    """
    if from_image_path:
        return data.add_patient_round(df)["_pid"]
    col = group_col or cfg.PID_COL
    if col not in df.columns:
        raise SystemExit(
            f"!! no patient column {col!r} in the table. Pass --group-col, or "
            "--group-from-image-path to parse the id out of the image path.\n"
            f"   columns: {list(df.columns)}")
    if df[col].isna().any():
        raise SystemExit(f"!! {col!r} has {int(df[col].isna().sum())} missing value(s); "
                         "every row must belong to a patient or the split can leak.")
    return df[col].astype(str)


def check_labels(df, label_col, drop_unlabelled=False):
    """Return the frame restricted to rows with a usable label.

    A row whose label is not one of the three RANO levels cannot be stratified, and
    would crash the trainer's class lookup if it reached training. Rather than quietly
    park such rows in the test set — which would corrupt the held-out metrics instead —
    this stops, unless dropping them is asked for explicitly.
    """
    bad = ~df[label_col].isin(VALID_LABELS)
    if not bad.any():
        return df
    values = sorted(set(df.loc[bad, label_col].astype(str)))[:8]
    if not drop_unlabelled:
        raise SystemExit(
            f"!! {int(bad.sum())} row(s) have a label outside {VALID_LABELS}: {values}\n"
            "   Fix them, or pass --drop-unlabelled to exclude them from the output.")
    print(f"  dropping {int(bad.sum())} row(s) with an unusable label: {values}")
    return df[~bad]


def _single_stratified_group_split(y, groups, fraction, seed):
    """Indices of one stratified, grouped holdout of about `fraction` of the rows."""
    n_splits = max(2, int(round(1 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(iter(splitter.split(np.zeros(len(y)), y, groups)))
    return held


def assign_folds(df, num_folds=None, test_fraction=0.15, group_col=None,
                 label_col=None, seed=1, from_image_path=False,
                 drop_unlabelled=False):
    """Add `Fold_0` … `Fold_{k-1}` columns to a copy of `df`.

    The held-out test patients are chosen first, stratified and grouped, then the
    remaining patients are split into `num_folds` cross-validation folds the same way.
    Pass ``test_fraction=0`` to use the whole cohort for cross-validation.
    """
    num_folds = cfg.NUM_FOLDS if num_folds is None else num_folds
    label_col = label_col or cfg.LABEL_MET
    if label_col not in df.columns:
        raise SystemExit(f"!! no label column {label_col!r}; pass --label-col.\n"
                         f"   columns: {list(df.columns)}")
    if not 0 <= test_fraction < 1:
        raise SystemExit("!! --test-fraction must be at least 0 and below 1.")

    df = check_labels(df, label_col, drop_unlabelled).copy()
    groups = resolve_groups(df, group_col, from_image_path).to_numpy()
    y = df[label_col].to_numpy()

    n_groups = len(set(groups))
    if n_groups < num_folds:
        raise SystemExit(f"!! only {n_groups} patient(s) for {num_folds} folds; each fold "
                         "needs at least one patient of its own.")

    # ---------- held-out test set, chosen by patient ----------
    pos = np.arange(len(df))
    if test_fraction > 0:
        held = _single_stratified_group_split(y, groups, test_fraction, seed)
        held_groups = set(groups[held])
        # Take WHOLE patients: the splitter already groups, this just makes it explicit
        # and keeps the invariant obvious to a reader.
        is_held = np.isin(groups, list(held_groups))
    else:
        is_held = np.zeros(len(df), dtype=bool)
    dev = pos[~is_held]

    if len(set(groups[dev])) < num_folds:
        raise SystemExit(f"!! after holding out the test set only "
                         f"{len(set(groups[dev]))} patient(s) remain for {num_folds} "
                         "folds. Lower --test-fraction or --num-folds.")

    # ---------- cross-validation folds over the development patients ----------
    fold_cols = [f"Fold_{k}" for k in range(num_folds)]
    assignment = np.full((len(df), num_folds), HELD_OUT, dtype=int)
    assignment[dev, :] = TRAIN
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for k, (_, val_local) in enumerate(
            splitter.split(np.zeros(len(dev)), y[dev], groups[dev])):
        assignment[dev[val_local], k] = VAL

    for k, col in enumerate(fold_cols):
        df[col] = assignment[:, k]
    _check_no_leakage(df, fold_cols, groups, is_held)
    return df


def _check_no_leakage(df, fold_cols, groups, is_held):
    """Assert the properties the rest of the pipeline relies on.

    These are the reason to use this script rather than a hand-rolled split, so they are
    checked rather than assumed.
    """
    arr = df[fold_cols].to_numpy()
    # every development row is validation in exactly one fold
    val_counts = (arr == VAL).sum(axis=1)
    dev_rows = ~is_held
    assert set(val_counts[dev_rows]) <= {1}, \
        "a development metastasis is validation in more than one fold"
    assert (val_counts[dev_rows] == 1).all(), \
        "a development metastasis is never used for validation"
    # held-out rows are held out everywhere
    assert (arr[is_held] == HELD_OUT).all(), "a held-out metastasis is in a fold"
    assert (val_counts[is_held] == 0).all(), "a held-out metastasis is validated on"
    # no patient spans development and held-out test
    assert not (set(groups[dev_rows]) & set(groups[is_held])), \
        "a patient appears in both the development and the held-out cohort"
    # within a fold, no patient is both trained on and validated on
    for k, col in enumerate(fold_cols):
        tr = set(groups[df[col].to_numpy() == TRAIN])
        va = set(groups[df[col].to_numpy() == VAL])
        assert not (tr & va), f"{col}: a patient is both trained on and validated on"


def summarize(df, num_folds=None, label_col=None, group_col=None,
              from_image_path=False):
    """Print fold sizes and class balance, so "evenly split" can be checked."""
    num_folds = cfg.NUM_FOLDS if num_folds is None else num_folds
    label_col = label_col or cfg.LABEL_MET
    fold_cols = [f"Fold_{k}" for k in range(num_folds)]
    groups = resolve_groups(df, group_col, from_image_path)

    in_cv = df[fold_cols].isin([TRAIN, VAL]).any(axis=1)
    dev, held = df[in_cv], df[~in_cv]
    print(f"\n{'':14}{'patients':>10}{'mets':>8}")
    print("-" * 32)
    for name, sub in (("DEVELOPMENT", dev), ("HELD-OUT TEST", held)):
        print(f"{name:14}{groups[sub.index].nunique():>10}{len(sub):>8}")

    print(f"\n{'fold':>5}{'train':>8}{'val':>7}{'val patients':>14}   "
          "validation class balance")
    print("-" * 74)
    for k, col in enumerate(fold_cols):
        tr = int((df[col] == TRAIN).sum())
        va_mask = df[col] == VAL
        va = int(va_mask.sum())
        bal = df.loc[va_mask, label_col].value_counts(normalize=True)
        shares = "  ".join(f"{lvl[:4]}={bal.get(lvl, 0):.2f}" for lvl in VALID_LABELS)
        print(f"{k:>5}{tr:>8}{va:>7}{groups[df.index[va_mask]].nunique():>14}   {shares}")

    overall = df[label_col].value_counts(normalize=True)
    print("-" * 74)
    print(f"{'whole cohort':>27}{'':>7}   " +
          "  ".join(f"{lvl[:4]}={overall.get(lvl, 0):.2f}" for lvl in VALID_LABELS))
    print("\nEach metastasis is validation in exactly one fold, so validation sizes sum "
          "to the development cohort.\nNo patient spans train and validation, or "
          "development and held-out test — asserted, not assumed.")


def make_folds(in_csv=None, out_csv=None, num_folds=None, test_fraction=0.15,
               group_col=None, label_col=None, seed=1, from_image_path=False,
               drop_unlabelled=False):
    """Read a cohort table, add fold columns, write it back out.

    Writes to a new file rather than in place, and keeps the index column, because the
    rest of the package reads these tables with ``index_col=0``.
    """
    in_csv = in_csv or cfg.folds_csv()
    if out_csv is None:
        raise SystemExit("!! --out is required: the input table is never overwritten.")
    df = data.load_folds(in_csv)
    print(f"read {len(df)} rows from {in_csv}")
    out = assign_folds(df, num_folds=num_folds, test_fraction=test_fraction,
                       group_col=group_col, label_col=label_col, seed=seed,
                       from_image_path=from_image_path,
                       drop_unlabelled=drop_unlabelled)
    summarize(out, num_folds=num_folds, label_col=label_col, group_col=group_col,
              from_image_path=from_image_path)
    out.to_csv(out_csv)                      # index kept: readers use index_col=0
    print(f"\nwrote {len(out)} rows with "
          f"{cfg.NUM_FOLDS if num_folds is None else num_folds} fold columns -> {out_csv}")
    return out_csv
