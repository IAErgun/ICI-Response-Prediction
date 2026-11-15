"""Write cohort subsets to CSV."""
import os

from . import config as cfg
from . import data

# Sequence-presence columns required for the common cohort (T1Post is assumed
# present for every met).
ALL_SEQ = ["t2", "flair"]


def write_common_cohort_csv(folds_csv=None, out_csv=None, lower=None):
    """Write the mets that survive the exclusion criteria AND have ALL sequences.

    This is the common-subset cohort for a fair sequence comparison: the exact same
    mets are usable by the T1Post, T2 and multisequence experiments, so differences
    in performance reflect the sequences, not the cohort.

    Only row selection is replicated. Column values are left untouched — the shape
    column ``abs(complex())`` conversion in the trainer is a value transform, not an
    exclusion, and does not change which rows remain.
    """
    folds_csv = folds_csv or cfg.folds_csv()
    out_csv = out_csv or os.path.join(
        os.path.dirname(folds_csv), "common_allseq_cohort_after_exclusions.csv")

    df = data.load_folds(folds_csv)
    print(f"raw rows: {len(df)}")
    d = data.prune(df, lower=lower, verbose=True)
    d = data.require_sequences(d, ALL_SEQ, verbose=True)

    d = d.reset_index(drop=True)
    d.to_csv(out_csv, index=False)
    print(f"\nCommon all-sequence cohort: {len(d)} metastases written -> {out_csv}")

    # Quick label breakdown as a sanity check on the written file.
    if cfg.LABEL_MET in d.columns:
        print(f"\n{cfg.LABEL_MET} breakdown:")
        print(d[cfg.LABEL_MET].value_counts(dropna=False).to_string())
    return out_csv
