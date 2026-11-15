"""Rebuild the Table 1 cohort with ONE treatment course per patient.

Table 1's unit was the treatment COURSE, so patients who received more than one
ICI course were counted more than once. This module keeps each patient's FIRST
course and re-computes the Table 1 aggregates on that basis.

THE RULE
    Keep each patient's LOWEST treatment round. Rounds are numbered from 1 within a
    patient, so this keeps round 1 wherever round 1 survived the cohort prune (see
    the caveat in :func:`explain_missing_round1`).

    Why the round, and nothing else:
      - It is unambiguous. Every patient's courses carry DISTINCT round numbers, so
        the lowest one is well defined without any tie-break. Verified by check 4 of
        :func:`cohort.checks.verify_course_identity`: keying on patient+round gives
        the same course count as the ``PreTreatment_Image`` key, which a repeated
        round would have driven below it. Asserted again at runtime here.
      - It is complete: ``Treatment_round`` is 100% numeric, so no course is left
        unplaced.
      - It is corroborated: the round parsed from the image path equals
        ``Treatment_round`` for every metastasis, and the folds and clinical copies
        agree on all comparable rows.

    Why not a treatment date: there isn't one. ``Study_Date`` is sparsely populated,
    and the only well-populated date columns record when a BMI was measured, not when
    the course happened — ordering by that was tried and picked a later course as the
    first.

    Why not ``Age_at_Treatment``: it is inconsistent. Age decreases as the round
    increases for some repeat-course patients (see
    :func:`cohort.checks.check_round_sort_and_age_delta`), which is impossible for a
    correctly anchored age. It was previously wired in as a tie-break, but the round
    leaves no ties for it to break, so it never fired. Relying on it would be worse
    than failing loudly, hence the assertion instead.

KNOWN LIMIT
    The round ordering is unambiguous, but this dataset cannot prove it is
    CHRONOLOGICAL — there is no usable treatment-date column to check it against. If
    ``Treatment_round`` were mis-assigned for a patient, the wrong course would be
    kept. Nothing here can detect that.

PHI-SAFE: prints only aggregate counts and column names, never an individual patient
row or identifier. ``write_lists=True`` optionally writes the kept/dropped course IDs
to CSV for downstream re-runs; those files are not printed.
"""
import os

import pandas as pd

from . import config as cfg
from . import data
from . import demographics as demo

ROUND_COL = "_round_num"


def build_round(courses):
    """Numeric treatment round, preferring the explicit column over the image path."""
    for col in ("Treatment_round", "Treatment_round_clin"):
        if col in courses.columns:
            v = pd.to_numeric(courses[col], errors="coerce")
            if v.notna().mean() > 0.80:
                return v, f"explicit {col!r} column ({v.notna().mean():.0%} numeric)"
    return courses["_round"].map(data.round_number), "digits parsed from the image path"


def check_round_usable(courses):
    """The rule is only well defined if the round is complete and unique per patient."""
    missing = int(courses[ROUND_COL].isna().sum())
    if missing:
        raise SystemExit(
            f"ABORT: {missing} course(s) have no usable treatment round. The first-round "
            "rule cannot place them, and there is no reliable secondary key (see this "
            "module's docstring on Age_at_Treatment). Investigate before rebuilding Table 1."
        )
    per = courses.groupby(["_cohort", "_pid"])[ROUND_COL].agg(["size", "nunique"])
    clashes = per[per["size"] != per["nunique"]]
    if len(clashes):
        raise SystemExit(
            f"ABORT: {len(clashes)} patient(s) have two courses sharing a round number, so "
            "'the first round' is ambiguous for them. This contradicts check 4 of "
            "verify_course_identity (patient+round == PreTreatment_Image) and means the "
            "cohort or the round column has changed. Investigate before rebuilding Table 1."
        )
    print(f"  round is complete and unique within every patient "
          f"({len(per)} patients, {len(courses)} courses)")


def explain_missing_round1(kept):
    """For patients kept at round > 1: did a round-1 course exist in the source at all?

    Two very different situations produce a kept round of 2:
      - a round-1 course exists in the raw file but no metastasis survived the prune,
        so Table 1 holds the patient's earliest EVALUABLE course, not their first ICI
        course;
      - no round-1 row exists at all, i.e. the numbering does not start at 1 for them,
        and the kept course IS their first recorded one.
    Table 1's footnote wording depends on which dominates, so count them.
    """
    late = kept[kept[ROUND_COL] != 1]
    if not len(late):
        return
    raw = data.add_patient_round(data.load_folds())
    raw_round = pd.to_numeric(raw.get("Treatment_round"), errors="coerce")
    had_round1 = set(raw["_pid"][raw_round == 1])
    pruned = int(late["_pid"].isin(had_round1).sum())
    print(f"    of these {len(late)}: {pruned} DID have a round-1 course in the raw file "
          "that the prune removed (so their kept course is not their first ICI course);")
    print(f"                      {len(late) - pruned} have no round-1 row at all "
          "(numbering does not start at 1 for them; the kept course is their first "
          "recorded one).")


def select_first_course(courses):
    """Split the courses into (kept, dropped) by each patient's lowest round.

    "First round" is round 1 unless round 1 failed the cohort prune (e.g. all of that
    course's metastases fell below the size threshold). Filtering on
    ``Treatment_round == 1`` would silently delete those patients, so this keeps the
    lowest SURVIVING round instead.
    """
    first = courses.groupby(["_cohort", "_pid"])[ROUND_COL].transform("min")
    return courses[courses[ROUND_COL] == first], courses[courses[ROUND_COL] != first]


def rebuild(write_lists=False, out_dir="."):
    """Rebuild and print the Table 1 aggregates on one course per patient."""
    key = cfg.MERGE_KEY
    courses = data.one_row_per_course(data.load_courses())

    # ---------- the ordering key ----------
    print("\n=== treatment round ===")
    rnd, how = build_round(courses)
    courses[ROUND_COL] = rnd
    print(f"  source: {how}")
    check_round_usable(courses)
    print(f"  round range: {courses[ROUND_COL].min():.0f} .. {courses[ROUND_COL].max():.0f}")

    # ---------- patients with more than one course ----------
    counts = courses.groupby(["_cohort", "_pid"]).size()
    multi = counts[counts > 1]
    print("\n=== patients with >1 course ===")
    for cohort in cfg.COHORTS:
        m = multi[multi.index.get_level_values("_cohort") == cohort] if len(multi) else multi
        print(f"  {cohort}: {len(m)} patients contributing "
              f"{int(m.sum()) if len(m) else 0} courses")

    kept, dropped = select_first_course(courses)

    print("\n=== which round was kept ===")
    for r, k in kept[ROUND_COL].value_counts().sort_index().items():
        print(f"  round {r:.0f}: {k} patients")
    not_one = int((kept[ROUND_COL] != 1).sum())
    if not_one:
        print(f"  NOTE: {not_one} patient(s) have no round-1 course in the pruned cohort; "
              "their earliest SURVIVING course is kept. A literal 'Treatment_round == 1' "
              "filter would have dropped them.")
        explain_missing_round1(kept)
    else:
        print("  every kept course is round 1 (round 1 survived the prune for all patients)")

    print("\n=== deduplication ===")
    print(f"  courses before: {len(courses)}   kept: {len(kept)}   dropped: {len(dropped)}")
    for cohort in cfg.COHORTS:
        c = courses[courses["_cohort"] == cohort]
        k = kept[kept["_cohort"] == cohort]
        print(f"  {cfg.COHORT_LABELS[cohort]}: {len(c)} courses -> {len(k)} patients "
              f"(unique patients in source: {c['_pid'].nunique()})")
        assert len(k) == c["_pid"].nunique(), "kept count must equal unique patient count"

    if write_lists:
        os.makedirs(out_dir, exist_ok=True)
        kept[[key]].to_csv(os.path.join(out_dir, "table1_courses_kept.csv"), index=False)
        dropped[[key]].to_csv(os.path.join(out_dir, "table1_courses_dropped.csv"), index=False)
        print(f"  wrote table1_courses_kept.csv / table1_courses_dropped.csv to {out_dir}")

    # ---------- Table 1 aggregates on one course per patient ----------
    print("\n" + "#" * 66)
    print("# TABLE 1 AGGREGATES — one course per patient (first round)")
    print("#" * 66)
    for cohort in cfg.COHORTS:
        demo.summarize(kept[kept["_cohort"] == cohort], cfg.COHORT_LABELS[cohort])
    print(cfg.EXTERNAL_NOTE)
    return kept
