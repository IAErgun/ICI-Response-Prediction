"""Cohort QC and sensitivity checks.

One function per check, each printing an aggregate report and returning None. All
are PHI-SAFE: counts, medians and column names only — never an individual patient
row or identifier.

Run any of them from the command line via ``python -m brainmets.cohort <check>``; see
:mod:`cohort.__main__`.
"""
import os

import numpy as np
import pandas as pd

from . import config as cfg
from . import data
from . import demographics as demo

AGE = "Age_at_Treatment"


# --------------------------------------------------------------------------- 1
def count_patients_per_cohort(folds_csv=None, lower=None, require_seqs=(),
                              num_folds=None):
    """Count PATIENTS (not mets) in the development vs held-out test cohorts.

    Patient identity is the ``Patient_Number`` column. Also flags any patient
    appearing in BOTH cohorts, which would be a leakage red flag (expect 0).
    """
    d = data.prune(data.load_folds(folds_csv), lower=lower)
    d = data.require_sequences(d, require_seqs)

    if cfg.PID_COL not in d.columns:
        raise SystemExit(f"!! {cfg.PID_COL!r} not in columns: {list(d.columns)}")

    tv, te = data.split_cv_test(d, num_folds)
    tv_pat = set(tv[cfg.PID_COL].unique())
    te_pat = set(te[cfg.PID_COL].unique())
    overlap = tv_pat & te_pat

    print(f"\ncohort{data.seq_tag(require_seqs)}: {len(d)} mets, "
          f"{d[cfg.PID_COL].nunique()} patients")
    print(f"{'':16}{'patients':>10}{'mets':>8}")
    print("-" * 34)
    print(f"{'TRAIN/VAL':16}{len(tv_pat):>10}{len(tv):>8}")
    print(f"{'HELD-OUT TEST':16}{len(te_pat):>10}{len(te):>8}")
    print(f"{'mets/patient':16}{'':>10}{'':>8}  "
          f"(train/val {len(tv) / max(1, len(tv_pat)):.2f}, "
          f"test {len(te) / max(1, len(te_pat)):.2f})")
    if overlap:
        # The count only. Which patients they are is one line of pandas away for whoever
        # holds the table; printing ids here would break this module's guarantee.
        print(f"\n!! LEAKAGE: {len(overlap)} patient(s) in BOTH train/val and test.")
    else:
        print("\nOK: no patient appears in both train/val and test "
              "(patient-level split is clean).")


# --------------------------------------------------------------------------- 2
def count_samples_per_fold(folds_csv=None, lower=None, require_seqs=(),
                           num_folds=None, npy_dir=None,
                           prefix="Fold_Binary_", tag=""):
    """Per-fold sample counts actually used for training.

    For each fold k: train = rows where ``Fold_k == 0``, val = rows where
    ``Fold_k == 1``. Each surviving row is one metastasis = one sample, so row
    counts equal data-list lengths.

    Pass `npy_dir` to cross-check against the saved ``*_data_list.npy`` lengths —
    the literal count the training run loaded. A mismatch means the data lists are
    stale with respect to the current CSV or pruning.
    """
    num_folds = cfg.NUM_FOLDS if num_folds is None else num_folds
    df = data.load_folds(folds_csv)
    print(f"raw rows: {len(df)}")
    d = data.require_sequences(data.prune(df, lower=lower), require_seqs)
    print(f"pruned cohort{data.seq_tag(require_seqs)}: {len(d)}")

    suffix = f"_{tag}" if tag else ""
    header = f"{'fold':>4} {'train(=0)':>10} {'val(=1)':>9} {'train+val':>10} {'other':>7}"
    if npy_dir:
        header += f" {'npy_train':>10} {'npy_val':>8}"
    print("\n" + header)
    print("-" * len(header))

    def npy_len(k, split):
        f = os.path.join(npy_dir, f"{prefix}{k}{suffix}_{split}_data_list.npy")
        return len(np.load(f, allow_pickle=True)) if os.path.exists(f) else None

    tot_tr = tot_va = 0
    for k in range(num_folds):
        col = f"Fold_{k}"
        if col not in d.columns:
            print(f"{k:>4}  (no column {col!r})")
            continue
        n_tr = int((d[col] == 0).sum())
        n_va = int((d[col] == 1).sum())
        tot_tr += n_tr
        tot_va += n_va
        line = f"{k:>4} {n_tr:>10} {n_va:>9} {n_tr + n_va:>10} {len(d) - n_tr - n_va:>7}"
        if npy_dir:
            lt, lv = npy_len(k, "train"), npy_len(k, "val")
            line += f" {('-' if lt is None else lt):>10} {('-' if lv is None else lv):>8}"
        print(line)

    print("-" * len(header))
    print(f"{'mean':>4} {tot_tr / num_folds:>10.0f} {tot_va / num_folds:>9.0f}")
    print("\nNote: train/val counts vary per fold by design "
          "(each met is val in exactly one fold).")
    if npy_dir:
        print("If npy_* differs from train(=0)/val(=1), the data lists are stale "
              "vs the current CSV/pruning.")


# --------------------------------------------------------------------------- 3
def count_train_val_classes(folds_csv=None, lower=None, require_seqs=(),
                            num_folds=None):
    """Class balance and zero-rule baselines, development vs held-out test.

    Reports 3-class and binary (Progression vs not) counts for each cohort, with
    the majority-class accuracy a trivial classifier would reach.
    """
    d = data.require_sequences(
        data.prune(data.load_folds(folds_csv), lower=lower), require_seqs)
    print(f"pruned cohort{data.seq_tag(require_seqs)}: {len(d)}")

    tv, te = data.split_cv_test(d, num_folds)
    for name, sub in [("TRAIN/VAL (CV cohort)", tv), ("HELD-OUT TEST", te)]:
        _report_classes(name, sub)


def _report_classes(name, sub):
    n = len(sub)
    if not n:
        print(f"\n=== {name}: 0 mets ===")
        return
    vc = sub[cfg.LABEL_MET].value_counts()
    prog = int(vc.get("Progression", 0))
    stable = int(vc.get("Stable", 0))
    resp = int(vc.get("Response", 0))
    not_prog = stable + resp
    print(f"\n=== {name}: {n} mets ===")
    print(f"  Progression: {prog:>5} ({prog / n:.3f})")
    print(f"  Stable:      {stable:>5} ({stable / n:.3f})")
    print(f"  Response:    {resp:>5} ({resp / n:.3f})")
    counts = [prog, stable, resp]
    print(f"  binary  (Prog vs not): {prog} vs {not_prog}  "
          f"-> zeror {max(prog, not_prog) / n:.3f} "
          f"(majority = {'Progression' if prog >= not_prog else 'not-Progression'})")
    print(f"  multiclass zeror: {max(counts) / n:.3f} (majority = "
          f"{['Progression', 'Stable', 'Response'][counts.index(max(counts))]})")


# --------------------------------------------------------------------------- 4
def verify_course_identity():
    """Verify that a treatment COURSE is uniquely and correctly identified.

    A patient who received one regimen in one course and a different one in the
    next must appear as TWO courses with a regimen change. The
    pipeline treats one ``PreTreatment_Image`` as one course and merges the
    clinical table on that key, which is only sound if (a) the clinical table is
    1:1 on that key, so the merge cannot fan out, and (b) no single image carries
    more than one distinct regimen.

    Checks: clinical rows per image; merge fan-out; distinct regimens per image;
    course counts under competing keys; folds-vs-clinical ``Treatment_round``
    agreement; and which file actually supplies each Table 1 variable.
    """
    folds = data.prune(data.load_folds())
    clin = data.load_clinical(map_gender=False)
    key = cfg.MERGE_KEY

    def reg(row):
        return demo.regimen(row, empty="NONE")

    print("=== 1. is the clinical table 1:1 on the course key? ===")
    print(f"  clinical rows: {len(clin)}   unique {key}: {clin[key].nunique()}")
    per_img = clin.groupby(key).size()
    for k, v in per_img.value_counts().sort_index().items():
        print(f"    {v} images have {k} clinical row(s)")
    dupes = int((per_img > 1).sum())
    if dupes:
        print(f"  !! {dupes} images have MULTIPLE clinical rows — the merge fans out and "
              "drop_duplicates(keep='first') would silently keep only one.")
        multi = clin[clin[key].isin(per_img[per_img > 1].index)].copy()
        multi["_reg"] = multi.apply(reg, axis=1)
        nreg = multi.groupby(key)["_reg"].nunique()
        print(f"     of those, {int((nreg > 1).sum())} carry MORE THAN ONE distinct regimen "
              "(i.e. genuinely different courses sharing one baseline scan)")
    else:
        print("  OK: clinical table is 1:1 on the course key.")

    print("\n=== 2. merge fan-out ===")
    merged = data.merge_clinical(folds, clin)
    print(f"  met rows before join: {len(folds)}   after join: {len(merged)}"
          f"   ({len(merged) - len(folds):+d})")
    if len(merged) != len(folds):
        print("  !! the join changed the row count — clinical rows are being duplicated.")

    print("\n=== 3. distinct regimens per course key ===")
    merged["_reg"] = merged.apply(reg, axis=1)
    nreg_img = merged.groupby(key)["_reg"].nunique()
    print(f"  images whose mets show >1 distinct regimen: {int((nreg_img > 1).sum())} "
          f"of {len(nreg_img)}")
    if int((nreg_img > 1).sum()):
        print("  !! one baseline scan maps to multiple regimens -> distinct courses are "
              "being COLLAPSED by the PreTreatment_Image key.")

    merged = data.add_patient_round(merged)

    print("\n=== 4. course counts under competing keys ===")
    keys = {
        "PreTreatment_Image (current)": [key],
        "patient + round": ["_pid", "_round"],
        "patient + round + regimen": ["_pid", "_round", "_reg"],
        "patient + regimen": ["_pid", "_reg"],
    }
    counts = {}
    for name, cols in keys.items():
        counts[name] = merged[cols].drop_duplicates().shape[0]
        print(f"  {name:32s} -> {counts[name]}")
    extra = counts["patient + round + regimen"] - counts["PreTreatment_Image (current)"]
    if extra > 0:
        print(f"  !! {extra} course(s) exist under patient+round+regimen that the current "
              "key does not distinguish — regimen changes are being hidden.")
    else:
        print("  OK: the image key is at least as fine-grained as patient+round+regimen.")

    print("\n=== 5. Treatment_round: folds vs clinical ===")
    if {"Treatment_round", "Treatment_round_clin"} <= set(merged.columns):
        a = pd.to_numeric(merged["Treatment_round"], errors="coerce")
        b = pd.to_numeric(merged["Treatment_round_clin"], errors="coerce")
        both = a.notna() & b.notna()
        print(f"  comparable rows: {int(both.sum())}   agree: {int((a[both] == b[both]).sum())}"
              f"   differ: {int((a[both] != b[both]).sum())}")
    else:
        present = [c for c in ("Treatment_round", "Treatment_round_clin") if c in merged.columns]
        print(f"  only present: {present or 'neither'}")

    if "Treatment_round" in merged.columns:
        rp = merged["_round"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        rc = pd.to_numeric(merged["Treatment_round"], errors="coerce")
        both = rp.notna() & rc.notna()
        if int(both.sum()):
            print(f"  path-derived round vs Treatment_round: agree "
                  f"{int((rp[both] == rc[both]).sum())} / {int(both.sum())}")

    print("\n=== 6. source and coverage of each Table 1 variable ===")
    _report_variable_sources(merged, folds, clin)


def _agree(a, b):
    """Count matching values, comparing NUMERICALLY when both sides are numeric.

    A plain ``astype(str)`` comparison reports 0 agreement on every integer-valued
    column: the left merge introduces NaN into the ``<var>_clin`` copy, promoting it
    to float, so "1" is compared against "1.0". That is a dtype artifact, not a
    data conflict.
    """
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    if na.notna().all() and nb.notna().all():
        return int(np.isclose(na.to_numpy(float), nb.to_numpy(float)).sum())
    return int((a.astype(str).str.strip() == b.astype(str).str.strip()).sum())


def _report_variable_sources(merged, folds, clin):
    """Which file supplies each Table 1 variable, and do the two copies agree?

    The folds CSV carries its own copy of the clinical variables and the merge adds
    a redundant ``<var>_clin`` set. ``summarize()`` reads the UNSUFFIXED column, i.e.
    the folds copy whenever both exist.
    """
    key = cfg.MERGE_KEY
    courses = data.one_row_per_course(merged)
    n = len(courses)
    matched = courses[key].isin(set(clin[key]))
    print(f"  courses: {n}   with a row in the clinical CSV: {int(matched.sum())} "
          f"({n - int(matched.sum())} unmatched)")
    hdr = (f"  {'variable':30s} {'in folds':>8} {'in clin':>7} {'source':>7} "
           f"{'cov(used)':>9} {'cov(clin)':>9} {'agree where both':>17}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for v in cfg.CONT + cfg.CAT + cfg.ICI:
        in_f, in_c = v in folds.columns, v in clin.columns
        alt = f"{v}_clin"
        used = v if v in courses.columns else None
        source = "folds" if in_f else ("clin" if in_c else "-")
        cov = f"{courses[used].notna().mean():.0%}" if used else "-"
        if alt in courses.columns:
            covc = f"{courses[alt].notna().mean():.0%}"
            both = courses[used].notna() & courses[alt].notna()
            agree = (f"{_agree(courses.loc[both, used], courses.loc[both, alt])}/{int(both.sum())}"
                     if both.any() else "n/a")
        else:
            covc, agree = (cov if (in_c and not in_f) else "-"), "-"
        print(f"  {v:30s} {str(in_f):>8} {str(in_c):>7} {source:>7} "
              f"{cov:>9} {covc:>9} {agree:>17}")
    print("  NOTE: 'source' is the column summarize() actually reads. If that is 'folds' while")
    print("        the clinical CSV covers fewer courses, Table 1 comes from the folds CSV.")


# --------------------------------------------------------------------------- 5
def check_round_sort_and_age_delta():
    """Diagnose the negative age delta reported for repeat-course patients.

    The delta came from :func:`describe_ici_other_and_repeat_courses`, which orders
    a patient's courses by the round DIRECTORY NAME, i.e. a STRING. Two candidate
    causes for a negative delta: a lexicographic sort ("Round_10" before
    "Round_2"), or ``Age_at_Treatment`` being genuinely inconsistent across a
    patient's courses. Recomputing under both orderings distinguishes them.

    Diagnostic only — Table 1 itself is unaffected either way, since
    :mod:`cohort.table1` orders by the explicit numeric ``Treatment_round``.
    """
    courses = data.one_row_per_course(data.load_courses(verbose=False))
    courses["_rnum"] = courses["_round"].map(data.round_number)

    per = courses.groupby(["_cohort", "_pid"]).size()
    pids = set(per[per > 1].index.get_level_values("_pid"))
    m = courses[courses["_pid"].isin(pids)].copy()
    print(f"repeat-course patients: {len(pids)}   contributing courses: {len(m)}")

    print("\n=== 1. round label shape ===")
    digits = courses["_round"].astype(str).str.extract(r"(\d+)")[0]
    print(f"  labels with no digits at all: {int(digits.isna().sum())}")
    print(f"  digit-length distribution (all courses): "
          f"{dict(digits.dropna().str.len().value_counts().sort_index())}")
    print(f"  numeric round range: {courses['_rnum'].min():.0f} .. "
          f"{courses['_rnum'].max():.0f}")
    alens = (m["_round"].astype(str).str.extract(r"(\d+)")[0].dropna().str.len()
             .value_counts().sort_index())
    print(f"  digit-length distribution (repeat-course patients only): {dict(alens)}")
    if len(alens) > 1:
        print("  -> MIXED digit lengths among affected patients: a string sort CAN misorder them.")
    else:
        print("  -> uniform digit length among affected patients: string and numeric sort agree,")
        print("     so a lexicographic sort cannot explain a negative delta.")

    print("\n=== 2. string sort vs numeric sort, per affected patient ===")
    differ = sum(
        1 for _, g in m.groupby("_pid")
        if list(g.sort_values("_round")[cfg.MERGE_KEY])
        != list(g.sort_values("_rnum")[cfg.MERGE_KEY]))
    print(f"  patients whose string-sorted course order differs from numeric: "
          f"{differ} / {len(pids)}")

    print("\n=== 3. Age_at_Treatment delta across the last consecutive transition ===")
    if AGE in m.columns:
        for label, key in (("string sort of round label", "_round"),
                           ("numeric round index", "_rnum")):
            d = m.sort_values(["_pid", key]).groupby("_pid")[AGE].agg(_last_diff)
            d = pd.to_numeric(d, errors="coerce").dropna()
            if len(d):
                print(f"  {label:26s}: median {d.median():+.1f}   "
                      f"range [{d.min():+.1f}, {d.max():+.1f}]   "
                      f"negative {int((d < 0).sum())}   zero {int((d == 0).sum())}")

    print("\n=== 4. age vs numeric round (every consecutive step, not just the last) ===")
    dec = inc = flat = 0
    for _, g in m.groupby("_pid"):
        step = pd.to_numeric(g.sort_values("_rnum")[AGE], errors="coerce").diff().dropna()
        if (step < 0).any():
            dec += 1
        elif (step > 0).any():
            inc += 1
        else:
            flat += 1
    print(f"  age strictly increases with round: {inc}")
    print(f"  age flat across all courses:       {flat}")
    print(f"  age DECREASES at some step:        {dec}")
    if dec == 0:
        print("  -> under the numeric ordering age is never negative: the lexicographic sort")
        print("     in describe_ici_other_and_repeat_courses fully explains the negative delta.")
    else:
        print("  -> age still decreases somewhere: Age_at_Treatment is genuinely inconsistent")
        print("     for at least one patient, independent of the sort.")

    print("\n=== 5. is age constant across a patient's courses? ===")
    nun = m.groupby("_pid")[AGE].apply(lambda s: pd.to_numeric(s, errors="coerce").nunique())
    print(f"  patients with the SAME recorded age on every course: "
          f"{int((nun <= 1).sum())} / {len(pids)}")
    print("  (a tie makes idxmin on the age key pick arbitrarily, so an age-vs-round key")
    print("   disagreement there does NOT imply the round ordering is wrong)")


def _last_diff(s):
    """Difference across the LAST consecutive transition."""
    v = pd.to_numeric(s, errors="coerce")
    return v.diff().iloc[-1] if len(v) > 1 else np.nan


# --------------------------------------------------------------------------- 6
def sensitivity_earliest_course_key():
    """COUNTERFACTUAL ONLY — what if age, not round, had picked each patient's course?

    Table 1 orders each patient's courses by ``Treatment_round``. The round leaves
    no ties (check 4 of :func:`verify_course_identity` shows patient+round yields
    the same course count as the image key) and the column is 100% numeric, so the
    age tie-break is never consulted and the known inconsistency in
    ``Age_at_Treatment`` cannot reach Table 1.

    This rebuilds the aggregates twice — round-then-age (what Table 1 ships) and
    age-then-round — and prints only the cells that DIFFER. It does not bound a
    live risk; keep it for the record. The genuinely open question, whether
    ``Treatment_round`` is chronological, no script here can answer: the dataset
    has no usable treatment-date column.
    """
    key = cfg.MERGE_KEY
    courses = data.one_row_per_course(data.load_courses(verbose=False))
    courses["_key_round"] = pd.to_numeric(courses.get("Treatment_round"), errors="coerce")
    courses["_key_age"] = pd.to_numeric(courses.get(AGE), errors="coerce")

    def keep_earliest(keys):
        return (courses.sort_values(["_cohort", "_pid"] + keys)
                       .drop_duplicates(subset=["_cohort", "_pid"], keep="first"))

    a = keep_earliest(["_key_round", "_key_age"])   # what Table 1 ships
    b = keep_earliest(["_key_age", "_key_round"])   # the alternative

    diff_ids = set(a[key]) ^ set(b[key])
    print(f"courses selected by round-first: {len(a)}   by age-first: {len(b)}")
    print(f"patients whose selected course differs: {len(diff_ids) // 2}")

    ca, cb = demo.table1_cells(a), demo.table1_cells(b)
    changed = [k for k in sorted(set(ca) | set(cb)) if ca.get(k, "-") != cb.get(k, "-")]
    print(f"\n=== Table 1 cells that differ ({len(changed)}) ===")
    if not changed:
        print("  none — the key choice does not change any Table 1 number.")
    else:
        print(f"  {'cohort':6} {'variable':18} {'level':34} "
              f"{'round-first':>16} {'age-first':>16}")
        for k in changed:
            coh, var, lvl = k
            print(f"  {coh:6} {var:18} {str(lvl):34} "
                  f"{ca.get(k, '-'):>16} {cb.get(k, '-'):>16}")
    print("\nNote: round-first is what the shipped Table 1 reports.")


# --------------------------------------------------------------------------- 7
def describe_ici_other_and_repeat_courses():
    """Detail for the Table 1 footnotes.

    (A) What the "Other or none" ICI-regimen courses actually received. The paper
    built the regimen label from IPI/NIVO/PEMBRO only (the other agents were too
    sparse in its cohort for subgroups of their own), so a course on any other
    checkpoint inhibitor fell into that bucket; this scans the clinical CSV for every
    agent-like column and reports what those courses were.

    (B) The patients who contributed more than one ICI treatment course, their
    regimen sequences, and whether the regimen changed between courses.
    """
    clin = data.load_clinical(map_gender=False)
    courses = data.one_row_per_course(data.load_courses(verbose=False))
    agent_cols = demo.agent_columns(clin)
    display = {"dev": "DEVELOPMENT", "test": "HELD-OUT TEST"}

    print("\n=== clinical CSV: agent-like columns detected ===")
    print(" ", agent_cols if agent_cols else "(none matched the token list)")
    extra = [c for c in agent_cols if c not in cfg.BASE3]
    print("  beyond IPI/NIVO/PEMBRO:", extra if extra else "(none)")

    def label(row):
        return demo.regimen(row, cols=agent_cols, empty="NO AGENT FLAGGED")

    for cohort in cfg.COHORTS:
        sub = courses[courses["_cohort"] == cohort]
        print(f"\n{'=' * 66}\n{display[cohort]}  (n = {len(sub)} courses, "
              f"{sub['_pid'].nunique()} unique patients)\n{'=' * 66}")

        # ---------- (A) the "Other or none" courses ----------
        base3 = [c for c in cfg.BASE3 if c in sub.columns]
        none3 = sub[~sub[base3].isin(cfg.POS).any(axis=1)]
        print(f"\n(A) Courses with none of IPI/NIVO/PEMBRO flagged: n = {len(none3)}")
        if len(none3):
            print("    What these courses actually received (any agent column):")
            for c in agent_cols:
                if c in none3.columns:
                    k = int(none3[c].isin(cfg.POS).sum())
                    if k:
                        print(f"      {c}: {k} ({100 * k / len(none3):.1f}% of the group)")
            print("    Regimen combinations within the group, n (%):")
            for val, k in none3.apply(label, axis=1).value_counts().items():
                print(f"      {val}: {k} ({100 * k / len(none3):.1f}%)")

        # ---------- (B) patients contributing more than one course ----------
        per_pat = sub.groupby("_pid").size()
        multi = per_pat[per_pat > 1]
        print(f"\n(B) Patients with >1 treatment course: {len(multi)} "
              f"(contributing {int(multi.sum())} courses)")
        if not len(multi):
            continue
        print("    Courses per patient, n patients:")
        for ncourse, k in multi.value_counts().sort_index().items():
            print(f"      {ncourse} courses: {k} patients")
        m = sub[sub["_pid"].isin(multi.index)].copy()
        m["_regimen"] = m.apply(label, axis=1)
        print("    Regimen sequence across a patient's courses, n patients (%):")
        seqs = (m.sort_values(["_pid", "_round"]).groupby("_pid")["_regimen"]
                 .apply(lambda s: " → ".join(s)))
        for val, k in seqs.value_counts().items():
            print(f"      {val}: {k} ({100 * k / len(multi):.0f}%)")
        print("    Did the regimen change between courses?")
        changed = seqs.map(lambda s: len(set(s.split(" → "))) > 1)
        print(f"      changed: {int(changed.sum())} | "
              f"same regimen repeated: {int((~changed).sum())}")
        for col in ("Number_of_BM", AGE):
            if col in m.columns:
                d = m.groupby("_pid")[col].agg(_last_diff)
                d = pd.to_numeric(d, errors="coerce").dropna()
                if len(d):
                    print(f"    Change in {col} from first to second course: "
                          f"median {d.median():.1f} [{d.min():.1f} to {d.max():.1f}]")


# --------------------------------------------------------------------------- 8
def _id_shapes(values, top=3):
    """The commonest shapes of an id column, with every digit as 9 and letter as A.

    ``P0042`` and ``P0917`` both become ``A9999``, so the output describes the format of
    an identifier without containing one.
    """
    shapes = (pd.Series(values).astype(str)
              .str.replace(r"[0-9]", "9", regex=True)
              .str.replace(r"[A-Za-z]", "A", regex=True))
    return ", ".join(f"{shape!r} x{n}" for shape, n in shapes.value_counts().head(top).items())


def compare_label_sources(folds_csv=None, centroids_csv=None, label=None):
    """Check whether the label source USED by the training pipeline disagrees with
    the lesion-centroid spreadsheet that was INTENDED.

    If the per-met labels match, the unused variable is harmless and the reported
    metrics stand. If they differ, the metrics were scored against the wrong ground
    truth. Deliberately defensive about column names, since the two files may have
    different schemas: it prints both schemas, auto-detects a join key and the
    label column, then reports how many overlapping mets disagree — in the raw
    3-class labels and in the binary collapse the trainer actually optimizes.

    To debug a join that finds nothing it prints the SHAPE of each key column (see
    :func:`_id_shapes`), not sample values: the patient key is a real identifier.
    """
    label = label or cfg.LABEL_MET
    folds = pd.read_csv(folds_csv or cfg.folds_csv())
    bm = pd.read_csv(centroids_csv or cfg.centroids_csv())
    for name, df in [("FOLDS (used)", folds), ("CENTROIDS (intended)", bm)]:
        print(f"\n=== {name} ===")
        print(f"rows: {len(df)}, cols: {len(df.columns)}")
        print("columns:", list(df.columns))

    def label_like(df):
        return [c for c in df.columns
                if any(t in c.lower() for t in ("rano", "response", "label"))]

    print("\nlabel-like columns in FOLDS    :", label_like(folds))
    print("label-like columns in CENTROIDS:", label_like(bm))

    # Resolve the label column per file (FOLDS carries the _Met suffix, the other doesn't).
    base = label[:-4] if label.endswith("_Met") else label

    def resolve(d, aliases, what):
        for a in aliases:
            if a in d.columns:
                return a
        raise SystemExit(f"!! Could not find {what}; tried {aliases}. "
                         f"Columns: {list(d.columns)}")

    flab = resolve(folds, [base + "_Met", base], "label col in FOLDS")
    blab = resolve(bm, [base + "_Met", base], "label col in CENTROIDS")
    print(f"\nlabel columns -> FOLDS: {flab} | CENTROIDS: {blab}")

    # FOLDS only stores a re-indexed Patient_Number, but the REAL patient id (matching
    # Patient_ID in the centroid file) and the round are embedded in the image path.
    # Reconstruct and join on (real patient id, round, metastasis), normalizing the met
    # so '1.0' and '1' match.
    path_col = next((c for c in ["PreTreatment_Image", "PreTreatment_image", "t1post"]
                     if c in folds.columns), None)
    if path_col is None:
        raise SystemExit("!! No image-path column in FOLDS to recover the real patient id. "
                         f"Columns: {list(folds.columns)}")

    def norm(s):
        x = pd.to_numeric(s, errors="coerce")
        return x.astype("Int64").astype(str).where(x.notna(), s.astype(str).str.strip())

    def parts(p):
        seg = os.path.split(str(p))[0].split("/")
        return pd.Series({"patient": seg[-3] if len(seg) >= 3 else "",
                          "round": seg[-2] if len(seg) >= 2 else ""})

    fparts = folds[path_col].apply(parts)
    f = pd.DataFrame({
        "patient": fparts["patient"].astype(str).str.strip(),
        "round": fparts["round"].astype(str).str.strip(),
        "met": norm(folds["Metastasis_ID"]),
        "label_used": folds[flab].values,
    })
    b = pd.DataFrame({
        "patient": bm["Patient_ID"].astype(str).str.strip(),
        "round": bm["Round_ID"].astype(str).str.strip(),
        "met": norm(bm["Metastasis_ID"]),
        "label_intended": bm[blab].values,
    })

    # The join key's SHAPE in each file, never its values: enough to see why two files
    # fail to join (zero-padding, a prefix, a different separator) without printing an id.
    print("\njoin-key shapes (digits shown as 9, letters as A; commonest first):")
    for nm, d in [("FOLDS    ", f), ("CENTROIDS", b)]:
        print(f"  {nm} " + " | ".join(f"{col}: {_id_shapes(d[col])}"
                                      for col in ("patient", "round", "met")))

    # Try the full key first; if rounds are encoded differently, fall back to patient+met.
    key_full, key_pm = ["patient", "round", "met"], ["patient", "met"]
    n_full = len(f.merge(b, on=key_full, how="inner"))
    n_pm = len(f.merge(b, on=key_pm, how="inner"))
    print(f"\noverlap on patient+round+met: {n_full} | on patient+met: {n_pm}")
    # Prefer the MOST SPECIFIC key when it already covers ~all of the centroid file. A
    # larger patient+met count is usually many-to-many inflation (the same met across
    # rounds), not better matching.
    keycols = key_full if n_full >= 0.9 * len(b) else key_pm
    print(f"using key: {keycols}")
    for nm, d in [("FOLDS", f), ("CENTROIDS", b)]:
        dup = int(d.duplicated(subset=keycols).sum())
        if dup:
            print(f"!! {nm}: {dup} rows share {keycols} -> join may be many-to-many.")
    merged = f.merge(b, on=keycols, how="inner")
    if len(merged) == 0:
        print("\n!! 0 overlap -> the patient id from the path doesn't match Patient_ID. "
              "Compare the join-key shapes above to find the right correspondence.")
        return

    print(f"\noverlapping mets (inner join): {len(merged)}")
    print(f"  only in FOLDS: {len(f) - len(merged)} | only in CENTROIDS: {len(b) - len(merged)}")

    both = merged.dropna(subset=["label_used", "label_intended"]).copy()
    print(f"\nmets with comparable labels: {len(both)}")

    # --- 3-class (raw RANO) comparison ---
    mismatch = both[both["label_used"].astype(str) != both["label_intended"].astype(str)]
    print(f"\n[3-CLASS] mismatches: {len(mismatch)}  "
          f"({100 * len(mismatch) / max(1, len(both)):.2f}%)")
    if len(mismatch):
        print("confusion (used -> intended):")
        print(pd.crosstab(mismatch["label_used"], mismatch["label_intended"]).to_string())

    # --- binary-collapsed comparison, matching class_dict in the trainer:
    #     {'Progression': 0, 'Stable': 1, 'Response': 1}
    def to_binary(v):
        s = str(v).strip().lower()
        if s in ("progression", "0"):
            return 0
        if s in ("stable", "response", "1"):
            return 1
        return None  # unrecognized label value

    both["bin_used"] = both["label_used"].map(to_binary)
    both["bin_intended"] = both["label_intended"].map(to_binary)
    unrec = both[both["bin_used"].isna() | both["bin_intended"].isna()]
    if len(unrec):
        print(f"\n!! {len(unrec)} mets had label values outside "
              "{Progression, Stable, Response} -> excluded from the binary check. "
              "Sample values:",
              sorted(set(unrec["label_used"]) | set(unrec["label_intended"]))[:10])
    binboth = both.dropna(subset=["bin_used", "bin_intended"])
    bin_mismatch = binboth[binboth["bin_used"] != binboth["bin_intended"]]
    print(f"\n[BINARY] (Progression vs not) mismatches: {len(bin_mismatch)}  "
          f"({100 * len(bin_mismatch) / max(1, len(binboth)):.2f}%)")
    if len(bin_mismatch):
        print("confusion (used -> intended):")
        print(pd.crosstab(bin_mismatch["bin_used"], bin_mismatch["bin_intended"]).to_string())

    print("\nVERDICT:")
    if len(mismatch) == 0:
        print("  Labels IDENTICAL on all overlapping mets -> the unused variable is harmless; "
              "all metrics stand.")
    elif len(bin_mismatch) == 0:
        print(f"  3-class labels DIFFER ({len(mismatch)} mets).")
        print("  BUT the binary collapse is IDENTICAL -> the BINARY results are UNAFFECTED; "
              "only multiclass would need re-running.")
    else:
        print(f"  3-class labels DIFFER ({len(mismatch)} mets).")
        print(f"  Binary also DIFFERS ({len(bin_mismatch)} mets) -> binary metrics used the "
              "wrong ground truth and need re-evaluation.")
