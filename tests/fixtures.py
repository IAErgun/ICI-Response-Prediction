"""Synthetic cohort data, so the package can be exercised without patient data.

The generated tables match the schema the code expects (see docs/configuration.md),
including the parts that are easy to get wrong: image paths deep enough
for the patient/round parsing, a few patients contributing several treatment courses, a
held-out group that is outside every fold, and a couple of patients whose recorded age
decreases with the round — the inconsistency the real data has and the QC checks look
for.

The numbers are fictional and the volumes do not exist on disk, so anything that reads
image files will not work against this; everything that reasons about the cohort tables
will.

Use it in a test:

    from fixtures import synthetic_cohort

    with synthetic_cohort() as paths:
        df = brainmets.cohort.data.load_folds(paths["folds"])

Or write a copy out to look at:

    python tests/fixtures.py /tmp/synthetic
"""
import contextlib
import os
import pathlib
import shlex
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

NUM_FOLDS = 5
N_PATIENTS = 40
REPEAT_COURSE_PATIENTS = 6      # patients 0-5 contribute more than one course
HELD_OUT_FROM = 32              # patients 32+ are the held-out test group
AGE_DECREASES_FOR = (2, 4)      # the deliberately inconsistent ages


def build_tables(seed=0):
    """Return (folds, clinical, centroids) as DataFrames."""
    rng = np.random.default_rng(seed)
    rows, course_id = [], 0
    for pi in range(N_PATIENTS):
        pid = f"P{pi:03d}"
        n_courses = int(rng.integers(2, 4)) if pi < REPEAT_COURSE_PATIENTS else 1
        held_out = pi >= HELD_OUT_FROM
        for ci in range(n_courses):
            rnd = ci + 1
            # base/<patient>/<round>/<lesion>/file — the parser takes [-3] and [-2] of
            # the dirname, so the path needs a level below the round.
            img = f"/data/preprocessed/{pid}/Round_{rnd}/lesion/_T1Post.nii.gz"
            age = 70 - ci if pi in AGE_DECREASES_FOR else 55 + pi % 25 + ci
            if held_out:
                folds = {f"Fold_{k}": 2 for k in range(NUM_FOLDS)}
            else:
                val = course_id % NUM_FOLDS
                folds = {f"Fold_{k}": (1 if k == val else 0) for k in range(NUM_FOLDS)}
            for met in range(1, int(rng.integers(2, 5))):
                rows.append({
                    **folds,
                    "Patient_Number": pi,
                    "Metastasis_ID": met,
                    "Treatment_round": rnd,
                    "PreTreatment_Image": img,
                    "PostTreatment_Image": img.replace("_T1Post", "_T1Post_post"),
                    "PreTreatment_Met": float(rng.integers(5, 40)),
                    "PostTreatment_Met": float(rng.integers(5, 40)),
                    "RANOResponse_-40_40_Met": rng.choice(
                        ["Progression", "Stable", "Response"]),
                    "RANOResponse_-40_40": rng.choice(
                        ["Progression", "Stable", "Response"]),
                    "rows_to_remove": int(rng.random() < 0.05),
                    "T2_present": bool(rng.random() < 0.8),
                    "FLAIR_present": bool(rng.random() < 0.8),
                    # stored as a complex-number string, as in the real table
                    "T1Post_original_shape_Elongation": complex(rng.random(), 0),
                    # the folds table carries its own copy of the clinical variables
                    "Age_at_Treatment": age,
                    "Number_of_BM": int(rng.integers(1, 12)),
                    "BMI_Result": round(20 + rng.random() * 12, 1),
                    "ALB_Latest_PreTreatment": round(3 + rng.random(), 1),
                    "HGB_Latest_PreTreatment": round(10 + rng.random() * 5, 1),
                    "PLT_Latest_PreTreatment": int(rng.integers(150, 400)),
                    "Gender": "Male" if pi % 2 else "Female",
                    "STEROIDS": int(rng.random() < 0.85),
                    "Histology_New": rng.choice(
                        ["Lung", "Melanoma", "Breast", "Other"]),
                    "IPI": int(rng.random() < 0.25),
                    "NIVO": int(rng.random() < 0.2),
                    "PEMBRO": int(rng.random() < 0.5),
                    "ATEZO": int(rng.random() < 0.05),
                    "DURVA": int(rng.random() < 0.02),
                })
            course_id += 1
    folds_df = pd.DataFrame(rows)

    clin_cols = ["Age_at_Treatment", "Number_of_BM", "BMI_Result",
                 "ALB_Latest_PreTreatment", "HGB_Latest_PreTreatment",
                 "PLT_Latest_PreTreatment", "STEROIDS", "Histology_New",
                 "IPI", "NIVO", "PEMBRO", "ATEZO", "DURVA", "Treatment_round"]
    clin = folds_df.drop_duplicates(subset="PreTreatment_Image")[
        ["PreTreatment_Image"] + clin_cols].copy()
    # Gender arrives encoded 0/1 in the clinical table, as in the real one
    clin["Gender"] = [i % 2 for i in range(len(clin))]
    clin["AVELUMAB"] = 0

    cen = folds_df.drop_duplicates(
        subset=["PreTreatment_Image", "Metastasis_ID"]).copy()
    cen["Patient_ID"] = cen["PreTreatment_Image"].map(lambda p: p.split("/")[-4])
    cen["Round_ID"] = cen["PreTreatment_Image"].map(lambda p: p.split("/")[-3])
    cen = cen[["Patient_ID", "Round_ID", "Metastasis_ID", "RANOResponse_-40_40"]]
    return folds_df, clin, cen


def write_tables(out_dir, seed=0):
    """Write the three CSVs into `out_dir`, plus an ``env.sh`` pointing the package at them.

    Returns the CSV paths. ``source <out_dir>/env.sh`` sets the three input variables, so
    the README's data-free quick start runs against the fixture.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds, clin, cen = build_tables(seed)
    paths = {"folds": out / "cohort_folds.csv",
             "clinical": out / "clinical_data.csv",
             "centroids": out / "lesion_centroids_rano.csv"}
    folds.to_csv(paths["folds"])                      # index written, as the real one is
    clin.to_csv(paths["clinical"], index=False)
    cen.to_csv(paths["centroids"], index=False)
    env = {"BM_FOLDS_CSV": paths["folds"], "BM_CLIN_CSV": paths["clinical"],
           "BM_CENTROIDS_CSV": paths["centroids"]}
    (out / "env.sh").write_text(
        "# Synthetic cohort written by tests/fixtures.py; not patient data.\n"
        + "".join(f"export {k}={shlex.quote(str(v.resolve()))}\n" for k, v in env.items()))
    return {k: str(v) for k, v in paths.items()}


@contextlib.contextmanager
def synthetic_cohort(seed=0):
    """Write the fixture to a temp directory and point the env vars at it.

    Restores the previous environment on exit, so tests cannot leak paths into each
    other.
    """
    tmp = tempfile.mkdtemp(prefix="brainmets-fixture-")
    saved = {k: os.environ.get(k)
             for k in ("BM_FOLDS_CSV", "BM_CLIN_CSV", "BM_CENTROIDS_CSV")}
    try:
        paths = write_tables(tmp, seed)
        os.environ["BM_FOLDS_CSV"] = paths["folds"]
        os.environ["BM_CLIN_CSV"] = paths["clinical"]
        os.environ["BM_CENTROIDS_CSV"] = paths["centroids"]
        yield paths
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "synthetic_cohort"
    written = write_tables(target)
    folds, clin, cen = build_tables()
    print(f"wrote synthetic tables to {target}/")
    for name, path in written.items():
        print(f"  {name:10} {path}")
    print(f"\n{len(folds)} metastasis rows, {len(clin)} treatment courses, "
          f"{folds['Patient_Number'].nunique()} patients")
    print(f"\nPoint the package at it with:\n  source {shlex.quote(target)}/env.sh")
