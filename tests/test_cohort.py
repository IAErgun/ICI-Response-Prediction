"""Tests for the cohort primitives — the definition of which metastases count.

These guard the functions every downstream number depends on. A silent error in `prune`
or `split_cv_test` would not crash anything; it would quietly change the cohort, the
training data and every reported metric at once, which is the kind of bug tests are
actually for.

Needs pandas and numpy, and the package installed (`pip install -e .`). No patient data:
everything runs against the synthetic fixture in `fixtures.py`.

    python -m unittest discover -s tests
"""
import contextlib
import io
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `fixtures`

from brainmets.cohort import config as cfg          # noqa: E402
from brainmets.cohort import checks, data, demographics, make_folds, table1  # noqa: E402
from fixtures import synthetic_cohort               # noqa: E402


def frame(**cols):
    """A minimal folds-table-shaped frame for the filter tests."""
    n = len(next(iter(cols.values())))
    base = {"rows_to_remove": [0] * n,
            "PreTreatment_Met": [20.0] * n,
            "PostTreatment_Met": [20.0] * n}
    base.update(cols)
    return pd.DataFrame(base)


class TestPrune(unittest.TestCase):
    """The trainer's exclusion criteria."""

    def test_drops_flagged_rows(self):
        df = frame(rows_to_remove=[0, 1, 0])
        self.assertEqual(len(data.prune(df)), 2)

    def test_threshold_is_strict(self):
        # "> lower", not ">=": a metastasis exactly at the threshold is excluded.
        df = frame(PreTreatment_Met=[9.0, 10.0, 11.0])
        kept = data.prune(df)
        self.assertEqual(list(kept["PreTreatment_Met"]), [11.0])

    def test_both_timepoints_must_clear(self):
        df = frame(PreTreatment_Met=[20.0, 20.0, 5.0],
                   PostTreatment_Met=[20.0, 5.0, 20.0])
        self.assertEqual(len(data.prune(df)), 1)

    def test_zero_pretreatment_excluded(self):
        self.assertEqual(len(data.prune(frame(PreTreatment_Met=[0.0]))), 0)

    def test_custom_threshold(self):
        df = frame(PreTreatment_Met=[12.0, 20.0])
        self.assertEqual(len(data.prune(df, lower=15)), 1)
        self.assertEqual(len(data.prune(df, lower=10)), 2)

    def test_does_not_mutate_input(self):
        df = frame(rows_to_remove=[0, 1])
        data.prune(df)
        self.assertEqual(len(df), 2)


class TestSequenceFilter(unittest.TestCase):
    def test_filters_on_presence_columns(self):
        df = frame(T2_present=[True, False, True], FLAIR_present=[True, True, False])
        self.assertEqual(len(data.require_sequences(df, ["t2"])), 2)
        self.assertEqual(len(data.require_sequences(df, ["t2", "flair"])), 1)

    def test_t1post_needs_no_column(self):
        # T1-post is assumed present for every metastasis.
        df = frame(PreTreatment_Met=[20.0, 20.0])
        self.assertEqual(len(data.require_sequences(df, ["t1post"])), 2)

    def test_empty_request_is_a_no_op(self):
        df = frame(PreTreatment_Met=[20.0])
        self.assertEqual(len(data.require_sequences(df, [])), 1)
        self.assertEqual(len(data.require_sequences(df, None)), 1)

    def test_missing_column_fails_loudly(self):
        with self.assertRaises(SystemExit):
            data.require_sequences(frame(PreTreatment_Met=[20.0]), ["t2"])

    def test_parse_option(self):
        self.assertEqual(data.parse_require_seqs("t2, flair"), ["t2", "flair"])
        self.assertEqual(data.parse_require_seqs(""), [])
        self.assertEqual(data.parse_require_seqs(None), [])


class TestFoldSplit(unittest.TestCase):
    """Development = in train or val of at least one fold; test = in neither, ever."""

    def folds(self, *rows):
        return pd.DataFrame(rows, columns=[f"Fold_{k}" for k in range(5)])

    def test_train_or_val_anywhere_is_development(self):
        df = self.folds([0, 2, 2, 2, 2], [2, 1, 2, 2, 2], [2, 2, 2, 2, 2])
        self.assertEqual(list(data.in_cv_mask(df)), [True, True, False])

    def test_split_is_a_partition(self):
        with synthetic_cohort():
            df = data.prune(data.load_folds())
            dev, test = data.split_cv_test(df)
            self.assertEqual(len(dev) + len(test), len(df))
            self.assertEqual(set(dev.index) & set(test.index), set())
            self.assertGreater(len(dev), 0)
            self.assertGreater(len(test), 0)

    def test_no_patient_spans_both_cohorts(self):
        # Leakage across the split would invalidate every held-out number.
        with synthetic_cohort():
            df = data.add_patient_round(data.prune(data.load_folds()))
            dev, test = data.split_cv_test(df)
            self.assertEqual(set(dev["_pid"]) & set(test["_pid"]), set())


class TestPatientRoundParsing(unittest.TestCase):
    """The real patient id lives in the image path, not in a column."""

    def test_parses_third_and_second_last_segments_of_the_dirname(self):
        df = pd.DataFrame({cfg.MERGE_KEY: [
            "/data/preprocessed/P001/Round_2/lesion/_T1Post.nii.gz"]})
        out = data.add_patient_round(df)
        self.assertEqual(out["_pid"].iloc[0], "P001")
        self.assertEqual(out["_round"].iloc[0], "Round_2")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({cfg.MERGE_KEY: ["/a/b/P1/R1/les/x.nii.gz"]})
        data.add_patient_round(df)
        self.assertNotIn("_pid", df.columns)

    def test_round_number_handles_multi_digit(self):
        # A string sort puts "Round_10" before "Round_2"; the numeric form must not.
        self.assertEqual(data.round_number("Round_10"), 10)
        self.assertEqual(data.round_number("Round_2"), 2)
        self.assertTrue(np.isnan(data.round_number("Round")))
        self.assertLess(data.round_number("Round_2"), data.round_number("Round_10"))


class TestCourseDeduplication(unittest.TestCase):
    def test_one_row_per_course(self):
        df = pd.DataFrame({cfg.MERGE_KEY: ["a", "a", "b"], "x": [1, 2, 3]})
        out = data.one_row_per_course(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["x"]), [1, 3])   # keeps the first

    def test_courses_outnumber_patients_in_the_fixture(self):
        # The fixture deliberately gives some patients several courses.
        with synthetic_cohort():
            courses = data.one_row_per_course(data.load_courses(verbose=False))
            self.assertGreater(len(courses), courses["_pid"].nunique())


class TestRegimenLabel(unittest.TestCase):
    def test_joins_positive_agents(self):
        row = {"IPI": 1, "NIVO": 1, "PEMBRO": 0, "ATEZO": 0, "DURVA": 0}
        self.assertEqual(demographics.regimen(row), "IPI + NIVO")

    def test_accepts_the_several_truthy_encodings(self):
        for truthy in (1, 1.0, "1", "Yes", "yes", "Y", "TRUE", "True", True):
            with self.subTest(value=truthy):
                row = {a: 0 for a in cfg.ICI}
                row["PEMBRO"] = truthy
                self.assertEqual(demographics.regimen(row), "PEMBRO")

    def test_empty_label_is_configurable(self):
        row = {a: 0 for a in cfg.ICI}
        self.assertEqual(demographics.regimen(row), "None/Other")
        self.assertEqual(demographics.regimen(row, empty="NONE"), "NONE")

    def test_includes_atezolizumab_and_durvalumab(self):
        # The paper used IPI/NIVO/PEMBRO only (these two were too sparse there); the
        # package deliberately counts all five. See the note on cohort.config.ICI.
        self.assertIn("ATEZO", cfg.ICI)
        self.assertIn("DURVA", cfg.ICI)
        row = {a: 0 for a in cfg.ICI}
        row["ATEZO"] = 1
        self.assertEqual(demographics.regimen(row), "ATEZO")


class TestAggregationHelpers(unittest.TestCase):
    def test_median_iqr_format(self):
        self.assertEqual(demographics.med_iqr(pd.Series([1, 2, 3, 4, 5])),
                         "3.0 [2.0–4.0]")

    def test_median_iqr_ignores_non_numeric(self):
        self.assertEqual(demographics.med_iqr(pd.Series(["x", "y"])), "n/a")
        self.assertEqual(demographics.med_iqr(pd.Series([])), "n/a")

    def test_category_counts_exclude_missing_from_the_denominator(self):
        body, n = demographics.cat_counts(pd.Series(["a", "a", "b", None]))
        self.assertEqual(n, 3)
        self.assertIn("a: 2 (66.7%)", body)


class TestLoadCourses(unittest.TestCase):
    """End-to-end on the fixture: the assembled frame downstream code receives."""

    def test_shape_and_helper_columns(self):
        with synthetic_cohort():
            merged = data.load_courses(verbose=False)
            for col in ("_cohort", "_pid", "_round"):
                self.assertIn(col, merged.columns)
            self.assertEqual(set(merged["_cohort"]) - {"dev", "test"}, set())

    def test_clinical_merge_does_not_fan_out(self):
        # A one-to-many join here would silently duplicate metastases.
        with synthetic_cohort():
            pruned = data.prune(data.load_folds())
            merged = data.load_courses(verbose=False)
            self.assertEqual(len(merged), len(pruned))

    def test_gender_is_decoded(self):
        with synthetic_cohort():
            clin = data.load_clinical()
            self.assertEqual(set(clin["Gender"]) - {"Male", "Female"}, set())
            raw = data.load_clinical(map_gender=False)
            self.assertEqual(set(raw["Gender"]) - {0, 1}, set())

    def test_sequence_requirement_shrinks_the_cohort(self):
        with synthetic_cohort():
            allseq = data.load_courses(verbose=False, require_seqs=["t2", "flair"])
            everything = data.load_courses(verbose=False)
            self.assertLess(len(allseq), len(everything))


class TestConfigResolution(unittest.TestCase):
    def test_fixture_env_file_points_at_the_tables(self):
        # the README's quick start is `source <dir>/env.sh`; check it in a subshell from
        # a directory with a space in its name, which is where quoting goes wrong
        import subprocess
        import tempfile
        from fixtures import write_tables
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "with space")
            write_tables(out)
            env_file = os.path.join(out, "env.sh")
            got = subprocess.run(
                ["bash", "-c", 'source "$1" && printf "%s\n" "$BM_FOLDS_CSV" "$BM_CLIN_CSV"',
                 "_", env_file], capture_output=True, text=True, cwd="/", check=True,
            ).stdout.split("\n")
            self.assertTrue(os.path.isfile(got[0]) and got[0].endswith("cohort_folds.csv"))
            self.assertTrue(os.path.isfile(got[1]) and got[1].endswith("clinical_data.csv"))

    def test_unset_input_fails_with_a_usable_message(self):
        saved = os.environ.pop("BM_FOLDS_CSV", None)
        try:
            with self.assertRaises(SystemExit) as ctx:
                cfg.folds_csv()
            self.assertIn("BM_FOLDS_CSV", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["BM_FOLDS_CSV"] = saved

    def test_missing_file_fails_with_a_usable_message(self):
        saved = os.environ.get("BM_FOLDS_CSV")
        os.environ["BM_FOLDS_CSV"] = "/nonexistent/cohort.csv"
        try:
            with self.assertRaises(SystemExit) as ctx:
                cfg.folds_csv()
            self.assertIn("does not exist", str(ctx.exception))
        finally:
            if saved is None:
                os.environ.pop("BM_FOLDS_CSV", None)
            else:
                os.environ["BM_FOLDS_CSV"] = saved

    def test_env_change_takes_effect_without_reimport(self):
        with synthetic_cohort() as paths:
            self.assertEqual(cfg.folds_csv(), paths["folds"])

    def test_unset_output_root_fails_instead_of_using_a_placeholder(self):
        # A placeholder default would have a run create and fill a literal directory
        # wherever it happened to have permission.
        for var, resolve in (("BM_WORK_ROOT", cfg.work_root), ("BM_CACHE_ROOT", cfg.cache_root)):
            saved = os.environ.pop(var, None)
            try:
                with self.subTest(var=var):
                    with self.assertRaises(SystemExit) as ctx:
                        resolve()
                    self.assertIn(var, str(ctx.exception))
            finally:
                if saved is not None:
                    os.environ[var] = saved

    def test_output_root_need_not_exist_yet(self):
        saved = os.environ.get("BM_WORK_ROOT")
        os.environ["BM_WORK_ROOT"] = "/nonexistent/created/on/first/use"
        try:
            self.assertEqual(cfg.work_root(), "/nonexistent/created/on/first/use")
        finally:
            if saved is None:
                os.environ.pop("BM_WORK_ROOT", None)
            else:
                os.environ["BM_WORK_ROOT"] = saved


class TestMakeFolds(unittest.TestCase):
    """Fold generation for a table that arrives without fold columns.

    The point of the generator is the guarantees, so these test the guarantees.
    """

    def table(self):
        """The fixture with its fold columns stripped off."""
        with synthetic_cohort() as paths:
            df = pd.read_csv(paths["folds"], index_col=0)
        return df.drop(columns=[c for c in df.columns if c.startswith("Fold_")])

    def test_creates_the_expected_columns(self):
        out = make_folds.assign_folds(self.table(), num_folds=5)
        self.assertEqual([f"Fold_{k}" for k in range(5)],
                         [c for c in out.columns if c.startswith("Fold_")])

    def test_honours_num_folds(self):
        out = make_folds.assign_folds(self.table(), num_folds=3)
        self.assertEqual(len([c for c in out.columns if c.startswith("Fold_")]), 3)

    def test_every_development_row_validates_exactly_once(self):
        out = make_folds.assign_folds(self.table(), num_folds=5)
        cols = [f"Fold_{k}" for k in range(5)]
        in_cv = out[cols].isin([0, 1]).any(axis=1)
        val_counts = (out.loc[in_cv, cols] == 1).sum(axis=1)
        self.assertEqual(set(val_counts), {1})

    def test_held_out_rows_are_held_out_everywhere(self):
        out = make_folds.assign_folds(self.table(), num_folds=5, test_fraction=0.2)
        cols = [f"Fold_{k}" for k in range(5)]
        held = ~out[cols].isin([0, 1]).any(axis=1)
        self.assertTrue(held.any(), "expected a non-empty held-out set")
        self.assertTrue((out.loc[held, cols] == 2).all().all())

    def test_no_patient_spans_train_and_validation(self):
        # The leak that matters: correlated lesions from one patient on both sides.
        out = make_folds.assign_folds(self.table(), num_folds=5)
        for k in range(5):
            col = f"Fold_{k}"
            train = set(out.loc[out[col] == 0, "Patient_Number"])
            val = set(out.loc[out[col] == 1, "Patient_Number"])
            with self.subTest(fold=k):
                self.assertEqual(train & val, set())

    def test_no_patient_spans_development_and_test(self):
        out = make_folds.assign_folds(self.table(), num_folds=5, test_fraction=0.2)
        cols = [f"Fold_{k}" for k in range(5)]
        in_cv = out[cols].isin([0, 1]).any(axis=1)
        self.assertEqual(set(out.loc[in_cv, "Patient_Number"]) &
                         set(out.loc[~in_cv, "Patient_Number"]), set())

    def test_folds_are_of_comparable_size(self):
        out = make_folds.assign_folds(self.table(), num_folds=5)
        sizes = [int((out[f"Fold_{k}"] == 1).sum()) for k in range(5)]
        # "evenly split": no fold more than twice any other
        self.assertLess(max(sizes), 2 * min(sizes))

    def test_validation_class_balance_tracks_the_cohort(self):
        out = make_folds.assign_folds(self.table(), num_folds=5)
        overall = out[cfg.LABEL_MET].value_counts(normalize=True)
        for k in range(5):
            share = out.loc[out[f"Fold_{k}"] == 1, cfg.LABEL_MET] \
                       .value_counts(normalize=True)
            for level in make_folds.VALID_LABELS:
                with self.subTest(fold=k, level=level):
                    self.assertLess(abs(share.get(level, 0) - overall.get(level, 0)), 0.25)

    def test_test_fraction_zero_holds_nothing_out(self):
        out = make_folds.assign_folds(self.table(), num_folds=5, test_fraction=0)
        cols = [f"Fold_{k}" for k in range(5)]
        self.assertTrue(out[cols].isin([0, 1]).any(axis=1).all())

    def test_is_reproducible_and_seed_dependent(self):
        df = self.table()
        a = make_folds.assign_folds(df, num_folds=5, seed=1)["Fold_0"].tolist()
        b = make_folds.assign_folds(df, num_folds=5, seed=1)["Fold_0"].tolist()
        c = make_folds.assign_folds(df, num_folds=5, seed=7)["Fold_0"].tolist()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_does_not_mutate_the_input(self):
        df = self.table()
        make_folds.assign_folds(df, num_folds=5)
        self.assertEqual([c for c in df.columns if c.startswith("Fold_")], [])

    def test_can_group_by_the_image_path_instead(self):
        out = make_folds.assign_folds(self.table(), num_folds=5, from_image_path=True)
        cols = [f"Fold_{k}" for k in range(5)]
        pid = out[cfg.MERGE_KEY].astype(str).map(
            lambda p: os.path.dirname(p).split("/")[-3])
        for k in range(5):
            train = set(pid[out[cols[k]] == 0])
            val = set(pid[out[cols[k]] == 1])
            with self.subTest(fold=k):
                self.assertEqual(train & val, set())

    def test_unusable_label_stops_unless_dropping_is_asked_for(self):
        df = self.table()
        df.loc[df.index[0], cfg.LABEL_MET] = "Equivocal"
        with self.assertRaises(SystemExit):
            make_folds.assign_folds(df, num_folds=5)
        out = make_folds.assign_folds(df, num_folds=5, drop_unlabelled=True)
        self.assertEqual(len(out), len(df) - 1)

    def test_missing_label_column_fails_clearly(self):
        df = self.table().drop(columns=[cfg.LABEL_MET])
        with self.assertRaises(SystemExit):
            make_folds.assign_folds(df, num_folds=5)

    def test_missing_patient_column_fails_clearly(self):
        df = self.table().drop(columns=["Patient_Number"])
        with self.assertRaises(SystemExit):
            make_folds.assign_folds(df, num_folds=5)

    def test_too_few_patients_for_the_requested_folds(self):
        df = self.table()
        df = df[df["Patient_Number"].isin(sorted(df["Patient_Number"].unique())[:3])]
        with self.assertRaises(SystemExit):
            make_folds.assign_folds(df, num_folds=5, test_fraction=0)

    def test_generated_folds_drive_the_pipeline(self):
        """The real check: the cohort code reads the generated columns correctly."""
        out = make_folds.assign_folds(self.table(), num_folds=5, test_fraction=0.2)
        dev, test = data.split_cv_test(data.prune(out))
        self.assertGreater(len(dev), 0)
        self.assertGreater(len(test), 0)
        dev_p = set(data.add_patient_round(dev)["_pid"])
        test_p = set(data.add_patient_round(test)["_pid"])
        self.assertEqual(dev_p & test_p, set())


class TestReportsPrintNoIdentifiers(unittest.TestCase):
    """Every cohort report promises aggregates only. Hold it to that.

    The fixture's patient ids look like ``P007`` and sit inside the image paths, which is
    where the real ones live too, so a report that echoes an id or a path is caught here.
    """

    IDENTIFIER = re.compile(r"P\d{3}|/data/preprocessed")

    REPORTS = {
        "count_patients_per_cohort": checks.count_patients_per_cohort,
        "count_samples_per_fold": checks.count_samples_per_fold,
        "count_train_val_classes": checks.count_train_val_classes,
        "verify_course_identity": checks.verify_course_identity,
        "check_round_sort_and_age_delta": checks.check_round_sort_and_age_delta,
        "sensitivity_earliest_course_key": checks.sensitivity_earliest_course_key,
        "describe_ici_other_and_repeat_courses": checks.describe_ici_other_and_repeat_courses,
        "compare_label_sources": checks.compare_label_sources,
        "demographics.report": demographics.report,
        "table1.rebuild": table1.rebuild,
    }

    @staticmethod
    def output_of(report, *args, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report(*args, **kwargs)
        return out.getvalue()

    def test_no_report_prints_a_patient_id_or_an_image_path(self):
        with synthetic_cohort():
            for name, report in self.REPORTS.items():
                with self.subTest(report=name):
                    text = self.output_of(report)
                    self.assertGreater(len(text), 0)
                    self.assertIsNone(self.IDENTIFIER.search(text),
                                      f"{name} printed an identifier")

    def test_leakage_is_reported_as_a_count(self):
        # Put one development patient into the held-out group as well. The report has to
        # say so, and say how many, without naming them.
        with synthetic_cohort() as paths:
            df = pd.read_csv(paths["folds"], index_col=0)
            df["Patient_Number"] = "ZQ" + df["Patient_Number"].astype(str)
            dev, test = data.split_cv_test(data.prune(df))
            leaked = dev["Patient_Number"].iloc[0]
            df.loc[test.index[0], "Patient_Number"] = leaked
            df.to_csv(paths["folds"])
            text = self.output_of(checks.count_patients_per_cohort)
        self.assertIn("LEAKAGE: 1 patient(s)", text)
        self.assertNotIn("ZQ", text)

    def test_join_key_shapes_describe_the_format_not_the_value(self):
        self.assertEqual(checks._id_shapes(["P0042", "P0917", "X1"]), "'A9999' x2, 'A9' x1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
