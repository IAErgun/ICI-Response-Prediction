"""Tests for the evaluation metrics: the code that produces the numbers that get reported.

Every expected value here is worked out by hand from a small constructed input, not
taken from running the code, so a test failing means the metric changed.

    python -m unittest tests.test_evaluation -v
"""
import importlib.util
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))                    # for `fixtures`

from brainmets.evaluation import bootstrap_sequence_comparison_ci as seqcmp  # noqa: E402
from brainmets.evaluation import bootstrap_test_ci as btci                    # noqa: E402
from brainmets.evaluation import compute_test_metrics_subgroups as subgroups  # noqa: E402

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None


def binary_frame(y, p, patient=None):
    """A predictions table in the schema `export_test_predictions` writes."""
    return pd.DataFrame({"y_true": y, "y_prob_ensemble": p,
                         "patient_id": patient if patient is not None else range(len(y))})


class TestBootstrapTestCI(unittest.TestCase):
    """`bootstrap_test_ci.bootstrap`: point estimates and percentile intervals."""

    def test_perfect_binary_classifier(self):
        df = binary_frame([0] * 10 + [1] * 10, np.r_[np.linspace(0, .4, 10), np.linspace(.6, 1, 10)])
        point, ci, n_valid = btci.bootstrap(df, "binary", n_boot=200, cluster=False, seed=0)
        self.assertEqual(point["AUROC"], 1.0)
        np.testing.assert_allclose(ci["AUROC"], [1.0, 1.0])
        self.assertEqual(n_valid, 200)

    def test_uninformative_classifier_interval_covers_chance(self):
        rng = np.random.default_rng(3)
        df = binary_frame(rng.integers(0, 2, 400), rng.random(400))
        point, ci, _ = btci.bootstrap(df, "binary", n_boot=300, cluster=False, seed=0)
        self.assertLess(abs(point["AUROC"] - 0.5), 0.1)
        lo, hi = ci["AUROC"]
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_patient_clustering_widens_the_interval(self):
        # 40 patients, each contributing 5 lesions with identical outcome and score: the
        # lesions are perfectly correlated, so there are really only 40 observations.
        # Resampling lesions treats them as 200 and must give a narrower interval;
        # resampling patients is the honest one. This is why `--cluster` exists.
        rng = np.random.default_rng(7)
        y_pat = rng.integers(0, 2, 40)
        p_pat = np.clip(0.5 + 0.25 * (y_pat - 0.5) + rng.normal(0, 0.25, 40), 0, 1)
        df = binary_frame(np.repeat(y_pat, 5), np.repeat(p_pat, 5), np.repeat(np.arange(40), 5))

        _, ci_lesion, _ = btci.bootstrap(df, "binary", n_boot=500, cluster=False, seed=0)
        _, ci_patient, _ = btci.bootstrap(df, "binary", n_boot=500, cluster=True, seed=0)
        lesion_width = ci_lesion["AUROC"][1] - ci_lesion["AUROC"][0]
        patient_width = ci_patient["AUROC"][1] - ci_patient["AUROC"][0]
        # ~sqrt(5) wider in expectation; 1.5x leaves a wide margin for sampling noise
        self.assertGreater(patient_width, 1.5 * lesion_width)

    def test_cluster_resampling_keeps_patients_whole(self):
        # Every patient's lesions share one label. If resampling split patients up this
        # would still hold, so instead check the property directly: a resample drawn the
        # way `bootstrap` draws it contains each sampled patient's lesions in full.
        df = binary_frame([0, 0, 0, 1, 1, 1, 1], [.1] * 7, ["a", "a", "a", "b", "b", "c", "c"])
        groups = df["patient_id"].to_numpy()
        pat_idx = {p: np.where(groups == p)[0] for p in np.unique(groups)}
        rng = np.random.default_rng(0)
        for _ in range(50):
            sampled = rng.choice(list(pat_idx), size=len(pat_idx), replace=True)
            idx = np.concatenate([pat_idx[p] for p in sampled])
            for p in set(sampled):
                self.assertEqual((groups[idx] == p).sum(), list(sampled).count(p) * len(pat_idx[p]))

    def test_single_class_resamples_are_skipped(self):
        # One positive among 20: many resamples miss it, and AUROC is undefined there.
        df = binary_frame([0] * 19 + [1], np.linspace(0, 1, 20))
        _, _, n_valid = btci.bootstrap(df, "binary", n_boot=300, cluster=False, seed=0)
        self.assertLess(n_valid, 300)
        self.assertGreater(n_valid, 0)

    def test_perfect_multiclass_classifier(self):
        y = np.repeat([0, 1, 2], 8)
        prob = np.eye(3)[y] * 0.8 + 0.2 / 3
        df = pd.DataFrame({"y_true": y, "p0": prob[:, 0], "p1": prob[:, 1], "p2": prob[:, 2],
                           "patient_id": range(len(y))})
        point, ci, _ = btci.bootstrap(df, "multiclass", n_boot=100, cluster=False, seed=0)
        self.assertEqual(point["kappa"], 1.0)
        self.assertEqual(point["AUROC_ovr"], 1.0)
        np.testing.assert_allclose(ci["kappa"], [1.0, 1.0])

    def test_seed_makes_it_reproducible(self):
        rng = np.random.default_rng(1)
        df = binary_frame(rng.integers(0, 2, 60), rng.random(60))
        a = btci.bootstrap(df, "binary", n_boot=100, cluster=False, seed=5)[1]["AUROC"]
        b = btci.bootstrap(df, "binary", n_boot=100, cluster=False, seed=5)[1]["AUROC"]
        np.testing.assert_array_equal(a, b)


class TestSubgroupMetrics(unittest.TestCase):
    """`compute_test_metrics_subgroups`: per-subgroup metrics and subgroup labels."""

    def test_binary_metrics_hand_computed(self):
        # y = [0,0,1,1], prob = [.1,.6,.4,.9]  ->  pred at 0.5 = [0,1,0,1]
        # AUROC: positive/negative pairs (.4>.1)(.4<.6)(.9>.1)(.9>.6) -> 3/4
        m = subgroups.binary_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .9]))
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["Progression%"], 0.5)
        self.assertAlmostEqual(m["AUROC"], 0.75)
        self.assertAlmostEqual(m["Accuracy"], 0.5)
        self.assertAlmostEqual(m["Sensitivity"], 0.5)
        self.assertAlmostEqual(m["Specificity"], 0.5)
        self.assertAlmostEqual(m["Precision"], 0.5)

    # A single-class subgroup must report undefined metrics as NaN. scikit-learn returns
    # numbers instead (AUPRC 1.0 or 0.0, specificity 0.0, kappa 0.0), which read as scores.

    def test_all_non_progression_subgroup(self):
        # y all 1: nothing to measure specificity on; sensitivity is still defined.
        # pred at 0.5 = [0, 1, 1] -> sensitivity 2/3, precision 1, F1 = 2*(2/3)/(5/3) = 0.8
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = subgroups.binary_metrics(np.array([1, 1, 1]), np.array([.2, .7, .9]))
        for undefined in ("AUROC", "AUPRC", "Kappa", "BalancedAcc", "Specificity"):
            self.assertTrue(np.isnan(m[undefined]), undefined)
        self.assertAlmostEqual(m["Sensitivity"], 2 / 3)
        self.assertAlmostEqual(m["Precision"], 1.0)
        self.assertAlmostEqual(m["F1"], 0.8)
        self.assertAlmostEqual(m["Accuracy"], 2 / 3)

    def test_all_progression_subgroup(self):
        # y all 0: no positives, so sensitivity and F1 are undefined; specificity = 1/3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = subgroups.binary_metrics(np.array([0, 0, 0]), np.array([.2, .7, .9]))
        for undefined in ("AUROC", "AUPRC", "Kappa", "BalancedAcc", "Sensitivity", "F1"):
            self.assertTrue(np.isnan(m[undefined]), undefined)
        self.assertAlmostEqual(m["Specificity"], 1 / 3)
        self.assertEqual(m["Progression%"], 1.0)

    def test_degenerate_predictions_keep_the_published_conventions(self):
        # Both classes present but every prediction negative: kappa is defined (0, chance
        # agreement) and precision keeps the zero-division convention the published
        # numbers used, rather than becoming NaN.
        m = subgroups.binary_metrics(np.array([0, 1, 0, 1]), np.array([.1, .2, .3, .4]))
        self.assertEqual(m["Kappa"], 0.0)
        self.assertEqual(m["Precision"], 0.0)
        self.assertEqual(m["Sensitivity"], 0.0)
        self.assertEqual(m["Specificity"], 1.0)

    def test_single_class_multiclass_subgroup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = subgroups.multiclass_metrics(np.array([2, 2, 2]), np.eye(3)[[0, 1, 2]])
        for undefined in ("Kappa_lw", "BalancedAcc", "AUROC_ovr", "AUROC_ovo"):
            self.assertTrue(np.isnan(m[undefined]), undefined)
        self.assertAlmostEqual(m["Accuracy"], 1 / 3)

    def test_multiclass_metrics_perfect(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        m = subgroups.multiclass_metrics(y, np.eye(3)[y])
        self.assertEqual(m["Kappa_lw"], 1.0)
        self.assertEqual(m["AUROC_ovr"], 1.0)
        self.assertEqual(m["AUROC_ovo"], 1.0)
        self.assertAlmostEqual(m["Stable%"], 1 / 3)

    def test_tertiles_split_evenly_and_keep_missing_separate(self):
        labels = subgroups.tertile_label(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]))
        self.assertEqual(labels.value_counts().to_dict(),
                         {"Low": 3, "Medium": 3, "High": 3, "Unknown": 1})
        self.assertEqual(list(labels[:3]), ["Low"] * 3)

    def test_ici_regimen_labels(self):
        none = {a: 0 for a in subgroups.cfg.ICI}
        self.assertEqual(subgroups.ici_regimen(none), "Other/none")
        self.assertEqual(subgroups.ici_regimen({**none, "IPI": 1, "NIVO": 1}), "Ipilimumab + nivolumab")
        self.assertEqual(subgroups.ici_regimen({**none, "PEMBRO": "Yes"}), "PEMBRO")
        # counted deliberately, unlike the paper's three-agent grouping (cohort.config.ICI)
        self.assertEqual(subgroups.ici_regimen({**none, "ATEZO": 1}), "ATEZO")

    def test_fmt_keeps_counts_integer_and_marks_undefined(self):
        self.assertEqual(subgroups.fmt({"n": 12, "AUROC": 0.71234, "Kappa": np.nan}),
                         "n=12  AUROC=0.712  Kappa=n/a")


class TestSequenceComparisonCI(unittest.TestCase):
    """`bootstrap_sequence_comparison_ci.auroc_ci`, which reads one predictions file."""

    def write(self, y, p):
        path = os.path.join(self.tmp.name, "pred.csv")
        binary_frame(y, p).to_csv(path, index=False)
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_perfect_predictions_format(self):
        text, n = seqcmp.auroc_ci(self.write([0, 0, 0, 1, 1, 1], [.1, .2, .3, .7, .8, .9]), 100, 0)
        self.assertEqual(text, "1.000 (1.000–1.000)")
        self.assertEqual(n, 6)

    def test_single_class_file_returns_none(self):
        self.assertIsNone(seqcmp.auroc_ci(self.write([1, 1, 1], [.2, .5, .9]), 100, 0))


@unittest.skipUnless(HAVE_MPL, "matplotlib is not installed")
class TestCalibration(unittest.TestCase):
    """`plot_diagnostics`: calibration error, Brier score and the AUROC interval."""

    @classmethod
    def setUpClass(cls):
        from brainmets.evaluation import plot_diagnostics
        cls.diag = plot_diagnostics

    def test_perfectly_calibrated(self):
        # four predictions of .25 with one positive, four of .75 with three positive
        y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        p = np.array([.25] * 4 + [.75] * 4)
        ece, brier, pts = self.diag.ece_and_brier(y, p)
        self.assertAlmostEqual(ece, 0.0)
        # per group: one miss at .75^2 and three hits at .25^2 -> (.5625 + 3*.0625) = .75
        self.assertAlmostEqual(brier, 1.5 / 8)
        self.assertEqual([n for *_, n in pts], [4, 4])

    def test_overconfident(self):
        # always 1.0, right half the time: off by .5 in the only bin
        ece, brier, _ = self.diag.ece_and_brier(np.array([0, 1, 0, 1]), np.ones(4))
        self.assertAlmostEqual(ece, 0.5)
        self.assertAlmostEqual(brier, 0.5)

    def test_probability_one_lands_in_the_last_bin(self):
        # the last bin is closed on the right, so p = 1.0 is not dropped
        _, _, pts = self.diag.ece_and_brier(np.array([1]), np.array([1.0]))
        self.assertEqual(len(pts), 1)

    def test_auroc_interval_for_perfect_predictions(self):
        lo, hi = self.diag.bootstrap_auc_ci(np.array([0, 0, 0, 1, 1, 1]),
                                            np.array([.1, .2, .3, .7, .8, .9]), n_boot=100)
        self.assertEqual((lo, hi), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
