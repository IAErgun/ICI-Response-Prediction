"""Tests for the training run configuration.

These pin the values the eight original per-experiment scripts hardcoded. The scripts
are gone, so this file is what keeps a preset from drifting away from the settings that
produced the published results.

Needs only the standard library — no torch, monai or data. Requires the package to be
installed (`pip install -e .`):

    python -m unittest discover -s tests -p 'test_config.py'

(Plain `discover` would also collect test_cnn.py, which does need torch.)
"""
import os
import unittest

from brainmets.training.config import (LEARNING_CURVE, LEARNING_CURVE_FRACTIONS,
                                       LEARNING_CURVE_REPEATS, PRESETS, RunConfig,
                                       folds_to_run)


class TestDerivedFields(unittest.TestCase):
    def test_binary_is_one_logit_not_two_classes(self):
        cfg = RunConfig(task="binary")
        self.assertEqual(cfg.num_classes, 1)
        self.assertFalse(cfg.multiclass)

    def test_multiclass_is_three(self):
        self.assertEqual(RunConfig(task="multiclass").num_classes, 3)

    def test_binary_collapses_stable_and_response(self):
        self.assertEqual(RunConfig(task="binary").class_dict,
                         {"Progression": 0, "Stable": 1, "Response": 1})

    def test_multiclass_keeps_three_levels(self):
        self.assertEqual(RunConfig(task="multiclass").class_dict,
                         {"Progression": 0, "Stable": 1, "Response": 2})

    def test_fold_prefix_follows_task(self):
        self.assertEqual(RunConfig(task="binary").fold_prefix, "Fold_Binary_")
        self.assertEqual(RunConfig(task="multiclass").fold_prefix, "Fold_MultiClass_")

    def test_metric_follows_task(self):
        self.assertEqual(RunConfig(task="binary").metric_name, "roc_auc")
        self.assertEqual(RunConfig(task="multiclass").metric_name, "cohen_kappa")

    def test_checkpoint_floor(self):
        # Binary must beat chance AUROC before a checkpoint is kept; weighted kappa
        # starts near zero, so the multiclass runs were given no floor.
        self.assertEqual(RunConfig(task="binary").key_metric_train_limit, 0.50)
        self.assertEqual(RunConfig(task="multiclass").key_metric_train_limit, 0.0)

    def test_seq_tag_joins_sequences(self):
        self.assertEqual(RunConfig(seqs=("t1post", "t2", "flair")).seq_tag,
                         "t1post_t2_flair")

    def test_presence_columns_skip_t1post(self):
        # T1-post is present for every met, so it has no presence column to filter on.
        self.assertEqual(RunConfig(seqs=("t1post",)).presence_columns, [])
        self.assertEqual(RunConfig(seqs=("t1post", "t2", "flair")).presence_columns,
                         ["T2_present", "FLAIR_present"])
        self.assertEqual(RunConfig(seqs=("t2",)).presence_columns, ["T2_present"])

    def test_rejects_unknown_task(self):
        with self.assertRaises(ValueError):
            RunConfig(task="regression")

    def test_rejects_empty_sequence_set(self):
        with self.assertRaises(ValueError):
            RunConfig(seqs=())

    def test_is_frozen(self):
        cfg = RunConfig()
        with self.assertRaises(Exception):
            cfg.task = "multiclass"

    def test_with_overrides_ignores_none(self):
        cfg = RunConfig(task="binary")
        self.assertEqual(cfg.with_overrides(task=None).task, "binary")
        self.assertEqual(cfg.with_overrides(task="multiclass").task, "multiclass")


class TestPublishedPresets(unittest.TestCase):
    """The settings that produced the published runs."""

    EXPECTED = {
        "t1post-binary": (("t1post",), "binary", "persistent_cache"),
        "t1post-multiclass": (("t1post",), "multiclass", "persistent_cache"),
        "multiseq-binary": (("t1post", "t2", "flair"), "binary",
                            "persistent_cache_t1post_t2_flair"),
        "multiseq-multiclass": (("t1post", "t2", "flair"), "multiclass",
                                "persistent_cache_t1post_t2_flair"),
        "t2-binary": (("t2",), "binary", "persistent_cache_t2"),
    }

    def test_preset_set_is_complete(self):
        self.assertEqual(set(PRESETS), set(self.EXPECTED))

    def test_presets_match_published_settings(self):
        for name, (seqs, task, cache) in self.EXPECTED.items():
            with self.subTest(preset=name):
                cfg = PRESETS[name]
                self.assertEqual(cfg.seqs, seqs)
                self.assertEqual(cfg.task, task)
                # Explicit, not derived: deriving would orphan the existing cache.
                self.assertEqual(cfg.cache_subdir, cache)

    def test_hyperparameters_are_shared_and_unchanged(self):
        for name, cfg in PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(cfg.seed, 1)
                self.assertEqual(cfg.bb_size, 120)
                self.assertEqual(cfg.batch_size, 16)
                self.assertEqual(cfg.batch_size_val, 16)
                self.assertEqual(cfg.num_workers, 0)
                self.assertEqual(cfg.lower_size_threshold, 10)
                self.assertEqual(cfg.num_folds, 5)
                self.assertEqual(cfg.learning_rate, 0.005)
                self.assertEqual(cfg.weight_decay, 0.000025)

    def test_learning_curve_writes_to_its_own_directory(self):
        # Many short runs must not mix with the full-data checkpoints.
        self.assertEqual(LEARNING_CURVE.out_subdir, "image_only_models_lc_t1post")
        self.assertEqual(LEARNING_CURVE.seqs, ("t1post",))
        self.assertEqual(LEARNING_CURVE.task, "binary")

    def test_learning_curve_sweep_settings(self):
        self.assertEqual(LEARNING_CURVE_FRACTIONS, (0.1, 0.25, 0.5, 1.0))
        self.assertEqual(LEARNING_CURVE_REPEATS, 1)


class TestPaths(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("BM_WORK_ROOT", "BM_CACHE_ROOT", "MAX_EPOCHS", "FOLD")}
        os.environ["BM_WORK_ROOT"] = "/w"
        os.environ["BM_CACHE_ROOT"] = "/c"
        os.environ.pop("MAX_EPOCHS", None)
        os.environ.pop("FOLD", None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cache_and_output_dirs(self):
        cfg = PRESETS["multiseq-binary"]
        self.assertEqual(cfg.cache_dir, "/c/persistent_cache_t1post_t2_flair/")
        self.assertEqual(cfg.output_dir, "/w/image_only_models_t1post_t2_flair/")

    def test_output_dir_derives_from_seq_tag_when_unset(self):
        self.assertEqual(RunConfig(seqs=("t2",)).output_dir, "/w/image_only_models_t2/")

    def test_data_lists_are_always_sequence_tagged(self):
        # Untagged names let different sequence sets overwrite each other's lists.
        cfg = PRESETS["multiseq-binary"]
        self.assertEqual(cfg.data_list_path("Fold_Binary_0", "train"),
                         "/w/Fold_Binary_0_t1post_t2_flair_train_data_list.npy")

    def test_fold_save_name(self):
        self.assertEqual(PRESETS["t1post-multiclass"].fold_save_name(3),
                         "Fold_MultiClass_3")

    def test_max_epochs_default_and_env(self):
        self.assertEqual(RunConfig().epochs, 300)
        os.environ["MAX_EPOCHS"] = "7"
        self.assertEqual(RunConfig().epochs, 7)
        self.assertEqual(RunConfig(max_epochs=12).epochs, 12)   # explicit wins

    def test_fold_env_restricts_to_one_fold(self):
        cfg = RunConfig()
        self.assertEqual(folds_to_run(cfg), [0, 1, 2, 3, 4])
        os.environ["FOLD"] = "2"
        self.assertEqual(folds_to_run(cfg), [2])

    def test_unset_roots_fail_naming_the_variable(self):
        # The trainer creates these directories, so it must not fall back to a
        # placeholder path and quietly write there.
        cfg = PRESETS["t1post-binary"]
        for var, read in (("BM_WORK_ROOT", lambda: cfg.output_dir),
                          ("BM_WORK_ROOT", lambda: cfg.data_list_path("Fold_Binary_0", "train")),
                          ("BM_CACHE_ROOT", lambda: cfg.cache_dir)):
            saved = os.environ.pop(var)
            try:
                with self.subTest(var=var):
                    with self.assertRaises(SystemExit) as ctx:
                        read()
                    self.assertIn(var, str(ctx.exception))
            finally:
                os.environ[var] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
