"""Run configuration for the CNN experiments.

One :class:`RunConfig` describes an experiment completely. The original pipeline had
one ~860-line script per experiment, differing only in the values collected here —
`seqs`, `task`, and the cache/output directory names — so the scripts were replaced by
these presets over a single training module.

Site paths come from environment variables (see docs/configuration.md); nothing here is
site-specific.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

# Sequence -> the cohort-table column saying that sequence exists for a metastasis
# (T1-post is assumed present for every met, so it has no entry). Imported rather than
# restated: the cohort package is the single definition of the dataset's conventions.
from ..cohort import config as cohort_cfg
from ..cohort.config import PRESENCE

# Radiomic shape columns stored as complex-number strings; the trainer takes their
# magnitude before use.
COMPLEX_SHAPE_COLS = [
    "T1Post_original_shape_Elongation",
    "T1Post_original_shape_Flatness",
    "T1Post_original_shape_LeastAxisLength",
    "T1Post_original_shape_MajorAxisLength",
    "T1Post_original_shape_MinorAxisLength",
]

RESPONSE_LABEL = "RANOResponse_-40_40"
RESPONSE_LABEL_MET = "RANOResponse_-40_40_Met"


# Every path is resolved by the cohort package, so there is one definition of where the
# data lives and where runs write (and one fail-fast message when either is not set).
data_root = cohort_cfg.data_root
folds_csv = cohort_cfg.folds_csv
clin_csv = cohort_cfg.clin_csv
work_root = cohort_cfg.work_root
cache_root = cohort_cfg.cache_root


def default_max_epochs():
    return int(os.environ.get("MAX_EPOCHS", 300))


@dataclass(frozen=True)
class RunConfig:
    """Everything that distinguishes one training run from another.

    The hyperparameter defaults are the published settings; they were identical across
    every original script, so they are defaults here rather than per-preset values.
    """

    # --- what makes this experiment different ---
    seqs: tuple = ("t1post",)
    task: str = "binary"                    # "binary" | "multiclass"
    # Cache and output directory names are given explicitly rather than derived, so each
    # preset keeps reusing the directories its published run already wrote. Deriving
    # them (e.g. "persistent_cache_t1post") would orphan an existing cache.
    cache_subdir: str = "persistent_cache"
    out_subdir: str | None = None           # default: image_only_models_<seq_tag>

    # --- fixed hyperparameters (identical in every original script) ---
    seed: int = 1
    bb_size: int = 120
    batch_size: int = 16
    batch_size_val: int = 16
    num_workers: int = 0
    lower_size_threshold: int = 10
    num_folds: int = 5
    learning_rate: float = 0.005
    weight_decay: float = 0.000025
    max_epochs: int | None = None           # None -> $MAX_EPOCHS, else 300
    use_persistent_dataset: bool = True
    rebuild_data_lists: bool = False

    def __post_init__(self):
        if self.task not in ("binary", "multiclass"):
            raise ValueError(f"task must be 'binary' or 'multiclass', got {self.task!r}")
        if not self.seqs:
            raise ValueError("seqs must name at least one sequence")

    # ------------------------------------------------------------------ derived
    @property
    def seq_tag(self) -> str:
        """Namespace tag so runs over different sequences cannot clobber each other."""
        return "_".join(self.seqs)

    @property
    def multiclass(self) -> bool:
        return self.task == "multiclass"

    @property
    def num_classes(self) -> int:
        """3 for multiclass; 1 for binary (a single logit, not 2 classes)."""
        return 3 if self.multiclass else 1

    @property
    def class_dict(self) -> dict:
        """RANO label -> class index. Binary collapses Stable and Response together."""
        if self.multiclass:
            return {"Progression": 0, "Stable": 1, "Response": 2}
        return {"Progression": 0, "Stable": 1, "Response": 1}

    @property
    def fold_prefix(self) -> str:
        return "Fold_MultiClass_" if self.multiclass else "Fold_Binary_"

    @property
    def metric_name(self) -> str:
        return "cohen_kappa" if self.multiclass else "roc_auc"

    @property
    def key_metric_train_limit(self) -> float:
        """Train-metric floor before a checkpoint may be saved.

        Binary runs must clear 0.50 (above chance AUROC); the multiclass runs were given
        no floor, because weighted kappa starts near 0 and the limit would never clear.
        """
        return 0.0 if self.multiclass else 0.50

    @property
    def epochs(self) -> int:
        return default_max_epochs() if self.max_epochs is None else self.max_epochs

    @property
    def presence_columns(self) -> list:
        """Presence columns to filter on, for the sequences this run needs."""
        return [PRESENCE[s] for s in self.seqs if s in PRESENCE]

    # -------------------------------------------------------------------- paths
    @property
    def cache_dir(self) -> str:
        return os.path.join(cache_root(), self.cache_subdir) + "/"

    @property
    def output_dir(self) -> str:
        sub = self.out_subdir or f"image_only_models_{self.seq_tag}"
        return os.path.join(work_root(), sub) + "/"

    def data_list_path(self, save_name: str, split: str) -> str:
        """Where a fold's cached data list lives.

        The sequence tag is always included. The earliest scripts omitted it and so
        overwrote each other's lists across sequence sets; tagging everything fixes that
        at the cost of one rebuild for anyone holding untagged files.
        """
        return os.path.join(
            work_root(), f"{save_name}_{self.seq_tag}_{split}_data_list.npy")

    def fold_save_name(self, fold: int) -> str:
        return f"{self.fold_prefix}{fold}"

    def with_overrides(self, **kw) -> "RunConfig":
        """Copy with fields replaced (used by the CLI)."""
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


# One preset per experiment in the paper. The names are the CLI's run identifiers.
PRESETS = {
    "t1post-binary": RunConfig(
        seqs=("t1post",), task="binary",
        cache_subdir="persistent_cache"),
    "t1post-multiclass": RunConfig(
        seqs=("t1post",), task="multiclass",
        cache_subdir="persistent_cache"),
    "multiseq-binary": RunConfig(
        seqs=("t1post", "t2", "flair"), task="binary",
        cache_subdir="persistent_cache_t1post_t2_flair"),
    "multiseq-multiclass": RunConfig(
        seqs=("t1post", "t2", "flair"), task="multiclass",
        cache_subdir="persistent_cache_t1post_t2_flair"),
    "t2-binary": RunConfig(
        seqs=("t2",), task="binary",
        cache_subdir="persistent_cache_t2"),
}

# The learning-curve sweep reuses the t1post binary setup but writes elsewhere, so its
# many short runs don't mix with the full-data checkpoints.
LEARNING_CURVE = RunConfig(
    seqs=("t1post",), task="binary",
    cache_subdir="persistent_cache",
    out_subdir="image_only_models_lc_t1post")

# Fractions of each fold's training set to train on, and how many independent
# subsample draws per fraction (>1 is worthwhile at low fractions, where the draw is
# high-variance). The full set has no randomness, so it is always drawn once.
LEARNING_CURVE_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
LEARNING_CURVE_REPEATS = 1


def folds_to_run(cfg: RunConfig) -> list:
    """Which folds this process should handle.

    ``FOLD=k`` restricts the process to one fold, so five processes can run the five
    folds concurrently on separate GPUs. Unset means run them all in sequence.
    """
    fold = os.environ.get("FOLD")
    return [int(fold)] if fold is not None else list(range(cfg.num_folds))
