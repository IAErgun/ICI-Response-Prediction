"""Scoring and diagnostics computed from data.

Every module here is a command-line entry point with its own ``--help``:

    python -m brainmets.evaluation.aggregate_fold_metrics
    python -m brainmets.evaluation.export_test_predictions --seqs t1post --task binary
    python -m brainmets.evaluation.bootstrap_test_ci

| Module | What it does |
| --- | --- |
| `aggregate_fold_metrics` | reads each fold's TensorBoard log into a CV table (mean ± SD) |
| `export_test_predictions` | scores the held-out test set with the 5-fold ensemble |
| `export_val_predictions` | pools out-of-fold validation predictions |
| `compute_test_metrics_subgroups` | test metrics split by clinical subgroup |
| `bootstrap_test_ci` | bootstrap confidence intervals on the test set |
| `bootstrap_sequence_comparison_ci` | the same for the T1-post vs T1+T2+FLAIR grid |
| `plot_learning_curve` | validation metric vs training-set size |
| `plot_diagnostics` | ROC, PR, calibration and confusion plots from pooled predictions |

These read run outputs and the cohort table. Modules that select cohort rows take them
from :mod:`brainmets.cohort.data`, so they score exactly the rows the trainer used.
"""
