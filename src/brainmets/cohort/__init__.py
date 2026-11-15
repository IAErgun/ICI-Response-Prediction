"""Cohort definition, QC checks and Table 1 construction.

The training pipeline's exclusion criteria, fold split and patient/round parsing are
defined once in :mod:`cohort.data` and reused by everything else here, so a change to
the cohort rule propagates to every check, table and export.

Modules
-------
:mod:`cohort.config`
    Input paths (from environment variables) and the cohort constants.
:mod:`cohort.data`
    Loading, pruning, cohort assignment and the clinical merge.
:mod:`cohort.demographics`
    Table 1 aggregation (median [IQR], n (%), regimen labels).
:mod:`cohort.checks`
    QC and sensitivity checks, one function each.
:mod:`cohort.table1`
    The one-course-per-patient Table 1 rebuild.
:mod:`cohort.make_folds`
    Patient-grouped, label-stratified fold columns for a dataset that has none.
:mod:`cohort.export`
    Cohort subset CSV writers.

Every report is PHI-SAFE: aggregates, counts and column names only.

Run a check from the command line with ``python -m brainmets.cohort <check>``, or import it:

>>> from brainmets.cohort import checks
>>> checks.count_patients_per_cohort(require_seqs=["t2", "flair"])
"""
import importlib

__all__ = ["checks", "config", "data", "demographics", "export", "make_folds", "table1"]


def __getattr__(name):
    """Import submodules on first use (PEP 562).

    Loading them eagerly would mean that reading a single constant out of
    `cohort.config` pulled in pandas and numpy through `cohort.checks`; the training
    package does exactly that, and wants to stay importable without them.
    """
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
