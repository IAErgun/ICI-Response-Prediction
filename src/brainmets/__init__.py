"""Brain-metastasis ICI response prediction.

Subpackages
-----------
:mod:`brainmets.cohort`
    The cohort definition — which metastases count — plus the QC checks and Table 1
    aggregation built on it. Everything else derives its rows from here.
:mod:`brainmets.training`
    The 3D CNN experiments: models, data pipeline, training loop, run presets.
:mod:`brainmets.evaluation`
    Scoring from data: cross-fold aggregation, held-out test predictions, bootstrap
    confidence intervals, subgroup metrics and diagnostic plots.

The dependency direction is one-way: `cohort` knows about nothing else, and `training`
and `evaluation` build on it.

Command-line entry points, either as ``python -m`` or through the installed command:

    python -m brainmets.cohort <check>                 brainmets cohort <check>
    python -m brainmets.training run <preset>          brainmets training run <preset>
    python -m brainmets.evaluation.<tool>              brainmets evaluation <tool>
"""
__all__ = ["cohort", "training", "evaluation"]
