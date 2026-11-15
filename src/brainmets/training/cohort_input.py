"""Turn the cohort table into the per-fold data lists the trainer consumes.

The exclusion criteria are not re-implemented here: :func:`cohort.data.prune` is the
single definition of which metastases count, shared with the Table 1 and QC reports, so
the trainer and the published cohort numbers cannot drift apart.
"""
import os

import numpy as np
import pandas as pd

from ..cohort.data import prune, require_sequences

from . import config as cfg_mod
from .data import load_data


def load_training_cohort(cfg, verbose=True):
    """Read the cohort table, apply the exclusions, and return the usable metastases.

    Takes the magnitude of the complex-valued radiomic shape columns (they are stored
    as strings like ``(1.2+0j)``), then applies the shared prune and this run's
    sequence-presence requirement.
    """
    df = pd.read_csv(cfg_mod.folds_csv(), index_col=0)
    for col in cfg_mod.COMPLEX_SHAPE_COLS:
        if col in df.columns:
            df[col] = [abs(complex(x)) for x in df[col]]

    pruned = prune(df, lower=cfg.lower_size_threshold, verbose=verbose)
    # Require every sequence this run needs; T1-post is assumed present for all. Without
    # this, load_data raises a KeyError on a met with no T2/FLAIR volume on disk.
    pruned = require_sequences(pruned, [s for s in cfg.seqs if s in cfg_mod.PRESENCE])
    pruned = pruned.reset_index(drop=True)
    if verbose:
        print(f"cohort for {cfg.seq_tag} / {cfg.task}: {len(pruned)} metastases")
    return pruned


def build_data_lists(cfg, pruned, folds, verbose=True):
    """Write each fold's train/val data list to .npy, skipping ones already built.

    Delete the .npy files (or pass ``rebuild_data_lists=True``) after changing the
    cohort table, the folds, the pruning thresholds, the sequence set, or the class map
    — the lists cache the resolved image paths and labels, not just the split.
    """
    os.makedirs(cfg_mod.work_root(), exist_ok=True)
    for fold in folds:
        save_name = cfg.fold_save_name(fold)
        train_npy = cfg.data_list_path(save_name, "train")
        val_npy = cfg.data_list_path(save_name, "val")
        if (not cfg.rebuild_data_lists
                and os.path.exists(train_npy) and os.path.exists(val_npy)):
            if verbose:
                print(f"  {save_name}: reusing cached data lists")
            continue
        if verbose:
            print(f"Building data lists for {save_name}...")
        fold_col = f"Fold_{fold}"
        train_list = load_data(pruned[pruned[fold_col] == 0], cfg_mod.data_root(),
                               cfg.class_dict, list(cfg.seqs), cfg_mod.RESPONSE_LABEL_MET)
        val_list = load_data(pruned[pruned[fold_col] == 1], cfg_mod.data_root(),
                             cfg.class_dict, list(cfg.seqs), cfg_mod.RESPONSE_LABEL_MET)
        if verbose:
            print(f"  {save_name}: {len(train_list)} train, {len(val_list)} val metastases")
        np.save(train_npy, train_list)
        np.save(val_npy, val_list)


def read_data_lists(cfg, fold):
    """Load one fold's cached train/val data lists."""
    save_name = cfg.fold_save_name(fold)
    train = list(np.load(cfg.data_list_path(save_name, "train"), allow_pickle=True))
    val = list(np.load(cfg.data_list_path(save_name, "val"), allow_pickle=True))
    return train, val
