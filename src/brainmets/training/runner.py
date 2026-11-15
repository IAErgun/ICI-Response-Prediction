"""The 5-fold cross-validation training run.

This is the loop the eight original per-experiment scripts each carried their own copy
of. It is now driven by a :class:`~training.config.RunConfig`.
"""
import json
import os
import random

import monai
import numpy as np
import torch
from monai.data import Dataset, PersistentDataset, DataLoader
from monai.data.utils import pad_list_data_collate

from . import compat  # noqa: F401 - patches torch.load on import
from . import config as cfg_mod
from .cohort_input import build_data_lists, load_training_cohort, read_data_lists
from .data import define_sampler, get_transforms
from .engine import train


def seed_everything(seed):
    """Pin every RNG the pipeline touches, so a fold is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    monai.utils.misc.set_determinism(seed=seed)
    return np.random.default_rng(seed)


def build_loaders(cfg, train_list, val_list, save_name):
    """Datasets and loaders for one fold.

    With `use_persistent_dataset` the preprocessed volumes are cached on disk per fold;
    the first epoch of a fresh cache is slow, every later one is fast.
    """
    train_tf = get_transforms(cfg.bb_size, list(cfg.seqs), mode="train")
    val_tf = get_transforms(cfg.bb_size, list(cfg.seqs), mode="val")
    if cfg.use_persistent_dataset:
        cache = cfg.cache_dir + save_name + "/"
        train_ds = PersistentDataset(data=train_list, transform=train_tf, cache_dir=cache)
        val_ds = PersistentDataset(data=val_list, transform=val_tf, cache_dir=cache)
    else:
        train_ds = Dataset(data=train_list, transform=train_tf)
        val_ds = Dataset(data=val_list, transform=val_tf)
    # The sampler equalizes the class imbalance in the training set only.
    sampler = define_sampler(train_list, weighting=None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers,
                              collate_fn=pad_list_data_collate, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_val,
                            num_workers=cfg.num_workers,
                            collate_fn=pad_list_data_collate)
    return train_loader, val_loader


def write_fold_metrics(cfg, out_dir, save_name, fold, result):
    """Persist one fold's metrics beside its checkpoint.

    Each fold writes its own file so cross-fold aggregation works whether the folds ran
    sequentially or as five independent one-GPU processes, which cannot see each other.
    """
    key_train, key_val, best, best_epoch, cm, recall = result
    metrics = {
        "save_name": save_name,
        "fold": fold,
        "num_classes": int(cfg.num_classes),
        "metric": cfg.metric_name,
        "best_val_metric": float(best),
        "best_metric_epoch": int(best_epoch),
        "last_train_metric": float(key_train),
        "last_val_metric": float(key_val),
    }
    if cfg.multiclass:
        metrics["best_val_confusion"] = cm          # rows=true, cols=pred, at the checkpoint epoch
        metrics["best_val_per_class_recall"] = recall
    path = os.path.join(out_dir, save_name + "_metrics.json")
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def run(cfg, folds=None, verbose=True):
    """Build the data lists, then train each fold.

    Returns ``{fold: best_val_metric}``. Pass `folds` to override the default, which is
    every fold unless ``FOLD=k`` restricts this process to one (see
    :func:`training.config.folds_to_run`).
    """
    folds = cfg_mod.folds_to_run(cfg) if folds is None else list(folds)
    seed_everything(cfg.seed)

    pruned = load_training_cohort(cfg, verbose=verbose)
    build_data_lists(cfg, pruned, folds, verbose=verbose)

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for fold in folds:
        seed_everything(cfg.seed)
        save_name = cfg.fold_save_name(fold)
        train_list, val_list = read_data_lists(cfg, fold)
        train_loader, val_loader = build_loaders(cfg, train_list, val_list, save_name)
        model_name = f"network_image_{save_name}.pth"
        if verbose:
            print(f"\n=== {save_name}: {len(train_list)} train / {len(val_list)} val, "
                  f"{cfg.task}, seqs={','.join(cfg.seqs)} ===")
        result = train(train_loader, val_loader, cfg.num_classes, list(cfg.seqs),
                       out_dir, model_name,
                       learning_rate=cfg.learning_rate, wd=cfg.weight_decay,
                       max_epochs=cfg.epochs,
                       key_metric_train_limit=cfg.key_metric_train_limit)
        path = write_fold_metrics(cfg, out_dir, save_name, fold, result)
        results[fold] = result[2]
        if verbose:
            print(f"  {save_name}: best val {cfg.metric_name} = {result[2]:.4f} "
                  f"(epoch {result[3]})  -> {path}")
    return results
