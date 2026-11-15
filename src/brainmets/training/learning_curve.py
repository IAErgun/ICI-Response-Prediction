"""Learning-curve sweep: val metric as a function of training-set size.

Trains each fold on a sequence of fractions of its training set, keeping the validation
set full so every point measures generalization against the same yardstick. The result
answers whether the model is data-limited: if the curve has flattened, more labelled
metastases will not help and the bottleneck is the representation.

Model initialisation and augmentation are held at the fixed seed; only the subsample
draw varies (by `repeat`), so the curve isolates the effect of n.
"""
import csv
import os

import numpy as np

from . import compat  # noqa: F401 - patches torch.load on import
from . import config as cfg_mod
from .cohort_input import build_data_lists, load_training_cohort, read_data_lists
from .engine import train
from .runner import build_loaders, seed_everything

CSV_COLUMNS = ["fold_prefix", "fold", "fraction", "repeat", "n_train",
               "n_class0", "n_class1", "best_val_metric", "best_metric_epoch",
               "last_val_metric", "metric"]


def stratified_subsample(data_list, fraction, rng):
    """Return a class-stratified subsample of data_list at the given fraction.

    Keeps each class's proportion intact (important at low fractions so a class
    isn't wiped out), and guarantees at least one sample per present class.
    `rng` is a numpy Generator for reproducibility.
    """
    if fraction >= 1.0:
        return list(data_list)
    by_label = {}
    for d in data_list:
        by_label.setdefault(d["label"], []).append(d)
    subsample = []
    for label, items in by_label.items():
        k = max(1, int(round(len(items) * fraction)))
        idx = rng.choice(len(items), size=k, replace=False)
        subsample.extend(items[i] for i in idx)
    rng.shuffle(subsample)
    return subsample

def _open_results_csv(path):
    """Create the results file with its header if it does not exist yet."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(CSV_COLUMNS)
    return path


def run(cfg=None, fractions=None, repeats=None, folds=None, verbose=True):
    """Run the sweep, appending one CSV row per (fold, fraction, repeat).

    Rows are appended as each run finishes, so a crash or a walltime kill leaves the
    completed points intact.
    """
    cfg = cfg_mod.LEARNING_CURVE if cfg is None else cfg
    fractions = cfg_mod.LEARNING_CURVE_FRACTIONS if fractions is None else fractions
    repeats = cfg_mod.LEARNING_CURVE_REPEATS if repeats is None else repeats
    folds = cfg_mod.folds_to_run(cfg) if folds is None else list(folds)

    seed_everything(cfg.seed)
    pruned = load_training_cohort(cfg, verbose=verbose)
    build_data_lists(cfg, pruned, folds, verbose=verbose)

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    results_csv = _open_results_csv(os.path.join(out_dir, "learning_curve_results.csv"))

    for fraction in fractions:
        # The full training set involves no random draw, so repeating it adds nothing.
        n_repeats = 1 if fraction >= 1.0 else repeats
        for repeat in range(n_repeats):
            for fold in folds:
                # Training randomness is pinned to the seed; only the draw varies.
                seed_everything(cfg.seed)
                rng = np.random.default_rng(cfg.seed + repeat)
                save_name = cfg.fold_save_name(fold)
                train_list, val_list = read_data_lists(cfg, fold)
                train_list = stratified_subsample(train_list, fraction, rng)
                labels = [d["label"] for d in train_list]
                n_train = len(train_list)
                n_class0, n_class1 = labels.count(0), labels.count(1)
                if verbose:
                    print(f"{save_name} | frac={fraction} rep={repeat} | "
                          f"n_train={n_train} (c0={n_class0}, c1={n_class1})")

                # The cache is keyed per fold on the FULL set, so subset items hit it.
                train_loader, val_loader = build_loaders(
                    cfg, train_list, val_list, save_name)

                frac_pct = int(round(fraction * 100))
                model_name = (f"network_image_lc_{save_name}"
                              f"_frac{frac_pct}_rep{repeat}.pth")
                key_train, key_val, best, best_epoch, _cm, _recall = train(
                    train_loader, val_loader, cfg.num_classes, list(cfg.seqs),
                    out_dir, model_name,
                    learning_rate=cfg.learning_rate, wd=cfg.weight_decay,
                    max_epochs=cfg.epochs,
                    key_metric_train_limit=cfg.key_metric_train_limit)

                # best_val_metric == -1 means no checkpoint cleared the train-metric
                # floor, so this point has no usable validation score.
                with open(results_csv, "a", newline="") as fh:
                    csv.writer(fh).writerow(
                        [cfg.fold_prefix, fold, fraction, repeat, n_train,
                         n_class0, n_class1, best, best_epoch, key_val,
                         cfg.metric_name])
    return results_csv
