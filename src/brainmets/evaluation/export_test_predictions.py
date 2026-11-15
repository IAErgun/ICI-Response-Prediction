"""Evaluate trained CNN fold-checkpoints on the held-out INTERNAL TEST set (the mets
that are ∉{0,1} in all Fold_k columns -- never seen in training).

For a given experiment (sequences + task + cohort), it:
  1. builds a fresh test data list from the held-out rows (prune + optional presence filter),
  2. runs each of the 5 fold checkpoints on that test set,
  3. reports per-fold metrics (mean±SD) AND the 5-fold probability ENSEMBLE,
  4. writes per-met ensemble predictions to CSV (input for figures / DeLong vs foundation).

Reuses the trainer's load_data / get_transforms(mode='val') / use_cropped_images_fn /
Custom3DCNN so inference matches how the model was validated.

Examples:
    # t1post binary (parallel checkpoints):
    python -m brainmets.evaluation.export_test_predictions --seqs t1post --task binary \
        --model-dir $BM_WORK_ROOT/image_only_models_t1post
    # multiseq multiclass:
    python -m brainmets.evaluation.export_test_predictions --seqs t1post,t2,flair --task multiclass --require-seqs t2,flair \
        --model-dir $BM_WORK_ROOT/image_only_models_t1post_t2_flair
"""
import os
import csv
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai.data import PersistentDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import Activations, Compose
from sklearn.metrics import roc_auc_score, cohen_kappa_score

# The model and data pipeline come from the training package, so this scores with exactly
# the code that trained.
from ..cohort import config as cohort_cfg
from ..cohort import data as cohort_data
from ..training import compat  # noqa: F401 - patches torch.load for the MONAI dataset cache
from ..training import config as training_cfg
from ..training.data import get_transforms, load_data
from ..training.models import Custom3DCNN, use_cropped_images_fn

LABEL_COL = cohort_cfg.LABEL_MET


def build_test_df(seqs_require: list[str], lower: int | None = None,
                  num_folds: int | None = None) -> pd.DataFrame:
    """The held-out test metastases: never in train or validation of any fold.

    The exclusion criteria and the fold split come from :mod:`brainmets.cohort.data`, so
    the rows scored here are exactly the rows the trainer held out.
    """
    d = cohort_data.load_folds()
    for c in training_cfg.COMPLEX_SHAPE_COLS:
        if c in d.columns:
            d[c] = [abs(complex(x)) for x in d[c]]
    d = cohort_data.require_sequences(cohort_data.prune(d, lower=lower), seqs_require)
    _, test = cohort_data.split_cv_test(d, num_folds)
    return test.reset_index(drop=True)


def metrics(y: np.ndarray, prob: np.ndarray, task: str) -> dict[str, float]:
    """The task's headline metrics for the ensembled predictions."""
    if task == "binary":
        return {"auroc": roc_auc_score(y, prob)}
    pred = prob.argmax(1)
    return {"kappa": cohen_kappa_score(y, pred, weights="linear"),
            "auroc_ovr": roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=[0, 1, 2])}


def main():
    """Score the held-out test set with the 5-fold ensemble and write the predictions."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="t1post", help="comma list, e.g. t1post or t1post,t2,flair")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--require-seqs", default="", help="cohort presence filter, e.g. t2,flair for multiseq")
    ap.add_argument("--model-dir", required=True, help="dir with network_image_<prefix><k>.pth checkpoints")
    ap.add_argument("--prefix", default=None, help="default Fold_Binary_ / Fold_MultiClass_ from --task")
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--bb-size", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None,
                    help="PersistentDataset cache for the test set "
                         "(default: $BM_CACHE_ROOT/test_cache_<seqs>/)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <model-dir>/test_predictions_<seqs>_<task>.csv)")
    args = ap.parse_args()

    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    require = [s.strip() for s in args.require_seqs.split(",") if s.strip()]
    prefix = args.prefix or ("Fold_Binary_" if args.task == "binary" else "Fold_MultiClass_")
    num_classes = 1 if args.task == "binary" else 3
    class_dict = {"Progression": 0, "Stable": 1, "Response": 1} if args.task == "binary" \
        else {"Progression": 0, "Stable": 1, "Response": 2}
    seq_tag = "_".join(seqs)
    out_csv = args.out or os.path.join(args.model_dir, f"test_predictions_{seq_tag}_{args.task}.csv")
    post_pred = Compose([Activations(sigmoid=True)]) if args.task == "binary" \
        else Compose([Activations(softmax=True, dim=1)])

    test_df = build_test_df(require)
    test_list = load_data(test_df, training_cfg.data_root(), class_dict, seqs, LABEL_COL)
    print(f"held-out test: {len(test_list)} mets ({args.task}, seqs={seqs}, require={require or 'none'})")

    cache_dir = args.cache_dir or os.path.join(cohort_cfg.cache_root(), f"test_cache_{seq_tag}")
    cache_dir = cache_dir.rstrip("/") + "/"
    ds = PersistentDataset(data=test_list, transform=get_transforms(args.bb_size, seqs, mode="val"), cache_dir=cache_dir)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=pad_list_data_collate)

    # Collect y_true + ids once, and each fold's probs (loader order is deterministic).
    y_true, mets, pids, fold_probs = None, [], [], []
    for k in range(args.num_folds):
        ckpt = os.path.join(args.model_dir, f"network_image_{prefix}{k}.pth")
        if not os.path.exists(ckpt):
            print(f"!! missing checkpoint, skipping fold {k}: {ckpt}")
            continue
        model = Custom3DCNN(num_classes=num_classes, in_channels=len(seqs)).to(args.device)
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        probs, ys, this_mets, this_pids = [], [], [], []
        # fold 0 builds the test cache (slow); folds 1-4 reuse it (fast).
        desc = f"fold {k} inference" + (" (+cache build)" if k == 0 else "")
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, total=len(test_list)):
                inputs = torch.cat([batch[s].to(args.device) for s in seqs], dim=1)
                out = use_cropped_images_fn(inputs, batch, model, device=args.device, use_cropped_images=True)
                p = post_pred(out if num_classes == 1 else out.reshape(1, num_classes))
                p = p.detach().cpu().numpy().reshape(-1)              # [1] binary or [3] multiclass
                probs.append(p[0] if num_classes == 1 else p)
                ys.append(int(np.asarray(batch["label"]).flatten()[0]))
                m = batch["met"]
                this_mets.append(int(m[0].item()) if torch.is_tensor(m) else int(np.asarray(m).flatten()[0]))
                this_pids.append(str(batch["patient_id"][0]))
        fold_probs.append(np.array(probs))
        if y_true is None:
            y_true, mets, pids = np.array(ys), this_mets, this_pids
        m = metrics(y_true, fold_probs[-1], args.task)
        print(f"  fold {k}: " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in m.items()))

    if not fold_probs:
        raise SystemExit("No checkpoints found -- check --model-dir / --prefix.")
    fold_probs = np.stack(fold_probs)                                 # [n_folds, N] or [n_folds, N, 3]

    # per-fold summary + ensemble
    print(f"\n=== {args.task} test metrics (n={len(y_true)}) ===")
    keys = ["auroc"] if args.task == "binary" else ["kappa", "auroc_ovr"]
    for key in keys:
        vals = np.array([metrics(y_true, fp, args.task)[key] for fp in fold_probs])
        print(f"  per-fold {key}: {vals.mean():.4f} ± {vals.std():.4f}  {[round(float(v),4) for v in vals]}")
    ens = fold_probs.mean(0)
    ens_m = metrics(y_true, ens, args.task)
    print("  ENSEMBLE (mean prob): " + ", ".join(f"{kk}={vv:.4f}" for kk, vv in ens_m.items()))

    # write per-met ensemble predictions (figure / DeLong input)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        if args.task == "binary":
            w.writerow(["met", "patient_id", "y_true", "y_prob_ensemble"])
            for i in range(len(y_true)):
                w.writerow([mets[i], pids[i], int(y_true[i]), float(ens[i])])
        else:
            w.writerow(["met", "patient_id", "y_true", "p0", "p1", "p2", "pred"])
            for i in range(len(y_true)):
                w.writerow([mets[i], pids[i], int(y_true[i]), float(ens[i, 0]), float(ens[i, 1]),
                            float(ens[i, 2]), int(ens[i].argmax())])
    print(f"\nwrote ensemble test predictions -> {out_csv}")


if __name__ == "__main__":
    main()
