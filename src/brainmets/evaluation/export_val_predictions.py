"""Export pooled out-of-fold (OOF) validation predictions for the t1post binary CNN.

For each of the 5 folds: load that fold's checkpoint, run inference on that fold's
VAL set, and record one row per metastasis. Since each met is in val exactly once
across the 5 folds, the concatenation is a clean OOF prediction over the whole
training cohort -- the input for ROC / PR / calibration / confusion / subgroup
figures (and later the DeLong test vs the foundation model).

Reuses the trainer's Custom3DCNN, get_transforms(mode='val'), and
use_cropped_images_fn so inference matches how the model was validated.

Output CSV, one row per metastasis:
    fold, patient_id, treatment_round, met, y_true, y_prob

``patient_id`` is whatever the image path carries (see docs/configuration.md). If those
are real identifiers rather than study ids, the file is patient-level data: keep it with
the data, not with the code. The repository's .gitignore and pre-commit hook refuse CSVs
for that reason.

Usage:
    python -m brainmets.evaluation.export_val_predictions

The defaults are where ``python -m brainmets.training run t1post-binary`` writes: its
checkpoints, its data lists and its dataset cache. ``--model-dir``, ``--tag`` and
``--cache-dir`` point it at a run laid out differently.
"""
import os
import csv
import argparse

import numpy as np
import torch
from monai.data import PersistentDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import Activations, Compose

# The model and data pipeline come from the training package, so this scores with exactly
# the code that trained.
from ..cohort import config as cohort_cfg
from ..training import compat  # noqa: F401 - patches torch.load for the MONAI dataset cache
from ..training.config import PRESETS
from ..training.data import get_transforms
from ..training.models import Custom3DCNN, use_cropped_images_fn

PRESET = PRESETS["t1post-binary"]       # the run whose validation predictions are pooled


def main():
    """Pool each fold's validation predictions into one out-of-fold predictions file.

    Every metastasis is in validation for exactly one fold, so the pooled file covers the
    development cohort once, each row scored by a model that did not train on it.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=None,
                   help="base dir holding the data lists and model dirs (default: $BM_WORK_ROOT)")
    p.add_argument("--model-dir", default=None,
                   help=f"dir with the fold checkpoints (default: <log-dir>/image_only_models_{PRESET.seq_tag})")
    p.add_argument("--cache-dir", default=None,
                   help=f"PersistentDataset cache (default: $BM_CACHE_ROOT/{PRESET.cache_subdir}/)")
    p.add_argument("--tag", default=PRESET.seq_tag,
                   help="sequence tag in the data-list filenames (default: %(default)s; "
                        "'' for untagged lists)")
    p.add_argument("--out", default=None, help="output CSV (default: <model-dir>/oof_predictions.csv)")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--bb-size", type=int, default=120)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Resolved here rather than at import, so explicit flags work without the env vars.
    log_dir = args.log_dir or cohort_cfg.work_root()
    model_dir = args.model_dir or os.path.join(log_dir, f"image_only_models_{PRESET.seq_tag}")
    cache_dir = args.cache_dir or PRESET.cache_dir
    out_csv = args.out or os.path.join(model_dir, "oof_predictions.csv")
    seq_of_interest = list(PRESET.seqs)
    tag = ("_" + args.tag) if args.tag else ""

    post_pred = Compose([Activations(sigmoid=True)])
    rows = []

    for fold in range(args.num_folds):
        save_name = f"Fold_Binary_{fold}"
        val_npy = os.path.join(log_dir, f"{save_name}{tag}_val_data_list.npy")
        ckpt = os.path.join(model_dir, f"network_image_{save_name}.pth")
        if not os.path.exists(ckpt):
            print(f"!! missing checkpoint, skipping fold {fold}: {ckpt}")
            continue
        if not os.path.exists(val_npy):
            print(f"!! missing val data list, skipping fold {fold}: {val_npy}")
            continue

        val_list = list(np.load(val_npy, allow_pickle=True))
        val_ds = PersistentDataset(
            data=val_list,
            transform=get_transforms(args.bb_size, seq_of_interest, mode="val"),
            cache_dir=os.path.join(cache_dir, save_name) + "/",
        )
        # batch_size=1 keeps the per-sample shapes simple and matches use_cropped_images_fn.
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=pad_list_data_collate)

        model = Custom3DCNN(num_classes=1, in_channels=len(seq_of_interest)).to(args.device)
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        n = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.cat([batch[s].to(args.device) for s in seq_of_interest], dim=1)
                out = use_cropped_images_fn(inputs, batch, model, device=args.device, use_cropped_images=True)
                prob = float(post_pred(out).detach().cpu().numpy().flatten()[0])
                label = int(np.asarray(batch["label"]).flatten()[0])
                met = batch["met"]
                met = int(met[0].item()) if torch.is_tensor(met) else int(np.asarray(met).flatten()[0])
                rows.append({
                    "fold": fold,
                    "patient_id": str(batch["patient_id"][0]),
                    "treatment_round": str(batch["treatment_round"][0]),
                    "met": met,
                    "y_true": label,
                    "y_prob": prob,
                })
                n += 1
        print(f"fold {fold}: {n} val mets")

    if not rows:
        raise SystemExit("No predictions written -- check model-dir / data-list --tag / paths.")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "patient_id", "treatment_round", "met", "y_true", "y_prob"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} pooled OOF predictions -> {out_csv}")


if __name__ == "__main__":
    main()
