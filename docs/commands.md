# Command reference

Every entry point is `python -m brainmets.<subpackage>`, and each has its own `--help`.
The installed `brainmets` command reaches the same ones by group:

| `python -m` form | `brainmets` form |
| --- | --- |
| `python -m brainmets.cohort count-patients` | `brainmets cohort count-patients` |
| `python -m brainmets.training run t1post-binary` | `brainmets training run t1post-binary` |
| `python -m brainmets.evaluation.bootstrap_test_ci` | `brainmets evaluation bootstrap-test-ci` |

`brainmets <group> --help` lists a group's commands.

## `brainmets.cohort`: the cohort definition

The exclusion criteria, fold split and patient/round parsing are defined once, in
`cohort/data.py`, and reused by every check, by training and by evaluation. Each check is
a function; run one with `python -m brainmets.cohort <command>`.

| Command | What it reports |
| --- | --- |
| `count-patients` | patients and mets per cohort, plus a train/test leakage check |
| `count-samples` | per-fold train/val counts, optionally cross-checked against the saved `*_data_list.npy` |
| `count-classes` | class balance and zero-rule baselines, development vs held-out test |
| `demographics` | Table 1 demographics, one row per treatment course |
| `table1` | Table 1 rebuilt with one course per patient (each patient's first round) |
| `verify-courses` | is a treatment course uniquely identified? clinical-merge fan-out, competing course keys, per-variable provenance |
| `check-round-age` | diagnoses the negative age delta seen for repeat-course patients |
| `sensitivity-course-key` | counterfactual: age instead of round as the course-selection key |
| `describe-ici` | what the "other or none" regimen courses received, and the repeat-course patients |
| `compare-labels` | do the used and intended label sources agree? (3-class and binary) |
| `make-folds` | generates fold columns for a table that has none (below) |
| `common-cohort-csv` | writes the all-sequence common cohort to CSV |

Commands that re-derive the cohort accept `--lower`, `--num-folds`, `--folds-csv` and
`--require-seqs t2,flair` (for the multisequence cohort). The functions are importable:

```python
from brainmets.cohort import checks, data

checks.count_patients_per_cohort(require_seqs=["t2", "flair"])
courses = data.one_row_per_course(data.load_courses())
```

Every report is PHI-safe: aggregates, counts and column names only, never an individual
patient row or identifier (`tests/test_cohort.py` runs each one on the fixture and checks).
Two commands write row-level files instead of printing a report, `table1 --write-lists`
and `common-cohort-csv`; those files hold image paths, so keep them with the data.

### Generating folds for a new dataset

```bash
python -m brainmets.cohort make-folds --in my_cohort.csv --out my_cohort_folds.csv
```

The columns use the encoding the trainer reads: `0` train, `1` validation, `2` held out
from every fold. So a development metastasis is `1` in exactly one fold and a held-out
one is `2` in all of them. Two properties matter more than equal fold sizes, and both are
enforced:

- **Grouped by patient.** A patient contributes several metastases with correlated
  outcomes. Splitting lesions independently puts a patient on both sides, and the
  validation score comes out optimistic.
- **Stratified by label.** The classes are imbalanced enough that an unstratified split
  can leave a fold with almost none of the minority class.

Both come from scikit-learn's `StratifiedGroupKFold`. The held-out test patients are
carved out first, then the rest are split, and the no-leakage properties are **asserted**
on the result. The command prints per-fold sizes and class balance:

```
 fold   train    val  val patients   validation class balance
    0      70     16             4   Prog=0.31  Stab=0.25  Resp=0.44
    1      68     18             6   Prog=0.28  Stab=0.39  Resp=0.33
...
       whole cohort                  Prog=0.29  Stab=0.34  Resp=0.36
```

Options: `--num-folds`, `--test-fraction` (`0` to cross-validate over everything),
`--group-col` or `--group-from-image-path` to say what a patient is, `--label-col`,
`--seed`, and `--drop-unlabelled` to exclude rows whose label is not one of the three
RANO levels (otherwise they stop the run). The input file is never overwritten.

## `brainmets.training`: the CNN experiments

One experiment is one `RunConfig` preset:

```bash
python -m brainmets.training presets              # list the runs and where they write
python -m brainmets.training run t1post-binary    # all five folds, in sequence
python -m brainmets.training learning-curve       # --fractions, --repeats
```

| Preset | Sequences | Task |
| --- | --- | --- |
| `t1post-binary` | T1-post | binary |
| `t1post-multiclass` | T1-post | 3-class |
| `multiseq-binary` | T1-post + T2 + FLAIR | binary |
| `multiseq-multiclass` | T1-post + T2 + FLAIR | 3-class |
| `t2-binary` | T2 | binary |
| `learning-curve` | T1-post | binary, training-set-size sweep |

`--fold k` (or `FOLD=k` in the environment) restricts a process to one fold, so the five
folds can run concurrently on separate GPUs:

```bash
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f FOLD=$f python -m brainmets.training run t1post-binary \
    > fold_${f}.log 2>&1 &
done
wait
```

On SLURM, `scripts/submit_folds.sh <preset> [fold ...]` submits one GPU job per fold;
set `PARTITION` and `ACCOUNT` at the top of the script or in the environment.

`--seqs`, `--task` and `--max-epochs` override a preset for one-off runs (`MAX_EPOCHS`
also sets the default of 300 from the environment), and `--rebuild-data-lists` regenerates
the cached per-fold data lists.

| Module | Contents |
| --- | --- |
| `config.py` | `RunConfig` and the presets |
| `models.py` | the 3D CNN and the cropped-input helper |
| `data.py` | data lists, the MONAI transform pipeline, class-balancing sampler |
| `transforms.py`, `crop.py` | custom transforms: per-lesion label selection, masking of neighbouring lesions, lesion-centred crop |
| `engine.py` | the training and evaluation loops |
| `cohort_input.py` | cohort table → per-fold data lists |
| `runner.py` | the 5-fold cross-validation run |
| `learning_curve.py` | validation metric vs training-set size |
| `compat.py` | restores pre-2.6 `torch.load`, so the MONAI cache unpickles |

### What a run writes

Everything lands under `$BM_WORK_ROOT`, except the dataset cache:

| Path | Contents |
| --- | --- |
| `<fold>_<seq_tag>_{train,val}_data_list.npy` | resolved image paths and labels per fold |
| `image_only_models_<seq_tag>/network_image_<fold>.pth` | best checkpoint for that fold |
| `image_only_models_<seq_tag>/<fold>_metrics.json` | best and last metric and epoch; for multiclass, the confusion matrix and per-class recall |
| `image_only_models_<seq_tag>/network_image_<fold>_<date>_<time>/` | TensorBoard event files |
| `image_only_models_lc_t1post/learning_curve_results.csv` | one row per (fold, fraction, repeat) |
| `image_only_models_<seq_tag>/test_predictions_<seq_tag>_<task>.csv` | held-out test predictions (`evaluation.export_test_predictions`) |
| `image_only_models_<seq_tag>/oof_predictions.csv` | pooled out-of-fold predictions (`evaluation.export_val_predictions`) |
| `$BM_CACHE_ROOT/<cache_subdir>/<fold>/` | MONAI `PersistentDataset` cache: large, and safe to delete |
| `$BM_CACHE_ROOT/test_cache_<seq_tag>/` | the same for the held-out test set (`evaluation.export_test_predictions`) |

`<fold>` is `Fold_Binary_k` or `Fold_MultiClass_k`; `<seq_tag>` is the sequence set joined
by underscores (`t1post`, `t1post_t2_flair`, `t2`).

## `brainmets.evaluation`: scoring from data

| Module | What it computes |
| --- | --- |
| `aggregate_fold_metrics` | reads each fold's TensorBoard log and reports the checkpoint-selected epoch as a CV table (mean ± SD) |
| `export_test_predictions` | scores the held-out test set with the 5-fold ensemble |
| `export_val_predictions` | pools out-of-fold validation predictions |
| `compute_test_metrics_subgroups` | test metrics split by clinical subgroup |
| `bootstrap_test_ci` | bootstrap confidence intervals on the test set; `--cluster` resamples patients rather than lesions |
| `bootstrap_sequence_comparison_ci` | the same for the T1-post vs T1+T2+FLAIR grid, from a manifest of prediction files |
| `plot_learning_curve` | validation metric vs training-set size |
| `plot_diagnostics` | ROC, precision-recall, calibration and confusion plots from pooled predictions |

Run with no arguments, the tools that read a run's outputs look where the T1-post presets
write (`$BM_WORK_ROOT/image_only_models_t1post`); `--logdir`, `--model-dir` or `--csv`
point them at another run, and then `BM_WORK_ROOT` need not be set at all.
`compute_test_metrics_subgroups` always needs `--pred` and `--task`. The prediction files
carry a `patient_id` column taken from the image path, so treat them as patient data.
