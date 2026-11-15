# Predicting immune-checkpoint-inhibitor response in brain metastases

![CI](https://github.com/IAErgun/ICI-Response-Prediction/actions/workflows/ci.yml/badge.svg)

Code from a study comparing **3D convolutional neural networks** against **tabular and
foundation-model classifiers** (TabPFN, CatBoost, conventional ML) for predicting RANO-BM
response of brain metastases to immune checkpoint inhibitor (ICI) therapy, at both the
**lesion** and **patient** level, with internal cross-validation, a held-out internal
test set, and external-site validation.

This repository holds the imaging arm and the reporting around it:

- the 3D CNN experiments: models, data pipeline, training, learning-curve sweep
- the cohort definition, with its QC checks and descriptive statistics
- evaluation: fold aggregation, held-out test scoring, bootstrap confidence intervals,
  subgroup metrics, diagnostic plots

The tabular models on clinical and radiomic features were developed by collaborators in a
separate pipeline and are **not** part of this repository. The code that renders the
paper's tables and figures carries the study's results, so it will be added when the
paper is out. No patient data, imaging, model weights, results or derived cohort tables
are included: this repository is code only.

## Try it without patient data

The cohort logic runs against a synthetic, schema-correct cohort that reproduces the
awkward parts of the real one (patients with several treatment courses, a held-out group,
inconsistent ages for the QC checks to find):

```bash
pip install -e .
python tests/fixtures.py /tmp/synthetic          # writes three CSVs and an env.sh
source /tmp/synthetic/env.sh

python -m brainmets.cohort count-patients        # cohort sizes, plus a train/test leakage check
python -m brainmets.cohort count-classes         # class balance and zero-rule baselines
python -m brainmets.cohort make-folds --in "$BM_FOLDS_CSV" --out /tmp/synthetic/refolded.csv
python -m unittest discover -s tests             # the full suite, also data-free
```

## Layout

```
src/brainmets/
  cohort/       the cohort definition (which metastases count), QC checks, Table 1
                aggregates, and fold generation for new datasets
  training/     the 3D CNN experiments: models, data pipeline, training loop, presets
  evaluation/   scoring from data: fold aggregation, test predictions, bootstrap CIs,
                subgroup metrics, diagnostic plots
scripts/        SLURM submission (one job per CV fold, one GPU each)
tests/          synthetic fixture, and tests for the cohort, evaluation metrics,
                run presets, CNN and package imports
docs/           configuration and command reference
```

Dependencies run one way: `cohort` imports nothing else from the package, and `training`
and `evaluation` build on it.

## How the pipeline fits together

```
BM_FOLDS_CSV  (one row per metastasis: labels, fold assignment, image paths, radiomics)
BM_CLIN_CSV   (one row per treatment course: demographics, treatment, labs)
      |
      |  cohort.data.prune  - the single definition of which metastases count
      |
      +--> cohort/       QC reports, Table 1 aggregates            -> printed, PHI-safe
      |
      +--> training/     per-fold data lists -> MONAI cache -> 5-fold training
      |                        -> checkpoint, metrics JSON, TensorBoard per fold
      |
      +--> evaluation/   fold aggregation, held-out test scoring, bootstrap CIs,
                         subgroup metrics, diagnostic plots
```

Training, evaluation and the cohort reports all take their rows from the same
`cohort.data.prune`, so the model's cohort and the published cohort counts cannot drift
apart.

## Running on real data

Inputs and outputs are located through environment variables (`BM_FOLDS_CSV`,
`BM_CLIN_CSV`, `BM_DATA_ROOT`, `BM_WORK_ROOT`, `BM_CACHE_ROOT`), described with the
expected table schema in [docs/configuration.md](docs/configuration.md). Training needs a
CUDA GPU: install a CUDA build of PyTorch from pytorch.org before `pip install -e .`.

```bash
# 1. cohort accounting
python -m brainmets.cohort count-patients
python -m brainmets.cohort count-samples

# 2. train one preset: all five folds in sequence, or one job per fold on SLURM
python -m brainmets.training run t1post-binary
bash scripts/submit_folds.sh t1post-binary

# 3. cross-validation metrics, then the held-out test set
python -m brainmets.evaluation.aggregate_fold_metrics
python -m brainmets.evaluation.export_test_predictions --seqs t1post --task binary \
    --model-dir "$BM_WORK_ROOT/image_only_models_t1post"
python -m brainmets.evaluation.bootstrap_test_ci --cluster
python -m brainmets.evaluation.compute_test_metrics_subgroups --task binary \
    --pred "$BM_WORK_ROOT/image_only_models_t1post/test_predictions_t1post_binary.csv"

# 4. baseline characteristics, one treatment course per patient
python -m brainmets.cohort table1
```

Installing also provides a `brainmets` command with the same groups, e.g.
`brainmets cohort count-patients` or `brainmets evaluation bootstrap-test-ci --cluster`;
`brainmets --help` lists them.

`python -m brainmets.training presets` lists the experiments (T1-post, multisequence and
T2; binary and 3-class; and a learning-curve sweep). If your cohort table has no fold
columns, `python -m brainmets.cohort make-folds` generates patient-grouped,
label-stratified ones. Every command, option and output file is listed in
[docs/commands.md](docs/commands.md).

## Tests and CI

```bash
python -m unittest discover -s tests
```

| Test file | Covers |
| --- | --- |
| `test_cohort.py` | exclusions, fold split and fold generation, patient/round parsing, regimen labels |
| `test_evaluation.py` | AUROC, kappa, calibration and bootstrap intervals, against hand-computed values |
| `test_config.py` | the run presets: the settings that produced the published results |
| `test_cli.py` | the `brainmets` command: dispatch, tool listing, error messages |
| `test_cnn.py` | forward and backward passes of the CNN, every norm type (skips without torch) |
| `test_imports.py` | every module imports, in a fresh interpreter, without writing anything |

`test_cohort.py` also holds the cohort reports to their promise of printing aggregates
only: each is run on the fixture and its output searched for a patient id or image path.

GitHub Actions runs two jobs on every push. The fast one runs lint, the suite and a scan
for site paths or identifier-like numbers on Python 3.10 and 3.12; it installs without the
deep-learning stack, so the tests that need torch skip and it takes under a minute. The
second installs everything (CPU-only PyTorch) and runs the suite again with nothing
skipped, which is what catches a broken import in the training code. To reproduce the fast
job locally:

```bash
pip install --no-deps -e . && pip install numpy pandas scikit-learn matplotlib ruff==0.17.4
ruff check src tests && python -m unittest discover -s tests
```

Enable the pre-commit hook once per clone with `git config core.hooksPath .githooks`. It
runs the same identifier scan (`scripts/check_identifiers.sh`) on what is about to be
committed, plus ruff, so a leak is stopped before it becomes public rather than after CI
sees it. Site-specific patterns go in a git-ignored `.identifiers-local`.

## Differences from the paper and known limitations

- **ICI regimen labels count all five agents** (ipilimumab, nivolumab, pembrolizumab,
  atezolizumab, durvalumab), which is how Table 1 lists them. The paper's subgroup
  analysis grouped regimens by the first three only: atezolizumab and durvalumab courses
  were too sparse in its cohort to support subgroups of their own, so they were reported
  under "Other or none". Counting all five generalises better to other datasets, but
  `compute_test_metrics_subgroups` will not reproduce the paper's regimen subgroups
  exactly. The paper's three are kept as `cohort.config.BASE3`.
- **Undefined subgroup metrics print as `n/a`.** In a subgroup containing a single class,
  scikit-learn returns confident-looking values for metrics that are actually undefined
  (AUPRC 1.0 or 0.0, specificity 0.0, kappa 0.0). `compute_test_metrics_subgroups`
  reports those as `n/a` instead. No subgroup in the paper's data was affected, and
  subgroups containing both classes are computed exactly as before.
- **Anything that loads images needs a CUDA GPU.** The training loop and the transform
  pipeline place tensors on `cuda` directly, as the runs that produced the results did,
  so training and the two prediction exporters cannot run on CPU. The cohort reports, the
  metrics and bootstrap intervals and the plots run anywhere.
- **`BM_DATA_ROOT` is checked, not used to build paths.** Image locations come from the
  cohort table's `PreTreatment_Image` column, so the table has to point at local storage.

## Documentation

- [docs/configuration.md](docs/configuration.md): installation, environment variables,
  the expected cohort table and image layout
- [docs/commands.md](docs/commands.md): every command, the training presets, and what a
  run writes

## License

MIT; see [LICENSE](LICENSE). One file, `src/brainmets/training/crop.py`, is adapted from
[MONAI](https://github.com/Project-MONAI/MONAI) and remains under the Apache License 2.0:
the notice is at the top of that file and the licence text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
