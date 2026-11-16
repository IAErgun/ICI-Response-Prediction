# Predicting immune-checkpoint-inhibitor response in brain metastases

![CI](https://github.com/IAErgun/ICI-Response-Prediction/actions/workflows/ci.yml/badge.svg)

## The question

**Can conventional MRI, on its own, carry a meaningful signal for whether a brain
metastasis will respond to immune checkpoint inhibitor (ICI) therapy?**

Every patient with brain metastases already has a pre-treatment MRI. If the images
predict response, that is a biomarker with no extra cost to the patient. If they do not,
that is worth knowing before more effort goes into image models for this task.

This repository is the imaging arm of a study built to answer that. It trains a 3D
convolutional network on the pre-treatment scan of each metastasis and asks whether it
can predict that lesion's RANO-BM response (progression, stable, or response) after
treatment. Much of the code exists to make sure that an answer, in either direction, can
be trusted:

| A "signal" could be an artefact of... | So the code... |
| --- | --- |
| class imbalance: a model that always says "no progression" looks accurate | reports zero-rule baselines next to every cohort (`cohort count-classes`) and scores with AUROC and weighted kappa, not accuracy |
| leakage: lesions from one patient are correlated, and a random split puts them on both sides | splits by patient, never by lesion, and asserts it (`cohort count-patients`, `cohort make-folds`) |
| tuning on the evaluation data: checkpoints are chosen on validation folds | keeps a held-out test set that no fold ever sees, scored once by the five-fold ensemble |
| too little data, not too little information | sweeps the training-set size (`training learning-curve`) to see whether the curve is still rising |
| the wrong MRI sequence | repeats the experiment with T2 and FLAIR added, on the lesions that have all three, so only the input changes |
| overconfident intervals: several lesions per patient are not independent observations | bootstraps confidence intervals by resampling patients (`bootstrap_test_ci --cluster`) |
| one subgroup carrying the result | reports test metrics within clinical subgroups, and prints `n/a` where a metric is undefined |
| a different cohort in each analysis | defines which metastases count once, in one function that training, evaluation and the cohort tables all call |

## What is here, and what is not

Here:

- the 3D CNN experiments: models, data pipeline, training, learning-curve sweep
- the cohort definition, with its quality checks and descriptive statistics
- evaluation: fold aggregation, held-out test scoring, bootstrap confidence intervals,
  subgroup metrics, diagnostic plots

Not here:

- The comparison models. In the study the CNN is compared with tabular and
  foundation-model classifiers (TabPFN, CatBoost, conventional ML) on clinical and
  radiomic features. Those were developed by collaborators in a separate pipeline.
- The code that renders the paper's tables and figures. It carries the study's results,
  so it will be added when the paper is out.
- Any patient data, imaging, model weights, results or derived cohort tables. This
  repository is code only.

## How the repository is organized, and why

```
src/brainmets/
  cohort/       which metastases count, quality checks, descriptive statistics,
                fold generation
  training/     the 3D CNN experiments: models, data pipeline, training loop, presets
  evaluation/   fold aggregation, test predictions, bootstrap intervals, subgroup
                metrics, diagnostic plots
scripts/        SLURM submission, and the identifier scan used by CI and the commit hook
tests/          a synthetic cohort, and tests that run without patient data
docs/           configuration and command reference
```

The three packages follow the three questions a reader has to be able to answer.

| Package | Question it answers | Why it is separate |
| --- | --- | --- |
| `cohort` | Who is in the study, and is the data sound? | Every number downstream depends on the cohort. It imports nothing else from the package and needs no GPU, so the cohort can be audited on a laptop. |
| `training` | What does the model learn from the images? | One training loop, with each experiment a named preset over it. The experiments began as near-identical scripts; as presets they cannot drift apart, and the settings that differ between them fit on one screen. |
| `evaluation` | How good is it, and how sure are we? | Scoring is kept apart from training so the same tools score any model that writes predictions in the same format. |

Dependencies run one way: `cohort` stands alone, and `training` and `evaluation` build
on it. Training, evaluation and the cohort reports all take their rows from the same
function, `cohort.data.prune`, so the model's cohort and the reported cohort cannot
differ.

```
BM_FOLDS_CSV  (one row per metastasis: labels, fold assignment, image paths)
BM_CLIN_CSV   (one row per treatment course: demographics, treatment, labs)
      |
      |  cohort.data.prune  - the single definition of which metastases count
      |
      +--> cohort/       quality checks, descriptive statistics   -> printed, aggregates only
      |
      +--> training/     per-fold data lists -> MONAI cache -> 5-fold training
      |                        -> checkpoint, metrics, TensorBoard log per fold
      |
      +--> evaluation/   fold aggregation, held-out test scoring, bootstrap intervals,
                         subgroup metrics, diagnostic plots
```

## What each part does

### `cohort`: who is in the study

A metastasis is included if its treatment course qualifies and the lesion exceeds a size
threshold both before and after treatment, so that a response can be measured at all.
Each command is one check, run as `python -m brainmets.cohort <command>`.

| Command | What it tells you | Why it matters |
| --- | --- | --- |
| `count-patients` | patients and lesions in the development and held-out test cohorts | also checks that no patient is in both, which would invalidate every held-out number |
| `count-classes` | class balance, and the accuracy of always guessing the majority class | the bar a model has to clear to mean anything |
| `count-samples` | train and validation sizes per fold | confirms the model trained on the cohort you think it did |
| `demographics`, `table1` | baseline characteristics, per course and per patient | `table1` keeps one course per patient, so patients treated more than once are not counted twice |
| `verify-courses` | whether a treatment course is uniquely identified in the data | a patient treated twice must appear as two courses, and the clinical join must not duplicate lesions |
| `check-round-age`, `sensitivity-course-key` | whether the ordering of a patient's courses can be trusted | which course is "first" decides who is in the per-patient table |
| `describe-ici` | what the less common treatment regimens were | explains the "other" category of the regimen breakdown |
| `compare-labels` | whether two sources of response labels agree | a model scored against the wrong labels is not evidence of anything |
| `make-folds` | writes cross-validation folds for a dataset that has none | grouped by patient and stratified by response, with the no-leakage properties asserted |
| `common-cohort-csv` | the lesions that have every MRI sequence | the shared cohort for a fair sequence comparison |

Every report prints aggregates only: counts, medians and column names, never a patient
row or identifier. A test runs each one on the synthetic cohort and checks.

### `training`: what the model sees and learns

The unit of prediction is one metastasis. A scan can contain several, each with its own
outcome, so the pipeline is built around isolating one lesion at a time:

1. **Select the lesion.** The segmentation labels every metastasis in the scan; the one
   this row refers to is kept.
2. **Mask its neighbours.** Voxels outside a small margin around the lesion are zeroed,
   so the model cannot use a neighbouring lesion, or the rest of the brain, to predict
   this one.
3. **Crop a fixed box centred on the lesion**, then crop each sample again to its own
   extent before the network, so most of what the network sees is lesion, not padding.
4. **Balance the classes** when sampling training batches, so the model cannot minimise
   its loss by ignoring the rarer outcome.
5. **Augment with flips only.** Other augmentations were tried during development and
   did not improve validation performance, so they are not in the pipeline.

The network is deliberately small: three convolution blocks and a linear layer. With one
label per lesion and a modest number of lesions, a deeper backbone overfits before it
generalises.

A checkpoint is saved when the validation metric improves, provided the training metric
is above chance: a model that validates well without having fit the training data got
there by luck. Because checkpoints are chosen on the validation folds, cross-validation
scores are optimistic, which is why the held-out test set exists.

Each experiment is one preset (`python -m brainmets.training run <preset>`):

| Preset | Input | Task | What it tests |
| --- | --- | --- | --- |
| `t1post-binary` | T1 post-contrast | progression vs not | the central question, in its simplest form |
| `t1post-multiclass` | T1 post-contrast | progression / stable / response | whether the signal separates all three RANO-BM levels |
| `multiseq-binary`, `multiseq-multiclass` | T1-post + T2 + FLAIR | both | whether further sequences add information |
| `t2-binary` | T2 | progression vs not | whether another sequence alone carries the signal |
| `learning-curve` | T1 post-contrast | progression vs not, on 10% to 100% of the training data | whether the model is limited by the amount of data or by what the images contain |

The model, the data pipeline and the training loop are the code that produced the
study's results, moved here unchanged: each function still parses to the same syntax
tree as the original scripts.

### `evaluation`: how good, and how sure

Each tool is a module, run as `python -m brainmets.evaluation.<tool>`.

| Tool | What it does | Why |
| --- | --- | --- |
| `aggregate_fold_metrics` | cross-validation table, mean and SD over folds | reads the training logs and replays the checkpoint rule, so it reports the epoch that was actually saved, not the best one in hindsight |
| `export_test_predictions` | scores the held-out test set with the five fold models averaged | the estimate that no tuning decision has touched |
| `export_val_predictions` | pools each fold's validation predictions | every development lesion scored once, by a model that did not train on it; the input for the diagnostic plots |
| `bootstrap_test_ci` | 95% confidence intervals on the test metrics | `--cluster` resamples patients, not lesions, because one patient's lesions are correlated |
| `bootstrap_sequence_comparison_ci` | intervals for the T1-post vs multisequence comparison | on the lesions both inputs share |
| `compute_test_metrics_subgroups` | test metrics by sex, histology, age, regimen, lesion size and lesion count | shows whether performance is uniform or carried by one group |
| `plot_diagnostics` | ROC, precision-recall, calibration, confusion matrices, score histograms | discrimination alone hides a miscalibrated model |
| `plot_learning_curve` | validation metric against training-set size | a curve that is still rising says "collect more data"; a flat one says the limit is elsewhere |

## Try it without patient data

The cohort logic runs against a synthetic cohort with the awkward features of a real one
(patients with several treatment courses, a held-out group, inconsistent ages for the
quality checks to find):

```bash
pip install -e .
python tests/fixtures.py /tmp/synthetic          # writes three CSVs and an env.sh
source /tmp/synthetic/env.sh

python -m brainmets.cohort count-patients        # cohort sizes, plus the leakage check
python -m brainmets.cohort count-classes         # class balance and zero-rule baselines
python -m brainmets.cohort make-folds --in "$BM_FOLDS_CSV" --out /tmp/synthetic/refolded.csv
python -m unittest discover -s tests             # the full suite, also data-free
```

## Running on real data

Inputs and outputs are located through environment variables (`BM_FOLDS_CSV`,
`BM_CLIN_CSV`, `BM_DATA_ROOT`, `BM_WORK_ROOT`, `BM_CACHE_ROOT`), described with the
expected table schema in [docs/configuration.md](docs/configuration.md). Training needs a
CUDA GPU: install a CUDA build of PyTorch from pytorch.org before `pip install -e .`.

```bash
# 1. who is in the study
python -m brainmets.cohort count-patients
python -m brainmets.cohort count-classes

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
`brainmets --help` lists them. Every command, option and output file is in
[docs/commands.md](docs/commands.md).

## Keeping patient data out, and the code honest

This is clinical research code published without its data, so two things are checked
automatically.

**Nothing identifying reaches the repository.** `scripts/check_identifiers.sh` refuses
data files, storage paths, email addresses and identifier-shaped numbers. It runs as a
pre-commit hook, which stops a leak before it is public, and again in CI. Enable the hook
once per clone with `git config core.hooksPath .githooks`; site-specific patterns go in a
git-ignored `.identifiers-local`.

**The code does what this page says.** The tests need no patient data:

| Test file | Covers |
| --- | --- |
| `test_cohort.py` | exclusions, the patient-level split, fold generation, and that no report prints an identifier |
| `test_evaluation.py` | AUROC, kappa, calibration and bootstrap intervals, against values worked out by hand |
| `test_config.py` | the presets: the settings that produced the study's results |
| `test_cli.py` | the `brainmets` command |
| `test_cnn.py` | forward and backward passes of the network (skips without torch) |
| `test_imports.py` | every module imports, in a fresh interpreter, without writing anything |

GitHub Actions runs two jobs on every push. The fast one runs lint, the tests and the
identifier scan on Python 3.10 and 3.12 without the deep-learning stack, in under a
minute. The second installs everything, with CPU-only PyTorch, and runs the tests again
with nothing skipped. To reproduce the fast job locally:

```bash
pip install --no-deps -e . && pip install numpy pandas scikit-learn matplotlib ruff==0.17.4
ruff check src tests && python -m unittest discover -s tests
```

## Differences from the paper and known limitations

- **ICI regimen labels count all five agents** (ipilimumab, nivolumab, pembrolizumab,
  atezolizumab, durvalumab), which is how the baseline table lists them. The paper's
  subgroup analysis grouped regimens by the first three only: atezolizumab and durvalumab
  courses were too sparse in its cohort to support subgroups of their own, so they were
  reported under "Other or none". Counting all five generalises better to other datasets,
  but `compute_test_metrics_subgroups` will not reproduce the paper's regimen subgroups
  exactly. The paper's three are kept as `cohort.config.BASE3`.
- **Undefined subgroup metrics print as `n/a`.** In a subgroup containing a single class,
  scikit-learn returns confident-looking values for metrics that are actually undefined
  (AUPRC 1.0 or 0.0, specificity 0.0, kappa 0.0). `compute_test_metrics_subgroups`
  reports those as `n/a` instead. No subgroup in the paper's data was affected.
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
