# Configuration

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

Installing is required: the code lives under `src/`, so `brainmets` is not importable
until it is installed. An editable install means edits take effect immediately.

GPU training needs a CUDA-enabled PyTorch build matching your driver. Install that from
pytorch.org **first**, then `pip install -e .` for the rest; otherwise pip resolves a
CPU-only torch. The dependency list lives in `pyproject.toml` alone; `requirements.txt`
is just `-e .`, so `pip install -r requirements.txt` works too.

## Environment variables

Every site-specific location is read from an environment variable; nothing is
hardcoded, and there are no placeholder defaults. They are resolved on each use rather
than at import, so setting one always takes effect, and a variable that is needed but
unset, or an input path that does not exist, fails immediately, naming the variable:

```
$ python -m brainmets.cohort count-patients
BM_FOLDS_CSV is not set, so the cohort table cannot be located.
  export BM_FOLDS_CSV=/your/path   (see docs/configuration.md)
```

| Variable | Contents | Needed by |
| --- | --- | --- |
| `BM_FOLDS_CSV` | Cohort table, one row per metastasis (schema below) | everything |
| `BM_CLIN_CSV` | Clinical table, one row per treatment course | demographics, Table 1, subgroup metrics |
| `BM_CENTROIDS_CSV` | Lesion centroid / RANO label table | `cohort compare-labels` only |
| `BM_DATA_ROOT` | Root of the preprocessed MRI tree; checked to exist, but image paths are taken from the cohort table (below) | training, test-set export |
| `BM_WORK_ROOT` | Output root: checkpoints, TensorBoard logs, data lists, predictions; created if missing | training, evaluation |
| `BM_CACHE_ROOT` | MONAI `PersistentDataset` cache (large; safe to delete); created if missing | training, prediction export |

```bash
export BM_DATA_ROOT=/data/brainmets/preprocessed
export BM_FOLDS_CSV=/data/brainmets/cohort_folds.csv
export BM_CLIN_CSV=/data/brainmets/clinical_data.csv
export BM_WORK_ROOT=/scratch/brainmets/work
export BM_CACHE_ROOT=/scratch/brainmets/cache
```

`python tests/fixtures.py <dir>` writes a synthetic set of the three input tables plus an
`env.sh` that sets the first three variables.

## The cohort table

`BM_FOLDS_CSV` is the central input, read with its first column as the index. The code
expects at least:

- `Patient_Number`, `Metastasis_ID`, `Treatment_round`: identifiers
- `PreTreatment_Image`, `PostTreatment_Image`: **absolute paths** to the NIfTI volumes
- `PreTreatment_Met`, `PostTreatment_Met`: lesion size before and after treatment; a
  metastasis is kept only if both exceed the threshold (10 by default, in the table's
  units; `--lower` changes it)
- `RANOResponse_-40_40` / `RANOResponse_-40_40_Met`: patient- and lesion-level labels at
  the ±40% RANO-BM thresholds (`Progression` / `Stable` / `Response`)
- `rows_to_remove`: exclusion flag for non-qualifying ICI regimens
- `T2_present`, `FLAIR_present`: sequence availability, for the multisequence runs
- `Fold_0` … `Fold_4`: the split, `0` = train, `1` = validation, `2` = held out from
  every fold (the test set). `cohort make-folds` generates these if they are missing.
- radiomic feature columns, e.g. `T1Post_original_shape_*`

Image locations come from the `PreTreatment_Image` column rather than from
`BM_DATA_ROOT`, so this table has to be regenerated against local storage before
training will find any volumes.

## Image layout

Within `BM_DATA_ROOT`, each lesion directory holds fixed-name files: `_T1Post.nii.gz`,
`_T1Pre.nii.gz`, `_T2.nii.gz`, `_FLAIR.nii.gz`, `_ADC.nii.gz`, `_SWI.nii.gz`,
`_DWI.nii.gz`, and the segmentation `_T1Post_TRACKED-label.nii.gz`.

The real patient id and treatment round are parsed from the image path, which must
therefore end `…/<patient>/<round>/<lesion>/<file>`. (`Patient_Number` in the table is a
re-indexed id; the path carries the one that links courses of the same patient.)
