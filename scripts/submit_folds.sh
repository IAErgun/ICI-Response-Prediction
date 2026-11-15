#!/bin/bash
# Submit one training preset to a SLURM cluster as one job per CV fold, each on its own
# GPU, so the five folds train in parallel. Each job sets FOLD, which restricts
# `python -m brainmets.training run` to that single fold.
#
#   bash scripts/submit_folds.sh <preset> [fold ...]
#
#   bash scripts/submit_folds.sh multiseq-binary          # all five folds
#   bash scripts/submit_folds.sh t1post-multiclass 3 4    # re-run a subset
#
# `python -m brainmets.training presets` lists the presets. The site settings below can be
# edited here or exported beforehand (e.g. `PARTITION=gpu ACCOUNT=mylab bash ...`); most
# sites require a partition and an account.

PARTITION=${PARTITION:-CHANGE_ME}                 # a GPU partition you can use (see `sinfo`)
ACCOUNT=${ACCOUNT:-CHANGE_ME}                     # SLURM account to charge
MEM=${MEM:-64G}                                   # headroom for loading 3D volumes
TIME=${TIME:-3-00:00:00}                          # 72h: 300 epochs plus the first-run cache build
CORES=${CORES:-3}                                 # CPU cores per GPU, for data loading
VENV_ACTIVATE=${VENV_ACTIVATE:-/path/to/venv/bin/activate}  # Python environment to activate
REPO_DIR=${REPO_DIR:-/path/to/ICI-Response-Prediction}  # where `pip install -e .` was run
SUBMIT=${SUBMIT:-sbatch}                          # or a site wrapper that takes sbatch flags

if [ $# -lt 1 ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi
PRESET=$1
shift
FOLDS=${*:-0 1 2 3 4}

if [ "$PARTITION" = "CHANGE_ME" ] || [ "$ACCOUNT" = "CHANGE_ME" ]; then
  echo "Set PARTITION and ACCOUNT (edit this script or export them) first." >&2
  exit 1
fi

for FOLD in $FOLDS; do
  echo "submitting $PRESET fold $FOLD"
  "$SUBMIT" -p "$PARTITION" -A "$ACCOUNT" --mem="$MEM" -t "$TIME" -c "$CORES" --gpus=1 \
    --job-name="bm-$PRESET-f$FOLD" \
    --wrap "source $VENV_ACTIVATE; cd $REPO_DIR; FOLD=$FOLD python -m brainmets.training run $PRESET"
done
