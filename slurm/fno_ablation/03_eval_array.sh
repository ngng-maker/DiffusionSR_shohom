#!/bin/bash
#SBATCH --job-name=fno_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=logs/fno_ablation/eval_%a_%j.log
#SBATCH --error=logs/fno_ablation/eval_%a_%j.err
#SBATCH --requeue
#
# Phase 3 — sample the full test set for each arm and upload predictions to W&B.
#
# Writes a predictions.npz artifact (pred + gt, both in physical units) onto each arm's existing
# W&B run. Those artifacts are what the comparison notebook reads to compute PSNR / SSIM / melt-pool
# depth / keyhole oscillation amplitude side by side.
#
# Requires Phase 2 (02_diffusion_array.sh) to have completed.
#
# --config is what lets the eval rebuild an FNO encoder instead of defaulting to RRDB. It must be
# the SAME yaml that trained the model, so the architecture matches the saved checkpoint exactly.
#
# To requeue a single failed task (e.g. task 2):
#   sbatch --array=2 slurm/fno_ablation/03_eval_array.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/fno_ablation
CFGS=$REPO/diffusionsr/configs/fno_ablation
DATA=/trace/group/forgelab/ngng/multifield/data_fields

CONDA_ENV="${CONDA_ENV:-/trace/group/forgelab/ngng/envs/diffusion_SR}"

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

TASK=$SLURM_ARRAY_TASK_ID

# Index order must match 01_setup_encoders.sh. Task 4 is the PhysicsNeMo
# calibration arm: identical to task 1 except for the backend.
ARMS=(rrdb fno_pre fno_spectral fno_conv fno_pre_pnemo)
ARM=${ARMS[$TASK]}
NAME="fno_abl_diff_${ARM}"

# Recover the W&B display name stamped at Phase 2's first launch. The date is not knowable here, so
# it is read from the run directory rather than recomputed — `date` would give the wrong answer if
# Phase 3 runs on a later calendar day than Phase 2.
WNAME_FILE="$RUNS/$NAME/wandb_run_name.txt"
if [ -f "$WNAME_FILE" ]; then
  WNAME=$(cat "$WNAME_FILE")
else
  # Fall back to today's date. If this misses, eval_predictions creates a separate *_eval run
  # rather than failing, and the artifact is still produced.
  WNAME="$(date +"%d_%b_%Y")_${NAME}"
  echo "WARNING: $WNAME_FILE not found; guessing wandb run name $WNAME"
fi

echo "=== Task $TASK: evaluating $NAME (encoder=$ARM, wandb=$WNAME) ==="

python -m diffusionsr.analysis.eval_predictions \
    --run_name       "$NAME" \
    --wandb_run_name "$WNAME" \
    --model_type     diffusion \
    --model_dir      "$RUNS/$NAME" \
    --enc_dir        "$RUNS/enc_${ARM}" \
    --config         "$CFGS/diff_${ARM}.yml" \
    --data_root      "$DATA" \
    --field_names    temperature \
    --n_steps        1 \
    --sampler        DDIM \
    --skip           50 \
    --batch_size     4

echo "=== eval $NAME complete ==="
