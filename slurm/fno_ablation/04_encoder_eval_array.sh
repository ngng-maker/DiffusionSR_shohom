#!/bin/bash
#SBATCH --job-name=fno_enc_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=logs/fno_ablation/enc_eval_%a_%j.log
#SBATCH --error=logs/fno_ablation/enc_eval_%a_%j.err
#SBATCH --requeue
#
# Phase 4 — encoder-only evaluation: runs each FNO/RRDB encoder on the full
# test set and uploads predictions as a W&B artifact named
# {run_name}_eval_predictions.  No diffusion sampling — single forward pass
# per sample, so each task takes ~2–5 min.
#
# Task mapping (must match 01_setup_encoders.sh):
#   0  rrdb          — RRDB CNN encoder (paper baseline)
#   1  fno_pre       — FNO-A: bicubic to HR, operator at HR
#   2  fno_spectral  — FNO-B: operator at LR, spectral upsampling
#   3  fno_conv      — FNO-C: operator at LR, RRDB conv upsampling
#   4  fno_pre_pnemo — FNO-A under PhysicsNeMo backend (calibration arm)
#
# To requeue a single task (e.g. task 1):
#   sbatch --array=1 slurm/fno_ablation/04_encoder_eval_array.sh
#
# Task 4 requires the physicsnemo env:
#   sbatch --array=4 --export=ALL,CONDA_ENV=/trace/group/forgelab/ngng/envs/diffusion_SR_fno \
#     slurm/fno_ablation/04_encoder_eval_array.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_enc_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
if [ -n "${PY_OVERLAY:-}" ]; then
  source "$PY_OVERLAY/bin/activate"
  echo "Using venv overlay: $PY_OVERLAY"
fi

TASK=$SLURM_ARRAY_TASK_ID

ARMS=(rrdb fno_pre fno_spectral fno_conv fno_pre_pnemo)
ARM=${ARMS[$TASK]}
NAME="fno_abl_enc_${ARM}"

# Read the W&B display name stamped at Phase 1 training time.
WNAME_FILE="$RUNS/enc_${ARM}/wandb_run_name.txt"
if [ -f "$WNAME_FILE" ]; then
  WNAME=$(cat "$WNAME_FILE")
else
  WNAME="$(date +"%d_%b_%Y")_${NAME}"
  echo "WARNING: $WNAME_FILE not found; guessing wandb run name $WNAME"
fi

echo "=== Task $TASK: encoder-only eval for enc_${ARM} (wandb=$WNAME) ==="

python -m diffusionsr.analysis.eval_predictions \
    --run_name       "$NAME" \
    --wandb_run_name "$WNAME" \
    --model_type     encoder \
    --model_dir      "$RUNS/enc_${ARM}" \
    --config         "$CFGS/enc_${ARM}.yml" \
    --data_root      "$DATA" \
    --field_names    temperature \
    --n_steps        1 \
    --batch_size     8

echo "=== enc_${ARM} eval complete ==="
