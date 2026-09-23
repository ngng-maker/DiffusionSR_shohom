#!/bin/bash
#SBATCH --job-name=fno_diffusion
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-4
#SBATCH --output=logs/fno_ablation/diff_%a_%j.log
#SBATCH --error=logs/fno_ablation/diff_%a_%j.err
#SBATCH --requeue
#
# Phase 2 — STAGE-2 COMPARISON: train a conditional DDPM on each frozen encoder.
#
# The diffusion side is held completely fixed across all four tasks (same U-Net, 1000 timesteps,
# linear schedule, Huber loss, same data and splits). The ONLY thing that varies is which frozen
# encoder supplies x_e. That is what makes this a clean test of the conditioning encoder.
#
# Requires Phase 1 (01_setup_encoders.sh) to have completed.
#
# Task mapping:
#   0  rrdb          — RRDB CNN encoder (paper baseline)
#   1  fno_pre       — FNO-A: bicubic to HR, operator at HR
#   2  fno_spectral  — FNO-B: operator at LR, spectral upsampling  (builtin backend only)
#   3  fno_conv      — FNO-C: operator at LR, RRDB's conv upsampling stack
#   4  fno_pre_pnemo — CALIBRATION: FNO-A again under the PhysicsNeMo backend. Compared against
#                     task 1 it isolates whether the backend choice is a confound. Not part of the
#                     primary comparison; requires physicsnemo to be installed.
#
# To requeue a single failed task (e.g. task 2):
#   sbatch --array=2 slurm/fno_ablation/02_diffusion_array.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/fno_ablation
CFGS=$REPO/diffusionsr/configs/fno_ablation

CONDA_ENV="${CONDA_ENV:-/trace/group/forgelab/ngng/envs/diffusion_SR}"

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

TASK=$SLURM_ARRAY_TASK_ID
DATE=$(date +"%d_%b_%Y")

# Index order must match 01_setup_encoders.sh. Task 4 is the PhysicsNeMo
# calibration arm: identical to task 1 except for the backend.
ARMS=(rrdb fno_pre fno_spectral fno_conv fno_pre_pnemo)
ARM=${ARMS[$TASK]}
NAME="fno_abl_diff_${ARM}"
# The date is stamped at first submission. On requeue, {force_run_dir}/wandb_run_id.txt re-attaches
# to the same W&B run, so this display name is only used on the very first launch.
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $NAME (encoder=$ARM, wandb=$WNAME) ==="

# Record the W&B display name so Phase 3 can find this run even if it executes on a later calendar
# day (WNAME embeds today's date). Written only once: on a requeue the original stamp must survive,
# otherwise the eval would look up a name that no run has.
mkdir -p "$RUNS/$NAME"
if [ ! -f "$RUNS/$NAME/wandb_run_name.txt" ]; then
  echo "$WNAME" > "$RUNS/$NAME/wandb_run_name.txt"
fi

# --force_enc_dir points at the Phase 1 output. encoder_type/encoder_kwargs come from the config,
# and MUST match enc_${ARM}.yml or load_state_dict will reject the checkpoint.
python -m diffusionsr.runners.train_srdiff \
  --config            "$CFGS/diff_${ARM}.yml" \
  --modeltype         diffusion \
  --gpu               0 \
  --force_run_dir     "$RUNS/$NAME" \
  --force_enc_dir     "$RUNS/enc_${ARM}" \
  --wandb_run_name    "$WNAME" \
  --resume_from_wandb "$WNAME"

echo "=== $NAME complete ==="
