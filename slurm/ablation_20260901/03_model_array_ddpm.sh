#!/bin/bash
#SBATCH --job-name=abl_ddpm
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=logs/ablation_20260901/ddpm_%a_%j.log
#SBATCH --error=logs/ablation_20260901/ddpm_%a_%j.err
#SBATCH --requeue
#
# Phase 3: train 4 pixel-space DDPM ablation models (standardize norm).
# Requires Phase 0a (00a_setup_encoders.sh) to have completed (enc tasks need encoder).
#
# Task mapping:
#   0  diffusion_enc_multifield  — DDPM + encoder + temp+sdfliqlabel + standardize
#   1  diffusion_enc_temp        — DDPM + encoder + temp only        + standardize
#   2  diffusion_noenc_multifield— DDPM + no encoder + temp+sdfliqlabel + standardize
#   3  diffusion_noenc_temp      — DDPM + no encoder + temp only        + standardize
#
# Reuses the same enc_multifield/enc_temp encoders as FM — no new encoder pretraining needed.
#
# To requeue a single failed task (e.g. task 1):
#   sbatch --array=1 slurm/ablation_20260901/03_model_array_ddpm.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

TASK=$SLURM_ARRAY_TASK_ID
DATE=$(date +"%d_%b_%Y")

NAMES=(abl_diffusion_enc_multifield   abl_diffusion_enc_temp \
       abl_diffusion_noenc_multifield  abl_diffusion_noenc_temp)

CFGFILES=(diffusion_enc_multifield   diffusion_enc_temp \
          diffusion_noenc_multifield  diffusion_noenc_temp)

NAME=${NAMES[$TASK]}
CFG=${CFGFILES[$TASK]}
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $NAME (modeltype=diffusion, wandb=$WNAME) ==="

case $TASK in
  0) ENC_ARG="--force_enc_dir $RUNS/enc_multifield" ;;
  1) ENC_ARG="--force_enc_dir $RUNS/enc_temp"       ;;
  *) ENC_ARG=""                                      ;;
esac

python -m diffusionsr.runners.train_srdiff \
  --config      "$CFGS/$CFG.yml" \
  --modeltype   diffusion \
  --gpu         0 \
  --force_run_dir "$RUNS/$NAME" \
  --wandb_run_name "$WNAME" \
  --resume_from_wandb "$WNAME" \
  $ENC_ARG

echo "=== $NAME complete ==="
