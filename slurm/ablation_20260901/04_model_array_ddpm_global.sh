#!/bin/bash
#SBATCH --job-name=abl_ddpm_g
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=logs/ablation_20260901/ddpm_global_%a_%j.log
#SBATCH --error=logs/ablation_20260901/ddpm_global_%a_%j.err
#SBATCH --requeue
#
# Phase 4: train 4 pixel-space DDPM ablation models (global_standardize norm).
# Requires Phase 0c (00c_setup_encoders_global.sh) to have completed.
#
# Task mapping:
#   0  diffusion_enc_multifield_global   — DDPM + encoder + temp+sdfliqlabel + global_std
#   1  diffusion_enc_temp_global         — DDPM + encoder + temp only        + global_std
#   2  diffusion_noenc_multifield_global — DDPM + no encoder + temp+sdfliqlabel + global_std
#   3  diffusion_noenc_temp_global       — DDPM + no encoder + temp only        + global_std
#
# To requeue a single failed task (e.g. task 0):
#   sbatch --array=0 slurm/ablation_20260901/04_model_array_ddpm_global.sh

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

NAMES=(abl_diffusion_enc_multifield_global   abl_diffusion_enc_temp_global \
       abl_diffusion_noenc_multifield_global  abl_diffusion_noenc_temp_global)

CFGFILES=(diffusion_enc_multifield_global   diffusion_enc_temp_global \
          diffusion_noenc_multifield_global  diffusion_noenc_temp_global)

NAME=${NAMES[$TASK]}
CFG=${CFGFILES[$TASK]}
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $NAME (modeltype=diffusion, wandb=$WNAME) ==="

case $TASK in
  0) ENC_ARG="--force_enc_dir $RUNS/enc_multifield_global" ;;
  1) ENC_ARG="--force_enc_dir $RUNS/enc_temp_global"       ;;
  *) ENC_ARG=""                                             ;;
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
