#!/bin/bash
#SBATCH --job-name=abl_enc_global
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/enc_global_%j.log
#SBATCH --error=logs/ablation_20260901/enc_global_%j.err
#SBATCH --requeue
#
# Phase 0c: train encoder prerequisites for the global_standardize ablation runs.
# Uses global_standardize normalization — separate encoders are required because the
# input distribution (and thus encoder output scale) differs from standardize encoders.

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR
mkdir -p logs/ablation_20260901

echo "=== [1/2] Encoder (global_std): temperature + sdfliqlabel ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_multifield_global.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_multifield_global" \
  --force_enc_dir "$RUNS/enc_multifield_global"

echo "=== [2/2] Encoder (global_std): temperature only ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_temp_global.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_temp_global" \
  --force_enc_dir "$RUNS/enc_temp_global"

echo "=== Encoder pretraining (global_standardize) complete ==="
