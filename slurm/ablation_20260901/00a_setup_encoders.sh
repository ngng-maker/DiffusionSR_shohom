#!/bin/bash
#SBATCH --job-name=abl_encoders
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/enc_%j.log
#SBATCH --error=logs/ablation_20260901/enc_%j.err
#SBATCH --requeue

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

echo "=== [1/2] Encoder: temperature + sdfliqlabel ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_multifield.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_multifield" \
  --force_enc_dir "$RUNS/enc_multifield"

echo "=== [2/2] Encoder: temperature only ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_temp.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_temp" \
  --force_enc_dir "$RUNS/enc_temp"

echo "=== Encoder pretraining complete ==="
