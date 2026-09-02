#!/bin/bash
#SBATCH --job-name=abl_enc_1Sep2026
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/enc_%j.log
#SBATCH --error=logs/ablation_20260901/enc_%j.err

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901

cd "$REPO"
conda activate diffusion_SR
mkdir -p logs/ablation_20260901

echo "=== [1/2] Encoder: temperature + sdfliqlabel ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_sdf.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_sdf" \
  --force_enc_dir "$RUNS/enc_sdf" \
  --wandb_run_name "1_Sep_2026_encoder_sdf"

echo "=== [2/2] Encoder: temperature only ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_temp.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_temp" \
  --force_enc_dir "$RUNS/enc_temp" \
  --wandb_run_name "1_Sep_2026_encoder_temp"

echo "=== Encoder pretraining complete ==="
