#!/bin/bash
#SBATCH --job-name=abl_setup_1Sep2026
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/setup_%j.log
#SBATCH --error=logs/ablation_20260901/setup_%j.err
#
# Phase 0: pre-train shared encoders + VAEs for the ablation_20260901 experiment.
# Run sequentially on one GPU so there are no write conflicts.
# Submit BEFORE the model-training array:
#   sbatch slurm/ablation_20260901/00_setup_enc_vae.sh
# Then submit the array with dependency:
#   sbatch --dependency=afterok:<SETUP_JID> slurm/ablation_20260901/01_model_array.sh
# Or use submit_all.sh which handles this automatically.

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
conda activate diffusion_SR

mkdir -p logs/ablation_20260901

echo "=== [1/4] Encoder: temperature + sdfliqlabel ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_sdf.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_sdf" \
  --force_enc_dir "$RUNS/enc_sdf" \
  --wandb_run_name "1_Sep_2026_encoder_sdf"

echo "=== [2/4] Encoder: temperature only ==="
python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/fm_enc_temp.yml" \
  --modeltype encoder \
  --gpu 0 \
  --force_run_dir "$RUNS/enc_temp" \
  --force_enc_dir "$RUNS/enc_temp" \
  --wandb_run_name "1_Sep_2026_encoder_temp"

echo "=== [3/4] VAE: temperature + sdfliqlabel ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_sdf" \
  --fields temperature_sdfliqlabel \
  --n_steps 3 \
  --epochs 100 \
  --wandb_run_name "1_Sep_2026_vae_sdf"

echo "=== [4/4] VAE: temperature only ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_temp" \
  --fields temperature \
  --n_steps 3 \
  --epochs 100 \
  --wandb_run_name "1_Sep_2026_vae_temp"

echo "=== Setup complete. All encoders and VAEs are in $RUNS ==="
