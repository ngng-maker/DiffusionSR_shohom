#!/bin/bash
#SBATCH --job-name=abl_vae_1Sep2026
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/vae_%j.log
#SBATCH --error=logs/ablation_20260901/vae_%j.err

set -eo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
source ~/.bashrc
conda activate diffusion_SR

echo "=== [1/2] VAE: temperature + sdfliqlabel ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_sdf" \
  --fields temperature_sdfliqlabel \
  --n_steps 3 \
  --epochs 100 \
  --wandb_run_name "1_Sep_2026_vae_sdf"

echo "=== [2/2] VAE: temperature only ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_temp" \
  --fields temperature \
  --n_steps 3 \
  --epochs 100 \
  --wandb_run_name "1_Sep_2026_vae_temp"

echo "=== VAE pretraining complete ==="
