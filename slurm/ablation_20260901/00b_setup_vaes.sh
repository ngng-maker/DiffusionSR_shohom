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
#SBATCH --requeue

set -eo pipefail

# Redirect W&B artifact cache to node-local /tmp (auto-cleaned, never hits home quota)
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"

# Always clean W&B artifact staging AND cache on exit (success or failure)
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

echo "=== [1/2] VAE: temperature + liqlabel ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_sdf" \
  --fields temperature_liqlabel \
  --n_steps 3 \
  --epochs 100 \
  --gpu 0 \
  --wandb_run_name "1_Sep_2026_vae_sdf"

echo "=== [2/2] VAE: temperature only ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir "$RUNS/vae_temp" \
  --fields temperature \
  --n_steps 3 \
  --epochs 100 \
  --gpu 0 \
  --wandb_run_name "1_Sep_2026_vae_temp"

echo "=== VAE pretraining complete ==="
