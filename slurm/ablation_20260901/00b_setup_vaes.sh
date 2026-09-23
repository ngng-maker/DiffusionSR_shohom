#!/bin/bash
#SBATCH --job-name=abl_vaes
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

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR
mkdir -p logs/ablation_20260901

echo "=== [1/2] VAE: temperature + sdfliqlabel ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir     "$RUNS/vae_multifield" \
  --fields      temperature_sdfliqlabel \
  --n_steps     3 \
  --epochs      100 \
  --gpu         0

echo "=== [2/2] VAE: temperature only ==="
python -m diffusionsr.scripts.pretrain_vae_standalone \
  --root_folder "$DATA" \
  --vae_dir     "$RUNS/vae_temp" \
  --fields      temperature \
  --n_steps     3 \
  --epochs      100 \
  --gpu         0

echo "=== VAE pretraining complete ==="
