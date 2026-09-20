#!/bin/bash
#SBATCH --job-name=eval_ldm_enc_sdf
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/eval_ldm_enc_sdf_%j.log
#SBATCH --error=logs/ablation_20260901/eval_ldm_enc_sdf_%j.err
#SBATCH --requeue

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

python -m diffusionsr.analysis.eval_predictions \
    --run_name        abl_ldm_enc_sdf \
    --wandb_run_name  1_Sep_2026_ldm_enc_sdf \
    --model_type      ldm \
    --model_dir       "$RUNS/abl_ldm_enc_sdf" \
    --enc_dir         "$RUNS/enc_sdf" \
    --vae_dir         "$RUNS/vae_sdf" \
    --data_root       "$DATA" \
    --field_names     temperature liqlabel \
    --batch_size      2

echo "=== eval_abl_ldm_enc_sdf complete ==="
