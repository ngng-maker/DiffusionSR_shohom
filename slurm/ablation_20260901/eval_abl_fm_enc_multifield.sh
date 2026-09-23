#!/bin/bash
#SBATCH --job-name=eval_fm_enc_multifield
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_20260901/eval_fm_enc_multifield_%j.log
#SBATCH --error=logs/ablation_20260901/eval_fm_enc_multifield_%j.err
#SBATCH --requeue
set -eo pipefail
export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields
# TODO: fill in after training starts — e.g. $(date +"%d_%b_%Y")_fm_enc_multifield
WANDB_RUN_NAME=""

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

python -m diffusionsr.analysis.eval_predictions \
    --run_name        abl_fm_enc_multifield \
    --wandb_run_name  "$WANDB_RUN_NAME" \
    --model_type      flow_matching \
    --model_dir       "$RUNS/abl_fm_enc_multifield" \
    --enc_dir         "$RUNS/enc_multifield" \
    --data_root       "$DATA" \
    --field_names     temperature sdfliqlabel \
    --batch_size      4

echo "=== eval_abl_fm_enc_multifield complete ==="
