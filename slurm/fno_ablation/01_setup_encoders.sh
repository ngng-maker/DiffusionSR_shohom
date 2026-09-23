#!/bin/bash
#SBATCH --job-name=fno_encoders
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fno_ablation/enc_%j.log
#SBATCH --error=logs/fno_ablation/enc_%j.err
#SBATCH --requeue
#
# Phase 1 — STAGE-1 COMPARISON: train all four conditioning encoders.
#
# This phase is itself a complete experiment. Each encoder is trained with plain L1 loss to map
# LR -> HR, so their train/test loss curves in W&B (project RRDN_Encoder) are a direct, apples-to-
# apples comparison of RRDB vs the three FNO variants as standalone super-resolvers. Read those
# curves before committing GPU-hours to Phase 2.
#
# Run sequentially on one GPU so the four runs cannot contend for memory.
# --requeue lets SLURM auto-restart on preemption; each runner resumes from encoder_ckpt.pth.

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/fno_ablation
CFGS=$REPO/diffusionsr/configs/fno_ablation

# Override at submit time if PhysicsNeMo lives in a cloned env:
#   sbatch --export=ALL,CONDA_ENV=/trace/group/forgelab/ngng/envs/diffusion_SR_fno <script>
CONDA_ENV="${CONDA_ENV:-/trace/group/forgelab/ngng/envs/diffusion_SR}"

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
mkdir -p logs/fno_ablation

# The four arms. rrdb is the paper's baseline; the three fno_* entries are the study treatments.
# Order matters only in that rrdb runs first, so a failure there aborts before any FNO time is spent.
ARMS=(rrdb fno_pre fno_spectral fno_conv)

for i in "${!ARMS[@]}"; do            # ${!ARRAY[@]} expands to the index list, giving both position and value
  ARM=${ARMS[$i]}
  echo "=== [$((i+1))/${#ARMS[@]}] Encoder: $ARM ==="
  # --force_run_dir and --force_enc_dir point at the same directory for encoder training: the runner
  # writes bestmodel_saved.pth there, and Phase 2 reads it back from the same path.
  python -m diffusionsr.runners.train_srdiff \
    --config        "$CFGS/enc_${ARM}.yml" \
    --modeltype     encoder \
    --gpu           0 \
    --force_run_dir "$RUNS/enc_${ARM}" \
    --force_enc_dir "$RUNS/enc_${ARM}"
done

echo "=== Stage-1 encoder training complete for: ${ARMS[*]} ==="
echo "Compare train/test L1 curves in W&B project RRDN_Encoder before submitting Phase 2."
