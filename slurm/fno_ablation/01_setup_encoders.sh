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
# Optional venv overlay. `conda create --clone` of a CUDA torch env copies 8-15 GB of small files
# across a shared parallel filesystem and can take an hour; a venv created with
# --system-site-packages inherits the conda env's packages and adds only the extra ones, in seconds.
# Both activations are required: conda first for the CUDA shared libraries, then the overlay so its
# site-packages take precedence. Use with:
#   sbatch --export=ALL,PY_OVERLAY=/trace/group/forgelab/ngng/envs/pnemo_overlay <script>
if [ -n "${PY_OVERLAY:-}" ]; then
  source "$PY_OVERLAY/bin/activate"
  echo "Using venv overlay: $PY_OVERLAY"
fi
mkdir -p logs/fno_ablation

# PRIMARY arms: the actual comparison. All FNO arms use the builtin backend so they are internally
# consistent with fno_spectral, which can only use builtin.
PRIMARY_ARMS=(rrdb fno_pre fno_spectral fno_conv)

# CALIBRATION arm: FNO-A again, but under PhysicsNeMo. Compared against fno_pre (the identical
# variant under builtin) it tells us whether the backend is a confound. Run LAST and treated as
# non-blocking: it is a methodology check, not part of the primary comparison, so if PhysicsNeMo is
# missing or broken it must not cost us the four arms that matter.
CALIBRATION_ARMS=(fno_pre_pnemo)

FAILED_PRIMARY=()   # Collected so one bad arm does not hide the status of the others
FAILED_CALIB=()

train_arm() {       # $1 = arm name; returns the trainer's exit status to the caller
  local arm=$1
  echo "=== Encoder: $arm ==="
  # --force_run_dir and --force_enc_dir point at the same directory for encoder training: the runner
  # writes bestmodel_saved.pth there, and Phase 2 reads it back from the same path.
  python -m diffusionsr.runners.train_srdiff \
    --config        "$CFGS/enc_${arm}.yml" \
    --modeltype     encoder \
    --gpu           0 \
    --force_run_dir "$RUNS/enc_${arm}" \
    --force_enc_dir "$RUNS/enc_${arm}"
}

# `|| FAILED+=(...)` keeps the loop going past a failure. `set -e` does not trigger on a command
# whose status is consumed by || , which is exactly the behaviour wanted here: a single arm dying
# should not discard the GPU-hours already spent on the arms before it.
for arm in "${PRIMARY_ARMS[@]}"; do
  train_arm "$arm" || FAILED_PRIMARY+=("$arm")
done

for arm in "${CALIBRATION_ARMS[@]}"; do
  train_arm "$arm" || FAILED_CALIB+=("$arm")
done

echo
echo "=== Stage-1 summary ==="
echo "  primary arms:     ${PRIMARY_ARMS[*]}"
if [ ${#FAILED_CALIB[@]} -gt 0 ]; then
  echo "  WARNING: calibration arm failed: ${FAILED_CALIB[*]}"
  echo "           Usually means physicsnemo is not installed or not importable."
  echo "           The primary comparison is unaffected, but you cannot yet rule out"
  echo "           the backend as a confound. See PLAN.md section 3.2."
fi
if [ ${#FAILED_PRIMARY[@]} -gt 0 ]; then
  echo "  ERROR: primary arms failed: ${FAILED_PRIMARY[*]}"
  echo "         Phase 2 will not start (afterok dependency)."
  exit 1   # Only a primary failure blocks the chain; a calibration failure does not
fi

echo "  all primary arms trained."
echo "Compare train/test L1 curves in W&B project RRDN_Encoder before submitting Phase 2."
