#!/bin/bash
#SBATCH --job-name=abl_v2_followup
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=logs/ablation_v2/followup_%a_%j.log
#SBATCH --error=logs/ablation_v2/followup_%a_%j.err
#SBATCH --requeue
# SIGUSR1 to the batch shell 10 min before the time limit, so the task requeues itself instead of
# ending as TIMEOUT. The batchpartition QoS caps jobs at 2 days, but a 600-epoch run takes ~55-85 h
# on an A40 (330-500 s/epoch, measured on the original v2 runs), so every task needs 2 segments.
#SBATCH --signal=B:USR1@600
#
# Ablation v2 follow-up runs that test the explanations for the differences between configurations:
#   encswap : DiffusionSR with the encoder trained under the OTHER normalisation (encoder_normalize), to test
#             whether the field x normalisation interaction comes from encoder quality rather than the
#             field or normalisation themselves.
#   seed1   : DiffusionSR repeat runs with seed 1, to measure training-seed noise.
#   bigunet : LDM with a 192-wide latent U-Net (~7.5M params vs 73k), to test the "U-Net too small" explanation.
# Same command line as the v2 runs, so each resumes from <run_dir>/ckpt.pth and its W&B run when requeued.
#
# Submit from the repo root on TRACE:
#   mkdir -p logs/ablation_v2 && sbatch --exclude=trace09 slurm/ablation_v2/followup_array.sh
#
# Task map:
#   0-3  diffusion_encswap_{temp_std,temp_gstd,multi_std,multi_gstd}
#   4-7  diffusion_seed1_{temp_std,temp_gstd,multi_std,multi_gstd}
#   8-9  ldm_bigunet_{temp_std,multi_std}

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_v2

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

NAMES=(diffusion_encswap_temp_std diffusion_encswap_temp_gstd diffusion_encswap_multi_std diffusion_encswap_multi_gstd
       diffusion_seed1_temp_std diffusion_seed1_temp_gstd diffusion_seed1_multi_std diffusion_seed1_multi_gstd
       ldm_bigunet_temp_std ldm_bigunet_multi_std)

TASK=${SLURM_ARRAY_TASK_ID:?run as a SLURM array task}
NAME=${NAMES[$TASK]}
case $NAME in ldm_*) MODELTYPE=ldm ;; *) MODELTYPE=diffusion ;; esac
CFG="diffusionsr/configs/ablation_v2/$NAME.yml"

echo "=== Task $TASK: $NAME (modeltype=$MODELTYPE) ==="
test -f "$CFG" || { echo "missing $CFG"; exit 1; }

# Requeue before the time limit (same pattern as slurm/interpolant/01_train_array.sh): train_srdiff
# writes ckpt.pth every epoch and resumes from it in the same W&B run (wandb_run_id.txt), so at most
# one partial epoch is lost per segment. Python runs in the background because bash defers traps
# until a foreground command returns.
requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing job $SLURM_JOB_ID to resume from ckpt.pth ==="
  scontrol requeue "$SLURM_JOB_ID"
  # Let SLURM end the job as REQUEUED rather than racing it to exit with a FAILED status.
  sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.runners.train_srdiff \
  --config         "$CFG" \
  --gpu            0 \
  --modeltype      "$MODELTYPE" \
  --force_run_dir  "$RUNS/$NAME" \
  --wandb_run_name "$NAME" \
  --resume_from_wandb "$NAME" &
PY_PID=$!
# Returns python's exit status (set -e then fails the task on a crash), or returns early with >128
# when the USR1 trap fires, by which point scontrol requeue is already ending this job.
wait $PY_PID

echo "=== $NAME complete ==="
