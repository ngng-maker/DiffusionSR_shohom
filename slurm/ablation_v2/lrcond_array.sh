#!/bin/bash
#SBATCH --job-name=abl_v2_lrcond
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=logs/ablation_v2/lrcond_%a_%j.log
#SBATCH --error=logs/ablation_v2/lrcond_%a_%j.err
#SBATCH --requeue
# SIGUSR1 to the batch shell 10 min before the time limit, so the task requeues itself instead of
# ending as TIMEOUT. The batchpartition QoS caps jobs at 2 days, but a 600-epoch run takes ~55-85 h
# on an A40 (330-500 s/epoch, measured on the original v2 runs), so every task needs 2 segments.
#SBATCH --signal=B:USR1@600
#
# Ablation v2, "lrcond" arm: no encoder, conditioned on the bicubic-upscaled LR (the paper's no-encoder
# ablation). The existing "noenc" arm uses conditioning=none, i.e. it never sees the LR input, so it
# measures conditioned-vs-unconditional rather than encoder-vs-no-encoder.
#
#   diffusion / fm : conditioning=explicit  (upscaled LR concatenated with x_t)
#   ldm            : conditioning=implicit  (VAE latent of upscaled LR added, as in its encoder arm);
#                    reuses the frozen v2 VAEs in runs/ablation_v2/vae_{temp,multi}_{std,gstd}
#
# Same command line as the original 24 v2 runs (train_srdiff, --force_run_dir runs/ablation_v2/<variant>,
# --wandb_run_name <variant>), so precompute_ablation_v2_preview.py and notebook 18 find them by name.
# A requeued, preempted or self-requeued (time limit) task resumes from <run_dir>/ckpt.pth and the same
# W&B run (wandb_run_id.txt).
#
# Submit from the repo root on TRACE:
#   mkdir -p logs/ablation_v2 && sbatch slurm/ablation_v2/lrcond_array.sh
# Re-run a single task (e.g. 4):
#   sbatch --array=4 slurm/ablation_v2/lrcond_array.sh
#
# Task map (index = 4*model + 2*field + norm):
#    0 diffusion_lrcond_temp_std    1 diffusion_lrcond_temp_gstd   2 diffusion_lrcond_multi_std    3 diffusion_lrcond_multi_gstd
#    4 fm_lrcond_temp_std           5 fm_lrcond_temp_gstd          6 fm_lrcond_multi_std           7 fm_lrcond_multi_gstd
#    8 ldm_lrcond_temp_std          9 ldm_lrcond_temp_gstd        10 ldm_lrcond_multi_std         11 ldm_lrcond_multi_gstd

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

MODELS=(diffusion fm ldm)
MODELTYPES=(diffusion flow_matching ldm)
FIELDS=(temp multi)
NORMS=(std gstd)

TASK=${SLURM_ARRAY_TASK_ID:?run as a SLURM array task}
M=$(( TASK / 4 )); F=$(( (TASK / 2) % 2 )); N=$(( TASK % 2 ))
NAME="${MODELS[$M]}_lrcond_${FIELDS[$F]}_${NORMS[$N]}"
CFG="diffusionsr/configs/ablation_v2/$NAME.yml"

echo "=== Task $TASK: $NAME (modeltype=${MODELTYPES[$M]}) ==="
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
  --modeltype      "${MODELTYPES[$M]}" \
  --force_run_dir  "$RUNS/$NAME" \
  --wandb_run_name "$NAME" \
  --resume_from_wandb "$NAME" &
PY_PID=$!
# Returns python's exit status (set -e then fails the task on a crash), or returns early with >128
# when the USR1 trap fires, by which point scontrol requeue is already ending this job.
wait $PY_PID

echo "=== $NAME complete ==="
