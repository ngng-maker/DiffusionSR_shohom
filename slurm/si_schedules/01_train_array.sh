#!/bin/bash
#SBATCH --job-name=sisched_train
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
# SBATCH directives are TRACE defaults; submit_all.sh overrides partition/gres/time from the cluster
# profile (slurm/cluster_env.sh) on the sbatch command line.
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=logs/si_schedules/train_%a_%j.log
#SBATCH --error=logs/si_schedules/train_%a_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#
# SI-schedules study, phase 1: 8 training runs (4 schedules x 2 paired bases). See arms.sh.
#
# Preemption / interruption / moving between TRACE and ORCHARD
# ------------------------------------------------------------
#  * ckpt.pth is written atomically every epoch; a requeued task resumes from it.
#  * Every ckpt_every (=5) epochs the full training state (ckpt.pth, bestmodel_saved.pth, loss files,
#    config, run id) is uploaded as the ONLY version of the artifact <config>_ckpt; older versions are
#    deleted after the new one is committed. One artifact per config, overwritten, never accumulated.
#  * If the run directory is missing or empty (scratch cleared, or the task restarted on the other
#    cluster), train_srdiff restores it from that artifact before training (--ckpt_artifact).
#  * The W&B run id is deterministic (--wandb_run_id), resumed with resume='allow', so every segment of
#    a config, on either cluster, logs into ONE run.
#  Do not run the same config on both clusters AT THE SAME TIME: both would write the one artifact.

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT

# Repo from SLURM_SUBMIT_DIR, not BASH_SOURCE (sbatch runs a spooled copy of this script).
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "$REPO_GUESS/slurm/cluster_env.sh" ]; then
  echo "ERROR: $REPO_GUESS/slurm/cluster_env.sh not found. Submit from the repo root."; exit 1
fi
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_schedules/arms.sh"

REPO="$CL_REPO"
RUNS="$CL_REPO/diffusionsr/runs/si_schedules"
CFGS="$CL_REPO/diffusionsr/configs/si_schedules"
ENC_DIR="$CL_ENC_DIR"

cd "$REPO"
cluster_activate_env
mkdir -p logs/si_schedules "$RUNS"

TASK=$SLURM_ARRAY_TASK_ID
CFG=${SI_CONFIGS[$TASK]}
RUN_DIR="$RUNS/$CFG"
RUN_ID=$(train_run_id "$CFG")
ART=$(ckpt_artifact "$CFG")
EPOCHS_ARG=""
if [ -n "${SI_EPOCHS:-}" ]; then EPOCHS_ARG="--epochs $SI_EPOCHS"; fi

echo "=== Task $TASK: $CFG  run_id=$RUN_ID  artifact=$ART  cluster=$CLUSTER ==="

# Same frozen encoder as DiffusionSR (task 0/11) and the previous SI arms. Without --force_enc_dir
# train_srdiff would train a FRESH encoder and break comparability, so refuse to run without it.
if [ ! -f "$ENC_DIR/bestmodel_saved.pth" ]; then
  echo "ERROR: frozen encoder missing: $ENC_DIR/bestmodel_saved.pth"; exit 1
fi

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (resumes from ckpt.pth) ==="
  scontrol requeue "$SLURM_JOB_ID"
  sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.runners.train_srdiff \
  --data_root      "$CL_DATA" \
  --config         "$CFGS/$CFG.yml" \
  --modeltype      coupled \
  --gpu            0 \
  --force_enc_dir  "$ENC_DIR" \
  --force_run_dir  "$RUN_DIR" \
  --wandb_run_name "$CFG" \
  --wandb_run_id   "$RUN_ID" \
  --ckpt_artifact  "$ART" \
  $EPOCHS_ARG &
PY_PID=$!
wait $PY_PID

echo "=== $CFG complete ==="
