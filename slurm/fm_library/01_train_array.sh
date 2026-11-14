#!/bin/bash
#SBATCH --job-name=fmlib_train
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=logs/fm_library/train_%a_%j.log
#SBATCH --error=logs/fm_library/train_%a_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#
# FM-library study: train standard flow matching (facebookresearch/flow_matching, single velocity head)
# in the SI harness. Index map: arms.sh. Identical to slurm/si_noise/01_train_array.sh apart from the
# arms, directories, W&B group (fm_library) and the encoder, which is read from each config.
#
# Preemption / time limit: SIGUSR1 10 min before the limit requeues the job; it resumes from ckpt.pth, or
# from the config's single W&B checkpoint artifact if the run directory is gone (--ckpt_artifact).
#
# Usage, from the repo root on TRACE:
#   sbatch slurm/fm_library/01_train_array.sh              # all four
#   sbatch --array=0-1 slurm/fm_library/01_train_array.sh  # single-field only (no 2-field encoder needed)
#   SI_EPOCHS=1 sbatch --array=1 ...                       # short run (smoke)

set -eo pipefail
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT

REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "$REPO_GUESS/slurm/cluster_env.sh" ]; then
  echo "ERROR: $REPO_GUESS/slurm/cluster_env.sh not found. Submit from the repo root."; exit 1
fi
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/fm_library/arms.sh"
REPO="$CL_REPO"
RUNS="${FM_RUNS:-$CL_REPO/diffusionsr/runs/fm_library}"
CFGS="$CL_REPO/diffusionsr/configs/fm_library"
cd "$REPO"
cluster_activate_env
mkdir -p logs/fm_library "$RUNS"
python -c "import flow_matching, torchdiffeq" || { echo "ERROR: pip install --no-deps flow_matching==1.0.10 torchdiffeq==0.2.5"; exit 1; }

TASK=$SLURM_ARRAY_TASK_ID
CFG=${FM_CONFIGS[$TASK]}
CFG_PATH="$CFGS/$CFG.yml"
RUN_NAME="${FM_RUN_PREFIX:-}$CFG"
RUN_DIR="$RUNS/$RUN_NAME"
RUN_ID=$(train_run_id "$RUN_NAME")
ART=$(ckpt_artifact "$RUN_NAME")
ENC_DIR=$(encoder_dir "$CFG_PATH")
EPOCHS_ARG=""
if [ -n "${SI_EPOCHS:-}" ]; then EPOCHS_ARG="--epochs $SI_EPOCHS"; fi
echo "=== Task $TASK: $CFG  run=$RUN_NAME  run_id=$RUN_ID  artifact=$ART  enc=$ENC_DIR  cluster=$CLUSTER ==="

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
  --config         "$CFG_PATH" \
  --modeltype      coupled \
  --gpu            0 \
  --force_enc_dir  "$ENC_DIR" \
  --force_run_dir  "$RUN_DIR" \
  --wandb_run_name "$RUN_NAME" \
  --wandb_run_id   "$RUN_ID" \
  --ckpt_artifact  "$ART" \
  $EPOCHS_ARG &
PY_PID=$!
wait $PY_PID
echo "=== $RUN_NAME complete ==="
