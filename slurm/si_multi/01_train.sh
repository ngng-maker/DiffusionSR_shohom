#!/bin/bash
#SBATCH -J simulti_train
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/si_multi/train_%a_%j.log
#
# One training run per array index = row (index + 1) of slurm/si_multi/train.tsv (scripts/gen_si_multi.py).
# Submit with slurm/si_multi/submit.sh, which picks the rows of a phase. Kinds:
#   encoder       frozen RRDB encoder pretraining (train_srdiff --modeltype encoder)
#   si | fm | ddpm_matched   the shared SI harness (--modeltype coupled; si_process in the config)
#   diffusionsr   the as-published DiffusionSR loop (--modeltype diffusion)
# Every run resumes from its own ckpt.pth after preemption or the requeue below.

set -eo pipefail
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_multi
export SI_WANDB_GROUP=si_multi

ROW=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == i' slurm/si_multi/train.tsv)
IFS=$'\t' read -r KIND NAME MTYPE CFG RUN ENC PHASE <<< "$ROW"
[ -n "$NAME" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }
DATA=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['root_folder'])" "$CFG")
RUN_ID=$(echo "$NAME-v1" | tr '[:upper:]_' '[:lower:]-')
echo "=== $SLURM_ARRAY_TASK_ID: $KIND $NAME (modeltype=$MTYPE, phase $PHASE) data=$DATA ==="
[ -d "$DATA/train" ] || { echo "ERROR: dataset missing: $DATA"; exit 1; }

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (resumes from ckpt.pth) ==="
  scontrol requeue "$SLURM_JOB_ID"; sleep 300
}
trap requeue_before_timeout USR1

if [ "$KIND" = encoder ]; then
  if [ -f "$ENC/bestmodel_saved.pth" ] && [ -f "$ENC/.done" ]; then echo "encoder already trained"; exit 0; fi
  python -m diffusionsr.runners.train_srdiff --data_root "$DATA" --config "$CFG" --modeltype encoder --gpu 0 \
    --force_enc_dir "$ENC" --wandb_run_name "$NAME" &
  wait $!
  touch "$ENC/.done"
else
  [ -f "$ENC/bestmodel_saved.pth" ] || { echo "ERROR: frozen encoder missing: $ENC (train phase 1 first)"; exit 1; }
  mkdir -p "$RUN"
  if [ "$KIND" = diffusionsr ] || [ "$KIND" = ddpm_matched ]; then
    RESUME="--resume_from_wandb $NAME"
  else
    RESUME="--wandb_run_id $RUN_ID --ckpt_artifact ${NAME}_ckpt"
  fi
  python -m diffusionsr.runners.train_srdiff --data_root "$DATA" --config "$CFG" --modeltype "$MTYPE" --gpu 0 \
    --force_enc_dir "$ENC" --force_run_dir "$RUN" --wandb_run_name "$NAME" $RESUME ${SI_EPOCHS:+--epochs $SI_EPOCHS} &
  wait $!
fi
echo "=== $NAME complete ==="
