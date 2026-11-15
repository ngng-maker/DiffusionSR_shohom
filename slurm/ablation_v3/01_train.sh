#!/bin/bash
#SBATCH -J abl3_train
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/ablation_v3/train_%a_%j.log
#
# Ablation v3 (scripts/gen_ablation_v3.py): one training per array index = row (index + 1) of $TSV
# (default slurm/ablation_v3/train.tsv). Submit with slurm/ablation_v3/submit.sh. Kinds:
#   encoder     shared frozen RRDB encoder (train_srdiff --modeltype encoder)         phase 1
#   vae         shared LDM VAE (train_srdiff --modeltype vae)                          phase 1
#   dsr|fm|ldm  the model, with the shared encoder (enc_dir) or none ('-')            phase 2
# Every run resumes from its own checkpoint after preemption or the requeue below (encoder_ckpt.pth,
# vae_ckpt.pth, ckpt.pth).

set -eo pipefail
TSV=${TSV:-slurm/ablation_v3/train.tsv}
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/ablation_v3

ROW=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == i' "$TSV")
IFS=$'\t' read -r KIND NAME MTYPE CFG RUN ENC PHASE <<< "$ROW"
[ -n "$NAME" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }
DATA=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['root_folder'])" "$CFG")
echo "=== $SLURM_ARRAY_TASK_ID: $KIND $NAME (modeltype=$MTYPE, phase $PHASE) data=$DATA ==="
[ -d "$DATA/train" ] || { echo "ERROR: dataset missing: $DATA"; exit 1; }

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (resumes from its checkpoint) ==="
  scontrol requeue "$SLURM_JOB_ID"; sleep 300
}
trap requeue_before_timeout USR1

case "$KIND" in
  encoder)
    if [ -f "$ENC/.done" ]; then echo "encoder already trained"; exit 0; fi
    python -m diffusionsr.runners.train_srdiff --data_root "$DATA" --config "$CFG" --modeltype encoder --gpu 0 \
      --force_enc_dir "$ENC" --wandb_run_name "$NAME" &
    wait $!
    touch "$ENC/.done" ;;
  vae)
    if [ -f "$RUN/.done" ]; then echo "VAE already trained"; exit 0; fi
    python -m diffusionsr.runners.train_srdiff --data_root "$DATA" --config "$CFG" --modeltype vae --gpu 0 \
      --force_run_dir "$RUN" --wandb_run_name "$NAME" &
    wait $!
    touch "$RUN/.done" ;;
  *)
    ENC_ARGS=()
    if [ "$ENC" != "-" ]; then
      [ -f "$ENC/.done" ] || { echo "ERROR: shared encoder not finished: $ENC (phase 1 first)"; exit 1; }
      ENC_ARGS=(--force_enc_dir "$ENC")
    fi
    if [ "$KIND" = ldm ]; then
      VAE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['vae_results_dir'])" "$CFG")
      [ -f "$VAE/.done" ] || { echo "ERROR: shared VAE not finished: $VAE (phase 1 first)"; exit 1; }
    fi
    mkdir -p "$RUN"
    python -m diffusionsr.runners.train_srdiff --data_root "$DATA" --config "$CFG" --modeltype "$MTYPE" --gpu 0 \
      "${ENC_ARGS[@]}" --force_run_dir "$RUN" --wandb_run_name "$NAME" &
    wait $! ;;
esac
echo "=== $NAME complete ==="
