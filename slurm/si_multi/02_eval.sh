#!/bin/bash
#SBATCH -J simulti_eval
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/si_multi/eval_%a_%j.log
#
# One evaluation per array index = row (index + 1) of slurm/si_multi/eval.tsv (scripts/gen_si_multi.py),
# through analysis/evaluate_si_schedules.py (SI, FM, DDPM baselines, encoder alone) with eval_extras.
# extra column: '-' | 'extras_only' (add the new outputs to an existing eval) | 'ckpt_artifact=<name>'.
# Sampling resumes from partial/ after preemption.

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

ROW=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == i' slurm/si_multi/eval.tsv)
IFS=$'\t' read -r NAME CFG MODEL ENC DATA SOLVER EPS NSTEPS K WEIGHTS OUT EXTRA PHASE <<< "$ROW"
[ -n "$NAME" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "=== $SLURM_ARRAY_TASK_ID: $NAME $SOLVER eps=$EPS n=$NSTEPS K=$K weights=$WEIGHTS extra=$EXTRA -> $OUT ==="

ARGS=()
case "$EXTRA" in
  extras_only)      ARGS+=(--extras_only) ;;
  ckpt_artifact=*)  ARGS+=(--ckpt_artifact "${EXTRA#ckpt_artifact=}") ;;
esac
# New data: overlay/trajectory frame = largest true melt pool; old data keeps frame 70 (notebooks 21/22).
case "$DATA" in *regen_2d*) ARGS+=(--plot_frame_idx -1) ;; esac
[ "$MODEL" = "-" ] && MODEL="$OUT/_no_model"

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (sampling resumes from partial/) ==="
  scontrol requeue "$SLURM_JOB_ID"; sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.analysis.evaluate_si_schedules \
  --train_name "$NAME" --config "$CFG" --model_dir "$MODEL" --enc_dir "$ENC" --data_root "$DATA" \
  --solver "$SOLVER" --eps "$EPS" --n_steps "$NSTEPS" --k_samples "$K" --weights "$WEIGHTS" \
  --out_root "$OUT" --run_tag v1 "${ARGS[@]}" &
wait $!
echo "=== eval complete ==="
