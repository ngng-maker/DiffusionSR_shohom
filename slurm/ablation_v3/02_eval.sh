#!/bin/bash
#SBATCH -J abl3_eval
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/ablation_v3/eval_%a_%j.log
#
# Ablation v3: one evaluation per array index = row (index + 1) of $TSV (default slurm/ablation_v3/eval.tsv),
# through analysis/evaluate_si_schedules.py (ablation_adapter for model_kind abl_*), so the metrics, P-V maps,
# timing, model size and GIFs match the SI studies. Full test split, K = 10 (encoder alone: K = 1).
# Sampling resumes from partial/ after preemption. EVAL_ARGS adds arguments (00_smoke.sh uses --max_frames).

set -eo pipefail
TSV=${TSV:-slurm/ablation_v3/eval.tsv}
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/ablation_v3
export SI_WANDB_GROUP=${SI_WANDB_GROUP:-ablation_v3}

ROW=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == i' "$TSV")
IFS=$'\t' read -r NAME CFG MODEL ENC DATA SOLVER EPS NSTEPS K WEIGHTS OUT EXTRA PHASE <<< "$ROW"
[ -n "$NAME" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "=== $SLURM_ARRAY_TASK_ID: $NAME $SOLVER n=$NSTEPS K=$K -> $OUT ==="
[ "$MODEL" = "-" ] && MODEL="$OUT/_no_model"
[ "$ENC" = "-" ] && ENC=""
ARGS=(--plot_frame_idx -1)          # regen data: overlay/trajectory frame = largest true melt pool
[ "$EXTRA" = extras_only ] && ARGS+=(--extras_only)

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (sampling resumes from partial/) ==="
  scontrol requeue "$SLURM_JOB_ID"; sleep 300
}
trap requeue_before_timeout USR1

# shellcheck disable=SC2086  # EVAL_ARGS is a word list on purpose
python -m diffusionsr.analysis.evaluate_si_schedules \
  --train_name "$NAME" --config "$CFG" --model_dir "$MODEL" --enc_dir "$ENC" --data_root "$DATA" \
  --solver "$SOLVER" --eps "$EPS" --n_steps "$NSTEPS" --k_samples "$K" --weights "$WEIGHTS" \
  --out_root "$OUT" --run_tag v1 "${ARGS[@]}" ${EVAL_ARGS:-} &
wait $!
echo "=== eval complete ==="
