#!/bin/bash
#SBATCH -J simulti_cascade
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH -o logs/si_multi/cascade_%a_%j.log
#
# One chained 40 -> 20 -> 10 um evaluation per array index = row (index + 1) of slurm/si_multi/cascade.tsv,
# through analysis/evaluate_cascade.py. Baseline chains carry solver 'ddpm:<weights>' (ancestral, both stages).

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_multi

ROW=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID + 2)) 'NR == i' slurm/si_multi/cascade.tsv)
IFS=$'\t' read -r CHAIN CA RA EA DA CB RB EB DB SOLVER EPS NSTEPS OUT PHASE <<< "$ROW"
[ -n "$CHAIN" ] || { echo "no row for index $SLURM_ARRAY_TASK_ID"; exit 1; }
WEIGHTS=raw
case "$SOLVER" in ddpm:*) WEIGHTS="${SOLVER#ddpm:}"; SOLVER=ddpm ;; esac
echo "=== $SLURM_ARRAY_TASK_ID: $CHAIN $SOLVER eps=$EPS n=$NSTEPS weights=$WEIGHTS ==="

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID ==="
  scontrol requeue "$SLURM_JOB_ID"; sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.analysis.evaluate_cascade --chain_name "$CHAIN" \
  --a_config "$CA" --a_model_dir "$RA" --a_enc_dir "$EA" --a_data_root "$DA" \
  --b_config "$CB" --b_model_dir "$RB" --b_enc_dir "$EB" --b_data_root "$DB" \
  --solver "$SOLVER" --eps "$EPS" --n_steps "$NSTEPS" --weights "$WEIGHTS" \
  --upscale_mode "${CASCADE_UPSCALE_MODE:-bicubic}" --out_root "$OUT" &
wait $!
echo "=== cascade eval complete ==="
