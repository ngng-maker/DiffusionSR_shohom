#!/bin/bash
#SBATCH --job-name=sisched_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-127
#SBATCH --output=logs/si_schedules/eval_%a_%j.log
#SBATCH --error=logs/si_schedules/eval_%a_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#
# SI-schedules study, phase 2: 128 evaluations = 8 trained runs x 16 (solver, eps). See arms.sh.
#
# Same metric code, test split, K = 10 and 20-step budget as the previous SI run (notebook 21), via
# diffusionsr/analysis/evaluate_si_schedules.py. Each task:
#   * resumes its own sampling after preemption (per-batch progress, per-batch seeds -> same samples);
#   * restores bestmodel_saved.pth from the config's checkpoint artifact if the run dir is missing
#     (e.g. trained on the other cluster);
#   * uploads ONE artifact <config>__<solver>_eps<eps>_eval_predictions, single version, overwritten.
#
# Rerun one eval:      sbatch --array=37 slurm/si_schedules/02_eval_array.sh
# All evals of a run:  sbatch --array=32-47 slurm/si_schedules/02_eval_array.sh   (train idx 2)

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT

REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "$REPO_GUESS/slurm/cluster_env.sh" ]; then
  echo "ERROR: $REPO_GUESS/slurm/cluster_env.sh not found. Submit from the repo root."; exit 1
fi
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_schedules/arms.sh"

REPO="$CL_REPO"
RUNS="$CL_REPO/diffusionsr/runs/si_schedules"
CFGS="$CL_REPO/diffusionsr/configs/si_schedules"
OUT="${SI_EVAL_OUT:-$CL_REPO/eval_results/si_schedules}"

cd "$REPO"
cluster_activate_env
mkdir -p logs/si_schedules "$OUT"

decode_eval "$SLURM_ARRAY_TASK_ID"
echo "=== Eval task $SLURM_ARRAY_TASK_ID: $SI_CFG  solver=$SI_SOLVER  eps=$SI_EPS  n_steps=$SI_N_STEPS  K=$SI_K_SAMPLES ==="

requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing $SLURM_JOB_ID (sampling resumes from partial/) ==="
  scontrol requeue "$SLURM_JOB_ID"
  sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.analysis.evaluate_si_schedules \
  --train_name    "$SI_CFG" \
  --config        "$CFGS/$SI_CFG.yml" \
  --model_dir     "$RUNS/$SI_CFG" \
  --enc_dir       "$CL_ENC_DIR" \
  --data_root     "$CL_DATA" \
  --solver        "$SI_SOLVER" \
  --eps           "$SI_EPS" \
  --n_steps       "$SI_N_STEPS" \
  --k_samples     "$SI_K_SAMPLES" \
  --out_root      "$OUT" \
  --ckpt_artifact "$(ckpt_artifact "$SI_CFG")" \
  --run_tag       "$SI_SCHED_TAG" \
  ${SI_EVAL_EXTRA:-} &
PY_PID=$!
wait $PY_PID

# Exact eval name from the same function the evaluator uses, so the check cannot accept another eps.
EVAL_NAME=$(python -c "import sys; from diffusionsr.analysis.evaluate_si_schedules import eval_name_for; \
print(eval_name_for(sys.argv[1], sys.argv[2], float(sys.argv[3])))" "$SI_CFG" "$SI_SOLVER" "$SI_EPS")
if [ ! -f "$OUT/$EVAL_NAME/${EVAL_NAME}_results.json" ]; then
  echo "ERROR: $OUT/$EVAL_NAME/${EVAL_NAME}_results.json was not written"; exit 1
fi
echo "=== eval complete ==="
