#!/bin/bash
#SBATCH -J si_ema_eval
#SBATCH -p batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-12
#SBATCH -o logs/si_noise/ema_eval_%a_%j.log
#
# Re-score the SI models shown in the deck from their EMA weights (checkpoint[5]).
#
# Why: the task-0 DDPM baseline was scored from EMA weights while every SI eval used raw weights
# (checkpoint[0]). Task 0's checkpoint no longer exists, so the matched comparison is made from the SI
# side: same solver, eps, steps, K and seed as the original evals, only --weights changes.
# Results go to eval_results/si_ema/<eval_name>/ with no W&B upload, so the raw-weight artifacts
# (one per eval config) are left untouched.
#
# Usage, from the repo root on TRACE:  sbatch slurm/si_noise/04_eval_ema.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_noise

#      config                     study          solver  eps
ARMS=("si_sched_linear_enc_temp   si_schedules   heun    0.0"
      "si_sched_trig_enc_temp     si_schedules   heun    0.0"
      "si_sched_vp_enc_temp       si_schedules   heun    0.0"
      "si_sched_bridge_enc_temp   si_schedules   heun    0.0"
      "si_sched_linear_ulr_temp   si_schedules   heun    0.0"
      "si_sched_trig_ulr_temp     si_schedules   heun    0.0"
      "si_sched_vp_ulr_temp       si_schedules   heun    0.0"
      "si_sched_bridge_ulr_temp   si_schedules   heun    0.0"
      "si_sched_linear_enc_temp   si_schedules   heun    20.0"
      "si_noise_g1_enc_temp       si_noise       heun    5.0"
      "si_noise_g2_enc_temp       si_noise       heun    5.0"
      "si_noise_s0p5_enc_temp     si_noise       heun    0.0"
      "si_noise_s0p5_enc_temp     si_noise       heun    5.0")
read -r CFG STUDY SOLVER EPS <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
OUT=eval_results/si_ema
echo "=== EMA eval $SLURM_ARRAY_TASK_ID: $CFG $SOLVER eps=$EPS ==="

python -m diffusionsr.analysis.evaluate_si_schedules \
  --train_name "$CFG" \
  --config     "$CL_REPO/diffusionsr/configs/$STUDY/$CFG.yml" \
  --model_dir  "$CL_REPO/diffusionsr/runs/$STUDY/$CFG" \
  --enc_dir    "$CL_ENC_DIR" \
  --data_root  "$CL_DATA" \
  --solver "$SOLVER" --eps "$EPS" --n_steps 20 --k_samples 10 \
  --weights ema --no_upload --out_root "$OUT"

EVAL_NAME=$(python -c "import sys; from diffusionsr.analysis.evaluate_si_schedules import eval_name_for; \
print(eval_name_for(sys.argv[1], sys.argv[2], float(sys.argv[3])))" "$CFG" "$SOLVER" "$EPS")
python -m diffusionsr.analysis.add_depth_metrics "$OUT/$EVAL_NAME"
echo "=== done: $OUT/$EVAL_NAME ==="
