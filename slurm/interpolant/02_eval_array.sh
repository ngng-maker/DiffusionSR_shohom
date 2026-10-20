#!/bin/bash
#SBATCH --job-name=si_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
# NOTE: SBATCH directives are parsed BEFORE the shell runs, so they cannot read
# $CL_PARTITION / $CL_GRES. These are TRACE defaults; submit_all.sh overrides them on the
# sbatch command line from the cluster profile, and a command-line value wins.
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=logs/interpolant/eval_%a_%j.log
#SBATCH --error=logs/interpolant/eval_%a_%j.err
#SBATCH --requeue
#
# Phase 2 -- score every arm ON THE SAME METRICS, so the numbers are comparable arm for arm.
#
# K-sample ensemble MAE/RMSE, ensemble-mean MAE, MP-MAE (melt-pool depth error in um), keyhole depth
# MAE, melt-pool depth/area, calibration coverage, and the per-pixel ensemble std map.
#
# Task 0 (the DDPM baseline) goes through analysis/evaluate_baseline_matched.py and tasks 1-10 go
# through analysis/evaluate_coupled.py. Those are DIFFERENT entry points but the SAME metric code:
# the baseline wrapper imports evaluate_coupled's `evaluate()` verbatim and only swaps the model it
# builds. Scoring the baseline with eval_predictions.py instead would produce a smaller, different
# metric set and the comparison would not line up.
#
# --k_samples is held equal across all tasks. It must be, or the diversity and calibration metrics
# are measured over different ensemble sizes and are not comparable.
#
# Requires 01_train_array.sh to have completed. Index order must match it exactly.
#
# To requeue one failed task:  sbatch --array=3 slurm/interpolant/02_eval_array.sh

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

# All cluster-specific values come from the profile, so this script is machine-agnostic.
# Locate the repo from SLURM_SUBMIT_DIR, NOT from BASH_SOURCE: sbatch runs a COPY of this script
# from /var/spool/slurmd/job<ID>/, so a path relative to the script resolves into the spool dir.
# See the same block in 01_train_array.sh.
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [ ! -f "$REPO_GUESS/slurm/cluster_env.sh" ]; then
  echo "ERROR: $REPO_GUESS/slurm/cluster_env.sh not found. Submit from the repo root."
  exit 1
fi
source "$REPO_GUESS/slurm/cluster_env.sh"

REPO="$CL_REPO"
RUNS="$CL_RUNS"
CFGS="$CL_CFGS"
ENC_DIR="$CL_ENC_DIR"
OUT="$CL_EVAL_OUT"
DATA="$CL_DATA"

cd "$REPO"
cluster_activate_env
mkdir -p logs/interpolant "$OUT"

TASK=$SLURM_ARRAY_TASK_ID

# Non-default sampling settings get their own results subdirectory, so a sweep point does not
# overwrite the default one. An empty tag leaves the default run's paths unchanged.
SWEEP_TAG=""
if [ "$EPS0" != "0.0" ] || [ "$N_STEPS" != "20" ]; then
  SWEEP_TAG="_eps${EPS0}_n${N_STEPS}"
fi

# MUST match 01_train_array.sh index for index.
CONFIGS=(baseline_ddpm_matched_temp \
         si_opt_a_sigma005_enc_temp si_opt_a_sigma010_enc_temp si_opt_a_sigma020_enc_temp \
         si_opt_b_c010_enc_temp     si_opt_b_c025_enc_temp \
         si_opt_a_sigma005_ulr_temp si_opt_a_sigma010_ulr_temp si_opt_a_sigma020_ulr_temp \
         si_opt_b_c010_ulr_temp     si_opt_b_c025_ulr_temp \
         baseline_diffusionsr_temp)

CFG=${CONFIGS[$TASK]}
NAME="si_${CFG}"
MODEL_DIR="$RUNS/$NAME"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: $MODEL_DIR does not exist. Run 01_train_array.sh --array=$TASK first."
  exit 1
fi

# Recover the date-stamped W&B display name written by Phase 1. Recomputing it with `date` would be
# wrong whenever eval runs on a later calendar day than training.
WNAME_FILE="$MODEL_DIR/wandb_run_name.txt"
if [ -f "$WNAME_FILE" ]; then
  WNAME=$(cat "$WNAME_FILE")
  WNAME_ARG="--wandb_run_name $WNAME"
else
  echo "WARNING: $WNAME_FILE not found; skipping the W&B artifact upload (results still land in $OUT)."
  WNAME_ARG=""
fi

echo "=== Task $TASK: evaluating $CFG ==="

if [ "$TASK" -eq 0 ] || [ "$TASK" -eq 11 ]; then
  # --- the two DDPM reference points ---
  # Task 0 was trained in the coupled harness, so its checkpoint carries EMA weights and must be
  # rebuilt as DDPMMatchedHarnessModel. Task 11 came from the plain DiffusionModel loop. Passing the
  # wrong --harness would load weights into the wrong class, so it is derived, not assumed.
  if [ "$TASK" -eq 0 ]; then HARNESS=coupled; else HARNESS=plain; fi
  echo "    DDPM baseline (harness=$HARNESS), DDIM skip=$SKIP (= $((1000/SKIP)) steps), K=$K_SAMPLES"
  python -m diffusionsr.analysis.evaluate_baseline_matched \
      --run_name    "$NAME" \
      --model_dir   "$MODEL_DIR" \
      --enc_dir     "$ENC_DIR" \
      --data_root   "$DATA" \
      --field_names temperature \
      --harness     "$HARNESS" \
      --skip        "$SKIP" \
      --k_samples   "$K_SAMPLES" \
      --out_dir     "$OUT/$NAME$SWEEP_TAG" \
      --wandb_entity "$WANDB_ENTITY" \
      $WNAME_ARG
else
  # --- the interpolant arms ---
  # sigma, gamma_c and si_cond are READ FROM THE TRAINING CONFIG rather than hardcoded here.
  # evaluate_coupled's --config only picks up root_folder and fields, so these three would
  # otherwise have to be duplicated on the command line -- and a mismatch would silently sample
  # from the wrong base distribution (sigma feeds make_x0) with no error.
  SIGMA=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['si_sigma'])"    "$CFGS/$CFG.yml")
  GAMMA=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['si_gamma_c'])"  "$CFGS/$CFG.yml")
  SICOND=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['si_cond'])"    "$CFGS/$CFG.yml")
  NEEDS_ENC=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('encoding', False))" "$CFGS/$CFG.yml")

  # Option A (gamma_c = 0) has no path noise and therefore no score, so it is ODE-only: eps0 must
  # stay 0. Option B (gamma_c > 0) trains the denoiser head and can be sampled stochastically.
  ARM_EPS0=$EPS0
  if [ "$GAMMA" = "0.0" ] && [ "$EPS0" != "0.0" ]; then
    echo "    NOTE: Option A (gamma_c=0) has no score to drive an SDE; forcing eps0=0.0"
    ARM_EPS0=0.0
  fi

  ENC_ARG=""
  if [ "$NEEDS_ENC" = "True" ]; then ENC_ARG="--enc_dir $ENC_DIR"; fi

  echo "    sigma=$SIGMA gamma_c=$GAMMA si_cond=$SICOND eps0=$ARM_EPS0 n_steps=$N_STEPS K=$K_SAMPLES"
  python -m diffusionsr.analysis.evaluate_coupled \
      --run_name    "$NAME" \
      --model_dir   "$MODEL_DIR" \
      --config      "$CFGS/$CFG.yml" \
      --sigma       "$SIGMA" \
      --gamma_c     "$GAMMA" \
      --si_cond     "$SICOND" \
      --sampler     heun \
      --n_steps     "$N_STEPS" \
      --eps0        "$ARM_EPS0" \
      --k_samples   "$K_SAMPLES" \
      --out_dir     "$OUT/$NAME$SWEEP_TAG" \
      --wandb_entity "$WANDB_ENTITY" \
      $ENC_ARG \
      $WNAME_ARG
fi

echo "=== eval $NAME complete; results in $OUT/$NAME$SWEEP_TAG ==="
