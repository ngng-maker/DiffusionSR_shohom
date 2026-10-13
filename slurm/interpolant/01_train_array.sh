#!/bin/bash
#SBATCH --job-name=si_train
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
# NOTE: SBATCH directives are parsed BEFORE the shell runs, so they cannot read
# $CL_PARTITION / $CL_GRES. These are TRACE defaults; submit_all.sh overrides them on the
# sbatch command line from the cluster profile, and a command-line value wins.
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=logs/interpolant/train_%a_%j.log
#SBATCH --error=logs/interpolant/train_%a_%j.err
#SBATCH --requeue
#
# Phase 1 -- train the DiffusionSR baseline and every stochastic-interpolant arm.
#
# THE COMPARISON THIS IS BUILT FOR
# --------------------------------
# Task 0 is DiffusionSR (DDPM + frozen RRDB encoder). Tasks 1-5 use the SAME frozen encoder
# (ablation_20260901/enc_temp) and the SAME implicit 64-channel conditioning, so against task 0 the
# ONLY difference is the generative process: DDPM -> stochastic interpolant. Those five are the
# attributable comparison.
#
# Tasks 6-10 are the ulr_concat twins. They drop the encoder AND change the conditioning to
# concatenated upscaled-LR, so relative to task 0 they change three things at once and are only
# interpretable against each other. They are the paper's own setup (Albergo et al. use bicubic with
# no encoder), kept as the secondary axis.
#
#   task  config                           encoder   conditioning    interpolant
#   ----  -------------------------------  --------  --------------  ---------------------
#    0    baseline_diffusionsr_temp        enc_temp  implicit 64ch   DDPM (the baseline)
#    1    si_opt_a_sigma005_enc_temp       enc_temp  implicit 64ch   A, sigma=0.05
#    2    si_opt_a_sigma010_enc_temp       enc_temp  implicit 64ch   A, sigma=0.10
#    3    si_opt_a_sigma020_enc_temp       enc_temp  implicit 64ch   A, sigma=0.20
#    4    si_opt_b_c010_enc_temp           enc_temp  implicit 64ch   B, c=0.10
#    5    si_opt_b_c025_enc_temp           enc_temp  implicit 64ch   B, c=0.25
#    6    si_opt_a_sigma005_ulr_temp       none      ulr_concat      A, sigma=0.05
#    7    si_opt_a_sigma010_ulr_temp       none      ulr_concat      A, sigma=0.10
#    8    si_opt_a_sigma020_ulr_temp       none      ulr_concat      A, sigma=0.20
#    9    si_opt_b_c010_ulr_temp           none      ulr_concat      B, c=0.10
#   10    si_opt_b_c025_ulr_temp           none      ulr_concat      B, c=0.25
#
# The enc arms come first deliberately: running only --array=0-5 gives the clean comparison.
#
# To run just the attributable comparison:   sbatch --array=0-5 slurm/interpolant/01_train_array.sh
# To requeue one failed task (e.g. task 3):  sbatch --array=3   slurm/interpolant/01_train_array.sh

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

# All cluster-specific values come from the profile, so this script is machine-agnostic.
REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_GUESS/slurm/cluster_env.sh"

REPO="$CL_REPO"
RUNS="$CL_RUNS"
CFGS="$CL_CFGS"
ENC_DIR="$CL_ENC_DIR"

cd "$REPO"
cluster_activate_env
mkdir -p logs/interpolant

TASK=$SLURM_ARRAY_TASK_ID
DATE=$(date +"%d_%b_%Y")

# Index order is load-bearing: it must match 02_eval_array.sh exactly, or an arm gets evaluated
# with another arm's settings.
CONFIGS=(baseline_ddpm_matched_temp \
         si_opt_a_sigma005_enc_temp si_opt_a_sigma010_enc_temp si_opt_a_sigma020_enc_temp \
         si_opt_b_c010_enc_temp     si_opt_b_c025_enc_temp \
         si_opt_a_sigma005_ulr_temp si_opt_a_sigma010_ulr_temp si_opt_a_sigma020_ulr_temp \
         si_opt_b_c010_ulr_temp     si_opt_b_c025_ulr_temp \
         baseline_diffusionsr_temp)

CFG=${CONFIGS[$TASK]}
# Task 11 is the AS-PUBLISHED DiffusionSR and must use the plain DiffusionModel loop (no EMA, no
# gradient clipping, fixed LR, SimulationXZDataset). It is a reference point, not the comparison.
# Every other task -- including task 0 -- runs through the SHARED coupled harness, where the
# config's si_process key selects DDPM vs interpolant. That is what makes task 0 vs task 2 differ
# in the generative process ALONE.
if [ "$TASK" -eq 11 ]; then MTYPE=diffusion; else MTYPE=coupled; fi
NAME="si_${CFG}"
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $CFG (modeltype=$MTYPE, wandb=$WNAME) ==="

# Read `encoding` straight from the YAML rather than hardcoding which tasks need the encoder, so the
# script cannot drift out of step with the configs.
NEEDS_ENC=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('encoding', False))" \
            "$CFGS/$CFG.yml")

# --force_enc_dir is MANDATORY for the encoder arms. Every config sets use_pretrained: False, so
# without it train_srdiff falls through to a datetime-stamped path and TRAINS A FRESH ENCODER --
# which would silently break the "same encoder as DiffusionSR" property the comparison rests on.
ENC_ARG=""
if [ "$NEEDS_ENC" = "True" ]; then
  if [ ! -f "$ENC_DIR/bestmodel_saved.pth" ]; then
    echo "ERROR: $CFG needs the frozen encoder but $ENC_DIR/bestmodel_saved.pth is missing."
    echo "       Train it first, or point ENC_DIR at an encoder that exists. Refusing to run,"
    echo "       because proceeding would train a fresh encoder and break comparability with task 0."
    exit 1
  fi
  ENC_ARG="--force_enc_dir $ENC_DIR"
  echo "    using frozen encoder: $ENC_DIR"
else
  echo "    no encoder (ulr_concat conditioning)"
fi

# Record the W&B display name so 02_eval_array.sh can find this run even if it executes on a later
# calendar day. Written once only, so a requeue keeps the original stamp.
mkdir -p "$RUNS/$NAME"
if [ ! -f "$RUNS/$NAME/wandb_run_name.txt" ]; then
  echo "$WNAME" > "$RUNS/$NAME/wandb_run_name.txt"
fi

python -m diffusionsr.runners.train_srdiff \
  --data_root         "$CL_DATA" \
  --config            "$CFGS/$CFG.yml" \
  --modeltype         "$MTYPE" \
  --gpu               0 \
  --force_run_dir     "$RUNS/$NAME" \
  --wandb_run_name    "$WNAME" \
  --resume_from_wandb "$WNAME" \
  $ENC_ARG

echo "=== $NAME complete ==="
