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
#SBATCH --array=0-12
#SBATCH --output=logs/interpolant/train_%a_%j.log
#SBATCH --error=logs/interpolant/train_%a_%j.err
#SBATCH --requeue
# Ask SLURM for SIGUSR1 to the batch shell 10 min before the time limit, so the job can requeue
# itself instead of ending as TIMEOUT (which is NOT requeued and would block the eval array).
# This is what makes a short --time safe: see requeue_before_timeout below.
#SBATCH --signal=B:USR1@600
#
# Phase 1 -- train the DiffusionSR baseline and every stochastic-interpolant arm.
#
# THE COMPARISON THIS IS BUILT FOR
# --------------------------------
# Task 0 is DiffusionSR (DDPM + frozen RRDB encoder). Tasks 1-5 use the SAME frozen encoder
# (fno_ablation/enc_rrdb) and the SAME implicit 64-channel conditioning, so against task 0 the
# ONLY difference is the generative process: DDPM -> stochastic interpolant. Those five are the
# attributable comparison.
#
# Tasks 6-10 are the ulr_concat twins. They drop the encoder AND change the conditioning to
# concatenated upscaled-LR, so relative to task 0 they change three things at once and are only
# interpretable against each other. They are the paper's own setup (Albergo et al. use bicubic with
# no encoder), kept as the secondary axis.
#
#   task  config                           encoder   conditioning    x0 base   process
#   ----  -------------------------------  --------  --------------  --------  ------------------
#    0    baseline_ddpm_matched_temp       enc_rrdb  implicit 64ch   noise     DDPM (the baseline)
#    1    si_opt_a_sigma005_enc_temp       enc_rrdb  implicit 64ch   bicubic   A, sigma=0.05
#    2    si_opt_a_sigma010_enc_temp       enc_rrdb  implicit 64ch   bicubic   A, sigma=0.10
#    3    si_opt_a_sigma020_enc_temp       enc_rrdb  implicit 64ch   bicubic   A, sigma=0.20
#    4    si_opt_b_c010_enc_temp           enc_rrdb  implicit 64ch   bicubic   B, c=0.10
#    5    si_opt_b_c025_enc_temp           enc_rrdb  implicit 64ch   bicubic   B, c=0.25
#    6    si_opt_a_sigma005_ulr_temp       none      ulr_concat      bicubic   A, sigma=0.05
#    7    si_opt_a_sigma010_ulr_temp       none      ulr_concat      bicubic   A, sigma=0.10
#    8    si_opt_a_sigma020_ulr_temp       none      ulr_concat      bicubic   A, sigma=0.20
#    9    si_opt_b_c010_ulr_temp           none      ulr_concat      bicubic   B, c=0.10
#   10    si_opt_b_c025_ulr_temp           none      ulr_concat      bicubic   B, c=0.25
#   11    baseline_diffusionsr_temp        enc_rrdb  implicit 64ch   noise     DDPM, plain loop
#   12    si_opt_a_sigma010_encbase_temp   enc_rrdb  implicit 64ch   enc(LR)   A, sigma=0.10
#
# Task 12 is task 2 with the transport starting at the encoder's own HR prediction instead of the
# bicubic upscale (si_base: "enc"), i.e. DiffusionSR with the diffusion model replaced outright and
# only the CNN encoder kept. Task 12 vs task 2 isolates the base.
#
# The enc arms come first deliberately: running only --array=0-5 gives the clean comparison.
#
# To run just the attributable comparison:   sbatch --array=0-5 slurm/interpolant/01_train_array.sh
# To requeue one failed task (e.g. task 3):  sbatch --array=3   slurm/interpolant/01_train_array.sh

set -eo pipefail

export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
# Single quotes: expanded at exit, after cluster_env.sh has (optionally) set a per-job WANDB_DATA_DIR.
# `|| true`: a cleanup failure (e.g. NFS "Directory not empty") must never turn a finished task
# into FAILED -- that blocks every afterok dependency even though the work succeeded.
trap 'rm -rf "${WANDB_DATA_DIR:-$HOME/.local/share/wandb}/artifacts/staging/" "$WANDB_CACHE_DIR" || true' EXIT

# All cluster-specific values come from the profile, so this script is machine-agnostic.
# Locate the repo from SLURM_SUBMIT_DIR, NOT from BASH_SOURCE: sbatch runs a COPY of this script
# from /var/spool/slurmd/job<ID>/, so a path relative to the script resolves into the spool dir.
# That bug failed all 12 tasks of the first ORCHARD array within seconds. submit_all.sh cds to the
# repo before calling sbatch, and the relative --output paths above already assume that.
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
         baseline_diffusionsr_temp \
         si_opt_a_sigma010_encbase_temp)

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
# A requeued segment that starts on a later day must keep the ORIGINAL name, not today's stamp.
WNAME=$(cat "$RUNS/$NAME/wandb_run_name.txt")

# Short-segment training: the time limit (CL_TRAIN_TIME in cluster_env.sh) can be far below a full
# run, because both training loops write ckpt.pth every epoch and resume from it. On SIGUSR1 (sent
# 10 min before the limit, per --signal above) the task requeues itself; the next segment resumes
# from the last completed epoch in the same W&B run. At most one partial epoch is lost per segment.
# Python runs in the background because bash defers traps until a foreground command returns.
requeue_before_timeout() {
  echo "=== $(date) time limit approaching: requeueing job $SLURM_JOB_ID to resume from ckpt.pth ==="
  scontrol requeue "$SLURM_JOB_ID"
  # Let SLURM end the job as REQUEUED rather than racing it to exit with a FAILED status.
  sleep 300
}
trap requeue_before_timeout USR1

python -m diffusionsr.runners.train_srdiff \
  --data_root         "$CL_DATA" \
  --config            "$CFGS/$CFG.yml" \
  --modeltype         "$MTYPE" \
  --gpu               0 \
  --force_run_dir     "$RUNS/$NAME" \
  --wandb_run_name    "$WNAME" \
  --resume_from_wandb "$WNAME" \
  $ENC_ARG &
PY_PID=$!
# `wait` returns python's exit status (set -e then fails the task on a crash), or returns early
# with >128 when the USR1 trap fires, by which point scontrol requeue is already ending this job.
wait $PY_PID

echo "=== $NAME complete ==="
