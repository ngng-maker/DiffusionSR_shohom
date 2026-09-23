#!/bin/bash
# Master submit script for the FNO-vs-CNN conditioning-encoder study (Track A).
#
# Usage (from repo root on TRACE login node):
#   cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
#   bash slurm/fno_ablation/submit_all.sh
#
# To use a conda env with PhysicsNeMo installed:
#   CONDA_ENV=/trace/group/forgelab/ngng/envs/diffusion_SR_fno bash slurm/fno_ablation/submit_all.sh
#
# What this compares
# ------------------
# Four arms, identical in every respect except which conditioning encoder supplies x_e:
#   rrdb          RRDB CNN          (the paper's baseline)          5,904,321 params
#   fno_pre       FNO-A, bicubic-to-HR then operator at HR          5,989,441 params  (1.014x)
#   fno_spectral  FNO-B, operator at LR + spectral upsampling       5,961,089 params  (1.010x)
#   fno_conv      FNO-C, operator at LR + RRDB's conv upsampling    6,071,873 params  (1.028x)
# plus one calibration arm:
#   fno_pre_pnemo FNO-A again under PhysicsNeMo. Compared against fno_pre (the same variant under
#                 the builtin backend) it tests whether the backend is a confound. Non-blocking in
#                 Phase 1: if physicsnemo is missing, the four primary arms still complete.
#                 NOTE: run count_encoder_params --match 5904321 --backend physicsnemo on TRACE and
#                 update its kwargs BEFORE trusting it, or it differs in size as well as backend.
#
# Phase 1  01_setup_encoders.sh   4 encoders, sequential, one GPU        -> stage-1 comparison
# Phase 2  02_diffusion_array.sh  4 DDPMs, parallel array, after Phase 1 -> stage-2 comparison
# Phase 3  03_eval_array.sh       4 evals, parallel array, after Phase 2 -> predictions.npz per arm
#
# Phase 1 is already a publishable result on its own (encoder-alone L1/PSNR/SSIM). The study plan
# gates Phase 2 on it: if no FNO variant lands within ~10% of RRDB's stage-1 loss, diagnose before
# spending the Phase 2 GPU-hours.
#
# To requeue one failed task:
#   sbatch --array=2 slurm/fno_ablation/02_diffusion_array.sh
#
# NOTE: this runs ONE seed. The study plan calls for 3 seeds per arm before drawing conclusions;
# re-run Phases 1-3 with a different torch seed and a distinct RUNS subdirectory for seeds 2 and 3.

set -eo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
mkdir -p logs/fno_ablation

# Propagate the env choice to every phase so all three run in the same interpreter.
EXPORT_ARG=""
if [ -n "${CONDA_ENV:-}" ]; then
  EXPORT_ARG="--export=ALL,CONDA_ENV=$CONDA_ENV"
  echo "Using conda env: $CONDA_ENV"
fi

echo "=== Phase 1: encoder pretraining (4 primary arms + 1 calibration arm) ==="
# --parsable makes sbatch print just the job ID, so it can be captured for the dependency chain.
ENC_JID=$(sbatch --parsable $EXPORT_ARG slurm/fno_ablation/01_setup_encoders.sh)
echo "  Encoder job ID: $ENC_JID"

echo "=== Phase 2: 5-run diffusion array — depends on $ENC_JID ==="
# afterok means the array only starts if Phase 1 exited cleanly; a failed encoder must not silently
# produce diffusion runs conditioned on a half-trained or missing checkpoint.
DIFF_JID=$(sbatch --parsable \
  --dependency=afterok:"$ENC_JID" \
  $EXPORT_ARG \
  slurm/fno_ablation/02_diffusion_array.sh)
echo "  Diffusion array job ID: $DIFF_JID"

echo "=== Phase 3: 5-run eval array — depends on $DIFF_JID ==="
EVAL_JID=$(sbatch --parsable \
  --dependency=afterok:"$DIFF_JID" \
  $EXPORT_ARG \
  slurm/fno_ablation/03_eval_array.sh)
echo "  Eval array job ID: $EVAL_JID"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor:      squeue -u ngng"
echo "  Logs:         $REPO/logs/fno_ablation/"
echo "  W&B stage 1:  project RRDN_Encoder           (encoder L1 curves — the stage-1 comparison)"
echo "  W&B stage 2:  project Flow3D_SuperResolution (diffusion runs + predictions.npz artifacts)"
echo ""
echo "Task -> arm mapping (Phases 2 and 3):"
echo "  0  rrdb          RRDB CNN baseline"
echo "  1  fno_pre       FNO-A  bicubic-to-HR, operator at HR"
echo "  2  fno_spectral  FNO-B  operator at LR, spectral upsampling (builtin backend)"
echo "  3  fno_conv      FNO-C  operator at LR, RRDB conv upsampling"
echo "  4  fno_pre_pnemo CALIBRATION: FNO-A under PhysicsNeMo (vs task 1 under builtin)"
echo ""
echo "Run slurm/fno_ablation/00_smoke_physicsnemo.sh FIRST if you have not already —"
echo "it verifies the environment and the RRDB regression gate before any training starts."
