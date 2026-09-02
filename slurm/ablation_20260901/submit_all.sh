#!/bin/bash
# Master submit script for the ablation_20260901 experiment.
#
# Usage (from repo root on TRACE login node):
#   cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
#   conda activate diffusion_SR
#   bash slurm/ablation_20260901/submit_all.sh
#
# What this does:
#   1. Submits 00_setup_enc_vae.sh  — trains 2 shared encoders + 2 shared VAEs
#   2. Submits 01_model_array.sh     — trains all 8 ablation models in parallel,
#      with a SLURM dependency so it waits for the setup to finish first.
#
# To REQUEUE a failed model task (e.g. task 3):
#   Add --resume_from_wandb <WANDB_NAME> to 01_model_array.sh, then:
#   sbatch --array=3 --dependency=afterok:<setup_jid_if_still_running> \
#          slurm/ablation_20260901/01_model_array.sh
# Task → W&B name mapping is in the header of 01_model_array.sh.

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
mkdir -p logs/ablation_20260901

echo "=== Submitting Phase 0: shared encoder + VAE setup ==="
SETUP_JID=$(sbatch --parsable slurm/ablation_20260901/00_setup_enc_vae.sh)
echo "  Setup job ID: $SETUP_JID"

echo "=== Submitting Phase 1: 8-run model array (depends on $SETUP_JID) ==="
ARRAY_JID=$(sbatch --parsable \
  --dependency=afterok:"$SETUP_JID" \
  slurm/ablation_20260901/01_model_array.sh)
echo "  Model array job ID: $ARRAY_JID"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor: squeue -u ngng"
echo "  W&B group: ablation_20260901  (project: Flow3D_SuperResolution)"
echo "  Logs: $REPO/logs/ablation_20260901/"
echo ""
echo "W&B run names:"
echo "  1_Sep_2026_encoder_sdf      (setup)"
echo "  1_Sep_2026_encoder_temp     (setup)"
echo "  1_Sep_2026_vae_sdf          (setup)"
echo "  1_Sep_2026_vae_temp         (setup)"
echo "  1_Sep_2026_fm_enc_sdf       (task 0)"
echo "  1_Sep_2026_fm_enc_temp      (task 1)"
echo "  1_Sep_2026_fm_noenc_sdf     (task 2)"
echo "  1_Sep_2026_fm_noenc_temp    (task 3)"
echo "  1_Sep_2026_ldm_enc_sdf      (task 4)"
echo "  1_Sep_2026_ldm_enc_temp     (task 5)"
echo "  1_Sep_2026_ldm_noenc_sdf    (task 6)"
echo "  1_Sep_2026_ldm_noenc_temp   (task 7)"
