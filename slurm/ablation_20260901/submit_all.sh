#!/bin/bash
# Master submit script for the ablation_20260901 experiment.
#
# Usage (from repo root on TRACE login node):
#   cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
#   conda activate diffusion_SR
#   bash slurm/ablation_20260901/submit_all.sh
#
# Three phases (each capped at 2-00:00:00 to fit QOS limit):
#   Phase 0a: 00a_setup_encoders.sh  — train enc_sdf, enc_temp  (sequential)
#   Phase 0b: 00b_setup_vaes.sh      — train vae_sdf, vae_temp  (sequential, after 0a)
#   Phase 1:  01_model_array.sh      — train 8 ablation models  (parallel array, after 0b)
#
# To REQUEUE a failed model task (e.g. task 3):
#   Add --resume_from_wandb <WANDB_NAME> to 01_model_array.sh, then:
#   sbatch --array=3 slurm/ablation_20260901/01_model_array.sh
# Task → W&B name mapping is in the header of 01_model_array.sh.

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
mkdir -p logs/ablation_20260901

echo "=== Phase 0a: encoder pretraining ==="
ENC_JID=$(sbatch --parsable slurm/ablation_20260901/00a_setup_encoders.sh)
echo "  Encoder job ID: $ENC_JID"

echo "=== Phase 0b: VAE pretraining (depends on $ENC_JID) ==="
VAE_JID=$(sbatch --parsable \
  --dependency=afterok:"$ENC_JID" \
  slurm/ablation_20260901/00b_setup_vaes.sh)
echo "  VAE job ID: $VAE_JID"

echo "=== Phase 1: 8-run model array (depends on $VAE_JID) ==="
ARRAY_JID=$(sbatch --parsable \
  --dependency=afterok:"$VAE_JID" \
  slurm/ablation_20260901/01_model_array.sh)
echo "  Model array job ID: $ARRAY_JID"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor: squeue -u ngng"
echo "  W&B group: ablation_20260901  (project: Flow3D_SuperResolution)"
echo "  Logs: $REPO/logs/ablation_20260901/"
echo ""
echo "W&B run names:"
echo "  1_Sep_2026_encoder_sdf      (phase 0a)"
echo "  1_Sep_2026_encoder_temp     (phase 0a)"
echo "  1_Sep_2026_vae_sdf          (phase 0b)"
echo "  1_Sep_2026_vae_temp         (phase 0b)"
echo "  1_Sep_2026_fm_enc_sdf       (task 0)"
echo "  1_Sep_2026_fm_enc_temp      (task 1)"
echo "  1_Sep_2026_fm_noenc_sdf     (task 2)"
echo "  1_Sep_2026_fm_noenc_temp    (task 3)"
echo "  1_Sep_2026_ldm_enc_sdf      (task 4)"
echo "  1_Sep_2026_ldm_enc_temp     (task 5)"
echo "  1_Sep_2026_ldm_noenc_sdf    (task 6)"
echo "  1_Sep_2026_ldm_noenc_temp   (task 7)"
