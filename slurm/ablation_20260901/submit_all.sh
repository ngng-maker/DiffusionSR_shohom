#!/bin/bash
# Master submit script for the ablation_20260901 experiment.
#
# Usage (from repo root on TRACE login node):
#   cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
#   conda activate diffusion_SR
#   bash slurm/ablation_20260901/submit_all.sh
#
# Three phases (each capped at 2-00:00:00 to fit QOS limit):
#   Phase 0a: 00a_setup_encoders.sh  — train enc_multifield, enc_temp  (sequential)
#   Phase 0b: 00b_setup_vaes.sh      — train vae_multifield, vae_temp  (sequential, after 0a)
#   Phase 1:  01_model_array.sh      — train 8 ablation models          (parallel array, after 0b)
#
# NOTE: phase 0a and 0b are independent (encoder and VAE share no dependencies).
# They are chained here only to avoid simultaneous GPU contention; submit them
# in parallel with two separate sbatch calls if cluster capacity allows.
#
# To REQUEUE a single failed model task (e.g. task 3, fm_noenc_temp):
#   sbatch --array=3 slurm/ablation_20260901/01_model_array.sh
# Task → name mapping is in the header of 01_model_array.sh.

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
mkdir -p logs/ablation_20260901

echo "=== Phase 0a: encoder pretraining (enc_multifield, enc_temp) ==="
ENC_JID=$(sbatch --parsable slurm/ablation_20260901/00a_setup_encoders.sh)
echo "  Encoder job ID: $ENC_JID"

echo "=== Phase 0b: VAE pretraining (vae_multifield, vae_temp) — depends on $ENC_JID ==="
VAE_JID=$(sbatch --parsable \
  --dependency=afterok:"$ENC_JID" \
  slurm/ablation_20260901/00b_setup_vaes.sh)
echo "  VAE job ID: $VAE_JID"

echo "=== Phase 1: 8-run model array — depends on $VAE_JID ==="
ARRAY_JID=$(sbatch --parsable \
  --dependency=afterok:"$VAE_JID" \
  slurm/ablation_20260901/01_model_array.sh)
echo "  Model array job ID: $ARRAY_JID"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor:    squeue -u ngng"
echo "  W&B group:  ablation_20260901  (project: Flow3D_SuperResolution)"
echo "  Logs:       $REPO/logs/ablation_20260901/"
echo ""
echo "W&B run names are date-stamped at first launch (see {run_dir}/wandb_run_id.txt)."
echo "Task → run name mapping (01_model_array.sh):"
echo "  0  abl_fm_enc_multifield"
echo "  1  abl_fm_enc_temp"
echo "  2  abl_fm_noenc_multifield"
echo "  3  abl_fm_noenc_temp"
echo "  4  abl_ldm_enc_multifield"
echo "  5  abl_ldm_enc_temp"
echo "  6  abl_ldm_noenc_multifield"
echo "  7  abl_ldm_noenc_temp"
