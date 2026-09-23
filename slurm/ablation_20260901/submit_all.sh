#!/bin/bash
# Master submit script for the ablation_20260901 experiment.
#
# Usage (from repo root on TRACE login node):
#   cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
#   conda activate diffusion_SR
#   bash slurm/ablation_20260901/submit_all.sh
#
# Ablation matrix: FM vs LDM  x  enc vs noenc  x  multifield vs temp  x  standardize vs global_standardize
# = 16 model runs (tasks 0-7 in Phase 1 = standardize; tasks 0-7 in Phase 2 = global_standardize)
#
# Phase 0a: 00a_setup_encoders.sh       — train enc_multifield, enc_temp        (standardize, sequential)
# Phase 0b: 00b_setup_vaes.sh           — train vae_multifield, vae_temp        (standardize, after 0a)
# Phase 1:  01_model_array.sh           — train 8 standardize models            (parallel array, after 0b)
#
# Phase 0c: 00c_setup_encoders_global.sh — train enc_multifield_global, enc_temp_global (global_std)
# Phase 0d: 00d_setup_vaes_global.sh     — train vae_multifield_global, vae_temp_global (global_std, after 0c)
# Phase 2:  02_model_array_global.sh     — train 8 global_std models            (parallel array, after 0d)
#
# NOTE: 0a/0b chain and 0c/0d chain are fully independent — submit them simultaneously.
#       0b depends on 0a (same GPU; avoid contention). 0d depends on 0c. Both chains
#       share the same GPU node sequentially; submit in the order shown here.
#
# To REQUEUE a single failed model task (e.g. task 3, Phase 1):
#   sbatch --array=3 slurm/ablation_20260901/01_model_array.sh
# To REQUEUE a single failed global task (e.g. task 3, Phase 2):
#   sbatch --array=3 slurm/ablation_20260901/02_model_array_global.sh

set -euo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
mkdir -p logs/ablation_20260901

# ── standardize chain ──────────────────────────────────────────────────────────
echo "=== Phase 0a: encoder pretraining [standardize] (enc_multifield, enc_temp) ==="
ENC_JID=$(sbatch --parsable slurm/ablation_20260901/00a_setup_encoders.sh)
echo "  Encoder job ID: $ENC_JID"

echo "=== Phase 0b: VAE pretraining [standardize] (vae_multifield, vae_temp) — depends on $ENC_JID ==="
VAE_JID=$(sbatch --parsable \
  --dependency=afterok:"$ENC_JID" \
  slurm/ablation_20260901/00b_setup_vaes.sh)
echo "  VAE job ID: $VAE_JID"

echo "=== Phase 1: 8-run standardize model array — depends on $VAE_JID ==="
ARRAY_JID=$(sbatch --parsable \
  --dependency=afterok:"$VAE_JID" \
  slurm/ablation_20260901/01_model_array.sh)
echo "  Model array job ID: $ARRAY_JID"

# ── global_standardize chain (independent of 0a/0b/Phase1) ───────────────────
echo "=== Phase 0c: encoder pretraining [global_std] (enc_multifield_global, enc_temp_global) ==="
ENC_G_JID=$(sbatch --parsable slurm/ablation_20260901/00c_setup_encoders_global.sh)
echo "  Encoder-global job ID: $ENC_G_JID"

echo "=== Phase 0d: VAE pretraining [global_std] (vae_multifield_global, vae_temp_global) — depends on $ENC_G_JID ==="
VAE_G_JID=$(sbatch --parsable \
  --dependency=afterok:"$ENC_G_JID" \
  slurm/ablation_20260901/00d_setup_vaes_global.sh)
echo "  VAE-global job ID: $VAE_G_JID"

echo "=== Phase 2: 8-run global_std model array — depends on $VAE_G_JID ==="
ARRAY_G_JID=$(sbatch --parsable \
  --dependency=afterok:"$VAE_G_JID" \
  slurm/ablation_20260901/02_model_array_global.sh)
echo "  Model-global array job ID: $ARRAY_G_JID"

echo ""
echo "=== All jobs submitted ==="
echo "  Monitor:    squeue -u ngng"
echo "  W&B group:  ablation_20260901  (project: Flow3D_SuperResolution)"
echo "  Logs:       $REPO/logs/ablation_20260901/"
echo ""
echo "W&B run names are date-stamped at first launch (see {run_dir}/wandb_run_id.txt)."
echo ""
echo "Phase 1 task → run name mapping (01_model_array.sh) [standardize]:"
echo "  0  abl_fm_enc_multifield"
echo "  1  abl_fm_enc_temp"
echo "  2  abl_fm_noenc_multifield"
echo "  3  abl_fm_noenc_temp"
echo "  4  abl_ldm_enc_multifield"
echo "  5  abl_ldm_enc_temp"
echo "  6  abl_ldm_noenc_multifield"
echo "  7  abl_ldm_noenc_temp"
echo ""
echo "Phase 2 task → run name mapping (02_model_array_global.sh) [global_standardize]:"
echo "  0  abl_fm_enc_multifield_global"
echo "  1  abl_fm_enc_temp_global"
echo "  2  abl_fm_noenc_multifield_global"
echo "  3  abl_fm_noenc_temp_global"
echo "  4  abl_ldm_enc_multifield_global"
echo "  5  abl_ldm_enc_temp_global"
echo "  6  abl_ldm_noenc_multifield_global"
echo "  7  abl_ldm_noenc_temp_global"
