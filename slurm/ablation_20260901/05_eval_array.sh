#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-23
#SBATCH --output=logs/ablation_20260901/eval_%a_%j.log
#SBATCH --error=logs/ablation_20260901/eval_%a_%j.err
#SBATCH --requeue
#
# Phase 5: sample the full test set for all 24 primary ablation models and
# upload predictions as W&B artifacts (loaded by notebooks locally).
#
# Task mapping:
#  FM standardize      0=enc_multifield  1=enc_temp  2=noenc_multifield  3=noenc_temp
#  FM global           4=enc_multifield  5=enc_temp  6=noenc_multifield  7=noenc_temp
#  LDM standardize     8=enc_multifield  9=enc_temp 10=noenc_multifield 11=noenc_temp
#  LDM global         12=enc_multifield 13=enc_temp 14=noenc_multifield 15=noenc_temp
#  DDPM standardize   16=enc_multifield 17=enc_temp 18=noenc_multifield 19=noenc_temp
#  DDPM global        20=enc_multifield 21=enc_temp 22=noenc_multifield 23=noenc_temp
#
# To requeue a single task (e.g. task 4):
#   sbatch --array=4 slurm/ablation_20260901/05_eval_array.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901
DATA=/trace/group/forgelab/ngng/multifield/data_fields

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

TASK=$SLURM_ARRAY_TASK_ID

# ── Run names (index = task ID) ───────────────────────────────────────────────
NAMES=(
  abl_fm_enc_multifield        abl_fm_enc_temp
  abl_fm_noenc_multifield      abl_fm_noenc_temp
  abl_fm_enc_multifield_global abl_fm_enc_temp_global
  abl_fm_noenc_multifield_global abl_fm_noenc_temp_global
  abl_ldm_enc_multifield       abl_ldm_enc_temp
  abl_ldm_noenc_multifield     abl_ldm_noenc_temp
  abl_ldm_enc_multifield_global abl_ldm_enc_temp_global
  abl_ldm_noenc_multifield_global abl_ldm_noenc_temp_global
  abl_diffusion_enc_multifield  abl_diffusion_enc_temp
  abl_diffusion_noenc_multifield abl_diffusion_noenc_temp
  abl_diffusion_enc_multifield_global abl_diffusion_enc_temp_global
  abl_diffusion_noenc_multifield_global abl_diffusion_noenc_temp_global
)

MODEL_TYPES=(
  flow_matching flow_matching flow_matching flow_matching
  flow_matching flow_matching flow_matching flow_matching
  ldm           ldm           ldm           ldm
  ldm           ldm           ldm           ldm
  diffusion     diffusion     diffusion     diffusion
  diffusion     diffusion     diffusion     diffusion
)

# "temperature sdfliqlabel" or "temperature"
FIELD_ARGS=(
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
  "temperature sdfliqlabel" "temperature" "temperature sdfliqlabel" "temperature"
)

NORMALIZES=(
  standardize standardize standardize standardize
  global_standardize global_standardize global_standardize global_standardize
  standardize standardize standardize standardize
  global_standardize global_standardize global_standardize global_standardize
  standardize standardize standardize standardize
  global_standardize global_standardize global_standardize global_standardize
)

# Encoder dir — empty string = no encoder
ENC_DIRS=(
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
)

# VAE dir — empty string = not LDM
VAE_DIRS=(
  "" "" "" ""
  "" "" "" ""
  "$RUNS/vae_multifield" "$RUNS/vae_temp" "$RUNS/vae_multifield" "$RUNS/vae_temp"
  "$RUNS/vae_multifield_global" "$RUNS/vae_temp_global" "$RUNS/vae_multifield_global" "$RUNS/vae_temp_global"
  "" "" "" ""
  "" "" "" ""
)

# Config files (used to reconstruct encoder architecture exactly)
CFGFILES=(
  "$CFGS/fm_enc_multifield.yml"        "$CFGS/fm_enc_temp.yml"
  "$CFGS/fm_noenc_multifield.yml"      "$CFGS/fm_noenc_temp.yml"
  "$CFGS/fm_enc_multifield_global.yml" "$CFGS/fm_enc_temp_global.yml"
  "$CFGS/fm_noenc_multifield_global.yml" "$CFGS/fm_noenc_temp_global.yml"
  "$CFGS/ldm_enc_multifield.yml"       "$CFGS/ldm_enc_temp.yml"
  "$CFGS/ldm_noenc_multifield.yml"     "$CFGS/ldm_noenc_temp.yml"
  "$CFGS/ldm_enc_multifield_global.yml" "$CFGS/ldm_enc_temp_global.yml"
  "$CFGS/ldm_noenc_multifield_global.yml" "$CFGS/ldm_noenc_temp_global.yml"
  "$CFGS/diffusion_enc_multifield.yml" "$CFGS/diffusion_enc_temp.yml"
  "$CFGS/diffusion_noenc_multifield.yml" "$CFGS/diffusion_noenc_temp.yml"
  "$CFGS/diffusion_enc_multifield_global.yml" "$CFGS/diffusion_enc_temp_global.yml"
  "$CFGS/diffusion_noenc_multifield_global.yml" "$CFGS/diffusion_noenc_temp_global.yml"
)

NAME=${NAMES[$TASK]}
MODEL_TYPE=${MODEL_TYPES[$TASK]}
FIELDS=${FIELD_ARGS[$TASK]}
NORMALIZE=${NORMALIZES[$TASK]}
ENC_DIR=${ENC_DIRS[$TASK]}
VAE_DIR=${VAE_DIRS[$TASK]}
CFG=${CFGFILES[$TASK]}

# Read W&B display name saved at training time (if present)
WNAME_FILE="$RUNS/$NAME/wandb_run_name.txt"
if [ -f "$WNAME_FILE" ]; then
  WANDB_RUN_NAME=$(cat "$WNAME_FILE")
else
  WANDB_RUN_NAME=""
fi

echo "=== Task $TASK: $NAME ==="
echo "    model=$MODEL_TYPE  normalize=$NORMALIZE  fields=$FIELDS"
echo "    enc=$ENC_DIR  vae=$VAE_DIR  wandb_run_name=$WANDB_RUN_NAME"

# Build optional args
ENC_ARG=""; [ -n "$ENC_DIR" ] && ENC_ARG="--enc_dir $ENC_DIR"
VAE_ARG=""; [ -n "$VAE_DIR" ] && VAE_ARG="--vae_dir $VAE_DIR"

# shellcheck disable=SC2086
python -m diffusionsr.analysis.eval_predictions \
  --run_name        "$NAME" \
  --wandb_run_name  "$WANDB_RUN_NAME" \
  --model_type      "$MODEL_TYPE" \
  --model_dir       "$RUNS/$NAME" \
  --config          "$CFG" \
  --data_root       "$DATA" \
  --field_names     $FIELDS \
  --normalize       "$NORMALIZE" \
  --batch_size      8 \
  $ENC_ARG \
  $VAE_ARG

echo "=== $NAME eval complete ==="
