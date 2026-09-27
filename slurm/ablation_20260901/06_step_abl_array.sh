#!/bin/bash
#SBATCH --job-name=abl_step
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-23
#SBATCH --output=logs/ablation_20260901/step_abl_%a_%j.log
#SBATCH --error=logs/ablation_20260901/step_abl_%a_%j.err
#SBATCH --requeue
#
# Phase 6: step-count / DDIM-skip ablation.
# Loops over sampling step counts inside each task and uploads one W&B
# artifact per step count, loadable locally without GPU or data.
#
# Artifact naming:
#   FM         → {run_name}_euler_abl_{n_steps}   (n_steps in 5 10 25 50 100)
#   LDM/Diffusion → {run_name}_ddim_abl_{skip}    (skip in 5 10 20 50 100)
#                   {run_name}_ddim_abl_ddpm       (full DDPM, 1000 steps)
#
# Task mapping matches 05_eval_array.sh:
#  FM std      0=enc_multifield  1=enc_temp  2=noenc_multifield  3=noenc_temp
#  FM global   4=enc_multifield  5=enc_temp  6=noenc_multifield  7=noenc_temp
#  LDM std     8=enc_multifield  9=enc_temp 10=noenc_multifield 11=noenc_temp
#  LDM global 12=enc_multifield 13=enc_temp 14=noenc_multifield 15=noenc_temp
#  DDPM std   16=enc_multifield 17=enc_temp 18=noenc_multifield 19=noenc_temp
#  DDPM global 20=enc_multifield 21=enc_temp 22=noenc_multifield 23=noenc_temp
#
# Requeue a single task:
#   sbatch --array=4 slurm/ablation_20260901/06_step_abl_array.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_stepabl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Arrays matching 05_eval_array.sh ─────────────────────────────────────────
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

ENC_DIRS=(
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
  "$RUNS/enc_multifield" "$RUNS/enc_temp"           "" ""
  "$RUNS/enc_multifield_global" "$RUNS/enc_temp_global" "" ""
)

VAE_DIRS=(
  "" "" "" ""
  "" "" "" ""
  "$RUNS/vae_multifield" "$RUNS/vae_temp" "$RUNS/vae_multifield" "$RUNS/vae_temp"
  "$RUNS/vae_multifield_global" "$RUNS/vae_temp_global" "$RUNS/vae_multifield_global" "$RUNS/vae_temp_global"
  "" "" "" ""
  "" "" "" ""
)

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

WNAME_FILE="$RUNS/$NAME/wandb_run_name.txt"
WANDB_RUN_NAME=$([ -f "$WNAME_FILE" ] && cat "$WNAME_FILE" || echo "")

ENC_ARG=""; [ -n "$ENC_DIR" ] && ENC_ARG="--enc_dir $ENC_DIR"
VAE_ARG=""; [ -n "$VAE_DIR" ] && VAE_ARG="--vae_dir $VAE_DIR"

echo "=== Task $TASK: $NAME  model=$MODEL_TYPE  normalize=$NORMALIZE ==="

COMMON_ARGS=(
  --run_name       "$NAME"
  --wandb_run_name "$WANDB_RUN_NAME"
  --model_type     "$MODEL_TYPE"
  --model_dir      "$RUNS/$NAME"
  --config         "$CFG"
  --data_root      "$DATA"
  --field_names    $FIELDS
  --normalize      "$NORMALIZE"
  --batch_size     8
  --max_batches    16
)
[ -n "$ENC_DIR" ] && COMMON_ARGS+=(--enc_dir "$ENC_DIR")
[ -n "$VAE_DIR" ] && COMMON_ARGS+=(--vae_dir "$VAE_DIR")

# ── FM: Euler step count sweep ────────────────────────────────────────────────
if [ "$MODEL_TYPE" = "flow_matching" ]; then
  for NS in 5 10 25 50 100; do
    echo "  → euler n_steps=$NS"
    python -m diffusionsr.analysis.eval_predictions \
      "${COMMON_ARGS[@]}" \
      --sampler euler \
      --fm_n_steps "$NS" \
      --artifact_name "${NAME}_euler_abl_${NS}"
  done

# ── LDM / Diffusion: DDIM skip sweep + full DDPM ─────────────────────────────
else
  for SKIP in 5 10 20 50 100; do
    echo "  → DDIM skip=$SKIP"
    python -m diffusionsr.analysis.eval_predictions \
      "${COMMON_ARGS[@]}" \
      --sampler DDIM \
      --skip    "$SKIP" \
      --artifact_name "${NAME}_ddim_abl_${SKIP}"
  done
  echo "  → DDPM (full 1000 steps)"
  python -m diffusionsr.analysis.eval_predictions \
    "${COMMON_ARGS[@]}" \
    --sampler DDPM \
    --artifact_name "${NAME}_ddim_abl_ddpm"
fi

echo "=== $NAME step ablation complete ==="
