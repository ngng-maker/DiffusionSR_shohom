#!/bin/bash
#SBATCH --job-name=abl_global
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=logs/ablation_20260901/model_global_%a_%j.log
#SBATCH --error=logs/ablation_20260901/model_global_%a_%j.err
#SBATCH --requeue
#
# Phase 2: train all 8 global_standardize ablation models in parallel (job array).
# Requires Phase 0c (00c_setup_encoders_global.sh) and Phase 0d (00d_setup_vaes_global.sh).
#
# Task mapping:
#   0  fm_enc_multifield_global   — FlowMatching + encoder + temp+sdfliqlabel + global_std
#   1  fm_enc_temp_global         — FlowMatching + encoder + temp only        + global_std
#   2  fm_noenc_multifield_global — FlowMatching + no encoder + temp+sdfliqlabel + global_std
#   3  fm_noenc_temp_global       — FlowMatching + no encoder + temp only        + global_std
#   4  ldm_enc_multifield_global  — LDM          + encoder + temp+sdfliqlabel + global_std
#   5  ldm_enc_temp_global        — LDM          + encoder + temp only        + global_std
#   6  ldm_noenc_multifield_global— LDM          + no encoder + temp+sdfliqlabel + global_std
#   7  ldm_noenc_temp_global      — LDM          + no encoder + temp only        + global_std
#
# --requeue is set so SLURM auto-restarts preempted tasks.
# W&B run name is date-stamped at first launch; wandb_run_id.txt persists it for restarts.
#
# To requeue a single failed task manually (e.g. task 2):
#   sbatch --array=2 slurm/ablation_20260901/02_model_array_global.sh

set -eo pipefail

export WANDB_ENTITY=ngng-
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WANDB_CACHE_DIR"
trap 'rm -rf ~/.local/share/wandb/artifacts/staging/ "$WANDB_CACHE_DIR"' EXIT

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901

cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

TASK=$SLURM_ARRAY_TASK_ID
DATE=$(date +"%d_%b_%Y")

# ── Per-task configuration ─────────────────────────────────────────────────────
NAMES=(abl_fm_enc_multifield_global   abl_fm_enc_temp_global \
       abl_fm_noenc_multifield_global  abl_fm_noenc_temp_global \
       abl_ldm_enc_multifield_global   abl_ldm_enc_temp_global \
       abl_ldm_noenc_multifield_global abl_ldm_noenc_temp_global)

CFGFILES=(fm_enc_multifield_global   fm_enc_temp_global \
          fm_noenc_multifield_global  fm_noenc_temp_global \
          ldm_enc_multifield_global   ldm_enc_temp_global \
          ldm_noenc_multifield_global ldm_noenc_temp_global)

MTYPES=(flow_matching flow_matching flow_matching flow_matching \
        ldm           ldm           ldm           ldm)

NAME=${NAMES[$TASK]}
CFG=${CFGFILES[$TASK]}
MTYPE=${MTYPES[$TASK]}
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $NAME (modeltype=$MTYPE, wandb=$WNAME) ==="

# Encoder dir: enc tasks use the global_std encoder from phase 0c
case $TASK in
  0|4) ENC_ARG="--force_enc_dir $RUNS/enc_multifield_global" ;;
  1|5) ENC_ARG="--force_enc_dir $RUNS/enc_temp_global"       ;;
  *)   ENC_ARG=""                                             ;;
esac

python -m diffusionsr.runners.train_srdiff \
  --config      "$CFGS/$CFG.yml" \
  --modeltype   "$MTYPE" \
  --gpu         0 \
  --force_run_dir "$RUNS/$NAME" \
  --wandb_run_name "$WNAME" \
  --resume_from_wandb "$WNAME" \
  $ENC_ARG

echo "=== $NAME complete ==="
