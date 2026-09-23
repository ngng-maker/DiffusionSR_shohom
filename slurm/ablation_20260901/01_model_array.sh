#!/bin/bash
#SBATCH --job-name=abl_models
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=logs/ablation_20260901/model_%a_%j.log
#SBATCH --error=logs/ablation_20260901/model_%a_%j.err
#SBATCH --requeue
#
# Phase 1: train all 8 ablation models in parallel (job array).
# Requires Phase 0a (00a_setup_encoders.sh) and Phase 0b (00b_setup_vaes.sh) to have completed.
#
# Task mapping:
#   0  fm_enc_multifield   — FlowMatching + encoder + temp+sdfliqlabel
#   1  fm_enc_temp         — FlowMatching + encoder + temp only
#   2  fm_noenc_multifield — FlowMatching + no encoder + temp+sdfliqlabel
#   3  fm_noenc_temp       — FlowMatching + no encoder + temp only
#   4  ldm_enc_multifield  — LDM          + encoder + temp+sdfliqlabel
#   5  ldm_enc_temp        — LDM          + encoder + temp only
#   6  ldm_noenc_multifield— LDM          + no encoder + temp+sdfliqlabel
#   7  ldm_noenc_temp      — LDM          + no encoder + temp only
#
# --requeue is set so SLURM auto-restarts preempted tasks.
# W&B run name is generated at first launch (date-stamped) and saved to
# {force_run_dir}/wandb_run_id.txt so restarts resume the same W&B run.
#
# To requeue a single failed task manually (e.g. task 3):
#   sbatch --array=3 slurm/ablation_20260901/01_model_array.sh

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
NAMES=(abl_fm_enc_multifield   abl_fm_enc_temp \
       abl_fm_noenc_multifield  abl_fm_noenc_temp \
       abl_ldm_enc_multifield   abl_ldm_enc_temp \
       abl_ldm_noenc_multifield abl_ldm_noenc_temp)

CFGFILES=(fm_enc_multifield   fm_enc_temp \
          fm_noenc_multifield  fm_noenc_temp \
          ldm_enc_multifield   ldm_enc_temp \
          ldm_noenc_multifield ldm_noenc_temp)

MTYPES=(flow_matching flow_matching flow_matching flow_matching \
        ldm           ldm           ldm           ldm)

NAME=${NAMES[$TASK]}
CFG=${CFGFILES[$TASK]}
MTYPE=${MTYPES[$TASK]}
# W&B display name: date is stamped at first submission; on requeue the
# wandb_run_id.txt mechanism re-attaches to the same run, so the name is unused.
WNAME="${DATE}_${NAME}"

echo "=== Task $TASK: $NAME (modeltype=$MTYPE, wandb=$WNAME) ==="

# Encoder dir: enc tasks use pre-trained encoder from phase 0a
case $TASK in
  0|4) ENC_ARG="--force_enc_dir $RUNS/enc_multifield" ;;
  1|5) ENC_ARG="--force_enc_dir $RUNS/enc_temp"       ;;
  *)   ENC_ARG=""                                      ;;
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
