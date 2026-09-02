#!/bin/bash
#SBATCH --job-name=fm_ldm_enc_sdf_ablation_1Sep2026
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=logs/ablation_20260901/model_%a_%j.log
#SBATCH --error=logs/ablation_20260901/model_%a_%j.err
#
# Phase 1: train all 8 ablation models in parallel (job array).
# Requires Phase 0 (00_setup_enc_vae.sh) to have completed first.
#
# Task mapping:
#   0  fm_enc_sdf   — FlowMatching + encoder + temp+sdf
#   1  fm_enc_temp  — FlowMatching + encoder + temp only
#   2  fm_noenc_sdf — FlowMatching + no encoder + temp+sdf
#   3  fm_noenc_temp— FlowMatching + no encoder + temp only
#   4  ldm_enc_sdf  — LDM         + encoder + temp+sdf
#   5  ldm_enc_temp — LDM         + encoder + temp only
#   6  ldm_noenc_sdf— LDM         + no encoder + temp+sdf
#   7  ldm_noenc_temp LDM         + no encoder + temp only
#
# To REQUEUE (after SLURM preemption), add --resume_from_wandb <WANDB_RUN_NAME>:
#   #SBATCH --array=<failed_task_id>
#   Add: --resume_from_wandb "${WANDB_NAMES[$SLURM_ARRAY_TASK_ID]}"

set -eo pipefail

REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
RUNS=$REPO/diffusionsr/runs/ablation_20260901
CFGS=$REPO/diffusionsr/configs/ablation_20260901

cd "$REPO"
source ~/.bashrc
conda activate diffusion_SR

TASK=$SLURM_ARRAY_TASK_ID

# ── Per-task configuration ─────────────────────────────────────────────────────
NAMES=(abl_fm_enc_sdf   abl_fm_enc_temp   abl_fm_noenc_sdf   abl_fm_noenc_temp \
       abl_ldm_enc_sdf  abl_ldm_enc_temp  abl_ldm_noenc_sdf  abl_ldm_noenc_temp)

CFGFILES=(fm_enc_sdf   fm_enc_temp   fm_noenc_sdf   fm_noenc_temp \
          ldm_enc_sdf  ldm_enc_temp  ldm_noenc_sdf  ldm_noenc_temp)

MTYPES=(flow_matching flow_matching flow_matching flow_matching \
        ldm           ldm           ldm           ldm)

WANDB_NAMES=(1_Sep_2026_fm_enc_sdf   1_Sep_2026_fm_enc_temp \
             1_Sep_2026_fm_noenc_sdf 1_Sep_2026_fm_noenc_temp \
             1_Sep_2026_ldm_enc_sdf  1_Sep_2026_ldm_enc_temp \
             1_Sep_2026_ldm_noenc_sdf 1_Sep_2026_ldm_noenc_temp)

NAME=${NAMES[$TASK]}
CFG=${CFGFILES[$TASK]}
MTYPE=${MTYPES[$TASK]}
WNAME=${WANDB_NAMES[$TASK]}

echo "=== Task $TASK: $NAME (modeltype=$MTYPE) ==="

# Encoder dir: enc tasks (0,4 → sdf; 1,5 → temp; 2,3,6,7 → no encoder)
case $TASK in
  0|4) ENC_ARG="--force_enc_dir $RUNS/enc_sdf"  ;;
  1|5) ENC_ARG="--force_enc_dir $RUNS/enc_temp" ;;
  *)   ENC_ARG=""                                ;;
esac

python -m diffusionsr.runners.train_srdiff \
  --config "$CFGS/$CFG.yml" \
  --modeltype "$MTYPE" \
  --gpu 0 \
  --force_run_dir "/scratch/ngng/runs/$NAME" \
  --wandb_run_name "$WNAME" \
  $ENC_ARG

echo "=== $NAME complete ==="
