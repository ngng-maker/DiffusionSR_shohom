#!/bin/bash
#SBATCH --job-name=diffsr_3f_n5
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/trace/group/forgelab/ngng/multifield/logs/3field_n5_%j.log

source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom/diffusionsr

export WANDB_ENTITY=ngng-
export PYTHONPATH=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom:$PYTHONPATH
export WANDB_DIR=/trace/group/forgelab/ngng/wandb
export WANDB_DATA_DIR=/trace/group/forgelab/ngng/wandb_data

python runners/train_srdiff.py \
  --config ss316l_3field_100step.yml \
  --gpu 0 \
  --modeltype diffusion \
  --force_enc_dir runs/direct/encoder/2026_07_10_19_42_41/standardize/n_steps_5 \
  --force_run_dir runs/direct/diffusionimplicitencoded/2026_07_10_19_42_41/standardize/n_steps_5
