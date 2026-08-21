#!/bin/bash
#SBATCH --job-name=cs_fm_temp
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/trace/group/forgelab/ngng/multifield/logs/cs_fm_temp_%j.log

source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR
cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom/diffusionsr
export WANDB_ENTITY="${WANDB_ENTITY:-ngng-}"
export PYTHONPATH=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom:$PYTHONPATH
export WANDB_DIR=/trace/group/forgelab/ngng/wandb
export WANDB_DATA_DIR=/trace/group/forgelab/ngng/wandb_data

python runners/train_srdiff.py   --config configs/multifield/flowmatch_temp_n3.yml   --gpu 0 --modeltype flow_matching   --force_enc_dir runs/direct/encoder/cs_temp_n3   --force_run_dir runs/direct/flowmatchingimplicitencoded/cs_temp_n3
