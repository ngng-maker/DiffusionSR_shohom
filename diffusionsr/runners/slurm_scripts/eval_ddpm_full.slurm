#!/bin/bash
#SBATCH --job-name=eval_ddpm_full
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/slurm_%j.out
#SBATCH --error=/trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/slurm_%j.err

mkdir -p /trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full

cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom

/trace/group/forgelab/ngng/envs/diffusion_SR/bin/python -m diffusionsr.analysis.multifield_eval --mode eval \
  2>&1 | tee /trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/eval_ddpm_full.log

echo "Done."
