#!/bin/bash
#SBATCH --job-name=regen_overlays
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/slurm_overlays_%j.out
#SBATCH --error=/trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/slurm_overlays_%j.err

cd /trace/group/forgelab/ngng/multifield/DiffusionSR_shohom

/trace/group/forgelab/ngng/envs/diffusion_SR/bin/python \
  -m diffusionsr.analysis.multifield_eval --mode overlays \
  2>&1 | tee /trace/group/forgelab/ngng/multifield/eval_consistency_ddpm_full/regen_overlays.log

echo "Done."
