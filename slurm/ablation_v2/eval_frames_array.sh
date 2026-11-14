#!/bin/bash
#SBATCH --job-name=abl_v2_eval38
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/ablation_v2/eval38_%a_%j.log
#SBATCH --requeue
#
# 38-frame evaluation (scripts/eval_ablation_v2_frames.py) of ablation v2 runs, one run per array task.
# Writes diffusionsr/runs/ablation_v2/_eval_frames/<run>.npz; an existing output is skipped unless --force.
#
# Submit from the repo root on TRACE, with the run names as arguments; the array size must match their count:
#   sbatch --exclude=trace09 --array=0-10 slurm/ablation_v2/eval_frames_array.sh run_a run_b ...

set -eo pipefail
REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
cd "$REPO"
source /trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate /trace/group/forgelab/ngng/envs/diffusion_SR

RUNS=("$@")
NAME=${RUNS[${SLURM_ARRAY_TASK_ID:?run as a SLURM array task}]:?no run name for this task}
echo "=== eval38 $NAME on $(hostname) ==="
python scripts/eval_ablation_v2_frames.py --out_dir diffusionsr/runs/ablation_v2/_eval_frames \
    --n_frames 38 --variants "$NAME"
echo "=== eval38 $NAME complete ==="
