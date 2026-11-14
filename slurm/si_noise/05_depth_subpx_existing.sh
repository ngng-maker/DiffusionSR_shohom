#!/bin/bash
#SBATCH -J depth_subpx
#SBATCH -p batch
#SBATCH --exclude=trace09
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/si_noise/depth_subpx_%j.log
#
# Back-fill sub-pixel depth metrics (analysis/add_depth_metrics.py -> depth_subpx.json) for every saved
# evaluation: the SI-schedules and SI-noise evals and the cached baselines. CPU only; no re-sampling.
#
# Usage, from the repo root on TRACE:  sbatch slurm/si_noise/05_depth_subpx_existing.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_noise

C=eval_results/_predictions_cache
ls -d "$C/si_baseline_ddpm_matched_temp" "$C/si_baseline_diffusionsr_temp" \
      "$C/si_si_opt_a_sigma010_enc_temp" "$C/si_si_opt_a_sigma010_encbase_temp" \
      eval_results/si_schedules/si_sched_*__*/ eval_results/si_noise/si_*__*/ \
  | xargs -P 8 -n 4 python -m diffusionsr.analysis.add_depth_metrics
echo "=== depth_subpx.json written ==="
