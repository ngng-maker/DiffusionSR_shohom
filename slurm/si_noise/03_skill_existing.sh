#!/bin/bash
#SBATCH --job-name=sinoise_skill
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/si_noise/skill_existing_%j.log
#SBATCH --error=logs/si_noise/skill_existing_%j.err
#SBATCH --requeue
#
# Adds the SI-noise study's ensemble-skill scores (fair CRPS, spread-skill ratio, rank histogram) to the
# evaluations that predate them, so notebook 24 compares like with like:
#   * the 128 SI-schedules evals            eval_results/si_schedules/*/skill.json
#   * the 4 K = 10 notebook-21 baselines     eval_results/_predictions_cache/<run>/skill.json
#   * the encoder alone (K = 1, CRPS = MAE)  eval_results/interpolant/encoder_only/skill.json
# The original results.json files are not touched. CPU work; the GPU is only what the partition requires.
# Idempotent: finished directories are skipped, so a requeue continues where it stopped.

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_noise

C=eval_results/_predictions_cache
python -m diffusionsr.analysis.add_skill_metrics \
  "$C/si_baseline_ddpm_matched_temp" "$C/si_baseline_diffusionsr_temp" \
  "$C/si_si_opt_a_sigma010_enc_temp" "$C/si_si_opt_a_sigma010_encbase_temp" \
  eval_results/interpolant/encoder_only \
  eval_results/si_schedules/si_sched_*__*/
echo "=== skill scores written ==="
