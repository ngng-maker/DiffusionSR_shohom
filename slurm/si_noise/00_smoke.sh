#!/bin/bash
#SBATCH --job-name=sinoise_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/si_noise/smoke_%j.log
#SBATCH --error=logs/si_noise/smoke_%j.err
#
# REAL-DATA smoke test of what is NEW in the SI-noise study, before the 5 + 73 task arrays (~30-45 min):
#   1. trains si_noise_g2_enc_temp (largest latent gamma) for 2 epochs into a throwaway run dir;
#   2. evaluates it on 16 test frames: euler and heun at eps = 5, rk4 at eps = 0;
#   3. Phase 0 path: the EXISTING si_sched_linear_enc_temp model on 16 frames, heun and euler at eps = 20;
#   4. the skill-score backfill on one existing baseline (timing for 03_skill_existing.sh).
# W&B runs OFFLINE here, so the smoke leaves no runs or artifacts online (the single-artifact upload and
# restore paths are unchanged since the SI-schedules smoke, job 707581, verified them). Everything local is
# prefixed smoke_ (runs/si_noise/smoke_*, eval_results/si_noise/smoke/).
#
#   sbatch --exclude=trace09 slurm/si_noise/00_smoke.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_noise/arms.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_noise
export WANDB_MODE=offline

CFG=si_noise_g2_enc_temp
SMOKE=smoke_${CFG}_$(date +%s)
RUN_DIR="$CL_REPO/diffusionsr/runs/si_noise/$SMOKE"
CFGS="$CL_REPO/diffusionsr/configs/si_noise"
OUT="$CL_REPO/eval_results/si_noise/smoke"
echo "=== 1. train $CFG for 2 epochs ==="
python -m diffusionsr.runners.train_srdiff --data_root "$CL_DATA" --config "$CFGS/$CFG.yml" \
  --modeltype coupled --gpu 0 --force_enc_dir "$CL_ENC_DIR" --force_run_dir "$RUN_DIR" \
  --wandb_run_name "$SMOKE" --wandb_run_id "$(echo "$SMOKE" | tr '_' '-')" \
  --ckpt_artifact "${SMOKE}_ckpt" --epochs 2
grep -q 'linear_gamma' "$RUN_DIR/configuration.yml" || { echo "configuration.yml lost si_schedule"; exit 1; }

ev() {  # name cfg_dir run_dir solver eps
  python -m diffusionsr.analysis.evaluate_si_schedules --train_name "$1" --config "$2/$1.yml" \
    --model_dir "$3" --enc_dir "$CL_ENC_DIR" --data_root "$CL_DATA" --solver "$4" --eps "$5" \
    --n_steps "$SI_N_STEPS" --k_samples "$SI_K_SAMPLES" --max_frames 16 --out_root "$OUT" --no_upload
}
echo "=== 2. evaluate the new arm on 16 frames ==="
cp "$CFGS/$CFG.yml" "$CFGS/$SMOKE.yml"; trap 'rm -f "$CFGS/$SMOKE.yml"' EXIT
for spec in "euler 5.0" "heun 5.0" "rk4 0.0"; do set -- $spec; ev "$SMOKE" "$CFGS" "$RUN_DIR" "$1" "$2"; done
echo "=== 3. Phase 0: existing linear model at eps = 20 on 16 frames ==="
for sv in heun euler; do
  ev "$P0_CFG" "$CL_REPO/diffusionsr/configs/si_schedules" "$CL_REPO/diffusionsr/runs/si_schedules/$P0_CFG" "$sv" 20.0
done
python - "$OUT" <<'PY'
import glob, json, math, sys
rows = sorted(glob.glob(f'{sys.argv[1]}/*/*_results.json'))
assert len(rows) == 5, rows
for f in rows:
    r = json.load(open(f))
    for k in ('MAE_ensemble_mean', 'calibration_2sigma', 'crps_all_K', 'crps_melt_K', 'ssr_all', 'ssr_melt'):
        assert math.isfinite(r[k]), (f, k, r[k])
    assert sum(r['rank_hist_all']) == r['N'] * 3600, f
    print(f"[smoke] {r['eval_name']:58s} ensMAE {r['MAE_ensemble_mean']:6.2f}  cal {r['calibration_2sigma']:.3f}  "
          f"CRPS melt {r['crps_melt_K']:6.1f}  SSR melt {r['ssr_melt']:.3f}  nfe {r['nfe_per_sample']:.0f}")
PY
echo "=== 4. skill-score backfill timing on one baseline ==="
time python -m diffusionsr.analysis.add_skill_metrics --force eval_results/_predictions_cache/si_baseline_ddpm_matched_temp
echo "=== SMOKE PASSED ==="
