#!/bin/bash
#SBATCH --job-name=sisched_baselines
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/si_schedules/baselines_%j.log
#SBATCH --error=logs/si_schedules/baselines_%j.err
#
# Rebuild the previous run's baselines (notebook 21) on THIS cluster, for notebook 22, WITHOUT retraining:
#
#   1. For tasks 0, 11, 2, 12: download the eval's results.json from its W&B artifact
#      <run>_eval_predictions into eval_results/interpolant/<run>/<run>_results.json. Those copies predate
#      the melt-pool metric fix (MP/KH/area wrong), but their MAE is right, which is what step 2 checks.
#   2. rescore_predictions.py: re-download predictions.npz, recompute EVERY metric with the current
#      compute_metrics, verify MAE reproduces, and rewrite results.json -- the same procedure that produced
#      the corrected numbers on ORCHARD (whose copies were lost).
#   3. evaluate_encoder_only.py: re-run the encoder-alone arm (deterministic; its predictions were never
#      on W&B).
#
#   sbatch slurm/si_schedules/00b_rebuild_baselines.sh            (TRACE)

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_schedules
export TMPDIR="${TMPDIR:-$CL_REPO/.tmp_rescore}"; mkdir -p "$TMPDIR"

OUT="$CL_REPO/eval_results/interpolant"
ARMS="si_baseline_ddpm_matched_temp si_baseline_diffusionsr_temp si_si_opt_a_sigma010_enc_temp si_si_opt_a_sigma010_encbase_temp"

echo "=== 1. seed results.json from the W&B artifacts ==="
python - "$OUT" $ARMS <<'PY'
import json, os, sys, wandb
out, arms = sys.argv[1], sys.argv[2:]
api = wandb.Api(timeout=120)
P = f"{os.environ['WANDB_ENTITY']}/Flow3D_SuperResolution"
for run in arms:
    d = os.path.join(out, run); os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, f'{run}_results.json')
    if os.path.exists(dest):
        print(f'{run}: {dest} exists, kept'); continue
    art = api.artifact(f'{P}/{run}_eval_predictions:latest')
    for name in ('results.json', 'overlay.png'):
        art.get_entry(name).download(root=d)
    os.replace(os.path.join(d, 'results.json'), dest)
    if os.path.exists(os.path.join(d, 'overlay.png')):
        os.replace(os.path.join(d, 'overlay.png'), os.path.join(d, 'overlay_frame70.png'))
    r = json.load(open(dest))
    print(f"{run}: {art.version}  MAE {r['MAE_mean']:.3f}  MP_MAE (pre-fix) {r.get('MP_MAE_mean')}")
PY

echo "=== 2. re-score from the saved predictions ==="
python -m diffusionsr.analysis.rescore_predictions --results_root "$OUT" --arms $ARMS --factor 2

echo "=== 3. encoder alone ==="
python -m diffusionsr.analysis.evaluate_encoder_only --data_root "$CL_DATA" --enc_dir "$CL_ENC_DIR" \
    --out_dir "$OUT/encoder_only"

rm -rf "$CL_REPO/.tmp_rescore" || true
echo "=== baselines rebuilt in $OUT ==="
for a in $ARMS encoder_only; do
  python -c "import json,sys; r=json.load(open(sys.argv[1])); print('%-36s ensMAE %7.3f  MP %6.3f  KH %6.3f  area %7.1f  cal %s' % (sys.argv[2], r['MAE_ensemble_mean'], r['MP_MAE_mean'], r['KH_MAE_mean'], r['area_mae_um2'], r['calibration_2sigma']))" "$OUT/$a/${a}_results.json" "$a"
done
