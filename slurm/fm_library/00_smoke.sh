#!/bin/bash
#SBATCH --job-name=fmlib_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=trace09
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fm_library/smoke_%j.log
#
# FM-library smoke test: train ONE epoch of fm_paired_enc_temp (or $SMOKE_CFG) into a throwaway run
# (smoke_ prefix, runs/fm_library_smoke/), then sample the first $SMOKE_FRAMES test frames with every
# solver from the raw and EMA weights and print NFE, wall time and the error against the truth.
# Delete the smoke_* W&B run and artifact afterwards. Usage: sbatch slurm/fm_library/00_smoke.sh

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/fm_library/arms.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/fm_library
SMOKE_CFG="${SMOKE_CFG:-fm_paired_enc_temp}"
SMOKE_FRAMES="${SMOKE_FRAMES:-8}"
IDX=""
for i in "${!FM_CONFIGS[@]}"; do
  if [ "${FM_CONFIGS[$i]}" = "$SMOKE_CFG" ]; then IDX=$i; fi
done
if [ -z "$IDX" ]; then echo "ERROR: $SMOKE_CFG is not in arms.sh"; exit 1; fi
export FM_RUNS="$CL_REPO/diffusionsr/runs/fm_library_smoke" FM_RUN_PREFIX=smoke_ SI_EPOCHS=1
export SLURM_ARRAY_TASK_ID=$IDX SLURM_ARRAY_JOB_ID=${SLURM_JOB_ID}
bash slurm/fm_library/01_train_array.sh

RUN_DIR="$FM_RUNS/smoke_$SMOKE_CFG"
CFG_PATH="$CL_REPO/diffusionsr/configs/fm_library/$SMOKE_CFG.yml"
python - "$CFG_PATH" "$RUN_DIR" "$(encoder_dir "$CFG_PATH")" "$CL_DATA" "$SMOKE_FRAMES" <<'PY'
import sys, time, types, yaml, numpy as np, torch
from diffusionsr.runners.train_fm_library import build_fm_model
cfg_path, run_dir, enc_dir, data_root, nf = sys.argv[1:6]
cfg = yaml.safe_load(open(cfg_path))
for weights in ('raw', 'ema'):
    args = types.SimpleNamespace(data_root=data_root, model_dir=run_dir, enc_dir=enc_dir, weights=weights)
    model, test_ds, fields = build_fm_model(args, cfg, 'cuda')
    items = [test_ds[i] for i in range(int(nf))]
    hr, lr, ulr = (torch.stack([it[j] for it in items]).cuda().float() for j in (1, 2, 3))
    x_e = model.compute_x_e(lr, ulr)
    gt = np.stack([test_ds.unscale_data(h.cpu().numpy(), input_type='hr') for h in hr])
    for solver, n in (('euler', 20), ('midpoint', 20), ('rk4', 20), ('dopri5', 20)):
        gen = torch.Generator(device='cuda').manual_seed(0)
        steps = []
        torch.cuda.synchronize(); t0 = time.time()
        x1, nfe = model.sample(ulr, x_e, lr=lr, solver=solver, n_steps=n, generator=gen,
                               callback=lambda i, t, x: steps.append(i))
        torch.cuda.synchronize(); dt = time.time() - t0
        pred = np.stack([test_ds.unscale_data(x.cpu().numpy(), input_type='hr') for x in x1])
        print(f'[{weights}] {solver:8s} n={n} nfe={nfe:4d} callback_pts={len(steps)} {dt:6.2f}s '
              f'MAE={np.abs(pred - gt).mean():8.2f} K  pred[{pred.min():.0f},{pred.max():.0f}] '
              f'gt[{gt.min():.0f},{gt.max():.0f}] finite={np.isfinite(pred).all()}', flush=True)
PY
echo "=== smoke done: delete $RUN_DIR and the W&B run/artifact smoke_${SMOKE_CFG}* ==="
