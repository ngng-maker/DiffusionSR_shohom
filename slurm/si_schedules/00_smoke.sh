#!/bin/bash
#SBATCH --job-name=sisched_smoke
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0-01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/si_schedules/smoke_%j.log
#SBATCH --error=logs/si_schedules/smoke_%j.err
#
# REAL-DATA smoke test of the whole pipeline, BEFORE the 8 + 128 task arrays. ~30-60 min on one GPU.
#   1. trains si_sched_bridge_enc_temp for 2 epochs (bridge: the schedule with the most singular
#      coefficients), into a throwaway run dir, with its own W&B run and single artifact;
#   2. deletes the local run dir and trains to epoch 3, which must RESTORE from the W&B artifact and
#      resume the same W&B run (the cross-cluster / scratch-loss path);
#   3. evaluates it on 16 test frames with every solver at eps = 0 and the two SDE solvers at eps = 0.5.
# Everything it creates is prefixed smoke_ (runs/si_schedules/smoke_*, eval_results/si_schedules/smoke,
# W&B runs and artifacts smoke_*); delete them afterwards if you like.
#
#   CLUSTER=orchard sbatch --partition=preempt --gres=gpu:1 --account=oogoke --qos=preempt_qos \
#       --export=ALL,CLUSTER=orchard slurm/si_schedules/00_smoke.sh
#   (on TRACE:  sbatch slurm/si_schedules/00_smoke.sh)

set -eo pipefail
REPO_GUESS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_schedules/arms.sh"
cd "$CL_REPO"
cluster_activate_env
mkdir -p logs/si_schedules

CFG=si_sched_bridge_enc_temp
SMOKE=smoke_${CFG}_$(date +%s)
RUN_DIR="$CL_REPO/diffusionsr/runs/si_schedules/$SMOKE"
CFGS="$CL_REPO/diffusionsr/configs/si_schedules"
train() {
  python -m diffusionsr.runners.train_srdiff --data_root "$CL_DATA" --config "$CFGS/$CFG.yml" \
    --modeltype coupled --gpu 0 --force_enc_dir "$CL_ENC_DIR" --force_run_dir "$RUN_DIR" \
    --wandb_run_name "$SMOKE" --wandb_run_id "$(echo "$SMOKE" | tr '_' '-')" \
    --ckpt_artifact "${SMOKE}_ckpt" --epochs "$1"
}
# ckpt_every=5 would not upload within 2 epochs; the final sync of each leg uploads regardless.
echo "=== 1. train 2 epochs ==="; train 2
echo "=== 2. drop the local run dir, resume from the W&B artifact to epoch 3 ==="
rm -rf "$RUN_DIR"; train 3
python - "$RUN_DIR" <<'PY'
import sys, numpy as np, torch
d = sys.argv[1]
n = len(np.loadtxt(f'{d}/loss_epoch.txt').reshape(-1))
e = torch.load(f'{d}/ckpt.pth', map_location='cpu')[2]
assert n == 3 and e == 2, f'resume from artifact failed: {n} loss rows, last epoch {e}'
print('[smoke] restored from the W&B artifact and resumed: 3 epochs of history, last epoch 2')
PY
echo "=== 3. evaluate on 16 frames ==="
for spec in "euler 0.0" "heun 0.0" "rk4 0.0" "dopri5 0.0" "euler 0.5" "heun 0.5"; do
  set -- $spec
  python -m diffusionsr.analysis.evaluate_si_schedules --train_name "$SMOKE" --config "$CFGS/$CFG.yml" \
    --model_dir "$RUN_DIR" --enc_dir "$CL_ENC_DIR" --data_root "$CL_DATA" --solver "$1" --eps "$2" \
    --n_steps 20 --k_samples 4 --max_frames 16 --out_root "$CL_REPO/eval_results/si_schedules/smoke" \
    --run_tag "$SI_SCHED_TAG"
done
echo "=== SMOKE PASSED: check W&B for exactly one version of ${SMOKE}_ckpt and of each smoke eval artifact ==="
