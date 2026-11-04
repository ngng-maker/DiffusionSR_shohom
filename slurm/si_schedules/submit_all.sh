#!/bin/bash
# SI-schedules study: paired-coupling stochastic interpolants x {linear, trig, vp, bridge} schedules
# x {enc, bicubic} bases, sampled with {euler, heun, rk4, dopri5} x eps in {0, .01, .05, .1, .25, .5, 1}.
# Compared against DiffusionSR (tasks 0 and 11 of the previous slurm/interpolant run, already evaluated
# with the same metric code) in diffusionsr/notebooks/22_si_schedules_results.ipynb.
#
# Usage, from the repo root on a login node:
#   CLUSTER=orchard bash slurm/si_schedules/submit_all.sh          # or CLUSTER=trace (default)
# Subsets:
#   TRAIN_ARMS="0 2"  bash slurm/si_schedules/submit_all.sh         # train idx 0 and 2, and their 32 evals
#   EVAL_ONLY=1       bash slurm/si_schedules/submit_all.sh         # (re)submit evals of finished runs
#
# Run slurm/si_schedules/00_smoke.sh once first: it exercises training, restore-from-artifact and every
# solver on real data in under an hour.
#
# Each training task gets its OWN eval array, dependent on that task alone (afterok:<job>_<idx>), so a
# schedule's 16 evals start as soon as it finishes, and one failed run does not block the others.

set -eo pipefail
REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_schedules/arms.sh"
cd "$CL_REPO"
mkdir -p logs/si_schedules

RESOURCE_ARG="--partition=$CL_PARTITION --gres=$CL_GRES $CL_EXTRA_SBATCH"
EXPORT_ARG="--export=ALL,CLUSTER=$CLUSTER,SI_SCHED_TAG=$SI_SCHED_TAG"
if [ -n "${PY_OVERLAY:-}" ]; then EXPORT_ARG="$EXPORT_ARG,PY_OVERLAY=$PY_OVERLAY"; fi
EVAL_TIME="${CL_EVAL_TIME:-08:00:00}"
TRAIN_ARMS="${TRAIN_ARMS:-$(seq -s ' ' 0 $(( N_TRAIN - 1 )))}"
echo "cluster=$CLUSTER partition=$CL_PARTITION gres=$CL_GRES  run-id tag=$SI_SCHED_TAG  train arms: $TRAIN_ARMS"

if [ -z "${WANDB_ENTITY:-}" ]; then echo "ERROR: WANDB_ENTITY is not set"; exit 1; fi
if [ ! -f "$CL_ENC_DIR/bestmodel_saved.pth" ]; then
  echo "ERROR: shared frozen encoder missing: $CL_ENC_DIR/bestmodel_saved.pth"; exit 1
fi
# The encoder must match the configs' n_steps / fields (a mismatch fails every task at load time).
python3 - "$CL_ENC_DIR/configuration.yml" "$CL_REPO/diffusionsr/configs/si_schedules" "${SI_CONFIGS[@]}" <<'PYCHECK' || exit 1
import os, re, sys
def keys(p):
    txt = open(p).read()
    g = lambda k: (re.search(rf'^{k}:\s*["\']?([^"\'\n#]+)', txt, re.M) or [None, None])[1]
    return g('n_steps'), g('fields')
enc = keys(sys.argv[1])
bad = [c for c in sys.argv[3:] if keys(os.path.join(sys.argv[2], c + '.yml')) != enc]
if bad:
    sys.exit(f'ERROR: encoder (n_steps, fields) = {enc} does not match {bad}')
print(f'Encoder matches all {len(sys.argv) - 3} configs {enc}')
PYCHECK

ARR=$(echo "$TRAIN_ARMS" | tr ' ' ',')
if [ -z "${EVAL_ONLY:-}" ]; then
  TRAIN_JID=$(sbatch --parsable --array="$ARR" --time="$CL_TRAIN_TIME" $RESOURCE_ARG $EXPORT_ARG \
              slurm/si_schedules/01_train_array.sh)
  echo "Training array: $TRAIN_JID (tasks $ARR, segment limit $CL_TRAIN_TIME, self-requeueing)"
fi

for i in $TRAIN_ARMS; do
  lo=$(( i * EVALS_PER_RUN )); hi=$(( lo + EVALS_PER_RUN - 1 ))
  DEP=""
  if [ -n "${TRAIN_JID:-}" ]; then DEP="--dependency=afterok:${TRAIN_JID}_${i}"; fi
  EJID=$(sbatch --parsable --array="$lo-$hi" --time="$EVAL_TIME" $DEP $RESOURCE_ARG $EXPORT_ARG \
         slurm/si_schedules/02_eval_array.sh)
  echo "  evals of [$i] ${SI_CONFIGS[$i]}: job $EJID (tasks $lo-$hi)"
done

cat <<EOF

Submitted. Monitor with: squeue -u \$USER
  logs:     $CL_REPO/logs/si_schedules/
  runs:     $CL_REPO/diffusionsr/runs/si_schedules/<config>/
  results:  $CL_REPO/eval_results/si_schedules/<config>__<solver>_eps<eps>/
  W&B:      project Flow3D_SuperResolution, group si_schedules
            8 checkpoint artifacts   <config>_ckpt                                 (1 version each)
            128 eval artifacts       <config>__<solver>_eps<eps>_eval_predictions  (1 version each)
Results notebook: diffusionsr/notebooks/22_si_schedules_results.ipynb
EOF
