#!/bin/bash
# SI-noise study (follow-up to slurm/si_schedules): linear path + latent gamma_t = c t(1-t), or a larger base
# noise sigma, so that eps has real noise to scale. 5 training runs (Phase 1) and 73 evals (65 Phase 1 +
# 8 Phase 0 on the EXISTING si_sched_linear_enc_temp model), plus the skill-score backfill of the
# SI-schedules evals and baselines. Results: diffusionsr/notebooks/24_si_noise_results.ipynb.
#
# Usage, from the repo root on a login node:
#   CLUSTER=trace CL_EXTRA_SBATCH="--exclude=trace09" bash slurm/si_noise/submit_all.sh
# Subsets:
#   TRAIN_ARMS="0 2"  bash slurm/si_noise/submit_all.sh    # train idx 0 and 2 and their 26 evals
#   EVAL_ONLY=1       bash slurm/si_noise/submit_all.sh    # (re)submit evals of finished runs
#   NO_PHASE0=1 / NO_SKILL=1                               # skip the Phase 0 evals / the backfill job
#
# Run slurm/si_noise/00_smoke.sh once first.
#
# Each training task gets its OWN eval array, dependent on that task alone (afterok:<job>_<idx>).

set -eo pipefail
REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_GUESS/slurm/cluster_env.sh"
source "$REPO_GUESS/slurm/si_noise/arms.sh"
cd "$CL_REPO"
mkdir -p logs/si_noise

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
python3 - "$CL_ENC_DIR/configuration.yml" "$CL_REPO/diffusionsr/configs/si_noise" "${NOISE_CONFIGS[@]}" <<'PYCHECK' || exit 1
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
              slurm/si_noise/01_train_array.sh)
  echo "Training array: $TRAIN_JID (tasks $ARR, segment limit $CL_TRAIN_TIME, self-requeueing)"
fi

for i in $TRAIN_ARMS; do
  lo=$(( i * EVALS_PER_RUN )); hi=$(( lo + EVALS_PER_RUN - 1 ))
  DEP=""
  if [ -n "${TRAIN_JID:-}" ]; then DEP="--dependency=afterok:${TRAIN_JID}_${i}"; fi
  EJID=$(sbatch --parsable --array="$lo-$hi" --time="$EVAL_TIME" $DEP $RESOURCE_ARG $EXPORT_ARG \
         slurm/si_noise/02_eval_array.sh)
  echo "  evals of [$i] ${NOISE_CONFIGS[$i]}: job $EJID (tasks $lo-$hi)"
done


if [ -z "${NO_PHASE0:-}" ]; then
  P0JID=$(sbatch --parsable --array="$P0_FIRST-$(( N_EVAL - 1 ))" --time="$EVAL_TIME" $RESOURCE_ARG $EXPORT_ARG \
          slurm/si_noise/02_eval_array.sh)
  echo "  Phase 0 (existing $P0_CFG, eps ${P0_EPS_LIST[*]}): job $P0JID (tasks $P0_FIRST-$(( N_EVAL - 1 )))"
fi
if [ -z "${NO_SKILL:-}" ]; then
  SKJID=$(sbatch --parsable $RESOURCE_ARG $EXPORT_ARG slurm/si_noise/03_skill_existing.sh)
  echo "  skill-score backfill of the SI-schedules evals and baselines: job $SKJID"
fi

cat <<EOF

Submitted. Monitor with: squeue -u \$USER
  logs:     $CL_REPO/logs/si_noise/
  runs:     $CL_REPO/diffusionsr/runs/si_noise/<config>/
  results:  $CL_REPO/eval_results/si_noise/<config>__<solver>_eps<eps>/
  W&B:      project Flow3D_SuperResolution, group si_noise
            5 checkpoint artifacts   <config>_ckpt                                 (1 version each)
            73 eval artifacts        <config>__<solver>_eps<eps>_eval_predictions  (1 version each)
Results notebook: diffusionsr/notebooks/24_si_noise_results.ipynb
EOF
