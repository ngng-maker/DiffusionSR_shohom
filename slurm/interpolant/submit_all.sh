#!/bin/bash
# Master submit script: DiffusionSR baseline vs stochastic interpolants.
#
# Usage (from the repo root on the TRACE login node):
#   cd <repo root on this cluster>
#   CLUSTER=trace    bash slurm/interpolant/submit_all.sh   # the default
#   CLUSTER=orchard  bash slurm/interpolant/submit_all.sh
#
# Only the attributable comparison (baseline + the five encoder arms):
#   ARMS=0-5 bash slurm/interpolant/submit_all.sh
#
# WHAT IS BEING COMPARED
# ----------------------
# Task 0 is DiffusionSR: DDPM conditioned on the frozen RRDB encoder
# (fno_ablation/enc_rrdb). Tasks 1-5 use THE SAME frozen encoder and THE SAME implicit
# 64-channel conditioning, so against task 0 the only thing that changes is the generative process:
#
#   DDPM            x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps,  starting from pure noise
#   Interpolant     I_t = (1-t) x0 + t x1,  starting from x0 = upscaled_lr + sigma*zeta
#
# Those five are the comparison that can attribute a difference to the process. Tasks 6-10 are the
# ulr_concat twins: they drop the encoder and change the conditioning as well, so relative to task 0
# they vary three things at once. They reproduce the paper's own setup (Albergo et al. 2024 use
# bicubic with no encoder) and are interpretable against each other, not against task 0.
#
# NOTE ON THE BASE. si_cond only selects the CONDITIONING. x0 is always
# make_x0(ulr_hr, sigma) = upscaled_lr + sigma*zeta (train_coupled.py:185) in every arm -- the
# encoder is never the transport's starting point. Starting the transport AT the encoder's own HR
# prediction is a separate idea and is not implemented; see FNO/INTERPOLANT_PLAN.md section 0.3.
#
# Phase 1  01_train_array.sh   11 runs, parallel array
# Phase 2  02_eval_array.sh    11 evals, parallel array, after Phase 1 (afterok)
#
# Everything is held equal across arms: dataset, splits, batch_size=40, epochs=300, lr=2e-4, l2
# loss, standardize normalisation, U-Net, and the sampling budget (20 steps / DDIM skip=50, K=10).

set -eo pipefail

REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_GUESS/slurm/cluster_env.sh"
REPO="$CL_REPO"
cd "$REPO"
mkdir -p logs/interpolant

# The SBATCH directives inside the array scripts are parsed before the shell runs and cannot
# read the profile, so the cluster-specific resource request is supplied here instead. A value
# given on the sbatch command line overrides the #SBATCH directive in the file.
RESOURCE_ARG="--partition=$CL_PARTITION --gres=$CL_GRES $CL_EXTRA_SBATCH"
echo "cluster=$CLUSTER  partition=$CL_PARTITION  gres=$CL_GRES  ${CL_EXTRA_SBATCH:-(no extra flags)}"

# ARMS selects which array indices to run. Default is all 13 (0-12); "0-5" gives just the attributable
# comparison, which is the cheaper and more informative first pass.
ARMS="${ARMS:-0-12}"

# CLUSTER must reach the compute node so the array scripts resolve the same profile.
EXPORT_ARG="--export=ALL,CLUSTER=$CLUSTER"
if [ -n "${PY_OVERLAY:-}" ]; then EXPORT_ARG="$EXPORT_ARG,PY_OVERLAY=$PY_OVERLAY"; fi

# Fail early rather than after queueing: tasks 0-5 all need this checkpoint, and without
# --force_enc_dir pointing at a real one, train_srdiff would train a FRESH encoder and silently
# break the "same encoder as DiffusionSR" property the whole comparison depends on.
ENC_DIR="$CL_ENC_DIR"
if [ ! -f "$ENC_DIR/bestmodel_saved.pth" ]; then
  echo "ERROR: the shared frozen encoder is missing:"
  echo "         $ENC_DIR/bestmodel_saved.pth"
  echo "       Tasks 0-5 (the baseline and all encoder arms) need it. Either train it, or run only"
  echo "       the ulr arms with:  ARMS=6-10 bash slurm/interpolant/submit_all.sh"
  exit 1
fi
echo "Shared frozen encoder found: $ENC_DIR"

# The encoder's input width is n_steps x num_fields from the config it was trained with, so an
# encoder trained with different n_steps or fields cannot load into these arms. That mismatch
# (enc_temp, n_steps=3, against n_steps=1 configs) failed every encoder task with a conv1 size
# mismatch seconds after it finally got a GPU, so compare before queueing anything.
# The arm list is read from 01_train_array.sh so the check covers exactly the configs being trained.
ARM_CFGS=$(sed -n '/^CONFIGS=(/,/)/p' slurm/interpolant/01_train_array.sh | sed 's/CONFIGS=//' | tr -d '()\\')
python3 - "$ENC_DIR/configuration.yml" "$CL_CFGS" $ARM_CFGS <<'PYCHECK' || exit 1
import sys, os, re
def keys(path):   # plain regex rather than yaml: the login node's system python may lack PyYAML
    txt = open(path).read()
    get = lambda k: (re.search(rf'^{k}:\s*["\']?([^"\'\n#]+)', txt, re.M) or [None, None])[1]
    return get('n_steps'), get('fields'), get('encoding')
enc = keys(sys.argv[1])[:2]
cfgs = [os.path.join(sys.argv[2], n + '.yml') for n in sys.argv[3:]]
bad = [os.path.basename(c) for c in cfgs if keys(c)[2] in ('True', 'true') and keys(c)[:2] != enc]
if bad:
    print(f"ERROR: the encoder was trained with n_steps={enc[0]}, fields={enc[1]}, but these configs differ:")
    for b in bad: print(f"         {b}: n_steps={keys(os.path.join(sys.argv[2], b))[0]}, fields={keys(os.path.join(sys.argv[2], b))[1]}")
    print("       Its weights would not load. Point CL_ENC_DIR at a matching encoder.")
    sys.exit(1)
print(f"Encoder matches all {len(cfgs)} arm configs (n_steps={enc[0]}, fields={enc[1]})")
PYCHECK

echo "=== Phase 1: training array (tasks $ARMS) ==="
# --parsable makes sbatch print only the job ID, so it can be captured for the dependency.
# The training time limit is per cluster, and only applies to training: each segment requeues
# itself before the limit and resumes from ckpt.pth, so it can be much shorter than a full run.
TRAIN_JID=$(sbatch --parsable --array="$ARMS" --time="$CL_TRAIN_TIME" $RESOURCE_ARG $EXPORT_ARG \
            slurm/interpolant/01_train_array.sh)
echo "  Training array job ID: $TRAIN_JID  (time limit per segment: $CL_TRAIN_TIME)"

echo "=== Phase 2: eval array (tasks $ARMS) -- depends on $TRAIN_JID ==="
# afterok on an array job waits for EVERY task to succeed. A single failed arm therefore blocks the
# whole eval phase; requeue that task and resubmit Phase 2 on its own rather than rerunning Phase 1.
EVAL_JID=$(sbatch --parsable \
  --dependency=afterok:"$TRAIN_JID" \
  --array="$ARMS" \
  $RESOURCE_ARG $EXPORT_ARG \
  slurm/interpolant/02_eval_array.sh)
echo "  Eval array job ID: $EVAL_JID"

echo ""
echo "=== Submitted ==="
echo "  Monitor:   squeue -u ngng"
echo "  Logs:      $REPO/logs/interpolant/"
echo "  Results:   $REPO/eval_results/interpolant/<arm>/   (results.json + overlay.png per arm)"
echo "  W&B:       project Flow3D_SuperResolution, one predictions.npz artifact per arm"
echo ""
echo "Task -> arm mapping:"
echo "   0  baseline_ddpm_matched_temp    DDPM, SI harness   <== THE EXPERIMENT (vs task 2)"
echo "   2  si_opt_a_sigma010_enc_temp    SI-A sigma=0.10    <== THE EXPERIMENT (vs task 0)"
echo "   1  si_opt_a_sigma005_enc_temp    SI-A sigma=0.05    (sigma sensitivity)"
echo "   3  si_opt_a_sigma020_enc_temp    SI-A sigma=0.20    (sigma sensitivity)"
echo "   4  si_opt_b_c010_enc_temp        SI-B c=0.10        (adds path noise + score head)"
echo "   5  si_opt_b_c025_enc_temp        SI-B c=0.25        (adds path noise + score head)"
echo "   6-10  the five _ulr_temp twins   no encoder, ulr_concat conditioning"
echo "  11  baseline_diffusionsr_temp     as-published DiffusionSR (plain loop, reference only)"
echo "  12  si_opt_a_sigma010_encbase_temp  SI-A sigma=0.10 from enc(LR)  (vs task 2: the base alone)"
echo ""
echo "THE MINIMAL EXPERIMENT is tasks 0 and 2. Identical encoder, conditioning, dataset, harness"
echo "(EMA + grad clip + StepLR + FlippedDataset) and hyperparameters; the configs differ in ONE"
echo "key, si_process. So any difference is attributable to the generative process alone:"
echo "    ARMS=0,2 bash slurm/interpolant/submit_all.sh"
echo ""
echo "Task 11 is NOT the comparison. DiffusionModel.train() has no EMA, no gradient clipping, a"
echo "fixed LR and no flip augmentation, so it differs from the SI arms by four harness features"
echo "as well as by the process. Keep it only as an as-published reference number."
echo ""
echo "AFTER the eval array finishes, the epsilon sweep costs no extra training. Option B arms"
echo "(tasks 4, 5, 9, 10) have a trained denoiser head, so they can be resampled stochastically:"
echo "  for e in 0.05 0.2 0.5; do"
echo "    sbatch --array=4,5 --export=ALL,EPS0=\$e slurm/interpolant/02_eval_array.sh"
echo "  done"
echo "Option A arms have no score and stay deterministic -- the eval script forces eps0=0 for them."
