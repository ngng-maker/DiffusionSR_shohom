#!/bin/bash
# One-time bootstrap for a cluster where nothing exists yet. Run on the LOGIN NODE.
#
#   bash slurm/orchard/00_bootstrap.sh
#
# Creates the conda environment and verifies it, then prints what still has to be transferred.
# It does NOT submit jobs and does NOT copy data: the transfers have to run from a machine that
# can see both clusters, so they are printed for you to run rather than attempted here.
#
# PREREQUISITE: the orchard block of slurm/cluster_env.sh must be filled in first. Run
#   bash slurm/discover_cluster.sh
# to get the values. This script refuses to continue while the placeholders are still in place.

set -eo pipefail

CLUSTER="${CLUSTER:-orchard}"
export CLUSTER
REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_GUESS/slurm/cluster_env.sh"   # exits if the profile is unconfirmed

echo "==================================================================="
echo " Bootstrap: cluster=$CLUSTER"
echo "==================================================================="
echo "repo        : $CL_REPO"
echo "data        : $CL_DATA"
echo "conda env   : $CL_CONDA_ENV"
echo "partition   : $CL_PARTITION"
echo "gres        : $CL_GRES"
echo "encoder     : $CL_ENC_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
echo "--- 1. Creating the conda environment ------------------------------"
if [ -n "$CL_CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CL_CONDA_SH"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. Either set CL_CONDA_SH in the profile, or load the"
  echo "       relevant module first (see section 5 of discover_cluster.sh)."
  exit 1
fi

if [ -d "$CL_CONDA_ENV" ]; then
  echo "env already exists, skipping creation: $CL_CONDA_ENV"
else
  # Python 3.11 matches the version TRACE runs, so the pinned dependencies behave the same.
  conda create -y -p "$CL_CONDA_ENV" python=3.11
fi
conda activate "$CL_CONDA_ENV"
echo "python: $(command -v python)  ($(python --version 2>&1))"
echo

# ---------------------------------------------------------------------------
# 2. PyTorch -- the step most likely to go wrong
# ---------------------------------------------------------------------------
echo "--- 2. Installing PyTorch ------------------------------------------"
# THE LESSON FROM TRACE: a torch wheel built against a newer CUDA than the driver supports installs
# and imports perfectly, then fails at the first .to('cuda') with
#   "The NVIDIA driver on your system is too old (found version 12030)"
# -- which cost a whole job array. So the CUDA build is chosen from the DRIVER version, not guessed.
#
# TORCH_CUDA may be set explicitly (cu121, cu124, cu126...). Otherwise it is inferred below.
if [ -z "${TORCH_CUDA:-}" ]; then
  echo "TORCH_CUDA not set; checking the driver on a GPU node..."
  DRIVER_CUDA=$(srun --partition="$CL_PARTITION" --gres="$CL_GRES" $CL_EXTRA_SBATCH \
                     --time=00:05:00 --pty nvidia-smi \
                     --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  if [ -z "$DRIVER_CUDA" ]; then
    echo "  could not reach a GPU node. Defaulting to cu121, the build known to work on TRACE."
    echo "  If ORCHARD has a newer driver you may prefer cu124; rerun with TORCH_CUDA=cu124."
    TORCH_CUDA=cu121
  else
    echo "  driver version: $DRIVER_CUDA"
    # Driver >= 525 supports CUDA 12.1; >= 550 supports 12.4. Conservative, since too old fails
    # loudly at runtime whereas too new merely forgoes a little performance.
    MAJOR=${DRIVER_CUDA%%.*}
    if [ "$MAJOR" -ge 550 ] 2>/dev/null; then TORCH_CUDA=cu124; else TORCH_CUDA=cu121; fi
    echo "  -> selecting $TORCH_CUDA"
  fi
fi

python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
echo "torch: $(python -c 'import torch; print(torch.__version__)')"
echo

# ---------------------------------------------------------------------------
# 3. Repo dependencies
# ---------------------------------------------------------------------------
echo "--- 3. Installing the repo dependencies ----------------------------"
if [ -f "$CL_REPO/requirements.txt" ]; then
  # requirements.txt caps numpy at <2.0. Note that TRACE actually runs numpy 2.4.4, so the cap is
  # stale -- but install as pinned here and only relax it if something demands it.
  python -m pip install --quiet -r "$CL_REPO/requirements.txt"
  python -m pip install --quiet -e "$CL_REPO"
  # pytest is dev-only and absent from requirements.txt, which already cost one failed TRACE job
  # whose only error was "No module named pytest".
  python -m pip install --quiet pytest
else
  echo "SKIPPED: $CL_REPO/requirements.txt not found -- transfer the repo first (section 5)."
fi
echo

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo "--- 4. Verifying the install ---------------------------------------"
python - <<'PYCHECK'
import sys
try:
    import torch, numpy
    print(f"  python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}   (False on a login node is normal)")
except Exception as exc:
    print(f"  IMPORT FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)
try:
    from diffusionsr.runners.train_coupled import CoupledInterpolantModel
    from diffusionsr.runners.train_ddpm_matched import DDPMMatchedHarnessModel
    print("  repo imports OK (CoupledInterpolantModel, DDPMMatchedHarnessModel)")
except Exception as exc:
    print(f"  repo import failed: {type(exc).__name__}: {exc}")
    print("  -> transfer the repo and rerun, or check `pip install -e` succeeded")
PYCHECK
echo
echo "Run the CPU-only test suite to confirm the install (should be 62 passed, 4 pre-existing fails):"
echo "  cd $CL_REPO && python -m pytest tests/test_interpolant.py -q"
echo

# ---------------------------------------------------------------------------
# 5. What still has to be transferred
# ---------------------------------------------------------------------------
echo "--- 5. Transfers still required ------------------------------------"
missing=0
check() {   # $1 = path, $2 = description
  if [ -e "$1" ]; then
    printf "  [ok]      %-22s %s\n" "$2" "$1"
  else
    printf "  [MISSING] %-22s %s\n" "$2" "$1"
    missing=$((missing + 1))
  fi
}
check "$CL_REPO/diffusionsr"                      "repo"
check "$CL_DATA"                                  "dataset"
check "$CL_DATA/statistics"                       "dataset statistics"
check "$CL_ENC_DIR/bestmodel_saved.pth"           "frozen encoder"

if [ "$missing" -gt 0 ]; then
  echo
  echo "Run these FROM A MACHINE THAT CAN REACH BOTH CLUSTERS (or stage via a laptop)."
  echo "Replace <orchard> with the SSH host and \$TRACE_REPO/\$TRACE_DATA with the TRACE paths."
  cat <<'EOF'

  TRACE_REPO=/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom
  TRACE_DATA=/trace/group/forgelab/ngng/multifield/data_fields

  # a) the repo (git is preferable to rsync -- it keeps history and the branch)
  ssh <orchard> "git clone -b fno-conditioning-encoder <your remote URL> $CL_REPO"

  # b) the dataset, including the statistics/ cache. WITHOUT statistics/, the dev and test splits
  #    raise rather than recomputing, and the train split would recompute them from scratch --
  #    which also risks a DIFFERENT normalisation than the TRACE runs used.
  rsync -avh --progress $TRACE_DATA/ <orchard>:$CL_DATA/

  # c) the frozen encoder. BOTH arms of the comparison condition on this exact checkpoint, so
  #    copying it is what keeps ORCHARD results comparable with TRACE ones. Retraining instead
  #    gives a different encoder and the two clusters' numbers stop being comparable.
  ssh <orchard> "mkdir -p $CL_ENC_DIR"
  rsync -avh --progress $TRACE_REPO/diffusionsr/runs/fno_ablation/enc_rrdb/ \
      <orchard>:$CL_ENC_DIR/
EOF
  echo
  echo "Bootstrap INCOMPLETE: $missing item(s) still missing."
  exit 1
fi

echo
echo "==================================================================="
echo " Bootstrap complete. Launch the experiment with:"
echo "   CLUSTER=$CLUSTER ARMS=0,2 bash slurm/interpolant/submit_all.sh"
echo "==================================================================="
