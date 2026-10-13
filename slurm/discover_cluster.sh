#!/bin/bash
# Print everything needed to fill in a cluster profile in slurm/cluster_env.sh.
#
# Read-only: queries the scheduler and the filesystem, submits nothing and installs nothing.
# Run it on the LOGIN NODE of the cluster being added:
#
#   bash slurm/discover_cluster.sh
#
# Then paste the reported values into the matching block of slurm/cluster_env.sh.

echo "==================================================================="
echo " Cluster discovery"
echo "==================================================================="
echo "hostname        : $(hostname)"
echo "user            : $(whoami)"
echo "home            : $HOME"
echo

echo "--- 1. Is this SLURM, and which version? --------------------------"
if command -v sbatch >/dev/null 2>&1; then
  echo "sbatch          : $(command -v sbatch)"
  sinfo --version 2>/dev/null || echo "  (sinfo --version unavailable)"
else
  echo "NO sbatch FOUND. This is not a SLURM cluster, so the job scripts need rewriting"
  echo "(headers, array syntax and the dependency chaining), not just re-pathing."
  command -v qsub  >/dev/null 2>&1 && echo "  found qsub  -> PBS/Torque"
  command -v bsub  >/dev/null 2>&1 && echo "  found bsub  -> LSF"
fi
echo

echo "--- 2. Partitions (-> CL_PARTITION) -------------------------------"
# %P partition (* marks the default), %l default time limit, %D node count, %G generic resources.
sinfo -o "%20P %10l %6D %25G %s" 2>/dev/null | head -25 || echo "  sinfo unavailable"
echo
echo "    Pick a partition that lists a gpu: entry under GRES. The one with * is the default."
echo

echo "--- 3. GPU types (-> CL_GRES) -------------------------------------"
# The exact string after gpu: is what --gres needs, e.g. gpu:a100:1
sinfo -o "%25N %30G" 2>/dev/null | grep -i gpu | sort -u | head -15 || echo "  no GPU nodes reported"
echo
echo "    CL_GRES must be gpu:<type>:1 using a type shown above (or plain gpu:1 if none is named)."
echo

echo "--- 4. Account / QoS, if required (-> CL_EXTRA_SBATCH) ------------"
if command -v sacctmgr >/dev/null 2>&1; then
  echo "associations for $(whoami):"
  sacctmgr -nP show assoc user="$(whoami)" format=Account,Partition,QOS 2>/dev/null | head -10 \
    || echo "  (sacctmgr query returned nothing)"
else
  echo "  sacctmgr unavailable; check local docs for whether --account is mandatory"
fi
echo
echo "    If an Account is listed and submission fails without it, set:"
echo "      CL_EXTRA_SBATCH=\"--account=<that account>\""
echo

echo "--- 5. Conda / modules (-> CL_CONDA_SH) ---------------------------"
if command -v conda >/dev/null 2>&1; then
  echo "conda already on PATH : $(command -v conda)"
  echo "  -> CL_CONDA_SH can stay EMPTY"
else
  echo "conda NOT on PATH. Looking for a conda.sh to source..."
  # The usual places a site-installed anaconda lives.
  for root in /opt /usr/local /sw /share /apps /packages /trace/packages "$HOME"; do
    [ -d "$root" ] || continue
    find "$root" -maxdepth 4 -name conda.sh -path '*profile.d*' 2>/dev/null | head -5
  done
  echo
  echo "  If nothing was found, check for a module instead:"
  module avail 2>&1 | grep -iE "anaconda|miniconda|python|cuda" | head -15 \
    || echo "    (no module system detected)"
fi
echo

echo "--- 6. GPU driver version (constrains the torch build) ------------"
# This is the check that matters: on TRACE, installing PhysicsNeMo pulled a torch compiled against
# a newer CUDA than the driver supported, and every job failed at .to('cuda').
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
    || echo "  nvidia-smi present but query failed (normal on a login node with no GPU)"
else
  echo "  no nvidia-smi on this node (normal for a login node). Check on a GPU node with:"
  echo "    srun --partition=<P> --gres=<G> --pty nvidia-smi"
fi
echo

echo "--- 7. Where to put the repo and data (-> CL_REPO / CL_DATA) ------"
echo "Candidate writable roots, with free space:"
for cand in "$HOME" /scratch /work /project /projects /data "/scratch/$(whoami)" "/work/$(whoami)"; do
  if [ -d "$cand" ] && [ -w "$cand" ]; then
    printf "  %-28s %s\n" "$cand" "$(df -h "$cand" 2>/dev/null | awk 'NR==2 {print $4" free of "$2}')"
  fi
done
echo
echo "    The dataset and checkpoints are large; prefer a scratch/project filesystem over \$HOME,"
echo "    which is usually quota-limited."
echo

echo "==================================================================="
echo " Paste into the matching block of slurm/cluster_env.sh:"
echo "==================================================================="
cat <<EOF
  CL_REPO="\${CL_REPO:-<a writable root from section 7>/DiffusionSR_shohom}"
  CL_DATA="\${CL_DATA:-<a writable root from section 7>/data_fields}"
  CL_CONDA_SH="\${CL_CONDA_SH:-<path from section 5, or empty>}"
  CL_CONDA_ENV="\${CL_CONDA_ENV:-<a writable root>/envs/diffusion_SR}"
  CL_PARTITION="\${CL_PARTITION:-<partition from section 2>}"
  CL_GRES="\${CL_GRES:-<gpu string from section 3>}"
  CL_EXTRA_SBATCH="\${CL_EXTRA_SBATCH:-<--account=... if section 4 requires it, else empty>}"
  CL_WANDB_ENTITY="\${CL_WANDB_ENTITY:-ngng-}"
  CL_ENC_DIR="\${CL_ENC_DIR:-\$CL_REPO/diffusionsr/runs/ablation_20260901/enc_temp}"
EOF
echo
echo "Then set CLUSTER_PROFILE_CONFIRMED=1 in that block to lift the placeholder guard."
