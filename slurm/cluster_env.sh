# Cluster profile, sourced by the job scripts under slurm/interpolant/.
#
# Everything cluster-specific lives HERE and nowhere else, so porting the experiment to a new
# machine means filling in one profile rather than editing every script. Sourced (not executed),
# so it only sets variables.
#
#   CLUSTER=trace    bash slurm/interpolant/submit_all.sh     # the default
#   CLUSTER=orchard  bash slurm/interpolant/submit_all.sh
#
# Any value can still be overridden per-invocation from the environment, e.g.
#   CLUSTER=orchard CONDA_ENV=/some/other/env bash slurm/interpolant/submit_all.sh
#
# To fill in a new profile, run `bash slurm/discover_cluster.sh` on that cluster's login node and
# paste the values it prints.

CLUSTER="${CLUSTER:-trace}"

case "$CLUSTER" in

  trace)
    # ---- TRACE (known good; these are the values every run in this repo has used) ----
    CL_REPO="${CL_REPO:-/trace/group/forgelab/ngng/multifield/DiffusionSR_shohom}"
    CL_DATA="${CL_DATA:-/trace/group/forgelab/ngng/multifield/data_fields}"
    CL_CONDA_SH="${CL_CONDA_SH:-/trace/packages/anaconda3/2023.03-1/etc/profile.d/conda.sh}"
    CL_CONDA_ENV="${CL_CONDA_ENV:-/trace/group/forgelab/ngng/envs/diffusion_SR}"
    CL_PARTITION="${CL_PARTITION:-batch}"
    CL_GRES="${CL_GRES:-gpu:a40:1}"
    CL_EXTRA_SBATCH="${CL_EXTRA_SBATCH:-}"          # TRACE needs no --account or --qos
    CL_TRAIN_TIME="${CL_TRAIN_TIME:-2-00:00:00}"     # a full 300-epoch run takes ~12-14 h on an A40
    CL_WANDB_ENTITY="${CL_WANDB_ENTITY:-ngng-}"
    # The frozen encoder both arms of the comparison condition on. It must match the configs'
    # n_steps and fields: the interpolant configs use n_steps=1, temperature, so this is the 1-step
    # RRDB temperature encoder. ablation_20260901/enc_temp was n_steps=3 (3 input channels) and
    # failed every encoder arm with a conv1 size mismatch. submit_all.sh now checks this.
    CL_ENC_DIR="${CL_ENC_DIR:-$CL_REPO/diffusionsr/runs/fno_ablation/enc_rrdb}"
    ;;

  orchard)
    # ---- ORCHARD (from discover_cluster.sh on orchard-login-001, 2026-10-13) --------------
    # SLURM 26.05; every node is 8x H100 80GB (driver 580, CUDA 13.0) and GRES names no type,
    # so gpu:1 is exact rather than generic. Account oogoke has two usable QoS:
    #   general_qos  -- 12 h wall limit, 1 GPU and 1 job per user: cannot fit a 2-day training array.
    #   preempt_qos  -- 2-day wall limit, up to 32 GPUs / 24 jobs. Jobs can be preempted, which is
    #                   safe here: the job scripts pass --requeue and train_coupled resumes from the
    #                   per-epoch ckpt.pth and the same W&B run.
    # STORAGE: $HOME has a hard 25 GB per-user quota that df does NOT show (df reports the whole
    # 5 TB NFS export). Filling it failed every training task with "Disk quota exceeded" and left
    # truncated loss files and a corrupt checkpoint. The repo (with all run outputs), the dataset
    # and W&B's staging area therefore live on /project/community/$USER (5 TB free, no quota hit
    # at 2 GB). ~/LPBFDiffusionSR_dev and ~/data_fields are symlinks there. Only Miniforge and the
    # conda env (~7 GB, not relocatable) stay in $HOME.
    # The login node's /tmp is full, so installs must set TMPDIR=$HOME/tmp.
    #
    # CL_ENC_DIR is the important one: the comparison conditions both arms on a specific trained
    # encoder checkpoint, which must be copied from TRACE (see slurm/orchard/00_bootstrap.sh).
    # Retraining it on ORCHARD makes the numbers incomparable with any TRACE results.
    CL_PROJECT="${CL_PROJECT:-/project/community/$USER}"
    CL_REPO="${CL_REPO:-$CL_PROJECT/LPBFDiffusionSR_dev}"
    CL_DATA="${CL_DATA:-$CL_PROJECT/data_fields}"
    # W&B stages every checkpoint artifact (~400 MB) under ~/.local/share/wandb before uploading;
    # with 12 tasks at once that alone can exhaust the home quota. One dir per job, so the EXIT trap
    # in the job scripts clears only its own staging, not uploads still in flight in other tasks.
    # Inside a job this deliberately IGNORES an inherited value: submit_all.sh sources this file on
    # the login node, which sets .../login, and sbatch --export=ALL then copied that into every job,
    # so all tasks shared one staging dir. A task's cleanup then hit "Directory not empty" while
    # another was still uploading, and the task ended FAILED despite a successful eval.
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      export WANDB_DATA_DIR="$CL_PROJECT/.wandb_data/$SLURM_JOB_ID"
    else
      export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$CL_PROJECT/.wandb_data/login}"
    fi
    CL_CONDA_SH="${CL_CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
    CL_CONDA_ENV="${CL_CONDA_ENV:-$HOME/envs/diffusion_SR}"
    CL_PARTITION="${CL_PARTITION:-preempt}"
    CL_GRES="${CL_GRES:-gpu:1}"
    CL_EXTRA_SBATCH="${CL_EXTRA_SBATCH:---account=oogoke --qos=preempt_qos}"
    # Per-segment training time limit. preempt is the lowest priority tier, so a 2-day request only
    # starts when a GPU is free for 2 days; a short one can be backfilled into gaps. 300 epochs took
    # ~12.5 h on a TRACE A40 (ddpm_base_temp, ~2.5 min/epoch), so ~5-8 h is expected on an H100.
    # A task that runs over requeues itself and resumes (see requeue_before_timeout in 01_train_array.sh).
    CL_TRAIN_TIME="${CL_TRAIN_TIME:-08:00:00}"
    CL_WANDB_ENTITY="${CL_WANDB_ENTITY:-ngng-}"
    CL_ENC_DIR="${CL_ENC_DIR:-$CL_REPO/diffusionsr/runs/fno_ablation/enc_rrdb}"
    CLUSTER_PROFILE_CONFIRMED=1

    if [ "${CLUSTER_PROFILE_CONFIRMED:-0}" != "1" ]; then
      echo "ERROR: the 'orchard' profile in slurm/cluster_env.sh still holds placeholder values."
      echo "       Running now would submit jobs with a guessed partition and GPU request."
      echo
      echo "       1. bash slurm/discover_cluster.sh          # on an ORCHARD login node"
      echo "       2. paste the values into the orchard block of slurm/cluster_env.sh"
      echo "       3. set CLUSTER_PROFILE_CONFIRMED=1 in that block to remove this guard"
      exit 1
    fi
    ;;

  *)
    echo "ERROR: unknown CLUSTER=$CLUSTER. Known profiles: trace, orchard."
    echo "       Add a block to slurm/cluster_env.sh for a new machine."
    exit 1
    ;;
esac

# Derived paths, identical on every cluster because they are relative to the repo.
CL_RUNS="$CL_REPO/diffusionsr/runs/interpolant"
CL_CFGS="$CL_REPO/diffusionsr/configs/stochastic_interpolant"
CL_EVAL_OUT="$CL_REPO/eval_results/interpolant"

# Activate the environment. Sourcing the conda profile is required in a non-interactive SLURM
# shell, which has no `conda` shell function until it is loaded; when CL_CONDA_SH is empty we
# assume conda is already available (e.g. provided by a module).
cluster_activate_env() {
  if [ -n "$CL_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CL_CONDA_SH"
  fi
  conda activate "$CL_CONDA_ENV"
  # Optional venv overlay layered on top, as used on TRACE for PhysicsNeMo.
  if [ -n "${PY_OVERLAY:-}" ]; then
    # shellcheck disable=SC1091
    source "$PY_OVERLAY/bin/activate"
    echo "Using venv overlay: $PY_OVERLAY"
  fi
  echo "cluster=$CLUSTER  env=$CL_CONDA_ENV  python=$(command -v python)"
}

export WANDB_ENTITY="$CL_WANDB_ENTITY"
