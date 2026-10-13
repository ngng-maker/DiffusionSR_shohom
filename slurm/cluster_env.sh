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
    CL_WANDB_ENTITY="${CL_WANDB_ENTITY:-ngng-}"
    # The frozen encoder both arms of the comparison condition on.
    CL_ENC_DIR="${CL_ENC_DIR:-$CL_REPO/diffusionsr/runs/ablation_20260901/enc_temp}"
    ;;

  orchard)
    # ---- ORCHARD (NOT YET FILLED IN) ----------------------------------------------------
    # Every value below is a guess and will not work as written. Run
    #     bash slurm/discover_cluster.sh
    # on an ORCHARD login node and replace these with what it prints.
    #
    # The CL_ENC_DIR one is the important one: the comparison conditions both arms on a specific
    # trained encoder checkpoint. It does not exist on ORCHARD yet, so it must either be copied
    # across (see slurm/orchard/00_bootstrap.sh) or retrained there first -- and if retrained, the
    # numbers are not comparable with any TRACE results.
    CL_REPO="${CL_REPO:-$HOME/DiffusionSR_shohom}"
    CL_DATA="${CL_DATA:-$HOME/data_fields}"
    CL_CONDA_SH="${CL_CONDA_SH:-}"                  # empty => assume conda is already on PATH
    CL_CONDA_ENV="${CL_CONDA_ENV:-$HOME/envs/diffusion_SR}"
    CL_PARTITION="${CL_PARTITION:-gpu}"             # PLACEHOLDER
    CL_GRES="${CL_GRES:-gpu:1}"                     # PLACEHOLDER: generic, no GPU type pinned
    CL_EXTRA_SBATCH="${CL_EXTRA_SBATCH:-}"          # add --account=... / --qos=... if required
    CL_WANDB_ENTITY="${CL_WANDB_ENTITY:-ngng-}"
    CL_ENC_DIR="${CL_ENC_DIR:-$CL_REPO/diffusionsr/runs/ablation_20260901/enc_temp}"

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
