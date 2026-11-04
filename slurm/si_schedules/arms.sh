# SI-schedules study: the ONE place that maps array indices to configurations.
# Sourced by 00_smoke.sh, 01_train_array.sh, 02_eval_array.sh and submit_all.sh, so they cannot drift.
#
# Training: 8 runs = 4 schedules x 2 paired bases. Every run conditions on the frozen RRDB encoder's
# features at every network evaluation (si_cond: enc_implicit), exactly as DiffusionSR does.
#
#   train idx  config                        schedule  base
#   ---------  ----------------------------  --------  --------------------------
#   0          si_sched_linear_enc_temp      linear    enc(LR) + sigma*zeta
#   1          si_sched_trig_enc_temp        trig      enc(LR) + sigma*zeta
#   2          si_sched_vp_enc_temp          vp        enc(LR) + sigma*zeta
#   3          si_sched_bridge_enc_temp      bridge    enc(LR) + sigma*zeta
#   4          si_sched_linear_ulr_temp      linear    bicubic(LR) + sigma*zeta
#   5          si_sched_trig_ulr_temp        trig      bicubic(LR) + sigma*zeta
#   6          si_sched_vp_ulr_temp          vp        bicubic(LR) + sigma*zeta
#   7          si_sched_bridge_ulr_temp      bridge    bicubic(LR) + sigma*zeta
#
# Evaluation: 16 sampling configurations per training run -> 128 eval tasks.
#   eval idx = 16 * train idx + sub
#   sub 0-6   euler (Euler-Maruyama when eps > 0), eps in EPS_LIST
#   sub 7-13  heun  (stochastic Heun when eps > 0),  eps in EPS_LIST
#   sub 14    rk4    eps = 0 (ODE only)
#   sub 15    dopri5 eps = 0 (ODE only, adaptive)
# e.g. all evals of the VP/enc run:  --array=32-47;  every eps=0 Heun eval: --array=7,23,39,55,71,87,103,119

SI_CONFIGS=(si_sched_linear_enc_temp si_sched_trig_enc_temp si_sched_vp_enc_temp si_sched_bridge_enc_temp
            si_sched_linear_ulr_temp si_sched_trig_ulr_temp si_sched_vp_ulr_temp si_sched_bridge_ulr_temp)
EPS_LIST=(0.0 0.01 0.05 0.1 0.25 0.5 1.0)
EVALS_PER_RUN=16
N_TRAIN=${#SI_CONFIGS[@]}
N_EVAL=$(( N_TRAIN * EVALS_PER_RUN ))

# Sampling budget held at the previous SI run's value (20 steps, K = 10), so these numbers line up with
# notebook 21. The network-evaluation count actually spent is recorded per eval (nfe_per_sample).
SI_N_STEPS="${SI_N_STEPS:-20}"
SI_K_SAMPLES="${SI_K_SAMPLES:-10}"

# Suffix of every deterministic W&B run id. W&B never lets a DELETED run's id be reused, so if runs
# are deleted and the study restarted from scratch, bump this (e.g. SI_SCHED_TAG=v2).
export SI_SCHED_TAG="${SI_SCHED_TAG:-v1}"

# decode_eval IDX -> sets SI_TRAIN_IDX, SI_CFG, SI_SOLVER, SI_EPS
decode_eval() {
  local idx=$1 sub
  SI_TRAIN_IDX=$(( idx / EVALS_PER_RUN ))
  sub=$(( idx % EVALS_PER_RUN ))
  SI_CFG=${SI_CONFIGS[$SI_TRAIN_IDX]}
  if   [ "$sub" -lt 7 ];  then SI_SOLVER=euler;  SI_EPS=${EPS_LIST[$sub]}
  elif [ "$sub" -lt 14 ]; then SI_SOLVER=heun;   SI_EPS=${EPS_LIST[$(( sub - 7 ))]}
  elif [ "$sub" -eq 14 ]; then SI_SOLVER=rk4;    SI_EPS=0.0
  else                         SI_SOLVER=dopri5; SI_EPS=0.0
  fi
}

# Deterministic names: no date stamps, so a requeue on a later day or on the OTHER cluster maps to the
# same W&B run and the same single artifact.
train_run_id()   { echo "$1-$SI_SCHED_TAG" | tr '[:upper:]_' '[:lower:]-'; }
ckpt_artifact()  { echo "${1}_ckpt"; }
