# SI-noise study (follow-up to slurm/si_schedules): the ONE place that maps array indices to configurations.
# Sourced by 00_smoke.sh, 01_train_array.sh, 02_eval_array.sh and submit_all.sh, so they cannot drift.
#
# Why: in the SI-schedules study eps barely changed linear / trig / VP, because with gamma = 0 the path's
# only noise is Gamma_t = alpha_t sigma <= 0.1, so eps_t = eps Gamma_t^2 injects almost nothing. These arms
# give eps real noise to scale. All use the linear path (the SI-schedules study's most accurate) and the
# encoder base (better than bicubic in every SI-schedules arm), and condition on the frozen encoder at
# every network evaluation as before.
#
# Phase 1 training: 5 runs
#   train idx  config                     si_schedule    gamma_t        sigma
#   ---------  -------------------------  -------------  -------------  -----
#   0          si_noise_g0p5_enc_temp     linear_gamma   0.5 t(1-t)     0.10
#   1          si_noise_g1_enc_temp       linear_gamma   1.0 t(1-t)     0.10
#   2          si_noise_g2_enc_temp       linear_gamma   2.0 t(1-t)     0.10
#   3          si_noise_s0p25_enc_temp    linear         0              0.25
#   4          si_noise_s0p5_enc_temp     linear         0              0.50
#
# Evaluation: eval idx 0-64 = Phase 1 (13 per run), 65-72 = Phase 0.
#   Phase 1: eval idx = 13 * train idx + sub
#     sub 0-5   euler (Euler-Maruyama when eps > 0), eps in EPS_LIST
#     sub 6-11  heun  (stochastic Heun when eps > 0),  eps in EPS_LIST
#     sub 12    rk4   eps = 0
#   Phase 0 (no retraining): the EXISTING SI-schedules model si_sched_linear_enc_temp, sampled with large
#   eps in P0_EPS_LIST (eps_t = eps Gamma_t^2 as before; stable to eps ~40 at 20 steps, see
#   tests/test_si_schedules.py): idx 65-68 euler, 69-72 heun.
# Results go to eval_results/si_noise/<config>__<solver>_eps<eps>/ for both phases.

NOISE_CONFIGS=(si_noise_g0p5_enc_temp si_noise_g1_enc_temp si_noise_g2_enc_temp
               si_noise_s0p25_enc_temp si_noise_s0p5_enc_temp)
EPS_LIST=(0.0 0.1 0.5 1.0 2.0 5.0)
EVALS_PER_RUN=13
N_TRAIN=${#NOISE_CONFIGS[@]}
N_EVAL_P1=$(( N_TRAIN * EVALS_PER_RUN ))       # 65

P0_CFG=si_sched_linear_enc_temp
P0_EPS_LIST=(2.0 5.0 10.0 20.0)
P0_FIRST=$N_EVAL_P1                             # 65
N_EVAL=$(( N_EVAL_P1 + 2 * ${#P0_EPS_LIST[@]} ))   # 73

# Same sampling budget as notebooks 21 and 22 (20 steps, K = 10).
SI_N_STEPS="${SI_N_STEPS:-20}"
SI_K_SAMPLES="${SI_K_SAMPLES:-10}"

# Suffix of every deterministic W&B run id (bump only if runs are deleted and the study restarted).
export SI_SCHED_TAG="${SI_SCHED_TAG:-v1}"
# W&B group for every run of this study (read by train_si_schedules and evaluate_si_schedules).
export SI_WANDB_GROUP=si_noise

# decode_eval IDX -> sets SI_PHASE, SI_TRAIN_IDX, SI_CFG, SI_CFG_DIR, SI_RUN_DIR, SI_SOLVER, SI_EPS
# (directories relative to the repo root)
decode_eval() {
  local idx=$1 sub n0
  if [ "$idx" -lt "$P0_FIRST" ]; then
    SI_PHASE=1
    SI_TRAIN_IDX=$(( idx / EVALS_PER_RUN ))
    sub=$(( idx % EVALS_PER_RUN ))
    SI_CFG=${NOISE_CONFIGS[$SI_TRAIN_IDX]}
    SI_CFG_DIR=diffusionsr/configs/si_noise
    SI_RUN_DIR=diffusionsr/runs/si_noise/$SI_CFG
    if   [ "$sub" -lt 6 ];  then SI_SOLVER=euler; SI_EPS=${EPS_LIST[$sub]}
    elif [ "$sub" -lt 12 ]; then SI_SOLVER=heun;  SI_EPS=${EPS_LIST[$(( sub - 6 ))]}
    else                         SI_SOLVER=rk4;   SI_EPS=0.0
    fi
  elif [ "$idx" -lt "$N_EVAL" ]; then
    SI_PHASE=0
    SI_TRAIN_IDX=p0
    sub=$(( idx - P0_FIRST )); n0=${#P0_EPS_LIST[@]}
    SI_CFG=$P0_CFG
    SI_CFG_DIR=diffusionsr/configs/si_schedules
    SI_RUN_DIR=diffusionsr/runs/si_schedules/$SI_CFG
    if [ "$sub" -lt "$n0" ]; then SI_SOLVER=euler; else SI_SOLVER=heun; fi
    SI_EPS=${P0_EPS_LIST[$(( sub % n0 ))]}
  else
    echo "decode_eval: index $idx out of range 0-$(( N_EVAL - 1 ))" >&2; return 1
  fi
}

# Deterministic names: no date stamps, so a requeue on a later day or on the OTHER cluster maps to the
# same W&B run and the same single artifact.
train_run_id()   { echo "$1-$SI_SCHED_TAG" | tr '[:upper:]_' '[:lower:]-'; }
ckpt_artifact()  { echo "${1}_ckpt"; }
