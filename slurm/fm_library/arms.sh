#!/bin/bash
# FM-library study (Experiment 2 of docs/plans/2026-11_si_multifield_fm_plan.md): the single index map.
# Sourced by 01_train_array.sh. Array index -> training config (diffusionsr/configs/fm_library/<cfg>.yml).
#
#   idx  config                    start (fm_source)              fields                     data
#   0    fm_noise_enc_temp         N(0, I)                        temperature                old, 2x
#   1    fm_paired_enc_temp        enc(LR) + 0.1*zeta (as SI)     temperature                old, 2x
#   2    fm_noise_enc_tempsdf      N(0, I)                        temperature + sdfliqlabel  old, 2x
#   3    fm_paired_enc_tempsdf     enc(LR) + 0.1*zeta (as SI)     temperature + sdfliqlabel  old, 2x
#
# The SI-FM comparison arm is configs/si_schedules/si_sched_linear_enc_temp (eps = 0); its multi-field
# and new-dataset counterparts come from Experiment 1. Tasks 2-3 need the 2-field encoder named in their
# configs (encoder_results_dir) to exist first.

FM_CONFIGS=(fm_noise_enc_temp fm_paired_enc_temp fm_noise_enc_tempsdf fm_paired_enc_tempsdf)
N_TRAIN=${#FM_CONFIGS[@]}

export SI_SCHED_TAG="${SI_SCHED_TAG:-v1}"
export SI_WANDB_GROUP=fm_library        # read by ScheduledInterpolantModel._wandb_init

train_run_id()   { echo "$1-$SI_SCHED_TAG" | tr '[:upper:]_' '[:lower:]-'; }
ckpt_artifact()  { echo "${1}_ckpt"; }
# encoder_dir CFG_PATH -> the config's encoder_results_dir (each config owns its frozen encoder)
encoder_dir()    { python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['encoder_results_dir'])" "$1"; }
