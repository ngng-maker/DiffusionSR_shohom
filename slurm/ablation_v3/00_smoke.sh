#!/bin/bash
# Ablation v3 smoke test, from the repo root on TRACE: 1-epoch versions of one run of every kind
# (shared encoders, shared VAE, DSR with and without encoder, FM with and without encoder, LDM width 48),
# then their evals on the first 8 test frames with K = 2 and no W&B upload. Writes only *_smoke directories.
#
#   python scripts/gen_ablation_v3.py --smoke && bash slurm/ablation_v3/00_smoke.sh
#   afterwards: rm -rf diffusionsr/runs/ablation_v3_smoke eval_results/ablation_v3_smoke
set -eo pipefail
T=slurm/ablation_v3/train_smoke.tsv
E=slurm/ablation_v3/eval_smoke.tsv
rows() { awk -F'\t' -v p="$2" 'NR > 1 && $NF == p {printf "%s%d", (n++ ? "," : ""), NR - 2}' "$1"; }
J1=$(sbatch --parsable --time=04:00:00 --array="$(rows $T 1)" --export=ALL,TSV=$T slurm/ablation_v3/01_train.sh)
J2=$(sbatch --parsable --time=04:00:00 --dependency=afterok:$J1 --array="$(rows $T 2)" --export=ALL,TSV=$T \
     slurm/ablation_v3/01_train.sh)
J3=$(sbatch --parsable --time=04:00:00 --dependency=afterok:$J2 --array="$(rows $E 3)" \
     --export="ALL,TSV=$E,SI_WANDB_GROUP=ablation_v3_smoke,EVAL_ARGS=--max_frames 8 --k_samples 2 --no_upload" \
     slurm/ablation_v3/02_eval.sh)
echo "smoke: phase 1 $J1, phase 2 $J2, evals $J3"
