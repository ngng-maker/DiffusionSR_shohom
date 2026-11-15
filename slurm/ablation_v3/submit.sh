#!/bin/bash
# Submit part of ablation v3, from the repo root on TRACE.
#
#   bash slurm/ablation_v3/submit.sh train <phase> [name-regex]   # 1 = shared encoders + VAEs, 2 = models
#   bash slurm/ablation_v3/submit.sh eval  3 [name-regex]
#   DEP=<jobid> bash slurm/ablation_v3/submit.sh ...              # start only after job <jobid> succeeded
#   DRY=1 ...                                                     # print the sbatch line only
#
# Array indices are 0-based TSV rows (header excluded); at most 20 tasks run at once (%20).
set -eo pipefail
WHAT=$1; PHASE=$2; FILTER=${3:-}
case "$WHAT" in
  train) TSV=slurm/ablation_v3/train.tsv; SCRIPT=slurm/ablation_v3/01_train.sh; PCOL=7; FCOL=2 ;;
  eval)  TSV=slurm/ablation_v3/eval.tsv;  SCRIPT=slurm/ablation_v3/02_eval.sh;  PCOL=13; FCOL=1 ;;
  *) echo "usage: $0 train|eval <phase> [name-regex]"; exit 1 ;;
esac
IDX=$(awk -F'\t' -v p="$PHASE" -v pc=$PCOL -v fc=$FCOL -v f="$FILTER" \
      'NR > 1 && $pc == p && (f == "" || $fc ~ f) {printf "%s%d", (n++ ? "," : ""), NR - 2}' "$TSV")
[ -n "$IDX" ] || { echo "no rows for $WHAT phase $PHASE ${FILTER:+filter $FILTER}"; exit 1; }
N=$(echo "$IDX" | tr ',' '\n' | wc -l)
CMD=(sbatch --array="${IDX}%20")
[ -n "${DEP:-}" ] && CMD+=(--dependency=afterok:$DEP)
CMD+=("$SCRIPT")
echo "$WHAT phase $PHASE ${FILTER:+($FILTER) }: $N tasks"
if [ -n "${DRY:-}" ]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
