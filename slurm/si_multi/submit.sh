#!/bin/bash
# Submit one phase of the SI multi-field / regen / FM study, from the repo root on TRACE.
#
#   bash slurm/si_multi/submit.sh train  <phase> [kind]   # rows of train.tsv in that phase (optionally one kind)
#   bash slurm/si_multi/submit.sh eval   <phase> [name-regex]
#   bash slurm/si_multi/submit.sh cascade
#   DEP=<jobid> bash slurm/si_multi/submit.sh ...         # start only after job <jobid> succeeded
#   DRY=1 ...                                             # print the sbatch line only
#
# Phases: 1 encoders | 2 old data (O1 extras, O2, FM on O1/O2) | 3 regen direct (N1, N2, FM) | 4 cascades.
# The array indices are 0-based row numbers of the TSV (header excluded). Max 20 array tasks at once (%20).
set -eo pipefail
WHAT=$1; PHASE=$2; FILTER=${3:-}
case "$WHAT" in
  train)   TSV=slurm/si_multi/train.tsv;   SCRIPT=slurm/si_multi/01_train.sh; PCOL=7; FCOL=1 ;;
  eval)    TSV=slurm/si_multi/eval.tsv;    SCRIPT=slurm/si_multi/02_eval.sh;  PCOL=13; FCOL=1 ;;
  cascade) TSV=slurm/si_multi/cascade.tsv; SCRIPT=slurm/si_multi/03_cascade.sh; PCOL=14; FCOL=1; PHASE=${PHASE:-4} ;;
  *) echo "usage: $0 train|eval|cascade <phase> [filter]"; exit 1 ;;
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
