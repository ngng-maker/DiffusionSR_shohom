#!/bin/bash
# Submit one phase of the SI multi-field / regen / FM study, from the repo root on TRACE.
#
#   bash slurm/si_multi/submit.sh train  <phase> [kind]   # rows of train.tsv in that phase (optionally one kind)
#   bash slurm/si_multi/submit.sh eval   <phase> [name-regex]
#   bash slurm/si_multi/submit.sh cascade
#   DEP=<jobid> bash slurm/si_multi/submit.sh ...         # start only after job <jobid> succeeded
#   DRY=1 ...                                             # print the sbatch line only
#   EXCLUDE=<regex> / THROTTLE=<n> ...                    # skip rows by name / max concurrent tasks (default 20)
#
# Phases: 1 encoders | 2 old data (O1 extras, O2, FM on O1/O2) | 3 regen direct (N1, N2, FM) | 4 cascades.
# Array index + IDX_OFFSET = 0-based row number of the TSV (header excluded). Max 20 array tasks at once (%20).
set -eo pipefail
WHAT=$1; PHASE=$2; FILTER=${3:-}
case "$WHAT" in
  train)   TSV=slurm/si_multi/train.tsv;   SCRIPT=slurm/si_multi/01_train.sh; PCOL=7; FCOL=1 ;;
  eval)    TSV=slurm/si_multi/eval.tsv;    SCRIPT=slurm/si_multi/02_eval.sh;  PCOL=13; FCOL=1 ;;
  cascade) TSV=slurm/si_multi/cascade.tsv; SCRIPT=slurm/si_multi/03_cascade.sh; PCOL=14; FCOL=1; PHASE=${PHASE:-4} ;;
  *) echo "usage: $0 train|eval|cascade <phase> [filter]"; exit 1 ;;
esac
# Optional EXCLUDE=<regex> drops rows whose name matches (e.g. a model still training; submit those with DEP).
IDXS=$(awk -F'\t' -v p="$PHASE" -v pc=$PCOL -v fc=$FCOL -v f="$FILTER" -v x="${EXCLUDE:-}" \
      'NR > 1 && $pc == p && (f == "" || $fc ~ f) && (x == "" || $fc !~ x) {print NR - 2}' "$TSV")
[ -n "$IDXS" ] || { echo "no rows for $WHAT phase $PHASE ${FILTER:+filter $FILTER}"; exit 1; }
echo "$WHAT phase $PHASE ${FILTER:+($FILTER) }${EXCLUDE:+(excluding $EXCLUDE) }: $(echo "$IDXS" | wc -l) tasks"
# Slurm rejects array indices >= MaxArraySize (1001 on TRACE): one sbatch per block of 1000 rows, with the block
# start passed as IDX_OFFSET (02_eval.sh / 03_cascade.sh add it back to SLURM_ARRAY_TASK_ID).
for OFF in $(echo "$IDXS" | awk '{print int($1 / 1000) * 1000}' | sort -nu); do
  IDX=$(echo "$IDXS" | awk -v o=$OFF 'int($1 / 1000) * 1000 == o {printf "%s%d", (n++ ? "," : ""), $1 - o}')
  # the total throttle (default 20) is shared between blocks in proportion to their size
  NB=$(echo "$IDX" | tr ',' '\n' | wc -l); NT=$(echo "$IDXS" | wc -l)
  T=$(( (${THROTTLE:-20} * NB + NT - 1) / NT ))
  CMD=(sbatch --array="${IDX}%${T}" --export=ALL,IDX_OFFSET=$OFF)
  [ -n "${DEP:-}" ] && CMD+=(--dependency=afterok:$DEP)
  CMD+=("$SCRIPT")
  echo "  rows $OFF + [$NB indices, max $T at once]"
  if [ -n "${DRY:-}" ]; then echo "  ${CMD[*]}"; else "${CMD[@]}"; fi
done
