#!/usr/bin/env bash
# Print the live status of every job listed in JOBS.md, plus anything in the queue
# that JOBS.md does not know about (which means somebody launched without recording it).
#
#   bash scripts/jobs_status.sh            # table for the current user
#   bash scripts/jobs_status.sh --since 2026-09-16   # also show finished/failed since a date
set -uo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
JOBS_MD=${JOBS_MD:-$REPO/JOBS.md}
SINCE=""
[[ ${1:-} == --since ]] && SINCE=${2:-}

printf '%-9s %-34s %-16s %-10s %s\n' JOBID NAME STATE ELAPSED "NODE/REASON"
printf '%-9s %-34s %-16s %-10s %s\n' --------- ---------------------------------- ---------------- ---------- -----------
squeue -h -u "$USER" -o "%i|%j|%T|%M|%R" | while IFS='|' read -r id name state elapsed reason; do
  printf '%-9s %-34s %-16s %-10s %s\n' "$id" "$name" "$state" "$elapsed" "$reason"
done

if [[ -n $SINCE ]]; then
  echo
  echo "Finished since ${SINCE} (non-COMPLETED first):"
  sacct -u "$USER" -S "$SINCE" -X --format=JobID,JobName%34,State%22,Elapsed,ExitCode \
    | grep -vE "PENDING|RUNNING" | grep -vE "^JobID|^---" \
    | sort -k3 | sed 's/^/  /'
fi

# Jobs in the queue that JOBS.md does not mention -- these are untracked.
if [[ -f $JOBS_MD ]]; then
  untracked=$(comm -23 \
    <(squeue -h -u "$USER" -o "%i" | sort -u) \
    <(grep -oE '\b18[0-9]{5}\b' "$JOBS_MD" | sort -u))
  if [[ -n $untracked ]]; then
    echo
    echo "!! In the queue but NOT in JOBS.md (record them with their launch command):"
    for j in $untracked; do
      printf '   %s  %s\n' "$j" "$(squeue -h -j "$j" -o '%j')"
    done
  fi
fi
