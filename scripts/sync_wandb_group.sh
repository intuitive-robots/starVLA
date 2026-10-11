#!/usr/bin/env bash
# Periodically push a group's OFFLINE W&B runs to the server and keep them in one group.
#
# Group-agnostic: WANDB_GROUP_ selects which runs this instance owns, so one copy of this
# script serves every experiment family and several instances can watch different groups
# side by side.
#
# train_libero_slurm.sh sets WANDB_MODE=offline (compute nodes have no route to
# wandb.ai), so nothing appears online until something syncs it. This is that something.
#
# Each run directory carries a canonical .wandb_run_id written at first init, and a
# .wandb_group pre-seeded by launch_piv4_imgdrop_libero.sh. Every offline stream in the
# run -- there is one per SLURM segment, so a 12h-wall continuation adds another -- is
# appended into that single canonical run id, which is what keeps a resumed arm as one
# W&B run instead of several unrelated ones.
#
# Usage
#   WANDB_GROUP_=<group> bash scripts/sync_wandb_group.sh                  # one pass
#   WANDB_GROUP_=<group> bash scripts/sync_wandb_group.sh --watch <jobids> # loop until
#                          those jobs end, then take a final snapshot
# Typically:
#   WANDB_GROUP_=piv4_imgdrop_libero nohup bash scripts/sync_wandb_group.sh \
#     --watch 1879710 ... > slurm_logs/wandb_sync_piv4_imgdrop.log 2>&1 &
#
# A run that was already past wandb.init when its .wandb_group file was written keeps its
# original (or absent) group locally; fix_groups below sets it server-side via the API, so
# grouping an already-running experiment retroactively is supported. Its next resumed
# segment then inits with the right group from the file.
#
# Runs on a LOGIN node: it needs outbound network, which the compute nodes do not have.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

WANDB_ENTITY_="${WANDB_ENTITY_:-niblank}"
WANDB_PROJECT_="${WANDB_PROJECT_:-starvla_libero}"
WANDB_GROUP_="${WANDB_GROUP_:?set WANDB_GROUP_ to the group this instance owns}"
INTERVAL="${INTERVAL:-900}"          # 15 min; these runs log every 100 steps
CKPT_ROOT=playground/Checkpoints

WATCH=0
JOB_IDS=()
if [[ "${1:-}" == "--watch" ]]; then
  WATCH=1; shift; JOB_IDS=("$@")
fi

# Every run directory whose .wandb_group matches ours. Discovering by the group file
# rather than a hardcoded list means an arm added later is picked up with no edit here.
run_dirs() {
  find "$CKPT_ROOT" -mindepth 2 -maxdepth 2 -name .wandb_group -print0 2>/dev/null \
    | while IFS= read -r -d '' f; do
        [[ "$(<"$f")" == "$WANDB_GROUP_" ]] && dirname "$f"
      done
}

sync_once() {
  local n=0
  while IFS= read -r run_dir; do
    [[ -f "$run_dir/.wandb_run_id" ]] || { echo "  skip $run_dir (no .wandb_run_id yet)"; continue; }
    local canonical; canonical="$(<"$run_dir/.wandb_run_id")"
    local streams; streams=$(find "$run_dir/wandb" -mindepth 2 -type f -name 'run-*.wandb' 2>/dev/null | sort)
    [[ -n "$streams" ]] || { echo "  skip $run_dir (no offline stream yet)"; continue; }
    echo "  $run_dir -> $canonical"
    while IFS= read -r stream; do
      # --append + --include-synced + --no-mark-synced: re-sending an already-sent stream
      # is how a still-growing segment keeps catching up without being marked done.
      wandb sync \
        --entity "$WANDB_ENTITY_" --project "$WANDB_PROJECT_" \
        --id "$canonical" --append --include-synced --no-mark-synced \
        "$stream" 2>&1 | sed 's/^/    /'
    done <<< "$streams"
    n=$((n+1))
  done < <(run_dirs)
  echo "  synced $n run(s)"
}

# The group is set at wandb.init from .wandb_group, but a run created before that file
# existed (or by a path that ignored it) would sit outside the group; fix it server-side.
fix_groups() {
  WANDB_ENTITY_="$WANDB_ENTITY_" WANDB_PROJECT_="$WANDB_PROJECT_" WANDB_GROUP_="$WANDB_GROUP_" \
  python - <<'PY' 2>&1 | sed 's/^/  /'
import os
from pathlib import Path
try:
    import wandb
except ImportError:
    raise SystemExit("wandb not importable here; skipping group fixup")
api = wandb.Api(timeout=30)
entity, project = os.environ["WANDB_ENTITY_"], os.environ["WANDB_PROJECT_"]
group = os.environ["WANDB_GROUP_"]
for gpath in Path("playground/Checkpoints").glob("*/.wandb_group"):
    if gpath.read_text().strip() != group:
        continue
    idpath = gpath.with_name(".wandb_run_id")
    if not idpath.exists():
        continue
    rid = idpath.read_text().strip()
    try:
        run = api.run(f"{entity}/{project}/{rid}")
    except Exception as exc:
        print(f"{rid}: not on the server yet ({type(exc).__name__})")
        continue
    if run.group != group:
        run.group = group
        run.update()
    print(f"{rid}: group={run.group} step={run.lastHistoryStep} {run.url}")
PY
}

jobs_active() {
  local id state
  for id in "${JOB_IDS[@]}"; do
    state="$(squeue -h -j "$id" -o '%T' 2>/dev/null || true)"
    [[ "$state" =~ ^(PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED)$ ]] && return 0
  done
  return 1
}

echo "=== W&B sync: group=$WANDB_GROUP_ project=$WANDB_PROJECT_ interval=${INTERVAL}s ==="
while :; do
  echo "--- $(date -Is)"
  sync_once
  fix_groups
  if (( WATCH == 0 )); then
    break
  fi
  if ! jobs_active; then
    echo "--- all watched jobs ended; taking the final snapshot"
    sync_once
    fix_groups
    break
  fi
  sleep "$INTERVAL"
done
echo "=== done $(date -Is) ==="
