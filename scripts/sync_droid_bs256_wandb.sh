#!/usr/bin/env bash
# Consolidate the controlled DROID batch-256 Slurm continuations into one
# W&B run per experiment. With job IDs as arguments, wait until every chain
# has terminated before taking the final, immutable snapshots.

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if (( $# > 0 )); then
    wait_job_ids=("$@")
    while true; do
        jobs_active=false
        for job_id in "${wait_job_ids[@]}"; do
            job_state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)"
            if [[ "$job_state" =~ ^(PENDING|RUNNING|CONFIGURING|COMPLETING)$ ]]; then
                jobs_active=true
                break
            fi
        done
        "$jobs_active" || break
        sleep 300
    done
fi

wandb_entity="niblank"
wandb_project="starvla_real_robot"
wandb_group="droid_pi05like_bs256_controlled"

run_dirs=(
    playground/Checkpoints/droid_pi05like_rel_eef_encoder_nocot_h16_100k
    playground/Checkpoints/droid_pi05like_abs_joint_encoder_state_drop05_nocot_h16_100k
    playground/Checkpoints/droid_pi05like_rel_eef_nostate_causal_bs256_fresh_100k
    playground/Checkpoints/droid_pi05like_abs_joint_state_drop05_causal_bs256_fresh_100k
)

for run_dir in "${run_dirs[@]}"; do
    canonical_id="$(<"$run_dir/.wandb_run_id")"
    echo "Consolidating $run_dir into W&B run $canonical_id"
    while IFS= read -r stream; do
        wandb sync \
            --entity "$wandb_entity" \
            --project "$wandb_project" \
            --id "$canonical_id" \
            --append \
            --include-synced \
            --no-mark-synced \
            "$stream"
    done < <(find "$run_dir/wandb/wandb" -mindepth 2 -maxdepth 2 -type f -name 'run-*.wandb' | sort)
done

WANDB_ENTITY="$wandb_entity" WANDB_PROJECT="$wandb_project" WANDB_GROUP="$wandb_group" \
python - <<'PY'
import os
from pathlib import Path

import wandb

api = wandb.Api(timeout=30)
for id_path in Path("playground/Checkpoints").glob("droid_pi05like*/.wandb_run_id"):
    group_path = id_path.with_name(".wandb_group")
    if not group_path.exists() or group_path.read_text().strip() != os.environ["WANDB_GROUP"]:
        continue
    run_id = id_path.read_text().strip()
    run = api.run(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{run_id}")
    run.group = os.environ["WANDB_GROUP"]
    run.update()
    print(f"{run_id}: step={run.lastHistoryStep} {run.url}")
PY
