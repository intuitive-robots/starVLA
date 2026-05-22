#!/bin/bash
# Submit all CoT experiment configs as SLURM jobs.
# Job name = config yaml stem (no extension).
# Usage: bash submit_cot_experiments.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

EXPERIMENT_DIR="examples/LIBERO/train_files/experiments_real_robot"

for cfg in "$EXPERIMENT_DIR"/*.yaml; do
  job_name="$(basename "$cfg" .yaml)"
  cmd=(sbatch --job-name="$job_name" train_libero_slurm.sh --config "$cfg")
  echo "  ${cmd[*]}"
  if [[ $DRY_RUN -eq 0 ]]; then
    "${cmd[@]}"
  fi
done
