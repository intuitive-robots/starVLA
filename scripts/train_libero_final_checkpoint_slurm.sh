#!/bin/bash
#SBATCH --job-name=starvla_final_ckpt_policy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/starvla_final_ckpt_%j.out
#SBATCH --error=slurm_logs/starvla_final_ckpt_%j.err

# Resolve a timestamped train_downstream endpoint only after its dependency and
# checkpoint-load smoke have passed, then run the ordinary four-GPU policy job.
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "usage: $0 CHECKPOINT_ROOT WORLD_SIZE EXPECTED_STEP CHECKPOINT_FLAG CONFIG RUN_ID SEED [TRAIN_OVERRIDES...]" >&2
    exit 2
fi

checkpoint_root="$1"
world_size="$2"
expected_step="$3"
checkpoint_flag="$4"
config="$5"
run_id="$6"
seed="$7"
shift 7

repo=/e/project1/m3/blank4/code/starVLA
resolver=/e/project1/m3/blank4/code/train_downstream/scripts/latest_complete_checkpoint.py
python_runtime="$repo/scripts/env/starvla_python"

checkpoint_path="$($python_runtime "$resolver" "$checkpoint_root" "$world_size")"
checkpoint_step="$($python_runtime -c 'import json,sys; print(json.load(open(sys.argv[1]))["global_step"])' "$checkpoint_path/trainer_state.json")"
if [[ "$checkpoint_step" != "$expected_step" ]]; then
    echo "[FAIL] resolved checkpoint step $checkpoint_step, expected $expected_step: $checkpoint_path" >&2
    exit 1
fi

echo "[INFO] resolved endpoint checkpoint: step=$checkpoint_step path=$checkpoint_path"
cd "$repo"
exec bash train_libero_slurm.sh \
    --config "$config" \
    --run_id "$run_id" \
    --seed "$seed" \
    "$checkpoint_flag" "$checkpoint_path" \
    "$@"
