#!/bin/bash
#SBATCH --job-name=droid_relq_stats
#SBATCH --account=m3
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=60
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/droid_relq_stats_%j.out
#SBATCH --error=slurm_logs/droid_relq_stats_%j.err

set -euo pipefail

STARVLA_REPO="${STARVLA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/e/scratch/m3/datasets/lerobot_3_0_transcoded/DROID/droid_success_180x320}"
PARTIAL_DIR="${PARTIAL_DIR:-/e/scratch/m3/blank4/droid_relative_joint_stats/${SLURM_JOB_ID}}"
OUTPUT="${OUTPUT:-$STARVLA_REPO/examples/DROID/stats/droid_dreamzero_relative_joint_abs_gripper.json}"

mkdir -p "$STARVLA_REPO/slurm_logs" "$PARTIAL_DIR"
export STARVLA_REPO DATA_ROOT PARTIAL_DIR OUTPUT

srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores --kill-on-bad-exit=1 bash -c '
        set -eo pipefail
        source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
        conda activate starVLA
        set -u
        cd "$STARVLA_REPO"
        python scripts/compute_droid_relative_joint_stats.py \
            --mode shard --data-root "$DATA_ROOT" --partial-dir "$PARTIAL_DIR" \
            --num-shards 4 --shard-id "$SLURM_PROCID" --horizon 16
    '

srun --ntasks=1 --nodes=1 --gpus-per-task=1 --cpus-per-task=8 --cpu-bind=cores bash -c '
    set -eo pipefail
    source /e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/etc/profile.d/conda.sh
    conda activate starVLA
    set -u
    cd "$STARVLA_REPO"
    python scripts/compute_droid_relative_joint_stats.py \
        --mode merge --data-root "$DATA_ROOT" --partial-dir "$PARTIAL_DIR" \
        --num-shards 4 --horizon 16 --output "$OUTPUT"
    python -m json.tool "$OUTPUT" >/dev/null
    echo "STATS_PASS output=$OUTPUT"
'
