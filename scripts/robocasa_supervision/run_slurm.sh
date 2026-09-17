#!/usr/bin/env bash
#SBATCH --job-name=rc_supervision
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/rc_supervision_%j.out
#SBATCH --error=slurm_logs/rc_supervision_%j.err
set -euo pipefail
REPO=/e/project1/m3/blank4/code/starVLA-upstream-merge
IMAGE=/e/project1/m3/blank4/code/starVLA/playground/sims/sif/robocasa365-main-arm64.sif
ROOT=/e/scratch/m3/blank4/rc365_supervision
MODE=${1:-pilot}
WORKERS=${2:-4}
OUT=${3:-$ROOT/pilot_labels_v1}
mkdir -p "$OUT"
cd "$REPO"
pids=()
for ((rank=0; rank<WORKERS; rank++)); do
    gpu=$((rank % 4))
    global_rank=$(( ${SHARD_GROUP:-0} * WORKERS + rank ))
    total_shards=$(( ${SHARD_GROUPS:-1} * WORKERS ))
    extra=()
    if [[ "$MODE" == pilot ]]; then extra+=(--pilot --overlays); fi
    if [[ -n ${EPISODE_IDS:-} ]]; then extra+=(--episode-ids "$EPISODE_IDS"); fi
    (
      export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      apptainer exec --nv --bind /e:/e "$IMAGE" python -u scripts/robocasa_supervision/generate.py \
        --manifest /e/project1/m3/blank4/code/starVLA/results_collected/robocasa_supervision/episode_sources.csv \
        --root "$ROOT" --dataset /e/scratch/m3/datasets/lerobot_3_0/robocasa365_target_atomic \
        --output "$OUT" --shard "$global_rank" --num-shards "$total_shards" "${extra[@]}" \
        > "$OUT/worker_${SLURM_JOB_ID}_${global_rank}.log" 2>&1
    ) &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
