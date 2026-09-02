#!/bin/bash
#SBATCH --job-name=starvla_vlabench_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:29:00
#SBATCH --output=slurm_logs/vlabench_%j.out
#SBATCH --error=slurm_logs/vlabench_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --exclude=jpbo-028-25

set -euo pipefail

STARVLA=/e/project1/m3/blank4/code/starVLA
VLABENCH=/e/project1/m3/blank4/code/VLABench
SERVER_PY=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
EVAL_PY=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/vlabench/bin/python
TRACK=track_1_in_distribution
NUM_EPISODES=10
BASE_PORT=10400
CKPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --track) TRACK="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
test -n "$CKPT" && test -f "$CKPT"
cd "$STARVLA"

tasks=(select_painting select_book select_drink select_chemistry_tube select_poker select_mahjong select_toy select_fruit add_condiment insert_flower)
RUN_ID=$(basename "$(dirname "$(dirname "$CKPT")")")
OUT=$(dirname "$(dirname "$CKPT")")/vlabench_eval/$TRACK
mkdir -p "$OUT"

export USE_TF=0 USE_TORCH=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLABENCH_ROOT="$VLABENCH/VLABench"

# One model server per GH200. The four server processes are kept in a separate Slurm step;
# simulator workers connect to the server assigned by task index modulo four.
srun --overlap --exact --ntasks=4 --cpus-per-task=8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 \
  bash -c '
    port=$((10400 + SLURM_LOCALID))
    exec "$0" deployment/model_server/server_policy.py --ckpt_path "$1" --port "$port" \
      --use_bf16 --idle_timeout -1 --max_batch_size 4 --max_wait_time 0.02
  ' "$SERVER_PY" "$CKPT" &
SERVER_STEP=$!
cleanup() { kill "$SERVER_STEP" 2>/dev/null || true; wait "$SERVER_STEP" 2>/dev/null || true; }
trap cleanup EXIT

TASKS_CSV=$(IFS=,; echo "${tasks[*]}")
TASKS_CSV="$TASKS_CSV" OUT="$OUT" TRACK="$TRACK" NUM_EPISODES="$NUM_EPISODES" \
STARVLA="$STARVLA" VLABENCH="$VLABENCH" EVAL_PY="$EVAL_PY" \
srun --overlap --exact --ntasks=10 --cpus-per-task=24 --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  IFS=, read -r -a tasks <<< "$TASKS_CSV"
  task="${tasks[$SLURM_LOCALID]}"
  gpu=$((SLURM_LOCALID % 4))
  port=$((10400 + gpu))
  export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export PYTHONPATH="$STARVLA:$VLABENCH:${PYTHONPATH:-}"
  "$EVAL_PY" "$STARVLA/examples/VLABENCH/eval_files/eval_starvla_vlabench.py" \
    --port "$port" --track "$TRACK" --tasks "$task" --num-episodes "$NUM_EPISODES" \
    --output "$OUT/$task"
'

cleanup
trap - EXIT
"$EVAL_PY" examples/VLABENCH/eval_files/aggregate_vlabench.py \
  --root "$OUT" --tasks "${tasks[@]}" --num-episodes "$NUM_EPISODES"
echo "VLABench evaluation complete: run=$RUN_ID result=$OUT/summary.json"
