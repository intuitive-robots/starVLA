#!/bin/bash
#SBATCH --job-name=starvla_eval_vla_arena
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/eval_vla_arena_%j.out
#SBATCH --error=slurm_logs/eval_vla_arena_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288

# Parallel, optionally MULTI-NODE VLA-Arena evaluation. Each node runs its own
# self-contained copy of the single-node pipeline (policy servers + sim workers,
# all node-local) against a distinct slice of the (suite x level) work units;
# shards land in a shared output_dir and this script aggregates once at the end.
# Wraps examples/VLA-Arena/eval_files/parallel_eval/auto_eval_vla_arena.sh.
#
# Difference vs eval_libero_plus_slurm.sh: LIBERO-plus splits one suite by task
# index (thousands of perturbed instances). eval_vla_arena.py has no task-range
# flag and always runs a whole suite, so the partition unit here is
# (suite x level) -- 11 suites x 3 levels = 33 units by default.
#
# Usage:
#   sbatch eval_vla_arena_slurm.sh --ckpt /path/to/pytorch_model.pt
#   sbatch --nodes=4 eval_vla_arena_slurm.sh --ckpt ... --levels "0 1" --num_trials 10
#   sbatch --nodes=2 eval_vla_arena_slurm.sh --ckpt ... --suites "long_horizon safety_cautious_grasp"
#
# Node count is set at SUBMISSION time via `sbatch --nodes=N` (SBATCH directives
# are parsed before this script runs), same as the LIBERO-plus wrapper.
#
# Flags:
#   --ckpt <path>            path to pytorch_model.pt (required; or $your_ckpt)
#   --suites <list>          space-separated suites, or "all" (default: all)
#   --levels <list>          space-separated levels     (default: "0 1 2")
#   --num_gpus <n>           GPUs per node              (default: nvidia-smi -L)
#   --workers_per_gpu <n>    concurrent sim workers/GPU (default: 2)
#   --servers_per_gpu <n>    policy servers per GPU     (default: 1)
#   --num_trials <n>         rollouts per task          (default: 10)
#   --tasks_per_unit <n>     tasks per work unit        (default: 1 = finest;
#                            0 = one unit per whole suite x level)
#   --max_batch_size <n>     server request batching    (default: 32; 1 = off)
#   --max_wait_time <s>      batch fill timeout         (default: 1.0)
#   --save_video_mode <m>    all|first_success_failure|none
#   --overlay_trace          draw the generated 2D trace on saved videos
#   --gpu_ids <csv>          explicit GPU ids per node

set -euo pipefail

AUTO_EVAL_SCRIPT="./examples/VLA-Arena/eval_files/parallel_eval/auto_eval_vla_arena.sh"
AGGREGATE_SCRIPT="./examples/VLA-Arena/eval_files/parallel_eval/aggregate_vla_arena_results.py"

# ── VLA-Arena environment (edit if your paths differ) ───────────────────────
export VLA_ARENA_HOME="${VLA_ARENA_HOME:-/e/project1/m3/blank4/code/VLA-Arena}"
export VLA_ARENA_python="${VLA_ARENA_python:-/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/vla-arena/bin/python}"
# Required for headless MuJoCo rendering on compute nodes; the shipped
# eval_vla_arena.sh never sets it and rendering fails silently without it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export POLICY_SERVER_CONDA_ENV="${POLICY_SERVER_CONDA_ENV:-starVLA}"

CKPT="${your_ckpt:-}"
SUITES="${suites:-all}"
LEVELS="${levels:-0 1 2}"
NUM_TRIALS="${num_trials:-10}"
TASKS_PER_UNIT="${tasks_per_unit:-1}"
WORKERS_PER_GPU="${workers_per_gpu:-2}"
SERVERS_PER_GPU="${servers_per_gpu:-1}"
MAX_BATCH_SIZE="${max_batch_size:-32}"
MAX_WAIT_TIME="${max_wait_time:-1.0}"
SAVE_VIDEO_MODE="${save_video_mode:-first_success_failure}"
OVERLAY_TRACE="${overlay_trace:-false}"
GPU_IDS_CSV="${gpu_ids_csv:-}"
NUM_GPUS="${num_gpus:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --suites) SUITES="$2"; shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --workers_per_gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        --servers_per_gpu) SERVERS_PER_GPU="$2"; shift 2 ;;
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --tasks_per_unit) TASKS_PER_UNIT="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_wait_time) MAX_WAIT_TIME="$2"; shift 2 ;;
        --save_video_mode) SAVE_VIDEO_MODE="$2"; shift 2 ;;
        --overlay_trace) OVERLAY_TRACE="true"; shift 1 ;;
        --gpu_ids) GPU_IDS_CSV="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${CKPT}" ]; then echo "[ERROR] pass --ckpt /path/to/pytorch_model.pt"; exit 1; fi
if [ ! -f "${CKPT}" ]; then echo "[ERROR] checkpoint not found: ${CKPT}"; exit 1; fi
if [ ! -d "${VLA_ARENA_HOME}" ]; then echo "[ERROR] VLA_ARENA_HOME not found: ${VLA_ARENA_HOME}"; exit 1; fi
if [ ! -x "${VLA_ARENA_python}" ]; then echo "[ERROR] VLA_ARENA_python not executable: ${VLA_ARENA_python}"; exit 1; fi
your_ckpt="${CKPT}"

# Resolved once so every node and the final aggregation agree without coordinating.
if [ -z "${output_dir:-}" ]; then
    output_dir="$(dirname "$(dirname "${your_ckpt}")")/results/vla-arena"
fi
mkdir -p "${output_dir}" slurm_logs

ml load CUDA
source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh
set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_NODES="${SLURM_NNODES:-1}"

echo "=========================================="
echo " VLA-Arena SLURM Eval (multi-node)"
echo "=========================================="
echo " Job ID:         ${SLURM_JOB_ID:-local}"
echo " Nodes:          ${NUM_NODES}"
echo " Checkpoint:     ${your_ckpt}"
echo " Output dir:     ${output_dir}"
echo " Suites:         ${SUITES}"
echo " Levels:         ${LEVELS}"
echo " GPUs/node:      ${NUM_GPUS:-<auto>}"
echo " Workers/GPU:    ${WORKERS_PER_GPU}   Servers/GPU: ${SERVERS_PER_GPU}"
echo " Trials/task:    ${NUM_TRIALS}   Tasks/unit: ${TASKS_PER_UNIT}"
echo " Max batch:      ${MAX_BATCH_SIZE} (max_wait_time=${MAX_WAIT_TIME}s)"
echo " VLA_ARENA_HOME: ${VLA_ARENA_HOME}"
echo " MUJOCO_GL:      ${MUJOCO_GL}"
echo "=========================================="

export your_ckpt output_dir
export SUITE_SUITES="${SUITES}" SUITE_LEVELS="${LEVELS}"
export SUITE_WORKERS_PER_GPU="${WORKERS_PER_GPU}" SUITE_SERVERS_PER_GPU="${SERVERS_PER_GPU}"
export SUITE_NUM_TRIALS="${NUM_TRIALS}" SUITE_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"
export SUITE_TASKS_PER_UNIT="${TASKS_PER_UNIT}"
export SUITE_MAX_WAIT_TIME="${MAX_WAIT_TIME}" SUITE_SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE}"
export SUITE_OVERLAY_TRACE="${OVERLAY_TRACE}"
export SUITE_GPU_IDS_CSV="${GPU_IDS_CSV}" SUITE_NUM_GPUS="${NUM_GPUS}"
export SUITE_NUM_NODES="${NUM_NODES}" SUITE_AUTO_EVAL_SCRIPT="${AUTO_EVAL_SCRIPT}"

# One SLURM task per node runs the whole single-node pipeline against its own
# partition_idx=$SLURM_PROCID slice. Inlined as a plain command string (not
# `export -f`) for the same reason as the LIBERO-plus wrapper: some clusters
# strip BASH_FUNC_* env vars across an srun hop.
NODE_CMD='
    node_num_gpus="${SUITE_NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
    echo "[node ${SLURM_PROCID}/$((SUITE_NUM_NODES - 1)) $(hostname)] gpus=${node_num_gpus}"
    SKIP_AGGREGATE=true \
        suites="${SUITE_SUITES}" levels="${SUITE_LEVELS}" \
        num_gpus="${node_num_gpus}" gpu_ids_csv="${SUITE_GPU_IDS_CSV}" \
        workers_per_gpu="${SUITE_WORKERS_PER_GPU}" servers_per_gpu="${SUITE_SERVERS_PER_GPU}" \
        num_trials="${SUITE_NUM_TRIALS}" tasks_per_unit="${SUITE_TASKS_PER_UNIT}" \
        max_batch_size="${SUITE_MAX_BATCH_SIZE}" max_wait_time="${SUITE_MAX_WAIT_TIME}" \
        save_video_mode="${SUITE_SAVE_VIDEO_MODE}" overlay_trace="${SUITE_OVERLAY_TRACE}" \
        partition_idx="${SLURM_PROCID}" num_partitions="${SUITE_NUM_NODES}" \
        bash "${SUITE_AUTO_EVAL_SCRIPT}"
'

srun --ntasks="${NUM_NODES}" --ntasks-per-node=1 bash -c "${NODE_CMD}"

echo "All ${NUM_NODES} partition(s) finished. Aggregating..."
python "${AGGREGATE_SCRIPT}" --root_path "${output_dir}"
echo "Combined results: ${output_dir}/overall_results.json"
