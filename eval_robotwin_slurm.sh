#!/bin/bash
#SBATCH --job-name=starvla_eval_robotwin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:59:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/eval_robotwin_%j.out
#SBATCH --error=slurm_logs/eval_robotwin_%j.err

# Evaluate one StarVLA checkpoint on RoboTwin. start_eval.sh assigns one
# policy-server/simulator pair to each visible GPU and dynamically schedules
# the task list over those four slots.
#
# Usage:
#   sbatch eval_robotwin_slurm.sh --ckpt /abs/path/model.pt --name run_name \
#       --mode demo_clean --num-trials 5 --tasks all

set -euo pipefail

STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
ROBOTWIN_REPO=/e/project1/m3/blank4/code/RoboTwin
STARVLA_PY=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/starVLA/bin/python
ROBOTWIN_PY=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/RoboTwin/bin/python

CKPT=""
NAME=""
MODE="demo_clean"
NUM_TRIALS=5
JOBS_PER_GPU=1
TASKS="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num-trials) NUM_TRIALS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[ERROR] --ckpt must name an existing checkpoint: $CKPT" >&2
    exit 2
fi
if [[ -z "$NAME" ]]; then
    echo "[ERROR] --name is required" >&2
    exit 2
fi

# Child launchers change into both repositories.  Resolve the checkpoint and
# derived log directory once so those paths remain valid after each cd.
CKPT="$(realpath "$CKPT")"
if [[ ! -x "$STARVLA_PY" || ! -x "$ROBOTWIN_PY" ]]; then
    echo "[ERROR] StarVLA or RoboTwin Python environment is missing" >&2
    exit 2
fi

cd "$STARVLA_REPO"
ml load CUDA
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

preflight_ok=1
for path in "$STARVLA_REPO" "$ROBOTWIN_REPO" "$CKPT" "$STARVLA_PY" "$ROBOTWIN_PY"; do
    stat "$path" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: cannot stat $path on $(hostname)"; preflight_ok=0; }
done
"$STARVLA_PY" -c "import torch, transformers, websockets" >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: StarVLA imports on $(hostname)"; preflight_ok=0; }
(
    cd "$ROBOTWIN_REPO"
    PYTHONPATH="$STARVLA_REPO:$STARVLA_REPO/examples/Robotwin/eval_files:${PYTHONPATH:-}" \
        "$ROBOTWIN_PY" -c "import sapien, mplib, torch, curobo; from envs.robot.planner import CuroboPlanner; import model2robotwin_interface"
) >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: RoboTwin imports on $(hostname)"; preflight_ok=0; }
if [[ "$preflight_ok" -ne 1 ]]; then
    exit 1
fi

export ROBOTWIN_PATH="$ROBOTWIN_REPO"
export STARVLA_PYTHON="$STARVLA_PY"
export ROBOTWIN_PYTHON="$ROBOTWIN_PY"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_TORCH=1
export ROBOTWIN_LOG_ROOT="$(dirname "$CKPT")/robotwin_eval_logs/${NAME}_${MODE}_${NUM_TRIALS}trials_job${SLURM_JOB_ID}"

echo "preflight OK on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint: $CKPT"
echo "Mode: $MODE"
echo "Trials/task: $NUM_TRIALS"
echo "Tasks: $TASKS"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Jobs/GPU: $JOBS_PER_GPU"
echo "Logs: $ROBOTWIN_LOG_ROOT"

bash examples/Robotwin/eval_files/start_eval.sh \
    --mode "$MODE" \
    --name "$NAME" \
    --ckpt "$CKPT" \
    --num-trials "$NUM_TRIALS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    "$TASKS"
