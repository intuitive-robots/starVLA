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
#
# Unit mode (2026-10-07, FLEX-pi 5-task pilot): --seed-groups K > 0 runs every
# task x mode (--modes, comma list) x RoboTwin seed group 0..K-1 as one unit of
# --num-trials episodes, all through the same per-GPU policy servers, longest
# tasks first as given in --tasks; aggregate_units.py sums the K groups per task.
#   sbatch eval_robotwin_slurm.sh --ckpt ... --name ... --modes demo_clean,demo_randomized \
#       --tasks stack_bowls_two,place_object_basket,... --seed-groups 4 --num-trials 25

set -euo pipefail

STARVLA_REPO=/e/project1/m3/blank4/code/starVLA
STARVLA_PY="$STARVLA_REPO/scripts/env/starvla_python"
ROBOTWIN_PY="$STARVLA_REPO/scripts/env/robotwin_python"

CKPT=""
NAME=""
MODE="demo_clean"
NUM_TRIALS=5
JOBS_PER_GPU=4
TASKS="all"
ACTION_MODE="auto"
TASK_SHARD_INDEX=0
TASK_SHARD_COUNT=1
EVAL_SEED=0
MODES=""
SEED_GROUPS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --num-trials) NUM_TRIALS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --action-mode) ACTION_MODE="$2"; shift 2 ;;
        --task-shard-index) TASK_SHARD_INDEX="$2"; shift 2 ;;
        --task-shard-count) TASK_SHARD_COUNT="$2"; shift 2 ;;
        --seed) EVAL_SEED="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --seed-groups) SEED_GROUPS="$2"; shift 2 ;;
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
    echo "[ERROR] StarVLA or RoboTwin Apptainer runtime wrapper is missing" >&2
    exit 2
fi

RUN_DIR="$(dirname "$(dirname "$CKPT")")"
TRAIN_CONFIG="$RUN_DIR/config.full.yaml"
if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "[ERROR] checkpoint training config is missing: $TRAIN_CONFIG" >&2
    exit 2
fi
"$STARVLA_PY" "$STARVLA_REPO/scripts/diag/check_robotwin_eval_contract.py" "$TRAIN_CONFIG"
TRAIN_ACTION_MODE="$($STARVLA_PY -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c.get("datasets",{}).get("vla_data",{}).get("action_mode","abs"))' "$TRAIN_CONFIG")"
case "$TRAIN_ACTION_MODE" in
    absolute) TRAIN_ACTION_MODE=abs ;;
    relative) TRAIN_ACTION_MODE=rel ;;
esac
if [[ "$ACTION_MODE" == "auto" ]]; then
    ACTION_MODE="$TRAIN_ACTION_MODE"
elif [[ "$ACTION_MODE" != "$TRAIN_ACTION_MODE" ]]; then
    echo "[ERROR] action-mode mismatch: checkpoint=$TRAIN_ACTION_MODE requested=$ACTION_MODE" >&2
    exit 2
fi
if [[ "$ACTION_MODE" != "abs" && "$ACTION_MODE" != "delta" && "$ACTION_MODE" != "rel" ]]; then
    echo "[ERROR] unsupported checkpoint action mode: $ACTION_MODE" >&2
    exit 2
fi

cd "$STARVLA_REPO"
ml load CUDA
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

preflight_ok=1
for path in "$STARVLA_REPO" "$CKPT" "$STARVLA_PY" "$ROBOTWIN_PY"; do
    stat "$path" >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: cannot stat $path on $(hostname)"; preflight_ok=0; }
done
"$STARVLA_PY" -c "import torch, transformers, websockets" >/dev/null 2>&1 \
    || { echo "PREFLIGHT FAIL: StarVLA imports on $(hostname)"; preflight_ok=0; }
"$ROBOTWIN_PY" -c "import os, pathlib, tempfile, mplib, sapien, torch, curobo; from envs.robot.planner import CuroboPlanner; import model2robotwin_interface; p=pathlib.Path(os.environ['WARP_CACHE_PATH']); p.mkdir(parents=True, exist_ok=True); f=tempfile.NamedTemporaryFile(dir=p, delete=False); f.close(); pathlib.Path(f.name).unlink()" \
    || { echo "PREFLIGHT FAIL: RoboTwin imports on $(hostname)"; preflight_ok=0; }
( ROBOTWIN_REQUIRE_PACKAGED_ASSETS=1 "$ROBOTWIN_PY" -c 'from pathlib import Path; root=Path("/opt/robotwin/runtime/assets"); required=[root/"objects",root/"embodiments",root/"background_texture"/"seen",root/"background_texture"/"unseen"]; missing=[str(p) for p in required if not p.is_dir()]; assert not missing, f"missing packaged RoboTwin assets: {missing}"; assert any((root/"background_texture"/"seen").iterdir()); assert any((root/"background_texture"/"unseen").iterdir()); print("packaged assets OK")' ) \
    || { echo "PREFLIGHT FAIL: packaged RoboTwin assets on $(hostname)"; preflight_ok=0; }
CUDA_VISIBLE_DEVICES=0 "$ROBOTWIN_PY" /opt/robotwin/runtime/script/test_render.py \
    || { echo "PREFLIGHT FAIL: RoboTwin renderer on $(hostname)"; preflight_ok=0; }
if [[ "$preflight_ok" -ne 1 ]]; then
    exit 1
fi

export STARVLA_PYTHON="$STARVLA_PY"
export ROBOTWIN_PYTHON="$ROBOTWIN_PY"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_TORCH=1
UNITS=()
if (( SEED_GROUPS > 0 )); then
    MODES="${MODES:-$MODE}"
    IFS=',' read -ra MODE_LIST <<< "$MODES"
    IFS=',' read -ra TASK_LIST <<< "$TASKS"
    if [[ "$TASKS" == "all" ]]; then
        echo "[ERROR] unit mode needs an explicit comma-separated --tasks list" >&2
        exit 2
    fi
    MODE="${MODE_LIST[0]}"
    # longest tasks first (order of --tasks), modes and seed groups interleaved
    for task in "${TASK_LIST[@]}"; do
        for (( g = EVAL_SEED; g < EVAL_SEED + SEED_GROUPS; ++g )); do
            for mode in "${MODE_LIST[@]}"; do
                UNITS+=("${task}@${mode}@${g}")
            done
        done
    done
    export ROBOTWIN_LOG_ROOT="$(dirname "$CKPT")/robotwin_eval_logs/${NAME}_units_${MODES//,/+}_${SEED_GROUPS}x${NUM_TRIALS}_seed${EVAL_SEED}_job${SLURM_JOB_ID}"
else
    export ROBOTWIN_LOG_ROOT="$(dirname "$CKPT")/robotwin_eval_logs/${NAME}_${MODE}_${NUM_TRIALS}trials_seed${EVAL_SEED}_shard${TASK_SHARD_INDEX}of${TASK_SHARD_COUNT}_job${SLURM_JOB_ID}"
fi
export ROBOTWIN_OUTPUT_ROOT="$ROBOTWIN_LOG_ROOT/simulator_outputs"

echo "preflight OK on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint: $CKPT"
echo "Mode: $MODE${MODES:+ (unit modes: $MODES)}"
echo "Units: ${#UNITS[@]} (seed groups: $SEED_GROUPS)"
echo "Trials/task: $NUM_TRIALS"
echo "Evaluation seed group: $EVAL_SEED (RoboTwin starts at $((100000 * (1 + EVAL_SEED))))"
echo "Tasks: $TASKS"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Jobs/GPU: $JOBS_PER_GPU"
echo "Render skip (ROBOTWIN_SKIP_SUBSTEP_RENDER): ${ROBOTWIN_SKIP_SUBSTEP_RENDER:-0}"
echo "Action mode: $ACTION_MODE (checkpoint: $TRAIN_ACTION_MODE)"
echo "Task shard: $TASK_SHARD_INDEX/$TASK_SHARD_COUNT"
echo "Logs: $ROBOTWIN_LOG_ROOT"

mkdir -p "$ROBOTWIN_LOG_ROOT"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5 > "$ROBOTWIN_LOG_ROOT/gpu_telemetry.csv" 2>&1 &
telemetry_pid=$!
cleanup_telemetry() {
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT INT TERM

if (( SEED_GROUPS > 0 )); then
    start_rc=0
    bash "$STARVLA_REPO/examples/Robotwin/eval_files/start_eval.sh" \
        --mode "$MODE" \
        --name "$NAME" \
        --ckpt "$CKPT" \
        --num-trials "$NUM_TRIALS" \
        --seed "$EVAL_SEED" \
        --jobs-per-gpu "$JOBS_PER_GPU" \
        --action-mode "$ACTION_MODE" \
        "${UNITS[@]}" || start_rc=$?
    # Aggregate whatever finished even if a unit failed, then fail closed.
    "$STARVLA_PY" "$STARVLA_REPO/examples/Robotwin/eval_files/aggregate_units.py" \
        "$ROBOTWIN_LOG_ROOT" --tasks "$TASKS" --modes "$MODES" \
        --seed-groups "$SEED_GROUPS" --first-seed "$EVAL_SEED" --trials-per-unit "$NUM_TRIALS" --allow-partial
    "$STARVLA_PY" "$STARVLA_REPO/examples/Robotwin/eval_files/aggregate_units.py" \
        "$ROBOTWIN_LOG_ROOT" --tasks "$TASKS" --modes "$MODES" \
        --seed-groups "$SEED_GROUPS" --first-seed "$EVAL_SEED" --trials-per-unit "$NUM_TRIALS"
    exit "$start_rc"
fi

bash examples/Robotwin/eval_files/start_eval.sh \
    --mode "$MODE" \
    --name "$NAME" \
    --ckpt "$CKPT" \
    --num-trials "$NUM_TRIALS" \
    --seed "$EVAL_SEED" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --action-mode "$ACTION_MODE" \
    --task-shard-index "$TASK_SHARD_INDEX" \
    --task-shard-count "$TASK_SHARD_COUNT" \
    "$TASKS"

# start_eval fails if any worker fails; additionally require a parseable final
# result for every launched task and persist a compact paper-facing aggregate.
"$STARVLA_PY" examples/Robotwin/eval_files/aggregate_eval.py \
    "$ROBOTWIN_LOG_ROOT" "$MODE" "$NUM_TRIALS" --seed "$EVAL_SEED"
