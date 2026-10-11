#!/bin/bash
#SBATCH --job-name=gripprobe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:40:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/gripprobe_%j.out
#SBATCH --error=slurm_logs/gripprobe_%j.err
#
# Starts one policy server (starVLA image) and runs scripts/diag/gripper_probe.py against it
# (LIBERO image). Modifies NO production eval file.
set -uo pipefail
cd /e/project1/m3/blank4/code/starVLA
CKPT="$1"; TAG="$2"; PORT="${3:-10193}"
# LIBERO lives in the repo, not in the image -- auto_eval_libero.sh:39-51 sets these and
# the probe fails with ModuleNotFoundError: No module named 'libero' without them.
# Deliberately NOT "${LIBERO_HOME:-...}": a LIBERO_HOME exported for LIBERO-plus would
# silently point the probe at the wrong fork.
export LIBERO_HOME="${VANILLA_LIBERO_HOME:-$PWD/playground/sims/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="$PWD:${LIBERO_HOME}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export APPTAINERENV_LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
SV=scripts/env/starvla_python
LB=scripts/env/libero_python

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  "$SV" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$PORT" --use_bf16 --idle_timeout -1 \
    > "slurm_logs/gripprobe_server_${TAG}_${SLURM_JOB_ID}.log" 2>&1 &
SRV=$!
trap 'kill -TERM -$SRV 2>/dev/null; kill $SRV 2>/dev/null' EXIT

echo "waiting for policy server on :$PORT ..."
for i in $(seq 1 120); do
  (echo >/dev/tcp/127.0.0.1/$PORT) >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
  sleep 1
done

"$LB" scripts/diag/gripper_probe.py --port "$PORT" --episodes 10 \
      --out "slurm_logs/gripper_probe_${TAG}.jsonl"
echo "=== done: $TAG ==="
