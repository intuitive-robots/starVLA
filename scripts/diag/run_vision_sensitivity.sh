#!/bin/bash
#SBATCH --job-name=vissens
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/vissens_%j.out
#SBATCH --error=slurm_logs/vissens_%j.err
#
# Four policy servers, one per GPU, each probed by scripts/diag/vision_sensitivity_probe.py.
# Answers: does the action depend on the image at all?  Modifies NO production eval file.
set -uo pipefail
cd /e/project1/m3/blank4/code/starVLA

export LIBERO_HOME="${VANILLA_LIBERO_HOME:-$PWD/playground/sims/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="$PWD:${LIBERO_HOME}"
export APPTAINERENV_PYTHONPATH="$PYTHONPATH"
export APPTAINERENV_LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH"
export APPTAINERENV_STARVLA_ROOT="$PWD"
export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
SV=scripts/env/starvla_python
LB=scripts/env/libero_python
CK=playground/Checkpoints

# tag|checkpoint  -- stage2ctrl is the 0% mystery; the other three calibrate the scale.
ARMS=(
  "stage2ctrl|$CK/ervla_lit_stage2ctrl_gr00t_b32_s42/final_model/pytorch_model.pt"
  "stage2trace|$CK/ervla_lit_stage2_gr00t_b32_s42/final_model/pytorch_model.pt"
  "qwen08b_ref|$CK/qwen08b_gr00t_b64_20k_s42/final_model/pytorch_model.pt"
  "stage1ctrl_blind|$CK/ervla_lit_stage1ctrl_gr00t_b32_s42/final_model/pytorch_model.pt"
)

run_arm () {
  local tag="$1" ckpt="$2" gpu="$3" port="$4"
  # Redirect INSIDE this shell: a login-side redirect dies with the client and orphans the step.
  exec > "slurm_logs/vissens_${tag}_${SLURM_JOB_ID}.log" 2>&1
  echo "=== $tag  gpu=$gpu port=$port  ckpt=$ckpt"
  [ -f "$ckpt" ] || { echo "MISSING CHECKPOINT: $ckpt"; return 1; }

  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PWD" \
    "$SV" deployment/model_server/server_policy.py \
      --ckpt_path "$ckpt" --port "$port" --use_bf16 --idle_timeout -1 \
      > "slurm_logs/vissens_server_${tag}_${SLURM_JOB_ID}.log" 2>&1 &
  local srv=$!
  trap 'kill -TERM -$srv 2>/dev/null; kill $srv 2>/dev/null' RETURN

  for i in $(seq 1 600); do
    (echo >/dev/tcp/127.0.0.1/$port) >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
  done

  env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl \
    "$LB" scripts/diag/vision_sensitivity_probe.py \
      --port "$port" --tag "$tag" --task_ids 0,3,6 --n_per_task 4 \
      --out "slurm_logs/vissens_${tag}.jsonl"
  echo "=== done: $tag"
}

g=0; p=10310
for a in "${ARMS[@]}"; do
  tag="${a%%|*}"; ckpt="${a##*|}"
  ( run_arm "$tag" "$ckpt" "$g" "$p" ) &
  g=$(( (g+1) % 4 )); p=$(( p + 1 ))
  [ "$g" = 0 ] && wait
done
wait

echo "=== SUMMARIES ==="
for a in "${ARMS[@]}"; do
  tag="${a%%|*}"
  echo -n "$tag: "; tail -1 "slurm_logs/vissens_${tag}.jsonl" 2>/dev/null || echo "(no output)"
done
