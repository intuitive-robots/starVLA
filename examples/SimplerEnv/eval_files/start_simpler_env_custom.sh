#!/bin/bash
# StarVLA SimplerEnv — simulation client for custom checkpoint
#
# Run from repo root in the simpler_env conda environment AFTER
# run_policy_server_custom.sh is up and accepting connections:
#   bash examples/SimplerEnv/eval_files/start_simpler_env_custom.sh [ckpt_path] [port]

###############################################################################
# === Please modify the variables below to match your environment ===

STARVLA_DIR=$(pwd)
sim_python=$(which python)                          # TODO: path to simpler_env python, e.g. /home/you/.conda/envs/simpler_env/bin/python
SimplerEnv_PATH=/path/to/SimplerEnv                # TODO: path to your SimplerEnv installation

# Default checkpoint / port (can be overridden via positional args)
default_ckpt=./playground/Checkpoints/libero_qwen4b_custom/checkpoints/steps_80000_pytorch_model.pt  # TODO
default_port=6694

# === End of configuration ===
###############################################################################

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${sim_python%/bin/*}/lib:${LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

MODEL_PATH=${1:-"${default_ckpt}"}
port=${2:-"${default_port}"}

ckpt_dir=$(dirname "${MODEL_PATH}")
ckpt_base=$(basename "${MODEL_PATH}")
ckpt_name="${ckpt_base%.*}"

output_eval_dir="${ckpt_dir}/output_eval_simplerenv"
mkdir -p "${output_eval_dir}"

# ── WidowX / Bridge eval scenes ──────────────────────────────────────────────
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# Add / remove tasks as needed
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
  StackGreenCubeOnYellowCubeInScene-v0
)

TSET_NUM=1   # number of evaluation runs per task

echo "Evaluating checkpoint : ${MODEL_PATH}"
echo "Server port           : ${port}"
echo "Output dir            : ${output_eval_dir}"

for env in "${ENV_NAMES[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶ Task [${env}] run#${run_idx} → ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path "${MODEL_PATH}" \
      --port "${port}" \
      --robot "${robot}" \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name "${scene_name}" \
      --rgb-overlay-path "${rgb_overlay_path}" \
      --robot-init-x "${robot_init_x}" \
      --robot-init-y "${robot_init_y}" \
      --num-episodes 50 \
      --output-dir "${output_eval_dir}" \
      2>&1 | tee "${task_log}"
  done
done

echo "Done. Results in ${output_eval_dir}"
