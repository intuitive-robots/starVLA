#!/bin/bash
# StarVLA SimplerEnv — policy server for custom Qwen2B / 4B checkpoint
#
# Run from repo root in the starVLA conda environment:
#   bash examples/SimplerEnv/eval_files/run_policy_server_custom.sh
#
# The server must be started BEFORE start_simpler_env_custom.sh.

###############################################################################
# === Please modify the variables below to match your environment ===

STARVLA_DIR=$(pwd)                       # assumes you run from repo root
STARVLA_PYTHON=$(which python)           # TODO: or set full path, e.g. /home/you/.conda/envs/starVLA/bin/python

# Path to your trained checkpoint (.pt file)
your_ckpt=./playground/Checkpoints/libero_qwen4b_custom/checkpoints/steps_80000_pytorch_model.pt  # TODO

gpu_id=0
port=6694

# === End of configuration ===
###############################################################################

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH}"

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"

output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

echo "Starting policy server on GPU ${gpu_id}, port ${port}"
echo "Checkpoint: ${your_ckpt}"
echo "Log: ${log_file}"

CUDA_VISIBLE_DEVICES=${gpu_id} ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16 \
    2>&1 | tee "${log_file}"
