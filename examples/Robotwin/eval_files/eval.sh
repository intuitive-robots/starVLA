#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> <action_mode> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

robotwin_eval_script="/opt/robotwin/runtime/script/eval_policy.py"

policy_name="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
action_mode="$7"
policy_port="${8:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${9:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    -e "s/^action_mode:.*/action_mode: \"${action_mode}\"/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH="${SCRIPT_DIR}"
STARVLA_PATH="${REPO_ROOT}"

echo "RoboTwin runtime: packaged ${robotwin_eval_script}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"
echo "action_mode: ${action_mode}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" "${robotwin_eval_script}" --config "${runtime_deploy_policy}" \
    --overrides \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --num_trials "${ROBOTWIN_NUM_TRIALS:-100}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
