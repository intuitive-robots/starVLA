#!/usr/bin/env bash
# RoboCasa365 walk-through evaluation — start the policy server in one terminal
# (starVLA env) and the simulation client in another (robocasa365 env).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
CKPT=${CKPT:-./playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt}
ENV_NAME=${ENV_NAME:-robocasa/OpenDrawer}
PORT=${PORT:-5678}
N_EPISODES=${N_EPISODES:-5}
N_ENVS=${N_ENVS:-1}
MAX_STEPS=${MAX_STEPS:-500}
N_ACT=${N_ACT:-8}
ROBOCASA365_RUNTIME=${ROBOCASA365_RUNTIME:-auto}
ROBOCASA365_SIF=${ROBOCASA365_SIF:-${REPO_ROOT}/playground/sims/sif/robocasa365-main-arm64.sif}

if [[ ${CKPT} != /* ]]; then
  CKPT="${REPO_ROOT}/${CKPT#./}"
fi

run_client() {
  local runtime=${ROBOCASA365_RUNTIME}
  if [[ ${runtime} == auto ]]; then
    if command -v apptainer >/dev/null 2>&1 && [[ -f ${ROBOCASA365_SIF} ]]; then
      runtime=apptainer
    else
      runtime=conda
    fi
  fi

  case "${runtime}" in
    apptainer)
      command -v apptainer >/dev/null 2>&1 || {
        echo "ERROR: apptainer is not available." >&2
        return 1
      }
      [[ -f ${ROBOCASA365_SIF} ]] || {
        echo "ERROR: RoboCasa365 SIF not found: ${ROBOCASA365_SIF}" >&2
        return 1
      }
      mkdir -p "${REPO_ROOT}/results/robocasa365_eval_test/videos"
      echo "RoboCasa365 simulator runtime: apptainer (${ROBOCASA365_SIF})"
      exec apptainer exec --cleanenv --writable-tmpfs --nv \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --pwd "${REPO_ROOT}" \
        --env "PYTHONPATH=${REPO_ROOT}" \
        --env PYTHONNOUSERSITE=1 \
        --env MUJOCO_GL=egl \
        --env PYOPENGL_PLATFORM=egl \
        --env EGL_PLATFORM=device \
        --env "NUMBA_CACHE_DIR=/tmp/robocasa365-numba-${USER:-user}" \
        --env "MPLCONFIGDIR=/tmp/robocasa365-mpl-${USER:-user}" \
        "${ROBOCASA365_SIF}" \
        python -m examples.Robocasa_365.eval_files.simulation_env \
          --args.pretrained-path "${CKPT}" \
          --args.env-name "${ENV_NAME}" \
          --args.port "${PORT}" \
          --args.n-episodes "${N_EPISODES}" \
          --args.n-envs "${N_ENVS}" \
          --args.max-episode-steps "${MAX_STEPS}" \
          --args.n-action-steps "${N_ACT}"
      ;;
    conda)
      echo "RoboCasa365 simulator runtime: current environment"
      exec python -m examples.Robocasa_365.eval_files.simulation_env \
        --args.pretrained-path "${CKPT}" \
        --args.env-name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n-episodes "${N_EPISODES}" \
        --args.n-envs "${N_ENVS}" \
        --args.max-episode-steps "${MAX_STEPS}" \
        --args.n-action-steps "${N_ACT}"
      ;;
    *)
      echo "ERROR: ROBOCASA365_RUNTIME must be auto, apptainer, or conda (got ${runtime})." >&2
      return 2
      ;;
  esac
}

case "${1:-}" in
  server)
    # Run inside the `starVLA` env
    exec python deployment/model_server/server_policy.py \
      --ckpt_path "${CKPT}" \
      --port "${PORT}" \
      --use_bf16
    ;;
  client)
    run_client
    ;;
  *)
    cat <<USAGE
Usage:
  # terminal 1 (conda env starVLA):
  bash examples/Robocasa_365/eval_files/run_eval.sh server
  # terminal 2 (automatically uses the RoboCasa365 SIF when present):
  bash examples/Robocasa_365/eval_files/run_eval.sh client

Override defaults with env vars: CKPT, ENV_NAME, PORT, N_EPISODES, N_ENVS, MAX_STEPS, N_ACT,
ROBOCASA365_RUNTIME (auto|apptainer|conda), and ROBOCASA365_SIF.
USAGE
    ;;
esac
