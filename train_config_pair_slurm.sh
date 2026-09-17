#!/bin/bash
# Generic two-config pair: slots two arbitrary configs on one 4-GPU node,
# 2 GPUs each, same seed. Sibling of train_seed_pair_slurm.sh (two seeds of one
# config) and train_v5_head_pair_slurm.sh (the hardcoded PI/GR00T pair).
#
#   sbatch --export=ALL,YAML_A=<cfg_a>,RUN_A=<prefix_a>,YAML_B=<cfg_b>,RUN_B=<prefix_b> \
#          train_config_pair_slurm.sh <seed>
#SBATCH --job-name=tr_cfg_pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/train_cfg_pair_%j.out
#SBATCH --error=slurm_logs/train_cfg_pair_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
#
# Trains BOTH action-head arms of one seed side by side on a single 4-GPU node:
#   slot 0 -> GPUs 0,1 -> PI head     (ervla_v5_pi_actiononly_pifix.yaml)
#   slot 1 -> GPUs 2,3 -> GR00T head  (ervla_v5_gr00t_actiononly.yaml)
# 2 GPUs x per_device_batch_size 32 = 64 effective per arm, the same shape the
# bidir/causal controls ran (their log header: "GPUs: 2", batch override 32).
#
# Pairing the two ARMS on one node rather than the two SEEDS means the heads
# being compared always share hardware, so a slow or flaky node shifts both arms
# together instead of biasing the comparison.
#
# Usage:  sbatch train_v5_head_pair_slurm.sh <seed>
# Logs:   slurm_logs/train_v5_pair_<jobid>_pi.log
#         slurm_logs/train_v5_pair_<jobid>_gr00t.log
set -euo pipefail

SEED="${1:-42}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

YAML_A="${YAML_A:?set YAML_A=<config> }"
YAML_B="${YAML_B:?set YAML_B=<config> }"
for f in "$YAML_A" "$YAML_B" train_libero_slurm.sh; do
    [ -f "$f" ] || { echo "[ERROR] missing $f"; exit 1; }
done


# ── FFmpeg shim, only when a config asks for torchcodec ───────────────────────
# starVLA.sif ships torchcodec but no system FFmpeg, so `video_backend: torchcodec`
# dies with "ImportError: torchcodec is not available." unless the shim is on the
# library path. The robocasa launcher has always exported this; a config routed
# through THIS launcher (the gr00t_deeps recipes carry torchcodec) did not, which
# is what killed job 1841895.
# See /e/project1/m3/blank4/containers/ffmpeg_shim/README.md
if grep -qE "^\s*video_backend:\s*torchcodec" "${YAML_A}" "${YAML_B}" 2>/dev/null; then
  FFMPEG_SHIM="${FFMPEG_SHIM:-/e/project1/m3/blank4/containers/ffmpeg_shim}"
  CUDA_LIB64="${CUDA_LIB64:-/e/software/default/stages/2026/software/CUDA/13/lib64}"
  SIF_SITE=/opt/conda/envs/starVLA/lib/python3.12/site-packages
  # -L not -e: symlinks point into the container image, so they look dangling from the host.
  [ -L "${FFMPEG_SHIM}/libavcodec.so.60" ] || { echo "[ERROR] FFmpeg shim missing: ${FFMPEG_SHIM}"; exit 1; }
  export APPTAINERENV_LD_LIBRARY_PATH="${FFMPEG_SHIM}:${CUDA_LIB64}:${SIF_SITE}/torch/lib:/opt/conda/envs/starVLA/lib"
  echo "torchcodec requested -> FFmpeg shim exported (${FFMPEG_SHIM})"
fi

# ── fla/triton preflight ──────────────────────────────────────────────────────
# Qwen3.5 backbones run their linear-attention layers through `fla`, which picks its
# device at import from triton's active driver. When that probe fails on a node, fla
# binds to torch.cpu and training dies ~3 min later inside the DeltaNet kernel with
#   AttributeError: module 'torch.cpu' has no attribute 'device'
# (torch 2.12 dropped it). It is node-dependent -- a probe on another node was fine --
# so fail fast here instead of burning a slot. Only matters for Qwen3.5 configs.
if grep -qiE "^\s*base_vlm:.*qwen3\.5" "${YAML_A}" "${YAML_B}" 2>/dev/null; then
  if ! /e/project1/m3/blank4/containers/envs/run_in_env.sh starVLA python -c "
import sys, fla.utils as u
sys.exit(0 if u.device == 'cuda' else 1)" 2>/dev/null; then
    echo "[ERROR] fla bound to a non-CUDA device on $(hostname) (triton probe failed);"
    echo "        Qwen3.5 training would crash in the DeltaNet kernel. Resubmit."
    exit 1
  fi
  echo "fla/triton preflight OK"
fi

echo "=========================================="
echo " config pair -- seed ${SEED}"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " slot 0  GPUs 0,1  A  ${RUN_A}_s${SEED}"
echo " slot 1  GPUs 2,3  B  ${RUN_B}_s${SEED}"
echo "=========================================="

pids=()
launch() {
    local slot="$1" arm="$2" yaml="$3" run_id="$4"
    local devs port log
    devs=$([ "$slot" -eq 0 ] && echo "0,1" || echo "2,3")
    # Distinct rendezvous port per slot: two accelerate groups on one node
    # otherwise collide on the default 29500 and one of them hangs in c10d init.
    port=$((37000 + slot))
    log="slurm_logs/train_cfg_pair_${SLURM_JOB_ID:-local}_${arm}.log"
    echo "seed=${SEED} arm=${arm} slot=${slot} CUDA_VISIBLE_DEVICES=${devs} run_id=${run_id}"
    (
        # NUM_PROCESSES must be set explicitly: train_libero_slurm.sh derives it
        # from `nvidia-smi -L | wc -l`, which still reports all four GPUs even
        # with CUDA_VISIBLE_DEVICES restricted, and would launch 4 ranks onto 2.
        # train_libero_slurm.sh does `export CUDA_VISIBLE_DEVICES=${STARVLA_CUDA_VISIBLE_DEVICES:-0,1,2,3}`,
        # so setting CUDA_VISIBLE_DEVICES here would be overwritten and BOTH arms
        # would land on all four GPUs. Set the variable it actually reads.
        export STARVLA_CUDA_VISIBLE_DEVICES="${devs}"
        export NUM_PROCESSES=2
        export MASTER_PORT="${port}"
        bash train_libero_slurm.sh \
            --config "${yaml}" \
            --run_id "${run_id}" \
            --seed "${SEED}"
    ) > "${log}" 2>&1 &
    pids+=($!)
}

launch 0 a "$YAML_A" "${RUN_A:?set RUN_A}_s${SEED}"
launch 1 b "$YAML_B" "${RUN_B:?set RUN_B}_s${SEED}"

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "[ERROR] at least one arm failed; see the per-arm logs above."
    exit 1
fi
echo "both arms finished for seed ${SEED}"
