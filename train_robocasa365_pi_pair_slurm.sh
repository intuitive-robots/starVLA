#!/bin/bash
#SBATCH --job-name=tr_rc365_pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/train_rc365_pair_%j.out
#SBATCH --error=slurm_logs/train_rc365_pair_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
#
# RoboCasa365 Atomic-Seen backbone comparison, PI action head, one seed per node:
#   slot 0 -> GPUs 0,1 -> causal Qwen3-VL-2B   (ervla_robocasa365_pi_causal.yaml)
#   slot 1 -> GPUs 2,3 -> v5 encoder-decoder   (ervla_robocasa365_pi_v5.yaml)
# 2 GPUs x per_device_batch_size 32 = 64 global per arm, same shape as the
# LIBERO v5 head-pair runs.
#
# Pairing the two BACKBONES on one node rather than the two SEEDS means the
# arms being compared always share hardware, so a slow or flaky node shifts
# both arms together instead of biasing the comparison.
#
# Usage:  sbatch train_robocasa365_pi_pair_slurm.sh <seed>
# Logs:   slurm_logs/train_rc365_pair_<jobid>_causal.log
#         slurm_logs/train_rc365_pair_<jobid>_v5.log
set -euo pipefail

SEED="${1:-42}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

CAUSAL_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_causal.yaml
V5_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_v5.yaml
DATA_ROOT=/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/robocasa365_target_atomic

for f in "$CAUSAL_YAML" "$V5_YAML" train_libero_slurm.sh; do
    [ -f "$f" ] || { echo "[ERROR] missing $f"; exit 1; }
done
# The loader writes meta/stats_gr00t.json and meta/steps_data_index.pkl into the
# dataset on first use; a read-only root fails deep inside the first epoch.
[ -w "${DATA_ROOT}/meta" ] || { echo "[ERROR] dataset meta/ not writable: ${DATA_ROOT}/meta"; exit 1; }
[ -f "${DATA_ROOT}/meta/modality.json" ] || { echo "[ERROR] missing ${DATA_ROOT}/meta/modality.json"; exit 1; }

# ── FFmpeg shim so torchcodec loads inside starVLA.sif ────────────────────────
# These configs use video_backend: torchcodec because decord's seek silently
# returns the WRONG FRAME on the RoboCasa365 H.264 mirrors (wrong on 9/12 random
# reads; torchcodec matched a sequential-decode reference 12/12). starVLA.sif
# ships torchcodec but no system FFmpeg, so it fails to load without this.
# See /e/project1/m3/blank4/containers/ffmpeg_shim/README.md
FFMPEG_SHIM="${FFMPEG_SHIM:-/e/project1/m3/blank4/containers/ffmpeg_shim}"
CUDA_LIB64="${CUDA_LIB64:-/e/software/default/stages/2026/software/CUDA/13/lib64}"
SIF_SITE=/opt/conda/envs/starVLA/lib/python3.12/site-packages
[ -e "${FFMPEG_SHIM}/libavcodec.so.60" ] || { echo "[ERROR] FFmpeg shim missing: ${FFMPEG_SHIM}"; exit 1; }
export APPTAINERENV_LD_LIBRARY_PATH="${FFMPEG_SHIM}:${CUDA_LIB64}:${SIF_SITE}/torch/lib:/opt/conda/envs/starVLA/lib"

# ── Stage the dataset into node RAM ───────────────────────────────────────────
# The dataset is only 39 mp4 files (mean 153 MB) but this node runs two arms x
# two ranks x 24 workers = 96 readers against them, each with its own decoder
# doing independent random seeks. On the shared exa_scratch filesystem that
# collides badly: observed data stalls of 11-16 s while model_times stayed at
# ~1.2 s. tmpfs has 239 GB and the copy takes about 4 s, so both arms read from
# RAM instead. STARVLA_RC365_STAGE=0 disables it.
STAGE_DATA="${STARVLA_RC365_STAGE:-1}"
if [ "${STAGE_DATA}" = "1" ]; then
    SHM_ROOT=/dev/shm/rc365_${SLURM_JOB_ID:-local}
    echo "staging dataset -> ${SHM_ROOT} (node RAM)"
    mkdir -p "${SHM_ROOT}"
    t0=$SECONDS
    # -a preserves the loader's generated meta/ cache (stats_gr00t.json and the
    # 20 MB steps_data_index.pkl), so indexing is not redone from scratch.
    cp -a "${DATA_ROOT}" "${SHM_ROOT}/" || { echo "[ERROR] staging failed"; exit 1; }
    echo "staged $(du -sh "${SHM_ROOT}" | cut -f1) in $((SECONDS-t0)) s; /dev/shm now $(df -h /dev/shm | awk 'NR==2{print $3" used, "$4" free"}')"
    DATA_ROOT_DIR_OVERRIDE="${SHM_ROOT}"
    cleanup_shm() { rm -rf "${SHM_ROOT}" 2>/dev/null || true; }
    trap cleanup_shm EXIT
else
    DATA_ROOT_DIR_OVERRIDE=""
fi

echo "=========================================="
echo " RoboCasa365 Atomic-Seen PI backbone comparison -- seed ${SEED}"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " slot 0  GPUs 0,1  causal  ervla_robocasa365_pi_causal_s${SEED}"
echo " slot 1  GPUs 2,3  v5      ervla_robocasa365_pi_v5_s${SEED}"
echo "=========================================="

pids=()
launch() {
    local slot="$1" arm="$2" yaml="$3" run_id="$4"
    local devs port log
    devs=$([ "$slot" -eq 0 ] && echo "0,1" || echo "2,3")
    # Distinct rendezvous port per slot: two accelerate groups on one node
    # otherwise collide on the default 29500 and one of them hangs in c10d init.
    port=$((37100 + slot))
    log="slurm_logs/train_rc365_pair_${SLURM_JOB_ID:-local}_${arm}.log"
    echo "seed=${SEED} arm=${arm} slot=${slot} CUDA_VISIBLE_DEVICES=${devs} run_id=${run_id}"
    (
        # train_libero_slurm.sh overwrites CUDA_VISIBLE_DEVICES from
        # STARVLA_CUDA_VISIBLE_DEVICES and derives NUM_PROCESSES from
        # `nvidia-smi -L | wc -l`, which still reports all four GPUs. Set both
        # explicitly or each arm lands on all four.
        # train_libero_slurm.sh defaults STARVLA_REPO to the live checkout and
        # cd's there; without this override both arms would train from that tree,
        # which does not contain these configs.
        export STARVLA_REPO="${REPO}"
        export DATA_ROOT_DIR_OVERRIDE
        export STARVLA_CUDA_VISIBLE_DEVICES="${devs}"
        export NUM_PROCESSES=2
        export MASTER_PORT="${port}"
        extra=()
        [ -n "${DATA_ROOT_DIR_OVERRIDE}" ] && extra+=(--datasets.vla_data.data_root_dir "${DATA_ROOT_DIR_OVERRIDE}")
        bash train_libero_slurm.sh \
            --config "${yaml}" \
            --run_id "${run_id}" \
            --seed "${SEED}" \
            "${extra[@]}"
    ) > "${log}" 2>&1 &
    pids+=($!)
}

launch 0 causal "$CAUSAL_YAML" "ervla_robocasa365_pi_causal_s${SEED}"
launch 1 v5     "$V5_YAML"     "ervla_robocasa365_pi_v5_s${SEED}"

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "[ERROR] at least one arm failed; see the per-arm logs above."
    exit 1
fi
echo "both arms finished for seed ${SEED}"
