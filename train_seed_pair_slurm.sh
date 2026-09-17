#!/bin/bash
#SBATCH --job-name=tr_seed_pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/train_seed_pair_%j.out
#SBATCH --error=slurm_logs/train_seed_pair_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
#
# Trains TWO SEEDS of one config side by side on a single 4-GPU node:
#   slot 0 -> GPUs 0,1 -> seed A
#   slot 1 -> GPUs 2,3 -> seed B
# 2 GPUs x per_device_batch_size 32 = 64 effective per run, the shape the
# bidir/causal controls used. Sibling of train_v5_head_pair_slurm.sh, which
# slots two ARMS of one seed instead; use that one when comparing two configs
# and this one when replicating a single config across seeds.
#
# Usage:  sbatch train_seed_pair_slurm.sh <config.yaml> <run_id_prefix> <seedA> <seedB> [trainer overrides...]
#   e.g.  sbatch train_seed_pair_slurm.sh \
#             examples/simBenchmarks/LIBERO/train_files/ervla_v5aux_pi_actiononly_pifix.yaml \
#             ervla_v5aux_pi_actiononly_pifix 42 43
# Produces run ids <run_id_prefix>_s<seed>.
# Logs:   slurm_logs/train_seed_pair_<jobid>_s<seed>.log
set -euo pipefail

YAML="${1:?usage: train_seed_pair_slurm.sh <config.yaml> <run_id_prefix> <seedA> <seedB>}"
PREFIX="${2:?missing run_id prefix}"
SEED_A="${3:?missing seed A}"
SEED_B="${4:?missing seed B}"
shift 4
EXTRA_TRAIN_ARGS=("$@")
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

for f in "$YAML" train_libero_slurm.sh; do
    [ -f "$f" ] || { echo "[ERROR] missing $f"; exit 1; }
done

echo "=========================================="
echo " seed pair -- $(basename "$YAML")"
echo " Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo " slot 0  GPUs 0,1  ${PREFIX}_s${SEED_A}"
echo " slot 1  GPUs 2,3  ${PREFIX}_s${SEED_B}"
echo " overrides ${EXTRA_TRAIN_ARGS[*]:-<none>}"
echo "=========================================="

pids=()
launch() {
    local slot="$1" seed="$2"
    local devs port log run_id
    devs=$([ "$slot" -eq 0 ] && echo "0,1" || echo "2,3")
    # Distinct rendezvous port per slot; two accelerate groups on one node
    # otherwise collide on the default 29500 and one hangs in c10d init.
    port=$((37000 + slot))
    run_id="${PREFIX}_s${seed}"
    log="slurm_logs/train_seed_pair_${SLURM_JOB_ID:-local}_s${seed}.log"
    echo "seed=${seed} slot=${slot} CUDA_VISIBLE_DEVICES=${devs} run_id=${run_id}"
    (
        # train_libero_slurm.sh does `export CUDA_VISIBLE_DEVICES=${STARVLA_CUDA_VISIBLE_DEVICES:-0,1,2,3}`,
        # so exporting CUDA_VISIBLE_DEVICES here would be overwritten and both slots
        # would land on all four GPUs. NUM_PROCESSES must also be explicit: that
        # script counts GPUs with `nvidia-smi -L`, which still reports all four.
        export STARVLA_CUDA_VISIBLE_DEVICES="${devs}"
        export STARVLA_REPO="${REPO}"
        export NUM_PROCESSES=2
        export MASTER_PORT="${port}"
        bash train_libero_slurm.sh \
            --config "${YAML}" \
            --run_id "${run_id}" \
            --seed "${seed}" \
            "${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"}"
    ) > "${log}" 2>&1 &
    pids+=($!)
}

launch 0 "${SEED_A}"
launch 1 "${SEED_B}"

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "[ERROR] at least one seed failed; see the per-seed logs above."
    exit 1
fi
echo "both seeds finished for ${PREFIX}"
