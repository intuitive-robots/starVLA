#!/bin/bash
#SBATCH --job-name=bench_lplus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/bench_lplus_%j.out
#SBATCH --error=slurm_logs/bench_lplus_%j.err
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --cpus-per-task=288
#
# A/B/C benchmark of the LIBERO-plus eval throughput changes, all three arms on
# the SAME node back to back (so node-to-node variance cannot explain the
# difference) against the same checkpoint and the same deterministic task sample.
#
#   A  8 workers/GPU, contiguous shards   -- the configuration the production
#                                            ep_* jobs ran with
#   B  8 workers/GPU, round-robin shards  -- isolates load balancing alone
#   C 24 workers/GPU, round-robin shards  -- balancing + the raised worker count
#
# The 1.0s dispatcher stall is NOT an arm here: the client-count cap makes it
# unreproducible by configuration (that is the fix). It is measured instead by
# comparing arm A's per-category episode times against the production run's,
# which used identical settings on the pre-fix code.
set -euo pipefail

# sbatch copies the script into the ParaStation job spool, so BASH_SOURCE does
# NOT point at the checkout -- resolve the repo from the submit directory, which
# sbatch preserves as the job's cwd (the same assumption eval_libero_plus_slurm.sh makes).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

CKPT="${bench_ckpt:-playground/Checkpoints/ervla_pi_bidir_actiononly_pifix_nolatent_nodrop_2gpu_s42/checkpoints/steps_20000_pytorch_model.pt}"
BENCH_ROOT="${bench_root:-${REPO_ROOT}/playground/bench/libero_plus_sharding_${SLURM_JOB_ID:-local}}"
SUITE="${bench_suite:-libero_10}"
EXACT="${bench_exact:-512}"
AUTO_EVAL="${REPO_ROOT}/examples/LIBERO-plus/eval_files/parallel_eval/auto_eval_libero_plus.sh"
SIF="${REPO_ROOT}/playground/sims/sif/libero-plus-v0.5.0-arm64.sif"

echo "=========================================="
echo " LIBERO-plus sharding/throughput benchmark"
echo " Job        : ${SLURM_JOB_ID:-local} on $(hostname)"
echo " Checkpoint : ${CKPT}"
echo " Suite      : ${SUITE}, ${EXACT} deterministic task instances per arm"
echo " Results    : ${BENCH_ROOT}"
echo "=========================================="
mkdir -p "${BENCH_ROOT}"

run_arm() {
    local name=$1 workers=$2 servers=$3 wait=$4 interleave=$5 port=$6
    local out="${BENCH_ROOT}/${name}"
    rm -rf "${out}"; mkdir -p "${out}"
    echo ""
    echo "############ ARM ${name}: workers/gpu=${workers} servers/gpu=${servers} max_wait_time=${wait} interleave=${interleave}"
    local t0 t1 rc=0
    t0=$(date +%s)
    STARVLA_INTERLEAVE_SHARDS="${interleave}" \
    output_dir="${out}" your_ckpt="${CKPT}" \
    exact_tasks_per_suite="${EXACT}" \
    max_batch_size=32 max_wait_time="${wait}" base_port="${port}" \
    LIBERO_PLUS_RUNTIME=apptainer LIBERO_PLUS_SIF="${SIF}" \
        bash "${AUTO_EVAL}" "${SUITE}" 4 0 "${workers}" 0 1 "0,1,2,3" "${servers}" 1 \
        > "${out}/arm.log" 2>&1 || rc=$?
    t1=$(date +%s)
    echo "BENCH_RESULT arm=${name} wall_s=$((t1 - t0)) rc=${rc}"
    tail -n 3 "${out}/arm.log" || true
}

run_arm A_contiguous_8w  8 2 1.0 0 10500
run_arm B_roundrobin_8w  8 2 0.0 1 10600
run_arm C_roundrobin_24w 24 2 0.0 1 10700

echo ""
echo "########## SUMMARY ##########"
grep -h "^BENCH_RESULT" "${SLURM_OUTPUT:-/dev/null}" 2>/dev/null || true
python3 "${REPO_ROOT}/examples/LIBERO-plus/eval_files/parallel_eval/bench_report.py" \
    --bench_root "${BENCH_ROOT}" --suite "${SUITE}"
