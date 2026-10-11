#!/bin/bash
#SBATCH --job-name=hk_bench_simscale
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --output=/e/scratch/m3/blank4/horeka_bench/logs/slurm_%j.out
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --exclude=jpbo-001-03,jpbo-013-26,jpbo-013-28,jpbo-016-32,jpbo-026-19,jpbo-031-15,jpbo-031-16,jpbo-047-46,jpbo-047-48,jpbo-053-11,jpbo-055-05,jpbo-055-23,jpbo-081-33
# HoreKa proposal benchmark B2: LIBERO evaluation scaling over GPUs/nodes at a fixed workload of 2,000 episodes
# (4 suites x 10 tasks x 50 trials), 16 envs per GPU (32 per 2 GPUs, the best setting of B), same policy as B.
#   sbatch --nodes=1 --export=ALL,HALF=1 bench_sim_scale_slurm.sh   # 2 GPUs: both halves of one node, two repeats
#   sbatch --nodes=1 bench_sim_scale_slurm.sh                       # 4 GPUs
#   sbatch --nodes=2 bench_sim_scale_slurm.sh                       # 8 GPUs
#   sbatch --nodes=4 bench_sim_scale_slurm.sh                       # 16 GPUs
#   add chain_suites=true to --export for the chained-suite launcher (results in sim_scale_chain/)
# Each node evaluates its slice of every suite; logs carry unix timestamps so start-up (job start -> policy servers
# ready) and workload time (servers ready -> last worker finished) are reported separately.
set -uo pipefail
ml load CUDA
REPO=/e/project1/m3/blank4/code/starVLA
OUT=/e/scratch/m3/blank4/horeka_bench/sim_scale
# chain_suites=true (passed through to auto_eval_libero.sh): workers run all suites back to back; separate output.
[ "${chain_suites:-false}" = true ] && OUT=${OUT}_chain
export CKPT=${CKPT:-$REPO/playground/Checkpoints/qwen08b_gr00t_b64_20k_s42/checkpoints/steps_20000_pytorch_model.pt}
export SUITES=libero_spatial,libero_object,libero_goal,libero_10 TRIALS=50 WPG=16
cd "$REPO"

# Plain command string, not `export -f`: BASH_FUNC_* variables do not survive srun to other nodes here
# (see eval_libero_slurm.sh). Args: <outdir> <num_gpus> <gpu csv> <cores> <base_port> <partition_idx> <num_partitions>
export RUN_CMD='cd /e/project1/m3/blank4/code/starVLA; mkdir -p "$1"
    your_ckpt="$CKPT" output_dir="$1" base_port=$5 SKIP_AGGREGATE=true servers_per_gpu=1 \
        max_batch_size=32 max_wait_time=0.01 \
        taskset -c "$4" bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
            "$SUITES" "$2" "$WPG" "$TRIALS" "$3" "$6" "$7" 2>&1 \
        | awk "{ print systime(), \$0; fflush() }"'
run() { bash -c "$RUN_CMD" _ "$@"; }

if [ "${HALF:-0}" = 1 ]; then
    for h in a b; do rm -rf "$OUT/g2$h"; mkdir -p "$OUT/g2$h"; date +%s > "$OUT/g2$h/job_start"; done
    run "$OUT/g2a" 2 0,1 0-143 10093 0 1 > "$OUT/g2a/node0.log" &
    run "$OUT/g2b" 2 2,3 144-287 10293 0 1 > "$OUT/g2b/node0.log" &
    wait
else
    NG=$((4 * SLURM_NNODES)); D=$OUT/g$NG
    rm -rf "$D"; mkdir -p "$D"; date +%s > "$D/job_start"
    export D
    # One task per node; each redirects its own log inside the remote shell.
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=288 \
        --mpi=none --cpu-bind=none bash -c \
        'exec > "$D/node$SLURM_NODEID.log" 2>&1; bash -c "$RUN_CMD" _ "$D" 4 0,1,2,3 0-287 10093 "$SLURM_NODEID" "$SLURM_NNODES"'
fi
echo "[$(date +%T)] done"
