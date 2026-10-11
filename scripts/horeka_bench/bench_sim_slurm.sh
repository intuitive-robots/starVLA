#!/bin/bash
#SBATCH --job-name=hk_bench_sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/m3/blank4/horeka_bench/logs/slurm_%j.out
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --exclude=jpbo-001-03,jpbo-013-26,jpbo-013-28,jpbo-016-32,jpbo-026-19,jpbo-031-15,jpbo-031-16,jpbo-047-46,jpbo-047-48,jpbo-053-11,jpbo-055-05,jpbo-055-23,jpbo-081-33
# HoreKa proposal benchmark B: LIBERO (libero_spatial, 10 tasks x 10 trials = 100 episodes) with a trained
# Qwen3.5-0.8B GR00T policy. Each setting is one half node: 2 GPUs + the 144 cores of their two Grace
# CPUs (taskset), so two settings run side by side: round 1 = 8 | 16 envs, round 2 = 32 | 64 envs.
# GPUs render (EGL) and serve the policy; CPU cores run MuJoCo in one process per env.
set -uo pipefail
ml load CUDA
REPO=/e/project1/m3/blank4/code/starVLA
OUT=/e/scratch/m3/blank4/horeka_bench/sim
CKPT=${CKPT:-$REPO/playground/Checkpoints/qwen08b_gr00t_b64_20k_s42/checkpoints/steps_20000_pytorch_model.pt}
SUITE=libero_spatial; TRIALS=10
mkdir -p "$OUT"
nvidia-smi -L; nvidia-smi topo -m; lscpu | grep -E "^CPU\(s\)|Model name|NUMA node[0-3] "
cd "$REPO"

python3 "$REPO/scripts/horeka_bench/node_sampler.py" "$OUT/sampler_${SLURM_JOB_ID}.csv" 0-143 144-287 2 &
SAMPLER=$!

half() {  # <envs total> <gpu csv> <cores> <base_port>
    local envs=$1 gpus=$2 cores=$3 port=$4 wpg=$(( $1 / 2 ))
    local d=$OUT/envs$envs
    rm -rf "$d"; mkdir -p "$d"
    (
        exec > "$d/run.log" 2>&1
        echo "envs=$envs gpus=$gpus cores=$cores workers/gpu=$wpg host=$(hostname)"
        your_ckpt="$CKPT" output_dir="$d" base_port=$port SKIP_AGGREGATE=true servers_per_gpu=1 \
            max_batch_size=$(( wpg > 32 ? wpg : 32 )) max_wait_time=0.01 \
            taskset -c "$cores" bash examples/LIBERO/eval_files/parallel_eval/auto_eval_libero.sh \
                "$SUITE" 2 "$wpg" "$TRIALS" "$gpus" 0 1 \
            | awk '{ print systime(), $0; fflush() }'
    )
}

for round in "8 16" "32 64"; do
    set -- $round
    echo "[$(date +%T)] round: $1 envs on GPUs 0,1 | $2 envs on GPUs 2,3"
    half "$1" 0,1 0-143 10093 & pa=$!
    half "$2" 2,3 144-287 10293 & pb=$!
    wait $pa $pb
    echo "[$(date +%T)] round done"
done
kill $SAMPLER
