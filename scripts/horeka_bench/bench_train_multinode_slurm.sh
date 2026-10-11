#!/bin/bash
#SBATCH --job-name=hk_bench_trainNn
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:40:00
#SBATCH --output=/e/scratch/m3/blank4/horeka_bench/logs/slurm_%j.out
#SBATCH --partition=booster
#SBATCH -A m3
# HoreKa proposal benchmark A, multi-node weak scaling of one model (default 9b) at the per-GPU batch
# found by bench_train_1node_slurm.sh (read from its 4-GPU result). Usage:
#   sbatch --nodes=2 bench_train_multinode_slurm.sh [model]     # 8 GPUs
#   sbatch --nodes=4 bench_train_multinode_slurm.sh [model]     # 16 GPUs
#   sbatch --nodes=2 --export=ALL,BENCH_BS=16 bench_train_multinode_slurm.sh 9b   # other per-GPU batch
set -uo pipefail
ml load CUDA
export SLURM_MPI_TYPE=none
M=${1:-9b}
R=/e/scratch/m3/blank4/horeka_bench/results
BASE=$(ls "$R"/${M}_g4_n1_b*_z${BENCH_ZERO:-2}.json 2>/dev/null | head -1)
[ -n "$BASE" ] || { echo "no 4-GPU single-node result for $M in $R"; exit 1; }
BS=$(python3 -c "import json;print(json.load(open('$BASE'))['per_gpu_batch'])")
BS=${BENCH_BS:-$BS}   # override, e.g. a batch that only fits once ZeRO shards over more GPUs
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000)) NUM_MACHINES=$SLURM_NNODES
NG=$((4 * SLURM_NNODES))
echo "model $M, $SLURM_NNODES nodes, $NG GPUs, per-GPU batch $BS (from $BASE), master $MASTER_ADDR"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --cpus-per-task=288 --mpi=none --cpu-bind=none --kill-on-bad-exit=1 \
    bash -c 'MACHINE_RANK=$SLURM_NODEID exec bash /e/project1/m3/blank4/code/starVLA/scripts/horeka_bench/run_train_one.sh "$@"' \
    _ "$M" "$NG" "$BS"
