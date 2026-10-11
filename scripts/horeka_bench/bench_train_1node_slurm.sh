#!/bin/bash
#SBATCH --job-name=hk_bench_train1n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/m3/blank4/horeka_bench/logs/slurm_%j.out
#SBATCH --partition=booster
#SBATCH -A m3
# HoreKa proposal benchmark A, single node: Qwen3.5 2B/4B/9B GR00T VLA, DeepSpeed ZeRO-2, 1/2/4 GPUs.
# Weak scaling: per model, the per-GPU batch is the largest power of 2 that fits at the smallest GPU
# count that runs at all (search descends from the first candidate), then held fixed for more GPUs.
# Each measurement: 41 steps = 10 warmup + 30 measured step intervals. Usage: sbatch bench_train_1node_slurm.sh [models...]
set -uo pipefail
ml load CUDA
B=/e/project1/m3/blank4/code/starVLA/scripts/horeka_bench
MODELS=("$@"); [ $# -eq 0 ] && MODELS=(2b 4b 9b)
# 2B b32 OOMed marginally on the shared login GPU (needed 3.2 GiB, 2.1 free); retried on a clean node.
declare -A CANDIDATES=([2b]="32 16 8 4" [4b]="16 8 4 2" [9b]="16 8 4 2")
# 9B searches its batch at 4 GPUs (the multi-node baseline), then runs 2 and 1 GPU at that batch; 1 GPU is
# expected to OOM (9.4B x 16 B = 150 GB of weights/grads/master/Adam > 95 GB HBM; ZeRO cannot shard on 1 GPU).
declare -A GPUS=([2b]="1 2 4" [4b]="1 2 4" [9b]="4 2 1")
nvidia-smi -L; nvidia-smi topo -m

for m in "${MODELS[@]}"; do
    bs=""
    for g in ${GPUS[$m]}; do
        if [ -z "$bs" ]; then
            for c in ${CANDIDATES[$m]}; do
                bash "$B/run_train_one.sh" "$m" "$g" "$c"; rc=$?
                [ $rc -eq 0 ] && { bs=$c; break; }
                [ $rc -ne 3 ] && break   # non-OOM failure: do not keep shrinking
            done
            [ -z "$bs" ] && echo "[$(date +%T)] $m g$g: no candidate batch fits"
        else
            bash "$B/run_train_one.sh" "$m" "$g" "$bs"
        fi
    done
    echo "[$(date +%T)] $m done, per-GPU batch ${bs:-none}"
done
