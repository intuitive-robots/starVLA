#!/bin/bash
#SBATCH --job-name=abc_sim_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=00:45:00
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/abc_sim_eval_%j.out
# ABC-Sim eval of an ABC-130K (m3 bridge) checkpoint, whole node: one policy server per GPU, CLIENTS_PER_GPU
# abc.sif simulator clients per GPU. Pass settings through --export (see eval_abc_sim_node.sh), e.g. from the live tree:
#   sbatch --export=ALL,CKPT=<.../steps_N_pytorch_model.pt>,NUM_EPISODES=50,CLIENTS_PER_GPU=12 \
#       --exclude=$(grep -v '^\s*#' config/libero_sim_node_blacklist.txt | sed 's/#.*//' | tr ',' '\n' | tr -d ' ' | grep ^jpbo | sort -u | paste -sd, -) \
#       examples/ABC/eval_files/eval_abc_sim_slurm.sh
# Results: $OUT/summary.json (default <run>/results/abc_sim/steps_N), per-episode JSONL, logs/.
set -uo pipefail
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-288}" \
    bash /e/project1/m3/blank4/code/starVLA/examples/ABC/eval_files/eval_abc_sim_node.sh
