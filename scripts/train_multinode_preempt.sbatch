#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --signal=B:USR1@900
# Multi-node counterpart of scripts/train_libero_preempt.sbatch: runs train_multinode_container_slurm.sh and,
# 15 min before the time limit, touches STARVLA_PREEMPT_FLAG (shared filesystem, polled by every rank) so the
# trainer saves weights + full training state and exits. Resume with --trainer.is_resume true.
# Usage: sbatch [--nodes=N] scripts/train_multinode_preempt.sbatch --config X [overrides...]
set -u
cd /e/project1/m3/blank4/code/starVLA
export STARVLA_PREEMPT_FLAG="/e/scratch/m3/blank4/starvla_preempt/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$STARVLA_PREEMPT_FLAG")"
rm -f "$STARVLA_PREEMPT_FLAG"
trap 'echo "$(date --iso-8601=seconds) signal: requesting checkpoint"; touch "$STARVLA_PREEMPT_FLAG"' USR1 TERM

bash /e/project1/m3/blank4/code/starVLA/train_multinode_container_slurm.sh "$@" &
pid=$!
# A trapped signal makes `wait` return early; keep waiting until the trainer really exits.
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
wait "$pid" 2>/dev/null; rc=${rc:-$?}
echo "$(date --iso-8601=seconds) train_multinode_container_slurm.sh exited rc=$rc"
exit "$rc"
