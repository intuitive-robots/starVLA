#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --partition=booster
#SBATCH -A m3
#SBATCH --output=slurm_logs/starvla_%j.out
#SBATCH --error=slurm_logs/starvla_%j.err
#SBATCH --signal=B:USR1@900
# train_libero_slurm.sh with a safe pre-time-limit checkpoint, for runs that must stop at a
# wall-clock boundary (maintenance) and resume later with --trainer.is_resume true.
# Slurm sends USR1 15 min before the limit; the trap touches STARVLA_PREEMPT_FLAG, which the
# trainer polls every STARVLA_PREEMPT_POLL_STEPS steps, agrees on across ranks, then saves
# (weights + full training state when save_full_training_state is on) and exits.
# Arguments are passed through to train_libero_slurm.sh unchanged.
set -u
# STARVLA_REPO may point at a worktree to train its code (default: the live tree)
export STARVLA_REPO=${STARVLA_REPO:-/e/project1/m3/blank4/code/starVLA}
cd "$STARVLA_REPO"
export STARVLA_PREEMPT_FLAG="/e/scratch/m3/blank4/starvla_preempt/${SLURM_JOB_ID}"
mkdir -p "$(dirname "$STARVLA_PREEMPT_FLAG")"
rm -f "$STARVLA_PREEMPT_FLAG"
trap 'echo "$(date --iso-8601=seconds) signal: requesting checkpoint"; touch "$STARVLA_PREEMPT_FLAG"' USR1 TERM

bash "$STARVLA_REPO/train_libero_slurm.sh" "$@" &
pid=$!
# A trapped signal makes `wait` return early; keep waiting until the trainer really exits.
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
wait "$pid" 2>/dev/null; rc=${rc:-$?}
echo "$(date --iso-8601=seconds) train_libero_slurm.sh exited rc=$rc"
exit "$rc"
