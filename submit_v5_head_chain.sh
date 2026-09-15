#!/bin/bash
# Submits the full v5 head-comparison chain for one seed:
#   train pair (PI + GR00T on one node)  ->  vanilla LIBERO eval per arm
#                                        ->  LIBERO-plus eval per arm
# Both eval jobs depend on the TRAIN job (afterok), not on each other, so the
# quick vanilla eval and the long Plus eval run in parallel once training lands.
set -euo pipefail
SEED="${1:?usage: submit_v5_head_chain.sh <seed>}"
cd "$(dirname "$0")"

tid=$(sbatch --parsable --job-name="tr_v5_s${SEED}" train_v5_head_pair_slurm.sh "${SEED}")
echo "train  ${tid}  tr_v5_s${SEED}  (PI + GR00T)"

for arm in pi gr00t; do
    case "$arm" in
        pi)    run="ervla_v5_pi_actiononly_pifix_s${SEED}" ;;
        gr00t) run="ervla_v5_gr00t_actiononly_s${SEED}" ;;
    esac
    ck="playground/Checkpoints/${run}/checkpoints/steps_20000_pytorch_model.pt"

    # Vanilla LIBERO: 10 trials/task x 4 suites = 400 episodes, as the controls ran.
    v=$(sbatch --parsable --dependency="afterok:${tid}" --job-name="ev_v5_${arm}_s${SEED}" \
        eval_libero_slurm.sh --ckpt "${ck}" --num_trials 10)
    # LIBERO-plus: exact-1000 per suite = 4000 episodes, the protocol every
    # current number on the index page uses.
    p=$(sbatch --parsable --dependency="afterok:${tid}" --job-name="ep_v5_${arm}_s${SEED}" \
        eval_libero_plus_slurm.sh --ckpt "${ck}" --exact_tasks_per_suite 1000)
    echo "  eval ${v}  vanilla LIBERO   ${run}"
    echo "  eval ${p}  LIBERO-plus 4k   ${run}"
done
