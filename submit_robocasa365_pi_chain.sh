#!/bin/bash
# Submits the RoboCasa365 Atomic-Seen PI backbone comparison for one seed:
#   one node, two arms (causal vs v5 encoder-decoder), 2 GPUs each.
#
#   ./submit_robocasa365_pi_chain.sh 42
#   ./submit_robocasa365_pi_chain.sh 4
#
# Evaluation is NOT chained here. RoboCasa365 eval needs a websocket policy
# server plus a simulator client, and run_eval.sh still defaults to
# MAX_STEPS=500 and N_ACT=8 -- 500 truncates 12 of the 18 atomic tasks, whose
# registry horizons run to 1050. Wire per-task horizons in before chaining it.
set -euo pipefail
SEED="${1:?usage: submit_robocasa365_pi_chain.sh <seed>}"
cd "$(dirname "$0")"

tid=$(sbatch --parsable --job-name="tr_rc365_s${SEED}" train_robocasa365_pi_pair_slurm.sh "${SEED}")
echo "train  ${tid}  tr_rc365_s${SEED}  (causal + v5)"
for arm in causal v5; do
    echo "  -> playground/Checkpoints/ervla_robocasa365_pi_${arm}_s${SEED}/checkpoints/"
done
