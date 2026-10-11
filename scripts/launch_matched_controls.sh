#!/usr/bin/env bash
echo "RETIRED: this launcher encodes the obsolete 80k, 2-GPU x 16 control recipe." >&2
echo "Use papers/BiBrain_ICLR_2027/RUN_PLAN.md C15: 30k, 4 GPUs x 8, 1k warmup." >&2
exit 2

# Prepared controls for the BiBrain paper (papers/BiBrain_ICLR_2027/RESULTS_AND_TODO.md §9).
# Prints the exact sbatch lines; submits ONLY with --submit.  Run from the starVLA repo root.
# Every pair job = train_libero_grid_pair_slurm.sh (2 GPUs x 16/device = global batch 32 per arm,
# 12 h auto-resume segments, exact4k LIBERO-Plus auto-submitted at 20/40/60/80k).
set -euo pipefail
cd /e/project1/m3/blank4/code/starVLA
T=examples/LIBERO/train_files
WANDB_GROUP=libero_matched_controls
SUBMIT=0; [ "${1:-}" = "--submit" ] && SUBMIT=1
G80="--datasets.vla_data.per_device_batch_size 16 --trainer.max_train_steps 80000 --trainer.num_warmup_steps 8000 --trainer.save_interval 20000 --trainer.eval_interval 1000 --trainer.save_full_training_state true --trainer.save_total_limit 32 --trainer.full_state_save_total_limit 2 --wandb_group $WANDB_GROUP"
run() { echo "+ $*"; if [ $SUBMIT = 1 ]; then "$@"; fi; }
group() { mkdir -p "playground/Checkpoints/$1"; echo "$WANDB_GROUP" > "playground/Checkpoints/$1/.wandb_group"; }

echo "# Pair A (seed 42, b32/80k): Ctrl-1 fully trainable causal GR00T  +  Ctrl-4 encoder = mask flip WITHOUT conversion checkpoint"
group ervla_ctrl1_causal_gr00t_b32_trainable_s42; group ervla_ctrl4_encfull_noconv_b32_s42
run sbatch --parsable --job-name=ctrl1_4 --time=12:00:00 train_libero_grid_pair_slurm.sh \
  $T/ervla_causal_gr00t_fullmem_noz_b64_trainable.yaml ervla_ctrl1_causal_gr00t_b32_trainable_s42 \
  $T/ervla_v5_gr00t_fullmem_noz_noconv.yaml ervla_ctrl4_encfull_noconv_b32_s42 42 $G80
echo
echo "# Pair B (seed 43, b32/80k): Ctrl-3 seed replicate of the headline pair, causal arm fully trainable"
group ervla_ctrl3_encfull_b32_s43; group ervla_ctrl3_causal_gr00t_b32_trainable_s43
run sbatch --parsable --job-name=ctrl3_s43 --time=12:00:00 train_libero_grid_pair_slurm.sh \
  $T/ervla_v5_gr00t_fullmem_noz.yaml ervla_ctrl3_encfull_b32_s43 \
  $T/ervla_causal_gr00t_fullmem_noz_b64_trainable.yaml ervla_ctrl3_causal_gr00t_b32_trainable_s43 43 $G80
echo "# Pair D (seed 42, b32/80k): Ctrl-5 fully trainable causal 4-token bottleneck  +  encoder bottleneck replicate (existing zonly recipe)"
group ervla_ctrl5_causal_zonly_b32_trainable_s42; group ervla_ctrl5_encoder_zonly_b32_s42rep
run sbatch --parsable --job-name=ctrl5_z --time=12:00:00 train_libero_grid_pair_slurm.sh \
  $T/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64_trainable.yaml ervla_ctrl5_causal_zonly_b32_trainable_s42 \
  $T/ervla_gr00t_sharedz_v5_zmem4_tokenonly_zonly.yaml ervla_ctrl5_encoder_zonly_b32_s42rep 42 $G80
echo
echo "# Deferred by RUN_PLAN.md: obsolete Ctrl-2 v2 and non-priority Qwen3.5-2B Ctrl-6 are NOT submitted by this launcher."
echo
echo "# Evals once a run reaches its endpoint (write to a FRESH output_dir; re-running into an existing dir merges):"
echo "#   sbatch --nodes=4 --export=ALL,output_dir=playground/Checkpoints/<run>/results/libero-plus-full-step80000 eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/<run>/checkpoints/steps_80000_pytorch_model.pt"
echo "#   sbatch eval_libero_slurm.sh --ckpt playground/Checkpoints/<run>/checkpoints/steps_30000_pytorch_model.pt            # plain LIBERO, 800 ep, state per checkpoint handshake"
echo "#   STARVLA_EVAL_INCLUDE_STATE=0 variant for the paired state-off control of Ctrl-2."
echo "# Eval-1: finish enc_v1_state libero_10; LIBERO-Plus full protocol for ervla_state_{gr00t,piv4}_{enc,causal}_v2_b32_s42 @30k."
[ $SUBMIT = 1 ] || echo "(dry run — nothing submitted; pass --submit to launch)"
