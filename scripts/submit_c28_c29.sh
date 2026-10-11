#!/bin/bash
# C28/C29 submission (2026-09-23): full six-paradigm matrix on LIBERO and RoboCasa.
# Every training is gated afterok on its own smoke; every eval on its training.
# Each submission is appended to $LOG as: name<TAB>jobid<TAB>dependency<TAB>command
set -u
LV=/e/project1/m3/blank4/code/starVLA; WT=/e/project1/m3/blank4/code/starVLA-upstream-merge
LOG=$LV/results_collected/submit_c28_c29_$(date +%Y%m%d_%H%M).tsv; : > $LOG
D=/e/scratch/m3/blank4/rc365_smoke
CP1=/e/project1/m3/blank4/code/train_downstream/train/outputs/insight_cp1_q3vl2b_causal_v5/2026-09-22_18-02-59_job1955957/checkpoints/checkpoint-28786
PRE=/e/project1/m3/blank4/code/train_downstream/train/outputs/insight_prefixlm_q3vl2b_v5/2026-09-23_03-07-52_job1958368/checkpoints/checkpoint-28786
PREDIR=/e/project1/m3/blank4/code/train_downstream/train/outputs/insight_prefixlm_q3vl2b_v5
LT=examples/LIBERO/train_files; RT=examples/simBenchmarks/Robocasa_365/train_files
CK=$LV/playground/Checkpoints
G30=(--datasets.vla_data.per_device_batch_size 8 --trainer.max_train_steps 30000 --trainer.num_warmup_steps 1000
     --trainer.save_interval 5000 --trainer.eval_interval 1000 --trainer.gradient_checkpointing false
     --trainer.save_full_training_state true --trainer.save_total_limit 2 --trainer.full_state_save_total_limit 2
     --framework.qwenvl.attn_implementation flash_attention_3 --wandb_group libero_matched_controls_30k)
DRY=${DRY:-1}

sb() {  # sb <name> <dep-or-"-"> <workdir> <command...>; prints the job id
  local name=$1 dep=$2 wd=$3; shift 3
  local depflag=(); [ "$dep" != "-" ] && depflag=(--dependency=$dep)
  local cmd=("$@")
  # insert the dependency right after the sbatch / sbatch_libero_sim.sh token
  local out
  if [ "$DRY" = 1 ]; then out="DRY$RANDOM"
  else
    if [ "${cmd[0]}" = sbatch ]; then out=$(cd $wd && sbatch --parsable -J $name "${depflag[@]}" "${cmd[@]:1}" 2>&1 | tail -1)
    else out=$(cd $wd && "${cmd[0]}" "${cmd[1]}" --parsable -J $name "${depflag[@]}" "${cmd[@]:2}" 2>&1 | tail -1); fi
  fi
  printf "%s\t%s\t%s\t(cd %s) %s\n" "$name" "$out" "$dep" "$wd" "${cmd[*]}" >> $LOG
  echo "$out"
}
libnom() { sb "$1" "$2" $LV bash scripts/sbatch_libero_sim.sh -t 01:00:00 --kill-on-invalid-dep=yes --export=ALL,output_dir=$CK/$3/results/libero-nominal-30k-s$4 eval_libero_slurm.sh --ckpt $CK/$3/checkpoints/steps_30000_pytorch_model.pt --workers_per_gpu 4 --max_concurrent_workers_per_gpu 2; }
libfull() { sb "$1" "$2" $LV bash scripts/sbatch_libero_sim.sh -t 05:00:00 --kill-on-invalid-dep=yes --export=ALL,output_dir=$CK/$3/results/libero-plus-full10030-30k-s$4 eval_libero_plus_slurm.sh --ckpt $CK/$3/checkpoints/steps_30000_pytorch_model.pt --workers_per_gpu 6 --max_concurrent_workers_per_gpu 6 --exact_tasks_per_suite 0 --resume; }
rceval() { sb "$1" "$2" $LV sbatch -t $4 -o $D/${1}_%j.out -e $D/${1}_%j.out --export=ALL,MANIFEST=$D/$3,OUTDIR=robocasa365_gr00tlad,SEED=$5,VIDEOS=0,CKPT_STEP=${6:-50000},UNIT_ATTEMPTS=5 $D/rc365_units.sbatch; }

echo "=== A1 PrefixLM s43 (smoke 1963769 already PASSED)"
R=insight_q3vl2_prefixlm_v5_gr00t_state_b32_s43
PF43=$(sb p0_c22_prefpol_s43 - $LV sbatch -t 08:00:00 scripts/train_libero_final_checkpoint_slurm.sh $PREDIR 8 28786 --framework.qwenvl.base_vlm $LT/ervla_v5_gr00t_state_prefixlm_v3_b32.yaml $R 43 --datasets.vla_data.per_device_batch_size 8 --trainer.max_train_steps 30000 --trainer.num_warmup_steps 1000 --trainer.gradient_checkpointing false --framework.qwenvl.attn_implementation flash_attention_3)
libnom p0_c22_prefpol_nom_s43 afterok:$PF43 $R 43 >/dev/null
libfull p0_c22_prefpol_lp10k_s43 afterok:$PF43 $R 43 >/dev/null

echo "=== A2 CP-1 -> bidirectional evals"
for s in 42 43; do J=$([ $s = 42 ] && echo 1967225 || echo 1967226); R=insight_cp1bidir_q3vl2b_noconv_state_gr00t30_s$s
  libnom p0_c27_cb_nom_s$s afterok:$J $R $s >/dev/null; libfull p0_c27_cb_lp10k_s$s afterok:$J $R $s >/dev/null; done

echo "=== A3 repairs"
sb p0_c20_c4_42_lp10k_r4 - $LV bash scripts/sbatch_libero_sim.sh -t 03:00:00 --export=ALL,output_dir=$CK/ctrl4_state_tokonly_enc_noconv_b32_30k_s42/results/libero-plus-full10030-reuse4k-30k,STARVLA_SKIP_EXISTING_EPISODES=1 eval_libero_plus_slurm.sh --ckpt $CK/ctrl4_state_tokonly_enc_noconv_b32_30k_s42/checkpoints/steps_30000_pytorch_model.pt --workers_per_gpu 6 --max_concurrent_workers_per_gpu 3 --exact_tasks_per_suite 0 --resume >/dev/null
sb p0_c19_modblk_lp10k_s43_r5 - $LV bash scripts/sbatch_libero_sim.sh -t 04:00:00 --export=ALL,output_dir=$CK/ervla_state_tokonly_gr00t_modblk_v3_b32_s43/results/libero-plus-full10030-30k-s43 eval_libero_plus_slurm.sh --ckpt $CK/ervla_state_tokonly_gr00t_modblk_v3_b32_s43/checkpoints/steps_30000_pytorch_model.pt --workers_per_gpu 6 --max_concurrent_workers_per_gpu 3 --exact_tasks_per_suite 0 --resume >/dev/null
sb p0_c19_modblk_nom_s43_r2 - $LV bash scripts/sbatch_libero_sim.sh -t 01:00:00 --export=ALL,output_dir=$CK/ervla_state_tokonly_gr00t_modblk_v3_b32_s43/results/libero-nominal-30k-s43 eval_libero_slurm.sh --ckpt $CK/ervla_state_tokonly_gr00t_modblk_v3_b32_s43/checkpoints/steps_30000_pytorch_model.pt --workers_per_gpu 4 --max_concurrent_workers_per_gpu 2 >/dev/null

echo "=== A4 LIBERO modality-z on every paradigm"
SMC=$(sb p0_c28_sm_mz_noconv - $LV sbatch -t 01:00:00 --export=ALL,SMOKE_PER_DEVICE_BATCH=8,SMOKE_EXTRA_B=--framework.qwenvl.base_vlm\ $CP1 smoke_insight_v3_pair_slurm.sh $LT/ervla_modz24_state_tokonly_noconv_b32_30k.yaml sm_mz_noconv_base $LT/ervla_modz24_state_tokonly_noconv_b32_30k.yaml sm_mz_noconv_cp1)
SMP=$(sb p0_c28_sm_mz_prefix - $LV sbatch -t 01:00:00 --export=ALL,SMOKE_PER_DEVICE_BATCH=8,SMOKE_EXTRA_A=--framework.qwenvl.base_vlm\ $PRE,SMOKE_EXTRA_B=--framework.qwenvl.base_vlm\ $PRE smoke_insight_v3_pair_slurm.sh $LT/ervla_modz24_state_tokonly_prefixlm_b32_30k.yaml sm_mz_prefix_a $LT/ervla_modz24_state_tokonly_prefixlm_b32_30k.yaml sm_mz_prefix_b)
W=scripts/train_libero_preempt.sbatch
for s in 42 43; do
  J=$(sb p0_c28_mzC_s$s afterok:$SMC $LV sbatch -t 06:00:00 $W --config $LT/ervla_modz24_state_tokonly_noconv_b32_30k.yaml --run_id p0_c28_modz24_noconv_s$s --seed $s "${G30[@]}"); libfull p0_c28_mzC_lp10k_s$s afterok:$J p0_c28_modz24_noconv_s$s $s >/dev/null
  J=$(sb p0_c28_mzCB_s$s afterok:$SMC $LV sbatch -t 06:00:00 $W --config $LT/ervla_modz24_state_tokonly_noconv_b32_30k.yaml --run_id p0_c28_modz24_cp1bidir_s$s --seed $s --framework.qwenvl.base_vlm $CP1 "${G30[@]}"); libfull p0_c28_mzCB_lp10k_s$s afterok:$J p0_c28_modz24_cp1bidir_s$s $s >/dev/null
  J=$(sb p0_c28_mzPF_s$s afterok:$SMP $LV sbatch -t 06:00:00 $W --config $LT/ervla_modz24_state_tokonly_prefixlm_b32_30k.yaml --run_id p0_c28_modz24_prefixlm_s$s --seed $s --framework.qwenvl.base_vlm $PRE "${G30[@]}"); libfull p0_c28_mzPF_lp10k_s$s afterok:$J p0_c28_modz24_prefixlm_s$s $s >/dev/null
done
J=$(sb p0_c28_mzE_s43 - $LV sbatch -t 06:00:00 $W --config $LT/ervla_modz24_state_tokonly_causal_b32_30k.yaml --run_id insight_cp2mz_q3vl2b_causal_modz24_s43 --seed 43 --framework.qwenvl.base_vlm $CP1 --framework.qwenvl.vlm_impl qwen3 "${G30[@]}")
libfull p0_c28_mzE_lp10k_s43 afterok:$J insight_cp2mz_q3vl2b_causal_modz24_s43 43 >/dev/null
libfull p0_c28_mzE_lp10k_s42 afterok:1970100 insight_cp2mz_q3vl2b_causal_modz24_s42_r2 42 >/dev/null
libfull p0_c28_mzD_lp10k_s43 afterok:1970101 p0_c15_modz24_state_tokonly_enc_b32_30k_s43_r2 43 >/dev/null
libfull p0_c28_mzA_lp10k_s42 - p0_c15_modz24_state_tokonly_causal_b32_30k_s42 42 >/dev/null
libfull p0_c28_mzD_lp10k_s42 - p0_c15_modz24_state_tokonly_enc_b32_30k_s42 42 >/dev/null

echo "=== A5 RoboCasa PrefixLM rung + A6/C29 RoboCasa modality-z (worktree)"
declare -A RCJ   # rung_seed -> training job id (for evals)
S=$(sb p0_c26_sm_rcg_prefixlm - $WT sbatch -t 01:00:00 train_seed_pair_slurm.sh $RT/ervla_robocasa365_gr00t_state_rr_prefixlm_noro_b64.yaml p0_c26_sm_rc365_gr00t_prefixlm_noro 4 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1)
rceval p0_c26_canary_prefixlm afterok:$S manifest_canary_prefixlm.txt 00:40:00 42 20 >/dev/null
for s in 4 42; do RCJ[prefixlm_$s]=$(sb p0_c26_rcg_prefixlm_s$s afterok:$S $WT sbatch -t 10:00:00 train_single_seed_4gpu_slurm.sh $RT/ervla_robocasa365_gr00t_state_rr_prefixlm_noro_b64.yaml p0_c26_rc365_gr00t_prefixlm_noro_b64_s$s $s); done
declare -A MZJ
for r in enc noconv causal cp2 cp1bidir prefixlm; do
  S=$(sb p0_c29_sm_$r - $WT sbatch -t 01:00:00 train_seed_pair_slurm.sh $RT/ervla_robocasa365_gr00t_state_rr_${r}_modz24_b64.yaml p0_c29_sm_rc365_${r}_modz24 4 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1)
  [ $r = causal ] && rceval p0_c29_canary_causal_modz afterok:$S manifest_canary_causal_modz.txt 00:40:00 42 20 >/dev/null
  for s in 4 42; do MZJ[${r}_$s]=$(sb p0_c29_rcmz_${r}_s$s afterok:$S $WT sbatch -t 10:00:00 train_single_seed_4gpu_slurm.sh $RT/ervla_robocasa365_gr00t_state_rr_${r}_modz24_b64.yaml p0_c29_rc365_gr00t_${r}_modz24_b64_s$s $s); done
done

echo "=== B4 RoboCasa evals (afterany: a failed/timed-out training makes its eval fail fast on the missing checkpoint)"
declare -A LAD=([enc_4]=1971277 [enc_42]=1967171 [noconv_4]=1967160 [noconv_42]=1967161 [causal_4]=1971273 [causal_42]=1971274 [cp2_4]=1971275 [cp2_42]=1971276 [cp1bidir_4]=1967228 [cp1bidir_42]=1967229)
LAD[prefixlm_4]=${RCJ[prefixlm_4]}; LAD[prefixlm_42]=${RCJ[prefixlm_42]}
for r in enc noconv causal cp2 cp1bidir prefixlm; do for s in 4 42; do
  rceval p0_c26_rclad_${r}_s$s afterany:${LAD[${r}_$s]} manifest_c26lad_${r}_s$s.txt 06:00:00 $s >/dev/null
  rceval p0_c29_rcmz_ev_${r}_s$s afterany:${MZJ[${r}_$s]} manifest_c29mz_${r}_s$s.txt 02:30:00 $s >/dev/null
done; done

echo "=== B3 state corruption (exact1k, s42) for E, C now; PrefixLM, CP-1->bidir after training"
for pair in "insight_cp2_q3vl2b_causal_state_gr00t30_s42 -" "ctrl4_state_tokonly_enc_noconv_b32_30k_s42 -" "insight_q3vl2_prefixlm_v5_gr00t_state_b32_s42 afterok:1963794" "insight_cp1bidir_q3vl2b_noconv_state_gr00t30_s42 afterok:1967225"; do
  set -- $pair; R=$1; dep=$2
  for MODE in shuffle random zero; do
    sb "p0_c28_int1k_${R:0:14}_${MODE}" $dep $LV bash scripts/sbatch_libero_sim.sh -t 02:00:00 --export=ALL,output_dir=$CK/$R/results/libero-plus-stateint-${MODE}-exact1k,STARVLA_STATE_CORRUPT=$MODE,STARVLA_STATE_CORRUPT_SEED=42,MAX_CONCURRENT_WORKERS_PER_GPU=3 eval_libero_plus_slurm.sh --ckpt $CK/$R/checkpoints/steps_30000_pytorch_model.pt --exact_tasks_per_suite 250 --workers_per_gpu 4 --max_concurrent_workers_per_gpu 3 --resume >/dev/null
  done
done
echo "done; log: $LOG"
