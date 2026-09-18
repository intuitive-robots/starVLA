# Job ledger

Every SLURM job we launch goes in here **at submit time**, with its launch command, so any
session can pick up monitoring without reconstructing what was running. Live status:

```bash
bash scripts/jobs_status.sh                      # queue + anything untracked
bash scripts/jobs_status.sh --since 2026-09-16   # also finished / failed
```

`jobs_status.sh` flags queued jobs missing from this file, so the ledger cannot silently rot.
When a job finishes, move its row to **Finished** with the outcome. Recover a lost launch
command with `sacct -j <id> -X -o SubmitLine%400`.

Last refreshed: 2026-09-18 00:58 CEST.

## Running / queued

Refreshed 2026-09-18 00:58 CEST from `squeue`, `sacct`, raw training logs and raw rollout JSONs (see `scripts/jobs_status.sh`).

| Job | Name | What | State |
|---|---|---|---|
| 1873664 / 1873665 | ep_gz100_10k42 / 43 | Exact4k LIBERO-plus at **step 10,000** of `ervla_gr00t_sharedz_v5_memdrop100_b64` (intermediate-checkpoint comparison against the 20k result) | PENDING (Priority), 2h limit. Submitted 12:3x from another session — recorded here so `jobs_status.sh` stays clean. |
| 1873117 | ae-prepare | other workstream (not this session) — `AE_Data_SelectioN_Robotic/slurm/prepare.jupiter.sbatch`, submitted Sep18 11:14, 6h limit | PENDING (Priority); listed only so `jobs_status.sh` stops flagging it |
| 1864839 / 1864841 | ep_cnoz_42 / 43 | Exact4k causal full-hidden/no-z GR00T after paired training1864837/38 | COMPLETED, CLEAN:66.200% /60.375%; mean63.288%. |
| 1864840 / 1864842 | ep_cz4_42 / 43 | Exact4k causal fixed four-token shared-z GR00T after paired training1864837/38 | COMPLETED, CLEAN:57.275% /56.950%; mean57.113%; paired−6.175pp. |
| 1864837 / 1864838 | tr_cz42 / tr_cz43 | Matched causal full-hidden/no-z vs fixed four-token z-only memory, one paired4-GPU node per seed,20k,batch64/arm | COMPLETED; all checkpoints and four exact4k evaluations clean. |
| 1864672 / 1864673 | ep_tl0a_42 / 43 | Exact4k LIBERO-plus after matched two-pass null-trace seeds1864670/71 | PENDING (Dependency) |
| 1864670 / 1864671 | tr_tl0a_42 / 43 | Full same-compute two-pass null-trace GR00T, one seed per4-GPU node,16/device, global64 | SUBMITTED — smokes1864418/19 passed |
| 1864642 / 1864643 | sm_cz42 / sm_cz43 | Matched causal full-memory GR00T no-z vs fixed four-token shared-z; paired arms on2GPUs each, seeds42/43,20 updates + eval10/20 | SUBMITTED |
| 1864756 | ep_cot_wrong | Causal CoT wrong-trace intervention, exact4k, foreign generated trace via `STARVLA_COT_CORRUPT=roll` | **CANCELLED by user** Sep18 01:24 after 01:39:04. 768/1,000 `libero_10` episodes rolled (partial 0.823 — note this is above the 0.80 read earlier in the run), **0 completed shards**, so nothing aggregatable is on disk; a rerun starts from scratch. Throughput was 504 eps/h, which was normal for its layout (mean episode 7.6 min vs 6.8 min for the clean CoT run; the gap is the intervention's extra full-horizon failures), not a misconfiguration. Submitted with the pre-tuning layout (16 workers/GPU, batch 16, servers co-located). Tuned relaunch ready at `/e/scratch/m3/blank4/starVLA_trace_eval/relaunch_wrongtrace_tuned.sh` (32 workers/GPU, servers on GPU0, `--nodes` via `NODES=`); it also archives the stale 64-shard debris and the zeroed `overall_results.json` left by cancelled 1864634. |
| 1864634 / 1864635 | ep_cot_wrong / prompt | **CANCELLED before rollout** — detached worktree lacked generated policy-server Python wrapper; replaced with absolute-wrapper jobs1864756/57 |
| 1864418 / 1864419 | sm_tl0_s42 / s43 | Matched two-pass null-trace GR00T control, one seed per4-GPU node,16/device, global64;20 updates + eval10/20 | SUBMITTED |
| 1864094 / 1864095 -> 1871859 | ep_tl2a_42 / 43 | Exact4k LIBERO-plus, tied two-pass predicted-trace | **BOTH COMPLETED, CLEAN** (n=4,000, no failed suites/shards, 0 aborts). s42 **0.7448**; s43 (resumed as1871859 after the bad-GPU cancel) **0.7252** — `libero_10` 0.601, `goal` 0.735, `object` 0.806, `spatial` 0.759. Seed mean **0.735**. |
| 1864092 / 1864093 | tr_tl2a_42 / 43 | Tied two-pass predicted-trace GR00T, one seed per4-GPU node,16/device, global64,20k | SUBMITTED — smokes1864038/39 passed |
| 1864074 / 1864075 -> 1871858 | ep_tl1a_42 / 43 | Exact4k LIBERO-plus, alternating one-pass trace head | **BOTH COMPLETED, CLEAN**. s42 **0.7360**; s43 (resumed as1871858) **0.7465** — `libero_10` 0.628, `goal` 0.748, `object` 0.845, `spatial` 0.765. Seed mean **0.741**; the 0.011 seed gap confirms the bad-GPU restart did not perturb the run. |
| 1864073 | tr_tl1_alt | Matched one-pass trace-head GR00T control, no readout/no z, alternating attention, seeds42/43,20k,batch64 | SUBMITTED — smoke1863878 passed |
| 1864038 / 1864039 | sm_tl2_s42 / s43 | **PASSED** — one seed per4-GPU node,16/device, global64;20 updates,eval10/20,finite losses,positive pass delta,complete checkpoints |
| 1863879 | sm_tl_2p | **FAILED smoke** — two simultaneous2-GPU runs at32/device exceeded95GB/GPU before update1; no result promoted |
| 1863878 | sm_tl_1p | **PASSED** — matched one-pass control, seeds42/43,20 updates,eval10/20,finite losses,complete checkpoints |
| 1863856 / 1863857 | ep_gz4f015_42 / 43 | Exact4k LIBERO-plus after fixed z-memory4 dropout0.15 full pair1863853 | PENDING (Dependency) |
| 1863854 / 1863855 | ep_gz4f100_42 / 43 | Exact4k LIBERO-plus after fixed z-memory4 dropout1 full pair1863852 | PENDING (Dependency) |
| 1863853 | tr_gz4f_015 | Fixed GR00T z-memory4 + full encoder memory dropout0.15, seeds42/43,20k,batch64 | RUNNING — smoke1863773 passed |
| 1863852 | tr_gz4f_100 | Fixed GR00T z-memory4, encoder memory fully masked, seeds42/43,20k,batch64 | RUNNING — smoke1863772 passed |
| 1863773 | sm_gz4f_015 | **PASSED** — full-hidden fixed arm;20 updates,eval10/20,finite loss,mean keep rates0.823/0.872,complete checkpoints |
| 1863772 | sm_gz4f_100 | **PASSED** — z-memory-only fixed arm;20 updates,eval10/20,finite loss,keep rate0,complete checkpoints |
| 1863684 | sm_gz4_015 | **PASSED but superseded** — used 32 learned readout tokens; not promoted because GR00T retained-memory test should use full hidden sequence |
| 1863683 | sm_gz4_100 | **PASSED but superseded** — same readout config; z-only input itself was unaffected, but paired design was replaced |
| 1861663 | ep_gz_015_43 | Exact4k LIBERO-plus; afterok:1861661, seed43 | SUBMITTED |
| 1861662 | ep_gz_015_42 | Exact4k LIBERO-plus; afterok:1861661, seed42 | SUBMITTED |
| 1861661 | tr_gz_015_b64 | LIBERO GR00T+z dropout015, seeds42/43,20k updates,batch64; smoke1861586 passed | RUNNING — all four runs passed20 updates |
| 1861660 | ep_gz_100_43 | Exact4k LIBERO-plus; afterok:1861658, seed43 | SUBMITTED |
| 1861659 | ep_gz_100_42 | Exact4k LIBERO-plus; afterok:1861658, seed42 | SUBMITTED |
| 1861658 | tr_gz_100_b64 | LIBERO GR00T+z dropout100, seeds42/43,20k updates,batch64; smoke1861585 passed | RUNNING — all four runs passed20 updates |
| 1861586 | sm_gz_memdrop015_b64 | GR00T+z memdrop015_b64, seeds42/43;20 updates, trainer eval10/20, batch64/run | **PASSED** —20 updates,eval10/20,finite losses,complete checkpoints |
| 1861585 | sm_gz_memdrop100_b64 | GR00T+z memdrop100_b64, seeds42/43;20 updates, trainer eval10/20, batch64/run | **PASSED** —20 updates,eval10/20,finite losses,complete checkpoints |
| 1858694 | tr_v5_piv4 | Full QwenPI_v4 LIBERO seeds42/43 at20k after passed smoke | RUNNING |
| 1858835 / 1858836 | ep_piv4_s42 / s43 | Exact4k LIBERO-plus evals after successful full QwenPI_v4 training1858694 | **RUNNING** since Sep18 00:07, ~3,000 eps/h, on `libero_object`. s42: `libero_10` 0.657, `libero_goal` 0.759, object 750/1,000. s43: `libero_10` 0.688, `libero_goal` 0.717, object 611/1,000. ETA ~01:45. |
| 1863830 | tr_q35enc_resume | Resume of the 12h-timed-out Qwen3.5 encoder-only pair (`deeps` + `deeps_aug`, seed42) to 30k; arm a resumed from 25,908, arm b from 20,916 | **RUNNING** since Sep17 22:33 |
| 1863831 / 1863832 | ep_q35_s42 / ep_q35_aug_s42 | Exact4k LIBERO-plus evals of the two q35 encoder-only arms at 30k, gated on1863830 | PENDING (Dependency) |
| 1858730 | tr_rc365_piv4 | Full QwenPI_v4 RoboCasa seeds4/42 at50k after passed smoke | **TIMEOUT** 12:01:02, ended Sep18 05:18 at step 30,000/50,000 — expected, the 12h QOS wall; `afterany` continuation1858851 picked it up |
| 1858851 | tr_rcpiv4_rs | Automatic RoboCasa continuation from latest checkpoint after first12h segment1858730 | **RUNNING** on jpbo-022-42, 4:14 elapsed / 7:46 left. Resumed at 30,000; now s4 39,415 and s42 39,637 of 50,000 at ~37 steps/min -> 50k in ~4.7h, ~3h of wall to spare. |
| 1858852 / 1858853 | rc_piv4_s4 / s42 | RoboCasa17-task ×48-episode evals at step50k after continuation1858851 | **PENDING (Dependency)** — gated `afterok:1858851`, so they only fire if the continuation *completes*; another TIMEOUT would cancel both and they would need resubmitting with the parent. |
| 1851384 | ep_zonly_gr00t_s43 | LIBERO-plus exact4k, zonly+GR00T s43 | **COMPLETED** — 3131/4000 = 78.275% |
| 1851383 | ep_zonly_gr00t_s42 | LIBERO-plus exact4k, zonly+GR00T s42 | **COMPLETED** — 3147/4000 = 78.675% |
| 1850693 | enc_dec_2b_v5_final_action_tracetime_w001 | other workstream (not this session) | RUNNING 21:52 |
| 1850350 / 1858474 | ep_t_trace_ours_v3_cotw01 / spatial resume | CoT-trace exact4k — ours_v3_cotw01 | first job **TIMEOUT**; resume **COMPLETED**,3323/4000=83.075%, empty failure sentinels |
| 1850351 / 1856685 | ep_ot_trace_det_v3_cotw01 / goal resume | CoT-trace exact4k — det_v3_cotw01 | first job **TIMEOUT**; resume **COMPLETED**,3249/4000=81.225% |
| 1850352 | ep_race_det_cotw1_readout | CoT-trace re-eval @1000 eps/suite — det_cotw1_readout (was 0.790) | RUNNING 33:30 |
| 1850353 | ep__trace_ours_full_cotw1 | CoT-trace re-eval @1000 eps/suite — ours_full_cotw1 (was 0.787) | RUNNING 33:30 |
| 1850294 | ep_zonly_v5_s42 | LIBERO-plus eval, zonly v5 s42 @1000 eps (rerun after CUDA abort) | RUNNING 38:18 |
| 1851073 | full_table | other workstream (not this session) | RUNNING 9:59 |
| 1850917 | trace_cls | other workstream (not this session) | RUNNING 17:11 |
| 1850162 | act_arm_probe2 | other workstream (not this session) | RUNNING 45:09 |

## Finished

| Job | Name | Outcome |
|---|---|---|
| 1858852 / 1858853 | rc_piv4_s4 / s42 | RoboCasa 17-task x48-episode eval at step50k, QwenPI_v4 (encoder-only, `skip_decoder`, dense layer-wise conditioning) | **RUNNING, 16/17 tasks, zero retries.** s4 **0.661** (508/768), s42 **0.659** (506/768). Directly comparable with 1842871 (same `rc365_units.sbatch`, same CKPT_STEP=50000, same 48 eps/task): on the 16 common tasks **piv4 0.661/0.659 vs pi_v5 enc-dec 0.379 and pi_causal 0.354** — a +0.28 absolute, ~+75% relative jump, the largest RoboCasa gain recorded so far. Biggest per-task movers: OpenCabinet 0.02/0.06 -> 0.67, NavigateKitchen 0.15/0.10 -> 0.77, PickPlaceToasterToCounter 0.17/0.08 -> 0.81, CloseFridge 0.17/0.21 -> 0.77. PickPlace tasks barely move (PickPlaceCounterToStove 0.65 -> 0.60, i.e. slightly down) and CloseBlenderLid stays near the floor (0.04/0.06 -> 0.15/0.33). |
| 1863854 / 1863855 | ep_gz4f100_42 / 43 | Exact4k LIBERO-plus — GR00T shared-z zmem4, memory_dropout 1.0 (z-only) | **COMPLETED, CLEAN** both seeds. s42 **0.7650** (0.657/0.797/0.830/0.776), s43 **0.7595** (0.609/0.783/0.845/0.801). Seed mean **0.762** — best of this batch, still below the zonly+GR00T 0.7847. |
| 1863856 / 1863857 | ep_gz4f015_42 / 43 | Exact4k LIBERO-plus — GR00T shared-z zmem4, memory_dropout 0.15 | **COMPLETED, CLEAN** both seeds. s42 **0.7598** (0.672/0.762/0.835/0.770), s43 **0.7568** (0.657/0.758/0.817/0.795). Seed mean **0.758** — best of this batch, still under the zonly+GR00T 0.7847. |
| 1864839–1864842 | ep_cnoz / ep_cz4, seeds42/43 | Exact4k LIBERO-plus — causal full-memory no-z versus causal four-token z-only replacement | **COMPLETED, CLEAN**. No-z:0.6620/0.6038; z-only:0.5728/0.5695. Paired z-only loss−0.0893/−0.0343 (mean−0.0618), driven by goal:−0.266/−0.115. This is a negative replacement-memory result, not an additive-z test. |
| 1864757 | ep_cot_prompt | Causal CoT prompt-only intervention, exact4k, identical weights, `generate_at_inference:false` | **COMPLETED** 01:05:55 (3,573 eps/h — action-head speed, no autoregressive pass). 4,000/4,000 eps, zero failed shards: `libero_10` 0.786, `libero_goal` 0.774, `libero_object` 0.867, `libero_spatial` 0.857 -> **mean 0.821** (3,284/4,000). `failed_suites.txt` in its output dir was stale debris from the cancelled 1864635 (names node jpbo-023-10; this job ran on jpbo-047-22) and is archived as `failed_suites.stale_from_1864635.txt`. |
| 1863688 / 1863690 / 1863692 | tr_rc_gz_42 / tr_rc_gz_4 / tr_rc_gzm | **CANCELLED by user at57s** — RoboCasa GR00T full runs stopped after a flaw was found in z-only conditioning; no result may be used |
| 1863689 / 1863691 / 1863693 | tr_rc_gz42_rs / tr_rc_gz4_rs / tr_rc_gzm_rs | **CANCELLED before start** — automatic resume dependencies for the stopped RoboCasa runs |
| 1863697–1863702 | rc_gnoz / rc_gz / rc_gm015, seeds4/42 | **CANCELLED before start** — chained18-task evaluations for the stopped RoboCasa runs |
| 1863647 / 1863648 | sm_rc_gz0 / sm_rc_gzm | **PASSED AS INTEGRATION TESTS ONLY** — four20-update runs,eval10/20,finite losses/gradients,all six z targets and complete checkpoints; architecture invalidated before full training |
| 1857753 | ix_piv4_smoke | **COMPLETED / released early** — interactive node ran both corrected two-seed smoke pairs; both returned0 and the allocation was relinquished after10m |
| 1857753 / LIBERO phase | ix_piv4_smoke | **PASSED** — both seeds completed20 updates and step10/20 eval; s42 loss1.675→0.904, MSE.01982→.01378; s43 loss1.049→0.846, MSE.01945→.01346; nonzero encoder gradients and complete step20 checkpoints |
| 1857753 / RoboCasa phase | ix_piv4_smoke | **PASSED** — both seeds completed20 updates and step10/20 eval; s4 loss1.436→0.969, MSE.01424→.01165; s42 loss1.388→0.964, MSE.01342→.01193; state retained, nonzero encoder gradients and complete step20 checkpoints |
| 1857343 | sm_v5_piv4 | **FAILED before update1** — v4 bypassed v3 constructor and lacked `cot_dropout_enabled`; fixed in4dde3dd |
| 1857345 | sm_rc365_piv4 | **FAILED before update1** — same missing constructor state; fixed in worktree commitd2a0690 |
| 1851382 | tr_zonly_gr00t | **COMPLETED** — seeds42/43 reached20k in5h55m; checkpoints ready and exact4k evals1851383/84 eligible |
| 1857317 / 1857332 | sm_v5_piv4 / sm_rc365_piv4 | **CANCELLED while pending** — resubmitted as two-hour backfill jobs1857343/1857345; no work ran |
| 1857258 | tr_v5_piv4 | **CANCELLED before start** — replaced by mandatory20-update/in-training-eval smoke gate |
| 1857259 / 1857260 | ep_piv4_s42 / s43 | **CANCELLED before start** with parent full run; these were simulator evals and are not part of the training smoke |
| 1850810 | rc365_nostate | **No-state ablation.** causal collapses to 0.00 on all three tasks; v5 keeps 0.83 / 0.48 / 0.46. The enc-dec arm is the one that does NOT need proprioception |
| 1850179 | rc365_eval_s42b | 15/34 units — EGL aborts again; seed-42 robocasa benchmark still incomplete |
| 1850996 | tr_q35enc_pair | FAILED — preflight printed OK then training still died: fla binds per PROCESS, not per node, so a separate-process probe cannot catch it |
| 1850178 | fla_probe | triton=cuda, fla=cuda on its node, with and without a prior CUDA touch |
| 1842871 | rc365_eval_s4 | 34/34 units. **causal 0.363 vs v5 0.373** over 17 tasks (816 rollouts/arm) — no difference |
| 1842595 | rc365_openloop | policies agree on identical states: cosine 0.964, gripper 98%, symmetric both directions |
| 1836346 / 1836350 | tr_zonly_v5 / tr_rc365_s42_rs | COMPLETED (7:31 / 7:37) |
| 1836353 | ep_zonly_v5_s42 | FAILED — CUDA abort on jpbo-058-08 → rerun as 1850294 |
| 1836354 | ep_zonly_v5_s43 | COMPLETED — **0.783**, the best protocol-complete LIBERO-plus number we have |
| 1826641 | tr_zonly_aug | COMPLETED — augmentation gives 0.763 vs 0.776 non-aug (n.s.) |

## Launch commands

### Step-10k shared-z evals 1873664 / 1873665

Run from `/e/project1/m3/blank4/code/starVLA`. Not submitted by this session;
recovered with `sacct -j <id> -X -o SubmitLine%500`.

```bash
sbatch --parsable --time=02:00:00 --job-name=ep_gz100_10k42 \
  --export=ALL,POLICY_SERVER_GPU=,output_dir=playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s42/results/libero-plus-step10000-exact4k-v1 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s42/checkpoints/steps_10000_pytorch_model.pt \
  --exact_tasks_per_suite 1000   # 1873664  (s43 identical -> 1873665)
```


### Bad-GPU recovery resumes 1871858 / 1871859

Run from `/e/project1/m3/blank4/code/starVLA`. Identical to the original
1864075 / 1864095 lines except the dependency is dropped (parents finished) and
`--resume` is added, which skips shards whose result JSON is already on disk.
The layout is deliberately unchanged: changing `workers_per_gpu` renumbers the
shards and would discard every banked shard.

```bash
sbatch --parsable --time=05:00:00 --job-name=ep_tl1a_43r --export=ALL,POLICY_SERVER_GPU= \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_1pass_noz_all_s43/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 \
  --max_batch_size 4 --max_wait_time 0.0 --resume   # 1871858

sbatch --parsable --time=05:00:00 --job-name=ep_tl2a_43r --export=ALL,POLICY_SERVER_GPU= \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_2pass_noz_pred_all_s43/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 \
  --max_batch_size 4 --max_wait_time 0.0 --resume   # 1871859
```


Run training/eval launchers from the tree that owns them. **RoboCasa365 lives in the
worktree** `/e/project1/m3/blank4/code/starVLA-upstream-merge` (branch `merge_upstream_2026_09`);
everything else from `/e/project1/m3/blank4/code/starVLA`.

### Fixed GR00T shared-z cross-memory smokes 1863772 / 1863773

The four projected memory tokens are decoded from the same 128D z bottleneck. They are
always visible to cross-attention; dropout masks the full final encoder hidden sequence.
Each job runs seeds42/43 concurrently, two GPUs/run and batch32/device (global64/run).
Smokes1863683/84 completed but used the 32-token learned readout projector; they were
superseded before full training because no prior GR00T retained-memory result motivates
that local fork component.

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_gz4f_100 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_memdrop100_b64.yaml ervla_gr00t_sharedz_v5_zmem4_fullmem_memdrop100_b64_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
sbatch --parsable -t 02:00:00 --job-name=sm_gz4f_015 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_memdrop015_b64.yaml ervla_gr00t_sharedz_v5_zmem4_fullmem_memdrop015_b64_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```

### Fixed GR00T shared-z full runs and exact4k evaluations 1863852–1863857

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_gz4f_100 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_memdrop100_b64.yaml ervla_gr00t_sharedz_v5_zmem4_memdrop100_b64 42 43
sbatch --parsable --time=12:00:00 --job-name=tr_gz4f_015 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_memdrop015_b64.yaml ervla_gr00t_sharedz_v5_zmem4_memdrop015_b64 42 43
```

Exact4k jobs1863854/55 depend afterok on1863852; jobs1863856/57 depend afterok on
1863853. Each uses the corresponding seed42/43 step20k checkpoint with1000 exact tasks
per suite,8 workers/GPU,2 servers/GPU,batch4 and zero batching wait.

### Alternating trace-loop smokes 1863878 / 1863879

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_tl_1p train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_1pass_noz_all.yaml ervla_v5_gr00t_traceloop_1pass_noz_all_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
sbatch --parsable -t 02:00:00 --job-name=sm_tl_2p train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_pred_all.yaml ervla_v5_gr00t_traceloop_2pass_noz_pred_all_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```

Job1863879 failed before update1 because the two-pass graph exceeded95GB/GPU at
32 samples/device. Its replacement keeps effective batch64 and gives each seed a full
4-GPU node at16 samples/device. The smoke now logs `trace_pass_delta`, trace loss,
coverage and prediction spread so a pass requires evidence that re-encoding changed the
pass2 hidden sequence.

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_tl2_s42 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_pred_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_pred_all_4gpu_smoke_s42 --seed 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
sbatch --parsable -t 02:00:00 --job-name=sm_tl2_s43 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_pred_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_pred_all_4gpu_smoke_s43 --seed 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```

The one-pass smoke1863878 passed for both seeds, so its matched full control and exact4k
evaluations were released without waiting for the two-pass smoke:

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_tl1_alt train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_1pass_noz_all.yaml ervla_v5_gr00t_traceloop_1pass_noz_all 42 43  # 1864073
sbatch --parsable --time=05:00:00 --dependency=afterok:1864073 --job-name=ep_tl1a_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_1pass_noz_all_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864074
sbatch --parsable --time=05:00:00 --dependency=afterok:1864073 --job-name=ep_tl1a_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_1pass_noz_all_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864075
```

Both four-GPU two-pass smokes passed: eval trace losses0.01683/0.01568,
prediction standard deviations0.0687/0.0649, trace-target coverage0.9375 and
pass1-to-pass2 hidden deltas3.142/2.792 for seeds42/43. Full jobs and their
separate exact4k dependencies were released:

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_tl2a_42 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_pred_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_pred_all_s42 --seed 42  # 1864092
sbatch --parsable --time=12:00:00 --job-name=tr_tl2a_43 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_pred_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_pred_all_s43 --seed 43  # 1864093
sbatch --parsable --time=05:00:00 --dependency=afterok:1864092 --job-name=ep_tl2a_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_2pass_noz_pred_all_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864094
sbatch --parsable --time=05:00:00 --dependency=afterok:1864093 --job-name=ep_tl2a_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_2pass_noz_pred_all_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864095
```

The same-compute null-trace control is claim-critical: it replaces predicted coordinates
with constant0.5 coordinates while retaining the second tied encoder pass, trace head/loss,
slots and action path. Its full run is gated on these smokes:

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_tl0_s42 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_null_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_null_all_4gpu_smoke_s42 --seed 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1  # 1864418
sbatch --parsable -t 02:00:00 --job-name=sm_tl0_s43 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_null_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_null_all_4gpu_smoke_s43 --seed 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1  # 1864419
```

Both null smokes passed with complete checkpoints and eval pass deltas2.994/3.024;
full runs and exact4k dependencies were released:

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_tl0a_42 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_null_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_null_all_s42 --seed 42  # 1864670
sbatch --parsable --time=12:00:00 --job-name=tr_tl0a_43 train_libero_slurm.sh --config examples/LIBERO/train_files/ervla_v5_gr00t_traceloop_2pass_noz_null_all.yaml --run_id ervla_v5_gr00t_traceloop_2pass_noz_null_all_s43 --seed 43  # 1864671
sbatch --parsable --time=05:00:00 --dependency=afterok:1864670 --job-name=ep_tl0a_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_2pass_noz_null_all_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864672
sbatch --parsable --time=05:00:00 --dependency=afterok:1864671 --job-name=ep_tl0a_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_v5_gr00t_traceloop_2pass_noz_null_all_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864673
```

### Orthogonal causal attribution controls 1864634–1864643

The rollout interventions use a detached clean worktree at commit62e8d98 so unrelated
working-tree edits cannot change evaluation behavior. Prompt-only uses a symlink to the
identical2.4GB weight file and changes only `framework.cot.generate_at_inference`.

```bash
# Submitted from /e/scratch/m3/blank4/starVLA_trace_eval; absolute paths abbreviated here.
sbatch --parsable --time=12:00:00 --job-name=ep_cot_wrong --export=ALL,STARVLA_COT_CORRUPT=roll,output_dir=<wrong-trace-output>,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt <ours-v3-cotw01>/final_model/pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 16 --servers_per_gpu 1 --max_batch_size 16 --max_wait_time 0.0 --sif <live-tree>/playground/sims/sif/libero-plus-v0.5.0-arm64.sif  # 1864634
sbatch --parsable --time=12:00:00 --job-name=ep_cot_prompt --export=ALL,output_dir=<prompt-only-output>,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt <promptonly-view>/final_model/pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 16 --servers_per_gpu 1 --max_batch_size 16 --max_wait_time 0.0 --sif <live-tree>/playground/sims/sif/libero-plus-v0.5.0-arm64.sif  # 1864635

sbatch --parsable -t 02:00:00 --job-name=sm_cz42 --export=ALL,YAML_A=examples/LIBERO/train_files/ervla_causal_gr00t_fullmem_noz_b64.yaml,RUN_A=ervla_causal_gr00t_fullmem_noz_b64_smoke,YAML_B=examples/LIBERO/train_files/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64.yaml,RUN_B=ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64_smoke train_config_pair_slurm.sh 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1  # 1864642
sbatch --parsable -t 02:00:00 --job-name=sm_cz43 --export=ALL,YAML_A=examples/LIBERO/train_files/ervla_causal_gr00t_fullmem_noz_b64.yaml,RUN_A=ervla_causal_gr00t_fullmem_noz_b64_smoke,YAML_B=examples/LIBERO/train_files/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64.yaml,RUN_B=ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64_smoke train_config_pair_slurm.sh 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1  # 1864643
```

Both causal paired smokes passed: all four runs reached20 updates and eval10/20 with
complete4.8GB checkpoints. Shared-z memory keep was0.0, effective ranks10.31/12.29,
and auxiliary transition/action losses were finite. Full paired training and exact4k
dependencies were released:

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_cz42 --export=ALL,YAML_A=examples/LIBERO/train_files/ervla_causal_gr00t_fullmem_noz_b64.yaml,RUN_A=ervla_causal_gr00t_fullmem_noz_b64,YAML_B=examples/LIBERO/train_files/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64.yaml,RUN_B=ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64 train_config_pair_slurm.sh 42  # 1864837
sbatch --parsable --time=12:00:00 --job-name=tr_cz43 --export=ALL,YAML_A=examples/LIBERO/train_files/ervla_causal_gr00t_fullmem_noz_b64.yaml,RUN_A=ervla_causal_gr00t_fullmem_noz_b64,YAML_B=examples/LIBERO/train_files/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64.yaml,RUN_B=ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64 train_config_pair_slurm.sh 43  # 1864838
sbatch --parsable --time=05:00:00 --dependency=afterok:1864837 --job-name=ep_cnoz_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_causal_gr00t_fullmem_noz_b64_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864839
sbatch --parsable --time=05:00:00 --dependency=afterok:1864837 --job-name=ep_cz4_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864840
sbatch --parsable --time=05:00:00 --dependency=afterok:1864838 --job-name=ep_cnoz_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_causal_gr00t_fullmem_noz_b64_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864841
sbatch --parsable --time=05:00:00 --dependency=afterok:1864838 --job-name=ep_cz4_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_causal_gr00t_sharedz_zmem4_memdrop100_b64_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0  # 1864842
```

Jobs1864634/35 were cancelled before any simulator worker launched because the detached
worktree does not contain the generated `scripts/env/starvla_python` wrapper. Replacements
1864756/57 add `POLICY_SERVER_PYTHON=<live-tree>/scripts/env/starvla_python`; every other
argument and output directory is unchanged.

```bash
sbatch --parsable --time=12:00:00 --job-name=ep_cot_wrong --export=ALL,STARVLA_COT_CORRUPT=roll,output_dir=<wrong-trace-output>,POLICY_SERVER_GPU=,POLICY_SERVER_PYTHON=<live-tree>/scripts/env/starvla_python eval_libero_plus_slurm.sh --ckpt <ours-v3-cotw01>/final_model/pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 16 --servers_per_gpu 1 --max_batch_size 16 --max_wait_time 0.0 --sif <live-tree>/playground/sims/sif/libero-plus-v0.5.0-arm64.sif  # 1864756
sbatch --parsable --time=12:00:00 --job-name=ep_cot_prompt --export=ALL,output_dir=<prompt-only-output>,POLICY_SERVER_GPU=,POLICY_SERVER_PYTHON=<live-tree>/scripts/env/starvla_python eval_libero_plus_slurm.sh --ckpt <promptonly-view>/final_model/pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 16 --servers_per_gpu 1 --max_batch_size 16 --max_wait_time 0.0 --sif <live-tree>/playground/sims/sif/libero-plus-v0.5.0-arm64.sif  # 1864757
```

```bash
# --- Interactive smoke allocation ----------------------------------------------------
# Inside this shell: run LIBERO seeds42/43 concurrently on GPUs0-1/2-3; if both
# pass, reuse the allocation for the RoboCasa seeds4/42 smoke. No full training.
salloc --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 \
  --partition=booster -A m3 --time=02:00:00 --job-name=ix_piv4_smoke \
  srun --pty bash -l  # 1857753

# --- RoboCasa365 training (worktree!) -------------------------------------------------
cd /e/project1/m3/blank4/code/starVLA-upstream-merge
# QwenPI_v4 smoke; inspect train/eval losses and steps_20 before full training.
sbatch --parsable -t 02:00:00 --job-name=sm_rc365_piv4 train_robocasa365_seed_pair_slurm.sh \
  examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_piv4.yaml \
  ervla_robocasa365_piv4_smoke 4 42 \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 \
  --trainer.eval_interval 10 --trainer.save_interval 20 \
  --trainer.logging_frequency 1  # 1857345

# Full pair, released only after the corrected smoke passed.
sbatch --parsable --job-name=tr_rc365_piv4 train_robocasa365_seed_pair_slurm.sh \
  examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_piv4.yaml \
  ervla_robocasa365_piv4 4 42  # 1858730

sbatch --parsable --dependency=afterany:1858730 --job-name=tr_rcpiv4_rs \
  train_robocasa365_seed_pair_slurm.sh \
  examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_piv4.yaml \
  ervla_robocasa365_piv4 4 42 --trainer.is_resume true  # 1858851

sbatch --parsable -J rc_piv4_s4 -t 03:00:00 --dependency=afterok:1858851 \
  -o /e/scratch/m3/blank4/rc365_smoke/rc_piv4_s4_%j.out \
  -e /e/scratch/m3/blank4/rc365_smoke/rc_piv4_s4_%j.out \
  --export=ALL,MANIFEST=/e/scratch/m3/blank4/rc365_smoke/manifest_piv4_s4.txt,OUTDIR=robocasa365_piv4_50k,SEED=42,VIDEOS=0,CKPT_STEP=50000 \
  /e/scratch/m3/blank4/rc365_smoke/rc365_units.sbatch  # 1858852

sbatch --parsable -J rc_piv4_s42 -t 03:00:00 --dependency=afterok:1858851 \
  -o /e/scratch/m3/blank4/rc365_smoke/rc_piv4_s42_%j.out \
  -e /e/scratch/m3/blank4/rc365_smoke/rc_piv4_s42_%j.out \
  --export=ALL,MANIFEST=/e/scratch/m3/blank4/rc365_smoke/manifest_piv4_s42.txt,OUTDIR=robocasa365_piv4_50k,SEED=42,VIDEOS=0,CKPT_STEP=50000 \
  /e/scratch/m3/blank4/rc365_smoke/rc365_units.sbatch  # 1858853

sbatch --parsable --job-name=tr_rc365_s42_rs --export=ALL,\
CAUSAL_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_causal.yaml,\
V5_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_v5.yaml,\
EXTRA_TRAIN_ARGS="--trainer.is_resume true" train_robocasa365_pi_pair_slurm.sh 42   # or 4

# --- LIBERO training pairs ------------------------------------------------------------
cd /e/project1/m3/blank4/code/starVLA
sbatch --parsable --job-name=tr_zonly_v5 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_zonly_pi_sharedz_ground_temporal_v5.yaml \
  ervla_zonly_pi_sharedz_ground_temporal_v5 42 43

# QwenPI_v4 smoke: exact full-run distribution/batch shape, validation at steps 10 and 20.
# Inspect finite training/eval losses and steps_20 checkpoint before submitting full.
sbatch --parsable -t 02:00:00 --job-name=sm_v5_piv4 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_actiononly.yaml \
  ervla_v5_piv4_actiononly_smoke 42 43 \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 \
  --trainer.eval_interval 10 --trainer.save_interval 20 \
  --trainer.logging_frequency 1  # 1857343

# Only after that smoke has completed and its logs/checkpoint have been inspected:
sbatch --parsable --job-name=tr_v5_piv4 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_actiononly.yaml \
  ervla_v5_piv4_actiononly 42 43  # 1858694

# Only after full training succeeds:
sbatch --parsable --dependency=afterok:1858694 --job-name=ep_piv4_s42 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_piv4_actiononly_s42/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000  # 1858835

sbatch --parsable --dependency=afterok:1858694 --job-name=ep_piv4_s43 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_piv4_actiononly_s43/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000  # 1858836

sbatch --parsable --job-name=tr_q35enc_pair --export=ALL,\
YAML_A=examples/LIBERO-plus/train_files/starvla_libero_plus_q35encdec_enconly_gr00t_deeps.yaml,\
RUN_A=libero_plus_q35enc_deeps,\
YAML_B=examples/LIBERO-plus/train_files/starvla_libero_plus_q35encdec_enconly_gr00t_deeps_aug.yaml,\
RUN_B=libero_plus_q35enc_deeps_aug train_config_pair_slurm.sh 42

# --- LIBERO-plus eval (queue with afterok on the training job) ------------------------
sbatch --parsable --dependency=afterok:<TRAIN_JOB> --job-name=ep_<tag> \
  eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/<run>/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000

# --- RoboCasa365 rollouts / throughput (manifest-driven, worktree code) ---------------
# manifest line: "<run> <env_name> <native|r224> <n_envs> <n_episodes> <tag>"
# Round-robin over 4 GPUs by line number, so order lines to keep one checkpoint per GPU.
sbatch -J rc365_pnp2 -t 02:00:00 \
  -o /e/scratch/m3/blank4/rc365_smoke/rc365_pnp2_%j.out -e /e/scratch/m3/blank4/rc365_smoke/rc365_pnp2_%j.out \
  --export=ALL,MANIFEST=/e/scratch/m3/blank4/rc365_smoke/manifest_pnp_rerun.txt,\
OUTDIR=robocasa365_pnp_30k,SEED=42,VIDEOS=1 /e/scratch/m3/blank4/rc365_smoke/rc365_units.sbatch
```

### Qwen3.5 encoder-only resume and evals 1863830-1863832

Run from `/e/project1/m3/blank4/code/starVLA`. The resume picks each arm's latest
checkpoint itself; `--trainer.is_resume true` is what distinguishes it from a fresh run.

```bash
sbatch --parsable --job-name=tr_q35enc_resume \
  --export=ALL,YAML_A=examples/LIBERO-plus/train_files/starvla_libero_plus_q35encdec_enconly_gr00t_deeps.yaml,RUN_A=libero_plus_q35enc_deeps,YAML_B=examples/LIBERO-plus/train_files/starvla_libero_plus_q35encdec_enconly_gr00t_deeps_aug.yaml,RUN_B=libero_plus_q35enc_deeps_aug \
  train_config_pair_slurm.sh 42 --trainer.is_resume true   # 1863830

sbatch --parsable --dependency=afterok:1863830 --job-name=ep_q35_s42 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/libero_plus_q35enc_deeps_s42/checkpoints/steps_30000_pytorch_model.pt \
  --exact_tasks_per_suite 1000   # 1863831

sbatch --parsable --dependency=afterok:1863830 --job-name=ep_q35_aug_s42 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/libero_plus_q35enc_deeps_aug_s42/checkpoints/steps_30000_pytorch_model.pt \
  --exact_tasks_per_suite 1000   # 1863832
```

### Eval layout 8/2/4 is fine — the s43 trace-loop evals were killed by a bad GPU

An earlier note here claimed `--workers_per_gpu 8 --servers_per_gpu 2
--max_batch_size 4` was 4x too small and that all queued evals needed
resubmitting. **That was wrong and the pending evals were left alone.** The
counter-evidence is direct: `ep_tl2a_42` (1864094) ran the identical layout and
finished all 4,000 episodes in **1:17:43 = 3,088 eps/h**, zero failed suites,
mean 0.7448; `ep_tl1a_42` (1864074) likewise, 1:21:39, mean 0.7360. The
`ep_gz_*` and `ep_tl0a_*` evals also completed in ~1:20 on this layout.

The 504 eps/h measured on the wrong-trace control is not comparable: that
checkpoint *generates* its trace autoregressively at inference, while the
trace-loop models predict it with a head.

What actually went wrong on the s43 pair: one GPU's block of shards (8-15 =
GPU1) aborted with rc=134, SIGABRT in the sim — the documented degraded-render
signature — on **two different nodes**, jpbo-007-38 and jpbo-012-20. The policy
servers on that GPU stayed healthy and listening, so it is the EGL side. Each
suite burned all 3 retries on those 8 shards and never aggregated, and the
retries re-rolled episodes: 2,257 episodes for a 1,000-episode budget on
1864075.

Recovery: cancelled both and resubmitted with `--resume` and the same layout, so
the 24/32 shards already on disk per suite are kept and only the missing 8 are
re-run. Same layout also keeps the shard partitioning, so the results stay
directly comparable with the s42 siblings.

### Lower a pending job's walltime in place — never resubmit for it

`scontrol update JobId=<id> TimeLimit=<hh:mm:ss>` works without admin rights as
long as the limit goes *down*, and it keeps the job's queue age. Resubmitting
resets age and pushes the job back, so it is the wrong tool for this.

Applied Sep18 09:45 to the ten pending LIBERO-plus evals (1864839-42, 1863854-57,
1871858/59): 05:00:00 -> 02:30:00. The measured runtime for a full exact4k eval
on a healthy node is 1:17-1:23 (1864074, 1864094, the `ep_gz_*` and `ep_tl0a_*`
jobs), so 5h was ~4x the real need. The two RoboCasa evals 1858852/53 were left
at 03:00:00 — no measured runtime for the 17-task x 48-episode unit yet.

Note this did **not** get nodes sooner: with the scheduler holding nodes,
`sbatch --test-only` returned the same estimated start (15:07) for 00:30:00,
01:30:00, 02:30:00 and 05:00:00 alike. The value is for later, when the
constraint is a backfill gap rather than a hold.

### BUG (not yet fixed): `--resume` never skips anything in shard mode

`eval_libero_in_one.sh:407` probes for a completed shard at

    ${output_dir}/logs/${suite}/${start}_to_${end}${result_tag}.json

but `eval_libero_model.py` writes that file with the shard suffix appended to
`sample_tag` whenever `num_shards > 1`:

    0_to_2519_exact1000_shard16of32.json    <- actually written
    0_to_2519_exact1000.json                <- what the resume probe looks for

So `-s "${shard_result}"` is never true and every shard re-runs. Two consequences:

1. `--resume` is a no-op for any sharded run, i.e. every LIBERO-plus eval with
   `workers_per_gpu > 1`. The recovery jobs 1871858/59 were submitted with
   `--resume` and are redoing all 32 shards instead of the 8 that were missing.
2. The per-suite retry loop sets `STARVLA_RESUME_EVAL=1` on attempts 2 and 3, so
   **the retries have never resumed either** — each attempt re-ran all 32 shards.
   That is the direct cause of 1864075 rolling 2,257 episodes against a
   1,000-episode budget.

Correctness is unaffected: a re-run overwrites its own shard json and the
filename carries the shard id, so aggregation cannot double-count.

Fix prepared but **not applied** — four evals are executing this script right now
and bash reads a script lazily by byte offset, so editing it mid-run corrupts
them. Patch:
`/tmp/claude-32646/-e-project1-m3-blank4-code-starVLA/96b45084-f7dd-4983-ba47-d90a3fae375c/scratchpad/resume_shard_path.patch`
Apply once 1871858/59, 1864841/42 have drained. The same wait applies to the
`failed_suites.txt` truncation fix noted above.

## Planned / not yet launched

| What | Why | Blocked on |
|---|---|---|
| Qwen3.5 encoder in starVLA's QWen3_EncDec | the q35 enc-only arm cannot run without it (below) | implementation decision |
| PickPlaceSinkToCounter rollouts | failed 3/3 attempts, every time in the renderer | needs the EGL abort handled, or the task skipped |
| RoboCasa365 resolution A/B (native 256 vs 224), 50 eps/cell | training packs frames at native 256; the bridge resized to 224 | nothing — manifest_res.txt ready |
| Multiple client processes per GPU (2-3 x n_envs=16) | load was only 63/288 at n_envs=24; our lockstep loop leaves CPU idle | needs `--max_batch_size` raised on the server |
| Proper `eval_robocasa365_slurm.sh` in-repo | the manifest launcher lives in /e/scratch and duplicates work another agent is doing in-tree | coordinate with the agent rewriting the eval stack |

## Known bug: one policy server, two concurrent clients

The VLM interfaces keep per-request state on the module -- `_last_encoder_attention_mask`
is set during the forward and read afterwards -- so a second in-flight request overwrites
it in between. Two clients sharing a server produce, on the server side:

```
ValueError: Layer number mismatch: got 15 VL layers, but project_layers has 28 layers.
RuntimeError: The expanded size of the tensor (284) must match the existing size (567) ...
```

and the client sees `KeyError: 'data'` because the reply carries an error instead. Observed
2026-09-16 in the paired open-loop harness (job 1842457), which queried both servers from
both directions at once.

Consequences: run **one client per server** (our eval launchers already do). It also blocks
the obvious throughput idea of pointing several sim worker processes at a shared server --
the fix is to thread that state through the call instead of storing it on the module.

## Operating notes

- **RoboCasa365 rollouts**: `n_envs=24` without videos (~929 rollouts/h/GPU, ~15 GB VRAM).
  With videos the recorder adds a second render context per env; 24 envs then peak ~60 GB
  and a worker gets killed (parent sees `EOFError` on the worker pipe). Use **n_envs=12**
  when `VIDEOS=1`. Episode counts snap up to a multiple of `n_envs`.
- The eval enforces a **seed contract**: the policy server must be started with the same
  `--seed` as the rollout, or the client aborts in ~12 s.
- `CANCELLED by 0` means the admin/node killed it during CONFIGURING — not our code.
  Just resubmit. `EGL_NOT_INITIALIZED` / `Aborted` in `mjr_readPixels` is a degraded render
  node; resubmit and it lands elsewhere.

## Blocked: Qwen3.5 encoder-only (q35) needs real work, not a config fix

`tr_q35enc_pair` cannot be made to run by tweaking YAML. The Qwen3.5 enc-dec checkpoint
(`train_downstream/.../q35_enc_dec_v5_tb18432`) stores its encoder as **318 top-level
`encoder_layers.*` tensors**, while starVLA's loader expects them nested at
`model.language_model.encoder_layers.*` (which is where the Qwen3-VL checkpoint puts its
308). A prefix remap would load the weights — and would still be wrong, because that
checkpoint also carries `bidir_gates`, `enc_norm.weight` and `enc_scale`, which
`train_downstream/train/models/qwen35_enc_dec.py` uses in the encoder forward:

* a **bidirectional scan** per DeltaNet layer, `mixed = fwd + bidir_gates[i] * bwd`
* an output `enc_norm` (RMSNorm) followed by `* enc_scale`

starVLA's `QWen3_EncDec` implements none of that — its encoder path was written for
Qwen3-VL, where "bidirectional" is only an attention-mask change and full-attention layers
have `self_attn`. Qwen3.5 is hybrid (`linear_attn` gated DeltaNet on most layers). Loading
the weights without the matching forward would silently run a *different* encoder than the
one that was trained.

Decision needed: port the Qwen3.5 encoder forward into starVLA, or drop the q35 arm.


### GR00T shared-z batch64 smoke 1861585

Working directory: `/e/project1/m3/blank4/code/starVLA`. Two2-GPU runs per4-GPU node; no full training before log/checkpoint inspection.

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_gz_100 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_memdrop100_b64.yaml ervla_gr00t_sharedz_v5_memdrop100_b64_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```


### GR00T shared-z batch64 smoke 1861586

Working directory: `/e/project1/m3/blank4/code/starVLA`. Two2-GPU runs per4-GPU node; no full training before log/checkpoint inspection.

```bash
sbatch --parsable -t 02:00:00 --job-name=sm_gz_015 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_memdrop015_b64.yaml ervla_gr00t_sharedz_v5_memdrop015_b64_smoke 42 43 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```


### tr_gz_100_b64 1861658

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_gz_100_b64 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_memdrop100_b64.yaml ervla_gr00t_sharedz_v5_memdrop100_b64 42 43
```


### ep_gz_100_42 1861659

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=05:00:00 --dependency=afterok:1861658 --job-name=ep_gz_100_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0
```


### ep_gz_100_43 1861660

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=05:00:00 --dependency=afterok:1861658 --job-name=ep_gz_100_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0
```


### tr_gz_015_b64 1861661

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=12:00:00 --job-name=tr_gz_015_b64 train_seed_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_memdrop015_b64.yaml ervla_gr00t_sharedz_v5_memdrop015_b64 42 43
```


### ep_gz_015_42 1861662

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=05:00:00 --dependency=afterok:1861661 --job-name=ep_gz_015_42 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop015_b64_s42/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0
```


### ep_gz_015_43 1861663

Working directory: `/e/project1/m3/blank4/code/starVLA`.

```bash
sbatch --parsable --time=05:00:00 --dependency=afterok:1861661 --job-name=ep_gz_015_43 --export=ALL,POLICY_SERVER_GPU= eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop015_b64_s43/checkpoints/steps_20000_pytorch_model.pt --exact_tasks_per_suite 1000 --workers_per_gpu 8 --servers_per_gpu 2 --max_batch_size 4 --max_wait_time 0.0
```

### RoboCasa supervision pilot 1862755

Submitted from `/e/project1/m3/blank4/code/starVLA-upstream-merge`:
```bash
sbatch --parsable scripts/robocasa_supervision/run_slurm.sh pilot 4 /e/scratch/m3/blank4/rc365_supervision/pilot_labels_v1
```
One4-GPU node, four independent annotation workers,2h limit. One full source episode per18 task families, saved-state replay, per-frame masks and object/gripper trajectories, plus overlay inspection. This is dataset generation; no policy training/evaluation was submitted. Full generation waits for the pilot audit.

### RoboCasa annotation interactive allocation1862781

Pilot1862755 failed before writing labels: saved XML omits external cameras. Fixed by restoring exact per-episode `cam_configs`. Allocation command (worktree):
```bash
salloc --no-shell --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --partition=booster -A m3 --time=02:00:00 --job-name=ix_rc_labels
srun --jobid=1862781 --nodes=1 --ntasks=1 bash scripts/robocasa_supervision/run_slurm.sh pilot 4 /e/scratch/m3/blank4/rc365_supervision/pilot_labels_v2
```
Use this allocation for iterative label validation; no new training jobs.

### RoboCasa full supervision array1862967_[0-3]

Pilot passed:18 task families/4,282 frames,54 RGB camera comparisons (mean MAE3.807/255, max5.746), then final shared-z conventions checked on4 episodes/1,182 frames with no audit errors. Full command from worktree:
```bash
sbatch --parsable --array=0-3 --time=04:00:00 --job-name=rc_labels_full --export=ALL,SHARD_GROUPS=4,SUPERVISION_CODE_DIR=/e/scratch/m3/blank4/rc365_supervision/code/fd7d801 scripts/robocasa_supervision/run_slurm.sh full 16 /e/scratch/m3/blank4/rc365_supervision/labels_v1
```
Four4-GPU nodes,16 annotation workers/node (four/GPU),64 disjoint episode shards. Scripts pinned at worktree commitfd7d801. Output `labels_v1/episode_*/{targets.npz,COMPLETE.json}`; failures remain explicit JSONL records. No training submitted. Final export/audit waits for completeness; multi-entity inferred boundaries remain marked for review. Interactive1862781 retained temporarily for diagnostics.

### RoboCasa correction validation allocation1863079

Interactive1862781 released after successful pilot/export validation. Full generation continues. Whole-data entity audit found151 scenes with another appliance also naming a start-button; corrected selector now matches the microwave fixture. Initial full generation also found38 overlapping door-motion intervals, caused in inspected curves by small bumps/settling outside the main manipulation. Added an explicit central90%-motion fallback that still rejects truly overlapping interactions.

From the RoboCasa worktree:
```bash
salloc --no-shell --nodes=1 --ntasks=1 --cpus-per-task=288 --gres=gpu:4 --partition=booster -A m3 --time=02:00:00 --job-name=ix_rc_label_fix
EPISODE_IDS=744,2529,8082,8083 srun --jobid=1863079 --nodes=1 --ntasks=1 bash scripts/robocasa_supervision/run_slurm.sh review 4 /e/scratch/m3/blank4/rc365_supervision/fix_pilot_v1
```
Four affected episodes, four GPUs, overlays and invariant checks. Retry affected labels only after this passes.

Correction pilot1863079 passed all4 affected episodes with zero audit errors. Within the same allocation, retrying189 episodes (151 microwave selectors +38 door-motion cases),16workers/4GPUs:
```bash
EPISODE_IDS=$(cat /e/scratch/m3/blank4/rc365_supervision/retry_episode_ids.txt) srun --jobid=1863079 --nodes=1 --ntasks=1 bash scripts/robocasa_supervision/run_slurm.sh full 16 /e/scratch/m3/blank4/rc365_supervision/retry_labels_v1
```
Retry outputs remain separate until audited and merged. Source-video alignment expanded to270 frame/camera comparisons across18 tasks: meanMAE3.719/255,max7.509.

Retry step1863079.1 was CPU-bound to one core by the interactive `srun` default (16workers each~5.5%CPU). Stopped only that annotation step and restarted with explicit full-node CPU binding; completed episode sentinels are resumable:
```bash
EPISODE_IDS=$(cat /e/scratch/m3/blank4/rc365_supervision/retry_episode_ids.txt) srun --jobid=1863079 --nodes=1 --ntasks=1 --cpus-per-task=288 --cpu-bind=none bash scripts/robocasa_supervision/run_slurm.sh full 16 /e/scratch/m3/blank4/rc365_supervision/retry_labels_v1
```
The original four batch-array tasks were already using their full CPU allocations.

### RoboCasa supervision completed

Array1862967_[0-3] finished in15m11s–16m53s, returning FAILED because189 explicit per-episode exceptions were retained. All189 were repaired in allocation1863079; final retry step1863079.3 completed successfully, and the allocation was released. The merged dataset has9,126/9,126 episodes,9,591 subtasks,2,231,347 frames, no unresolved failures.465 multi-part episodes retain inferred-boundary flags. Full native and both whole/remaining object/gripper CoT exports are under `/e/scratch/m3/blank4/rc365_supervision/mappings_v1/`. Final raw-count reports are in `results_collected/robocasa_supervision/{generation_audit,export_audit}.json`. No training jobs launched.

### RoboCasa GR00T no-z versus z-only smoke 1863647

Submitted from `/e/project1/m3/blank4/code/starVLA-upstream-merge`. One4-GPU node, two independent2-GPU runs,32 samples/device (global64/run),20 optimizer updates, trainer evaluation at steps10/20 and a step20 checkpoint. This is not a simulator rollout.

```bash
sbatch --parsable --time=02:00:00 --job-name=sm_rc_gz0 --export=ALL,STARVLA_REPO=/e/project1/m3/blank4/code/starVLA-upstream-merge,YAML_A=examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_gr00t_v5_noz.yaml,RUN_A=ervla_rc365_gr00t_v5_noz_smoke,YAML_B=examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_gr00t_sharedz_v5_zonly.yaml,RUN_B=ervla_rc365_gr00t_sharedz_v5_zonly_smoke /e/project1/m3/blank4/code/starVLA/train_config_pair_slurm.sh 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```

### RoboCasa GR00T z-only versus retained-memory smoke 1863648

Submitted from `/e/project1/m3/blank4/code/starVLA-upstream-merge` with the same gate. The duplicated z-only arm provides a cross-node consistency check; the retained-memory arm keeps GR00T readout memory on85% of action examples.

```bash
sbatch --parsable --time=02:00:00 --job-name=sm_rc_gzm --export=ALL,STARVLA_REPO=/e/project1/m3/blank4/code/starVLA-upstream-merge,YAML_A=examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_gr00t_sharedz_v5_zonly.yaml,RUN_A=ervla_rc365_gr00t_sharedz_v5_zonly_smoke_b,YAML_B=examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_gr00t_sharedz_v5_mem015.yaml,RUN_B=ervla_rc365_gr00t_sharedz_v5_mem015_smoke /e/project1/m3/blank4/code/starVLA/train_config_pair_slurm.sh 42 --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1
```

Both smokes completed in7m07s with exit0, but they establish software/data-path health only. A z-only conditioning flaw was identified immediately afterward. Full jobs1863688/90/92 had run for57s and were cancelled; resume jobs1863689/91/93 and chained evals1863697–702 were cancelled before start. The older RoboCasa PI-v4 job1858730 and all LIBERO jobs were left untouched as requested. Do not resume or evaluate these GR00T RoboCasa run IDs.

The cancelled full layout was no-z versus z-only at seed42 (1863688), no-z versus z-only at seed4 (1863690), and retained-memory seeds4/42 (1863692), each2GPUs/run and batch64, with12h resume dependencies and all18-task evals. Booster rejected an initial30h request before creating a job because the QOS maximum is12h.

### Batch64 shared-z 10k equal-example checkpoint evaluations 1873664 / 1873665

Submitted from `/e/project1/m3/blank4/code/starVLA`. These evaluate the intermediate10k checkpoints of the legacy batch64/dropout1 shared-z+GR00T control. At10k updates, each batch64 run has seen640,000 sampled examples, matching the historical batch32 run at20k updates. This diagnoses whether the batch64 final checkpoint degraded after equal example exposure, but it is not a fully matched training-budget result because the checkpoint is halfway through the20k cosine schedule. Each job uses the same exact4k task count and pinned evaluation layout as the corresponding20k control; output is isolated from existing results.

Initial5h jobs1873637/1873639 used the older pinned8-worker/2-server/batch4 layout. They were cancelled while pending, before allocation. Replacement commands use the measured-throughput launcher defaults:32 workers/GPU, one server/GPU, dynamic batch ceiling equal to clients/server, zero wait and resumable shards. The2h limit is close to the observed1.3–1.7h exact4k runtime and improves queue placement.

```bash
sbatch --parsable --time=02:00:00 --job-name=ep_gz100_10k42 --export=ALL,POLICY_SERVER_GPU=,output_dir=/e/project1/m3/blank4/code/starVLA/playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s42/results/libero-plus-step10000-exact4k-v1 eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s42/checkpoints/steps_10000_pytorch_model.pt --exact_tasks_per_suite 1000 --resume

sbatch --parsable --time=02:00:00 --job-name=ep_gz100_10k43 --export=ALL,POLICY_SERVER_GPU=,output_dir=/e/project1/m3/blank4/code/starVLA/playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s43/results/libero-plus-step10000-exact4k-v1 eval_libero_plus_slurm.sh --ckpt playground/Checkpoints/ervla_gr00t_sharedz_v5_memdrop100_b64_s43/checkpoints/steps_10000_pytorch_model.pt --exact_tasks_per_suite 1000 --resume
```

Submitted September18. Both replacement jobs were pending for priority at the initial queue check.
