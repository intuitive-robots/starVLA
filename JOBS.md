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

Last refreshed: 2026-09-17 22:44 CEST.

## Running / queued

Refreshed 2026-09-17 09:56 from `squeue` (see `scripts/jobs_status.sh`).

| Job | Name | What | State |
|---|---|---|---|
| 1863773 | sm_gz4f_015 | Fixed GR00T+z-memory4 + full encoder memory dropout0.15, seeds42/43; 20 updates, trainer eval10/20, batch64/run | SUBMITTED |
| 1863772 | sm_gz4f_100 | Fixed GR00T+z-memory4 + full encoder memory dropout1, seeds42/43; 20 updates, trainer eval10/20, batch64/run | SUBMITTED |
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
| 1858835 / 1858836 | ep_piv4_s42 / s43 | Exact4k LIBERO-plus evals after successful full QwenPI_v4 training1858694 | PENDING (Dependency) |
| 1858730 | tr_rc365_piv4 | Full QwenPI_v4 RoboCasa seeds4/42 at50k after passed smoke | RUNNING |
| 1858851 | tr_rcpiv4_rs | Automatic RoboCasa continuation from latest checkpoint after first12h segment1858730 | PENDING (Dependency) |
| 1858852 / 1858853 | rc_piv4_s4 / s42 | RoboCasa17-task ×48-episode evals at step50k after continuation1858851 | PENDING (Dependency) |
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
