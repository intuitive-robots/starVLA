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

Last refreshed: 2026-09-17 16:13 CEST.

## Running / queued

Refreshed 2026-09-17 09:56 from `squeue` (see `scripts/jobs_status.sh`).

| Job | Name | What | State |
|---|---|---|---|
| 1857317 | sm_v5_piv4 | QwenPI_v4 seeds42/43; 20 optimizer updates at 2GPUs/run, batch32/GPU; in-training eval at steps10/20 | PENDING |
| 1857332 | sm_rc365_piv4 | RoboCasa QwenPI_v4 seeds4/42; 20 optimizer updates at 2GPUs/run, batch32/GPU; in-training eval at steps10/20 | PENDING |
| 1851382 | tr_zonly_gr00t | zonly + GR00T head, seeds 42/43 — the two biggest effects combined | CONFIGURING  |
| 1851384 | ep_zonly_gr00t_s43 | LIBERO-plus eval, zonly+GR00T s43 @1000 eps (dep 1851382) | PENDING  |
| 1851383 | ep_zonly_gr00t_s42 | LIBERO-plus eval, zonly+GR00T s42 @1000 eps (dep 1851382) | PENDING  |
| 1850693 | enc_dec_2b_v5_final_action_tracetime_w001 | other workstream (not this session) | RUNNING 21:52 |
| 1850350 | ep_t_trace_ours_v3_cotw01 | CoT-trace re-eval @1000 eps/suite — ours_v3_cotw01 (was 0.826 @1153) | RUNNING 33:30 |
| 1850351 | ep_ot_trace_det_v3_cotw01 | CoT-trace re-eval @1000 eps/suite — det_v3_cotw01 (was 0.822 @1153) | RUNNING 33:30 |
| 1850352 | ep_race_det_cotw1_readout | CoT-trace re-eval @1000 eps/suite — det_cotw1_readout (was 0.790) | RUNNING 33:30 |
| 1850353 | ep__trace_ours_full_cotw1 | CoT-trace re-eval @1000 eps/suite — ours_full_cotw1 (was 0.787) | RUNNING 33:30 |
| 1850294 | ep_zonly_v5_s42 | LIBERO-plus eval, zonly v5 s42 @1000 eps (rerun after CUDA abort) | RUNNING 38:18 |
| 1851073 | full_table | other workstream (not this session) | RUNNING 9:59 |
| 1850917 | trace_cls | other workstream (not this session) | RUNNING 17:11 |
| 1850162 | act_arm_probe2 | other workstream (not this session) | RUNNING 45:09 |

## Finished

| Job | Name | Outcome |
|---|---|---|
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

```bash
# --- RoboCasa365 training (worktree!) -------------------------------------------------
cd /e/project1/m3/blank4/code/starVLA-upstream-merge
# QwenPI_v4 smoke; inspect train/eval losses and steps_20 before full training.
sbatch --parsable --job-name=sm_rc365_piv4 train_robocasa365_seed_pair_slurm.sh \
  examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_piv4.yaml \
  ervla_robocasa365_piv4_smoke 4 42 \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 \
  --trainer.eval_interval 10 --trainer.save_interval 20 \
  --trainer.logging_frequency 1  # 1857332

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
sbatch --parsable --job-name=sm_v5_piv4 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_actiononly.yaml \
  ervla_v5_piv4_actiononly_smoke 42 43 \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 \
  --trainer.eval_interval 10 --trainer.save_interval 20 \
  --trainer.logging_frequency 1  # 1857317

# Only after that smoke has completed and its logs/checkpoint have been inspected:
sbatch --parsable --job-name=tr_v5_piv4 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_actiononly.yaml \
  ervla_v5_piv4_actiononly 42 43

# Only after full training succeeds:
sbatch --parsable --job-name=ep_piv4_s42 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_piv4_actiononly_s42/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000

sbatch --parsable --job-name=ep_piv4_s43 \
  eval_libero_plus_slurm.sh \
  --ckpt playground/Checkpoints/ervla_v5_piv4_actiononly_s43/checkpoints/steps_20000_pytorch_model.pt \
  --exact_tasks_per_suite 1000

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
