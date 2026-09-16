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

Last refreshed: 2026-09-16 14:20 CEST.

## Running / queued

| Job | Name | What | Status | Follow-up |
|---|---|---|---|---|
| 1826524 | readout_gaps | other workstream (not this session) | RUNNING | — |
| 1826641 | tr_zonly_aug | zonly sharedz, **old** backbone + augmentation, seeds 42/43 | RUNNING | 1826645, 1826646 |
| 1836350 | tr_rc365_s42_rs | RoboCasa365 PI causal+v5, seed 42, resume from 30k | PENDING | — |
| 1836346 | tr_zonly_v5 | zonly sharedz, **v5** backbone, seeds 42/43 | PENDING | 1836353, 1836354 |
| 1826645 | ep_ervla_zonly_pi_s42 | LIBERO-plus eval, zonly aug s42 @1000 eps | PENDING (dep 1826641) | — |
| 1826646 | ep_ervla_zonly_pi_s43 | LIBERO-plus eval, zonly aug s43 @1000 eps | PENDING (dep 1826641) | — |
| 1836353 | ep_zonly_v5_s42 | LIBERO-plus eval, zonly v5 s42 @1000 eps | PENDING (dep 1836346) | — |
| 1836354 | ep_zonly_v5_s43 | LIBERO-plus eval, zonly v5 s43 @1000 eps | PENDING (dep 1836346) | — |
| 1830054 | enc_dec_2b_v5_final_action_linear | other workstream (not this session) | PENDING | — |
| 1834470 | probe_smoke_enc_dec_2b_v5_final_action_tracetime | other workstream | PENDING | — |
| 1836154 | eval_sweep | other workstream | PENDING | — |
| 1836158 | act_arm_probe | other workstream | PENDING | — |
| 1836286 | patchmap | other workstream | PENDING | — |
| 1836294 | sim_qstate | other workstream (resubmit of 1826711) | PENDING | — |

## Finished

| Job | Name | Outcome |
|---|---|---|
| 1826547 | tr_rc365_s4_rs | COMPLETED 7:37:52 — seed 4 reached 50k, both arms |
| 1840354 | ep_zonly_1000 | COMPLETED — **0.776 mean** at ~1038 eps/suite (object .847, spatial .785, goal .784, long .689). The 0.812 from 64 eps did not hold |
| 1840069 | rc365_pnp5 | 4/6 units OK. faulthandler CONFIRMED the failure: `Fatal Python error: Aborted` in `binding_utils.py:174 read_pixels` — an EGL abort in a sim worker, NOT VRAM |
| 1840351 | tr_q35enc_pair | FAILED — 318 encoder keys missing. **Blocked, needs implementation**: see below |
| 1840352/53 | q35 evals | CANCELLED with their parent |
| 1826988 / 1826989 | enc_dec_2b_v5_final_action_{dec,head} | COMPLETED, ~1:59 each |
| 1828647 | rc365_smoke | COMPLETED — RoboCasa eval verified end-to-end on a compute node |
| 1828808 | rc365_roll30k | COMPLETED — 40 rollouts, videos, first valid success rates |
| 1829028 | rc365_thru | COMPLETED — n_envs sweep 1/4/8/16 |
| 1829920 | rc365_scale | COMPLETED — production-layout scaling, 24 is the operating point |
| 1836345 | tr_q35enc_pair | FAILED — separate_cross_attention: true crashes on Qwen3.5 (hybrid layers have linear_attn, not self_attn) and is inert under skip_decoder. Fixed in 224d783 -> 1840351 |
| 1836355 | ep_zonly_1000 | FAILED — EGL_NOT_INITIALIZED on jpbo-045-09 (2nd degraded node) -> 1840354 |
| 1839196 | rc365_pnp4 | NO-OP — rc=127: a comment inside the backslash-continued apptainer command broke it |
| 1838352 | rc365_pnp3 | FAILED 6/6 — every unit lost a sim worker (silent EOFError). Not memory: MaxRSS 270GB of 858GB, no OOM kill. Faulthandler was off, so a SIGABRT worker died mutely |
| 1842871 | rc365_eval_s4 | RoboCasa365 benchmark eval, seed 4 @50k, 17 atomic tasks x causal/v5, 48 eps | PENDING | — |
| 1842872 | rc365_eval_s42 | same for seed 42 @50k | PENDING (dep 1836350) | — |
| 1842656 | tr_q35enc_pair | Qwen3.5 encoder-only pair, 4th attempt (ported encoder + FFmpeg shim) | PENDING | 1842657, 1842658 |
| 1842595 | rc365_openloop | paired open loop, both directions | COMPLETED — policies agree closely on identical states (cosine 0.964) |
| 1834289 | rc365_pnp2 | NO-OP — servers died at once: the launcher started the LIVE tree's policy server (relative path) against the worktree's client, so `--seed` was unrecognized. Fixed to an absolute worktree path -> 1838352 |
| 1830609 | rc365_pnp | PARTIAL — 4/10 units; 4 lost a worker to VRAM, 2 hit the new seed contract |
| 1826546, 1826640, 1826712, 1826711 | training jobs | **CANCELLED by 0** (admin/node failure during CONFIGURING, no logs) -> resubmitted as 1836350, 1836345, 1836346, 1836294 |
| 1826642 | ep_zonly_1000 | FAILED — `EGL_NOT_INITIALIZED` on jpbo-106-04 (degraded render node) -> resubmitted as 1836355 |
| 1826643/44, 1826713/14 | queued evals | CANCELLED with their parent -> resubmitted as 1836351/52, 1836353/54 |

## Launch commands

Run training/eval launchers from the tree that owns them. **RoboCasa365 lives in the
worktree** `/e/project1/m3/blank4/code/starVLA-upstream-merge` (branch `merge_upstream_2026_09`);
everything else from `/e/project1/m3/blank4/code/starVLA`.

```bash
# --- RoboCasa365 training (worktree!) -------------------------------------------------
cd /e/project1/m3/blank4/code/starVLA-upstream-merge
sbatch --parsable --job-name=tr_rc365_s42_rs --export=ALL,\
CAUSAL_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_causal.yaml,\
V5_YAML=./examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_pi_v5.yaml,\
EXTRA_TRAIN_ARGS="--trainer.is_resume true" train_robocasa365_pi_pair_slurm.sh 42   # or 4

# --- LIBERO training pairs ------------------------------------------------------------
cd /e/project1/m3/blank4/code/starVLA
sbatch --parsable --job-name=tr_zonly_v5 train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_zonly_pi_sharedz_ground_temporal_v5.yaml \
  ervla_zonly_pi_sharedz_ground_temporal_v5 42 43

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
