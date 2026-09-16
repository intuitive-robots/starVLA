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
| 1826547 | tr_rc365_s4_rs | RoboCasa365 PI causal+v5, seed 4, resume from 30k | RUNNING | — |
| 1826641 | tr_zonly_aug | zonly sharedz, **old** backbone + augmentation, seeds 42/43 | RUNNING | 1826645, 1826646 |
| 1836350 | tr_rc365_s42_rs | RoboCasa365 PI causal+v5, seed 42, resume from 30k | PENDING | — |
| 1840351 | tr_q35enc_pair | Qwen3.5 encoder-only GR00T deeps, aug + no-aug, seed 42 | PENDING | 1840352, 1840353 |
| 1836346 | tr_zonly_v5 | zonly sharedz, **v5** backbone, seeds 42/43 | PENDING | 1836353, 1836354 |
| 1840069 | rc365_pnp5 | RoboCasa365 pick-and-place, 6 reruns, n_envs=12, videos, faulthandler on | PENDING | — |
| 1826645 | ep_ervla_zonly_pi_s42 | LIBERO-plus eval, zonly aug s42 @1000 eps | PENDING (dep 1826641) | — |
| 1826646 | ep_ervla_zonly_pi_s43 | LIBERO-plus eval, zonly aug s43 @1000 eps | PENDING (dep 1826641) | — |
| 1840352 | ep_libero_plus_q3_s42 | LIBERO-plus eval, q35 enc no-aug @1000 eps | PENDING (dep 1840351) | — |
| 1840353 | ep_libero_plus_q3_aug_s42 | LIBERO-plus eval, q35 enc aug @1000 eps | PENDING (dep 1840351) | — |
| 1836353 | ep_zonly_v5_s42 | LIBERO-plus eval, zonly v5 s42 @1000 eps | PENDING (dep 1836346) | — |
| 1836354 | ep_zonly_v5_s43 | LIBERO-plus eval, zonly v5 s43 @1000 eps | PENDING (dep 1836346) | — |
| 1840354 | ep_zonly_1000 | LIBERO-plus re-eval of the best zonly ckpt @1000 eps (3rd try) | PENDING | — |
| 1830054 | enc_dec_2b_v5_final_action_linear | other workstream (not this session) | PENDING | — |
| 1834470 | probe_smoke_enc_dec_2b_v5_final_action_tracetime | other workstream | PENDING | — |
| 1836154 | eval_sweep | other workstream | PENDING | — |
| 1836158 | act_arm_probe | other workstream | PENDING | — |
| 1836286 | patchmap | other workstream | PENDING | — |
| 1836294 | sim_qstate | other workstream (resubmit of 1826711) | PENDING | — |

## Finished

| Job | Name | Outcome |
|---|---|---|
| 1826988 / 1826989 | enc_dec_2b_v5_final_action_{dec,head} | COMPLETED, ~1:59 each |
| 1828647 | rc365_smoke | COMPLETED — RoboCasa eval verified end-to-end on a compute node |
| 1828808 | rc365_roll30k | COMPLETED — 40 rollouts, videos, first valid success rates |
| 1829028 | rc365_thru | COMPLETED — n_envs sweep 1/4/8/16 |
| 1829920 | rc365_scale | COMPLETED — production-layout scaling, 24 is the operating point |
| 1836345 | tr_q35enc_pair | FAILED — separate_cross_attention: true crashes on Qwen3.5 (hybrid layers have linear_attn, not self_attn) and is inert under skip_decoder. Fixed in 224d783 -> 1840351 |
| 1836355 | ep_zonly_1000 | FAILED — EGL_NOT_INITIALIZED on jpbo-045-09 (2nd degraded node) -> 1840354 |
| 1839196 | rc365_pnp4 | NO-OP — rc=127: a comment inside the backslash-continued apptainer command broke it |
| 1838352 | rc365_pnp3 | FAILED 6/6 — every unit lost a sim worker (silent EOFError). Not memory: MaxRSS 270GB of 858GB, no OOM kill. Faulthandler was off, so a SIGABRT worker died mutely |
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
| RoboCasa365 resolution A/B (native 256 vs 224), 50 eps/cell | training packs frames at native 256; the bridge resized to 224 | nothing — manifest_res.txt ready |
| Multiple client processes per GPU (2-3 x n_envs=16) | load was only 63/288 at n_envs=24; our lockstep loop leaves CPU idle | needs `--max_batch_size` raised on the server |
| Proper `eval_robocasa365_slurm.sh` in-repo | the manifest launcher lives in /e/scratch and duplicates work another agent is doing in-tree | coordinate with the agent rewriting the eval stack |

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
