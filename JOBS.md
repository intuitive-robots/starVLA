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

Last refreshed: 2026-09-18 17:59 CEST.

## Running / queued

Refreshed 2026-09-18 17:59 CEST from `squeue`, `sacct`, raw training logs and raw rollout JSONs (see `scripts/jobs_status.sh`).

| Job | Name | What | State |
|---|---|---|---|
| 1893711 | ep_causal_b32_80k_resume | `--resume` top-up of the `causal_b32` **80k endpoint** eval into the same `output_dir` (`results/libero-plus-step80000-exact4k-v1`): skips the 36 shards already on disk, gives the 12 exhausted shards fresh attempts on a different node. Launched from the recovered submit line of 1892946 | SUBMITTED 2026-09-19 16:0x |
| 1893392 | rc_piv4c_s42now | RoboCasa eval of the **finished** causal-arm seed42 at 50k, submitted directly instead of waiting on the chain | PENDING — gives half the 2x2 answer ~5h early |
| 1888091 | v2_g32zt | Auto-resume segment of 1881737 (sweep v2 b32 pair tokfull + tokcam) via the grid launcher's self-resubmit; ids recorded in `slurm_logs/libero_grid_job_ledger.tsv` | RUNNING |
| 1888096 | v2_g32zp | Auto-resume segment of 1881739 (sweep v2 b32 pair zonly + piv4) | RUNNING |
| 1889365 | v2_g128e | Sweep v2 `ervla_gridv2_encfull_b128_s42`, resumed onto the target-aware launcher, 30k target | RUNNING |
| 1889366 | v2_g256zc | Sweep v2 `ervla_gridv2_tokcam_b256_s42`, resumed onto the target-aware launcher, 30k target, 4 nodes | RUNNING |
| 1889367 | v2_g256e | Sweep v2 `ervla_gridv2_encfull_b256_s42`, resumed onto the target-aware launcher, 30k target, 4 nodes | RUNNING |
| 1892943 / 1892944 | ep_1888087_60000 / 80000 | Auto-submitted exact4k of `ervla_gridv2_encfull_b32_s42` at steps 60k and **80k (endpoint)**; inherited the new `workers_per_gpu` default of 12 — the fix propagates to auto-submitted evals | RUNNING |
| 1892945 / 1892946 | ep_1888087_60000 / 80000 | Same for `ervla_gridv2_causal_b32_s42` at 60k and **80k (endpoint)** — first true b32 endpoint pair, and it includes the causal GR00T arm | 60k RUNNING, healthy. **80k (1892946) CANCELLED at 57 min**: 1,054/4,000 episodes (siblings were at ~3,000), 12 shards permanently exhausted, i.e. a marginal render node — the same signature at 12 workers that 32 workers produced everywhere. Replaced by 1893711 with `--resume`; nothing completed was lost |
| 1892334 / 1892335 / 1892561 / 1892562 | ae-fa3bisect / -rs / ae-aeseed7 / -rs | other workstream (not this session) — listed only so `jobs_status.sh` stops flagging them | RUNNING/PENDING |
| 1891593 | tr_rc365_piv4c_rs2 | Third and final segment: s4 resumes from 40,000, needs 10,000 steps (~4.2h); s42 is already at 50,000 and should exit immediately | **PENDING, dependency cleared** — waiting only on scheduling |
| 1891595 / 1891596 | rc_piv4c2_s4 / s42 | Replacement RoboCasa evals behind the second continuation | PENDING `afterok:1891593` |
| 1878223 / 1878224 | rc_piv4c_s4 / s42 | Original evals behind1878219 | **CANCELLED** by the `afterok` failure, as intended — superseded by1891595/96 and by the direct seed-42 eval1893392 |
| 1885185 → 1885186 / 1885187 | tr_anneal_late → ep_anneal_late_s42/s43 | **Late-decay anneal.** Blind through step 10,000, then linear 1.0→0.0 by 20,000, so the rate reaches 0 only at the final step. Its 10k blind budget matches the sequential arm's stage 1 exactly, making it the one-stage counterpart at equal blind exposure — LA4VLA's "mixed vs sequential" comparison. Evals chained `afterok` | SUBMITTED 2026-09-19 |
| 1883887 → 1883888 → 1883889 | tr_seq1_s42 → tr_seq2_s42 → ep_seq2_s42 | **Sequential arm, 10k+20k.** Stage 1 vision-free 10,000 x 128, whole VLM frozen; stage 2 sighted 20,000 x 64 from its `steps_10000`; then exact4k. Fully chained `afterok` | SUBMITTED 2026-09-18 23:15 |
| 1883890 → 1883891 → 1883892 | tr_seq1_s43 → tr_seq2_s43 → ep_seq2_s43 | Same, seed 43 | SUBMITTED |
| 1881708 → 1881720 / 1881721 | tr_state → ep_state_s42/s43 | **chunk 8 restart.** Image-dropout control: state, no dropout | SUBMITTED 2026-09-18 20:29 |
| 1881710 → 1881722 / 1881723 | tr_blind → ep_blind_s42/s43 | **chunk 8 restart.** State-only ceiling — read first | SUBMITTED |
| 1881719 → 1881724 / 1881725 | tr_imgdrop_anneal → ep_* | **chunk 8 restart.** Image dropout annealed 1.0→0.0 over steps 2k→10k | SUBMITTED |
| 1881726 / 1881729 | tr_seq1_s42 / s43 | Sequential stage 1 at 4k × 128 | **COMPLETED** 00:39:10 / 00:38:43, exit 0:0 — superseded by the 10k stage 1 (1883887/1883890); 4,000 frozen-backbone steps were too few for a prior to form |
| 1881727 / 1881730 → 1881728 / 1881731 | tr_seq2_s42 / s43 → ep_* | Sequential stage 2 from `steps_4000` | **CANCELLED at ~step 8,000** and relaunched as the 10k+20k chain. Before cancelling it confirmed the handoff works in production: `📦 loading checkpoint: .../steps_4000_pytorch_model.pt`, and the prior transferred — first three logged losses 0.0874/0.0580/0.1149 against the cold control's 0.3947/0.3503/0.2483 |
| 1881737 | v2_g32zt | **chunk 8 restart.** Sweep v2 b32 pair tokfull + tokcam, 80k / warmup 2000 | SUBMITTED |
| 1881738 | v2_g32ct | **chunk 8 restart.** Sweep v2 b32 encfull + causal | SUBMITTED |
| 1881739 | v2_g32zp | **chunk 8 restart.** Sweep v2 b32 zonly + piv4 | SUBMITTED |
| 1881740 / 1881741 | v2_g64zf / v2_g64zc | **chunk 8 restart.** Sweep v2 b64 tokfull / tokcam, 40k | SUBMITTED |
| 1881742 / 1881743 | v2_g128zc / v2_g128e | **chunk 8 restart.** Sweep v2 b128 tokcam / encfull, 30k, 2 nodes | SUBMITTED |
| 1881744 / 1881745 | v2_g256zc / v2_g256e | **chunk 8 restart.** Sweep v2 b256 tokcam / encfull, 30k, 4 nodes | SUBMITTED |
| 1881040 | v2_g32zt | **Sweep v2** b32 pair: `ervla_gridv2_tokfull_b32_s42` + `ervla_gridv2_tokcam_b32_s42`, 80k steps / warmup 2000 | SUBMITTED 2026-09-18 |
| 1881041 | v2_g32ct | **Sweep v2** b32 controls: `ervla_gridv2_encfull_b32_s42` + `ervla_gridv2_causal_b32_s42`, 80k / 2000 | SUBMITTED |
| 1881043 | v2_g32zp | **Sweep v2** b32 pair: `ervla_gridv2_zonly_b32_s42` + `ervla_gridv2_piv4_b32_s42`, 80k / 2000 | SUBMITTED |
| 1881044 | v2_g64zf | **Sweep v2** b64 `ervla_gridv2_tokfull_b64_s42`, 40k / 2000 | SUBMITTED |
| 1881045 | v2_g64zc | **Sweep v2** b64 `ervla_gridv2_tokcam_b64_s42`, 40k / 2000 | SUBMITTED |
| 1881046 | v2_g128zc | **Sweep v2** b128 `ervla_gridv2_tokcam_b128_s42`, 30k / 2000, 2 nodes | SUBMITTED |
| 1881047 | v2_g128e | **Sweep v2** b128 `ervla_gridv2_encfull_b128_s42`, 30k / 2000, 2 nodes | SUBMITTED |
| 1881048 | v2_g256zc | **Sweep v2** b256 `ervla_gridv2_tokcam_b256_s42`, 30k / 2000, 4 nodes | SUBMITTED |
| 1881049 | v2_g256e | **Sweep v2** b256 `ervla_gridv2_encfull_b256_s42`, 30k / 2000, 4 nodes | SUBMITTED |
| 1878150 | tr_g32zt | Global-batch **32** pair: `ervla_grid_tokfull_b32_s42` + `ervla_grid_tokcam_b32_s42` | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878151 | tr_g32ct | Global-batch **32** controls: `ervla_grid_encfull_b32_s42` + `ervla_grid_causal_b32_s42` | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878152 | tr_g64zf | Global-batch **64**: `ervla_grid_tokfull_b64_s42`, one 4-GPU node | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878153 | tr_g64zc | Global-batch **64**: `ervla_grid_tokcam_b64_s42` | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878154 | tr_g128zc | Global-batch **128**: `ervla_grid_tokcam_b128_s42`, 2 nodes | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878155 | tr_g128e | Global-batch **128**: `ervla_grid_encfull_b128_s42`, 2 nodes | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878156 | tr_g256zc | Global-batch **256**: `ervla_grid_tokcam_b256_s42`, 4 nodes | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878157 | tr_g256e | Global-batch **256**: `ervla_grid_encfull_b256_s42`, 4 nodes | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1878254 | tr_g32zp | Global-batch **32** pair: `ervla_grid_zonly_b32_s42` + `ervla_grid_piv4_b32_s42` | **CANCELLED 2026-09-18 and relaunched as sweep v2** — every arm ran 80k steps / 8k warmup regardless of batch, which is 256k warmup samples at b32 but 2.05M at b256 |
| 1879710 → 1879713 / 1879714 | tr_state → ep_state_s42/43 | **Control** for the image-dropout family: piv4 + quantised state, no dropout. Seeds 42/43, 20k, one 4-GPU node. Evals chained `afterok` | SUBMITTED 2026-09-18 19:02 |
| 1879711 → 1879715 / 1879716 | tr_blind → ep_blind_s42/43 | **The state-only ceiling — read this one first.** `vision_rate: 1.0` at train *and* eval. If it lands near 1879710 then LIBERO success cannot discriminate between visual representations and nothing in the other arms is interpretable | SUBMITTED 2026-09-18 19:02 |
| 1879712 → 1879717 / 1879718 | tr_imgdrop_anneal → ep_imgdrop_anneal_s42/s43 | **Headline arm**: image dropout annealed 1.0→0.0 over steps 2k→10k, one run, no second checkpoint. The two-stage curriculum as a schedule | SUBMITTED 2026-09-18 19:02 |
| 1879719 / 1879723 | tr_seq1_s42 / s43 | Sequential **stage 1**: `removal: tokens` (no image tokens, no vision tower), whole VLM frozen, 4k × 128 global on 4 GPUs, one seed per node | SUBMITTED 2026-09-18 19:02 |
| 1879720 / 1879724 → 1879721 / 1879725 | tr_seq2_s42 / s43 → ep_seq_stage2_s42/s43 | Sequential **stage 2**: ordinary sighted 12k × 64 from stage 1's `steps_4000` via `pretrained_checkpoint`. `afterok` on the matching stage 1. 4k×128 + 12k×64 = 1.28M samples = the control's 20k×64, matched on samples not steps | SUBMITTED 2026-09-18 19:02 |
| 1879430 | sm_piv4_imgdrop | Smoke rerun 2, now asserting on the metrics rather than the exit status | **ALL FIVE PASSED**, 00:06:53, exit 0:0. `anneal` traced its schedule exactly — `1.0` held for six steps, then `0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1`, then `0.0` for the rest (compressed to start 5/end 15 for the smoke) — which is what proves `_publish_train_step` reaches the model and the rate is not frozen at its start value. `blind` and `seq1` logged `vision_keep_rate` 0.0 on every step. Full family released. |
| 1879173 | sm_piv4_imgdrop | Smoke rerun after the 1878940 GPU-binding bug | **ALL FIVE COMPLETED, 00:07:13, exit 0:0** — 20 updates, eval 10/20 and a complete `steps_20` checkpoint per arm; losses finite and falling, and the arms diverge from `state` at identical seed/data (step 20: state 0.9504, blind 0.9880, anneal 0.9211, seq1 1.0303), which is what says the blinding changes the forward. **But `vision_keep_rate`/`vision_drop_rate` never appeared in the logs:** `_train_step` copies an explicit allowlist of keys out of the framework's output dict, so the two new metrics were returned and dropped. Plumbed at both call sites plus `wandb.define_metric`; resubmitted as 1879430 to confirm. Nothing was wrong with the dropout itself — only with the evidence it leaves. |
| 1878940 | sm_piv4_imgdrop | **PARTIAL — 3/5 PASS, resubmitted as 1879173.** `seq1` (token removal + whole VLM frozen), `seq2` (stage-1→stage-2 handoff) and `state` (new control, state_dim 8) all passed 20 updates + eval 10/20. `anneal` and `blind` died with **CUDA OOM in the DeepSpeed gradient partition — a launcher bug, not the dropout code**: the script exported `CUDA_VISIBLE_DEVICES`, but `train_libero_slurm.sh` overwrites it from `STARVLA_CUDA_VISIBLE_DEVICES`, so every slot saw all four GPUs and three processes piled onto GPU 0 (33 + 22 + 1 GiB). `train_seed_pair_slurm.sh` documents exactly this trap. The two `removal: pixels` arms are the heaviest (full ViT, full sequence) so they lost the contention; `seq1` runs no vision tower at all, which is why it survived. Fixed by launching four background slots with `STARVLA_CUDA_VISIBLE_DEVICES=$slot`, the proven pattern. Original description: `anneal` (scheduled pixel zeroing, compressed to start 5/end 15 so 20 steps cross blind→ramp→sighted), `blind` (rate 1.0 at train *and* eval), `seq1` (`removal: tokens` — no image tokens, no `pixel_values` key, whole VLM frozen), `seq2` (stage-1→stage-2 handoff against a real piv4 checkpoint), `state` (the new no-dropout control, `state_dim: 8`). Five arms on four GPUs, 20 updates + eval 10/20 each | SUBMITTED 2026-09-18 18:41 |
| 1878833 | sm_piv4_imgdrop | Superseded first attempt | **CANCELLED before it read any config** — every arm had `include_state: false`; proprioception must reach the encoder as quantised tokens, so all configs were regenerated with `state_dim: 8` / `include_state: true` and a new state-carrying control was added. Resubmitted as 1878940. |
| 1878217 | tr_rc365_piv4c | RoboCasa causal backbone + LayerwiseFM_v4, seeds4/42, 50k | **TIMEOUT** 12:01:03 at Sep19 06:03 — expected, the 12h wall. Reached s4 28,036 / s42 30,648. |
| 1878219 | tr_rc365_piv4c_rs | Continuation of1878217 | **CANCELLED by user at 9:11**, deliberately and with nothing lost: s42 had already written its 50,000 checkpoint (13:45) and s4's next save was at 50,000, which it could not reach (projected ~48,490 against 2.84h of wall), so the remaining compute would have produced no checkpoint. Cancelling freed the node ~3h early and released segment3. |
| 1878223 / 1878224 | rc_piv4c_s4 / s42 | RoboCasa 17-task x48-ep evals at 50k, same harness/manifests as1858852/53 so results compare directly | PENDING `afterok:1878219` |
| 1878149 | sm_rc_piv4c | **Smoke** for the missing 2x2 cell: QwenPI_v4 head on a **causal** backbone, RoboCasa, seeds4/42, 20 updates + eval10/20 | **PASSED** 00:05:21, exit0:0. Both seeds ran all 20 updates, evaluated at 10 and 20, and wrote complete step-20 checkpoints (5.84 GB each). Loss finite and falling: s4 1.3276 -> action_dit_loss 0.9931, s42 1.3197 -> 0.9894. The only `Error`/`NaN` matches in the logs are an offline albumentations version check and an `NCCL_ASYNC_ERROR_HANDLING` deprecation notice — no traceback, no NaN. Unfreezing the whole backbone (`freeze_modules: ''`) works with LayerwiseFM_v4, which was the risk this smoke existed to test. Chain 1878217 released. |
| 1878156 / 1878157 | tr_g256zc / tr_g256e | Seed42 global-batch256 boundary: token-only shared-z with structured camera dropout versus encoder full-memory/no-z;16 GPUs/run,16/device,80k with20k milestones | **RUNNING**, four nodes each,12h segment. Auto-resume/eval IDs go to `slurm_logs/libero_grid_job_ledger.tsv`. |
| 1878154 / 1878155 | tr_g128zc / tr_g128e | Seed42 global-batch128: token-only shared-z with structured camera dropout versus encoder full-memory/no-z;8 GPUs/run,16/device,80k | **RUNNING**, two nodes each,12h segment; exact4k at20/40/60/80k auto-submits. |
| 1878152 / 1878153 | tr_g64zf / tr_g64zc | Seed42 global-batch64 new token-only shared-z variants: all encoder memory versus30% structured one-camera dropout; z-AdaLN disabled | **RUNNING**, one4-GPU node each,12h segment. Existing encoder/causal batch64 controls are reused. |
| 1878150 | tr_g32zt | Seed42 global-batch32 token-only pair: full encoder memory versus structured camera dropout; two GPUs/run,16/device | **RUNNING**, one4-GPU node,12h segment. |
| 1878151 | tr_g32ct | Seed42 global-batch32 control pair: encoder full-memory/no-z versus causal full-memory/no-z; two GPUs/run,16/device | **RUNNING**, one4-GPU node,12h segment. |
| 1878254 | tr_g32zp | Added seed42 batch32 pair: token-only z bottleneck (all ordinary encoder memory masked) and QwenPI-v4 encoder; two GPUs/run,16/device | **PENDING**,12h. Both20-step/in-training-eval smokes passed in interactive allocation1873548. |
| 1873664 / 1873665 | ep_gz100_10k42 / 43 | Exact4k LIBERO-plus at **step 10,000** of `ervla_gr00t_sharedz_v5_memdrop100_b64` (intermediate-checkpoint comparison against the 20k result) | PENDING (Priority), 2h limit. Submitted 12:3x from another session — recorded here so `jobs_status.sh` stays clean. |
| 1880757 | ae-policy-rs | other workstream (not this session) — listed only so `jobs_status.sh` stops flagging it | RUNNING |
| 1877960 | ae-policy | other workstream (not this session) — listed only so `jobs_status.sh` stops flagging it | RUNNING |
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
| 1858852 / 1858853 | rc_piv4_s4 / s42 | RoboCasa 17-task x48-episode eval at step50k, QwenPI_v4 (v5 encoder, `skip_decoder`, dense layer-wise conditioning) | **COMPLETED, CLEAN** (25:01 / 24:56, exit 0:0, zero retries). s4 **0.6630** (541/816), s42 **0.6544** (534/816). Same 17 tasks as 1842871 — identical task set, no subset caveat — same `rc365_units.sbatch`, CKPT_STEP=50000, 48 eps/task: **piv4 0.663/0.654 vs pi_v5 enc-dec 0.3725 (304/816) vs pi_causal 0.3627 (296/816)**. +0.29 absolute, +78% relative. `ervla_robocasa365_piv4.yaml` and `ervla_robocasa365_pi_v5.yaml` share `encdec_ckpt=enc_dec_2b_v5_tb9216/checkpoint-30054`, `enc_dec: true`, `skip_decoder: true`, `collect_encoder_layers: true`; they differ only in head topology (`QwenPI_v3`+`DiT-B`+`layerwise_attention_layout: alternating` vs `QwenPI_v4`+`LayerwiseFM_v4`). So the same encoder yields 0.37 or 0.66 depending only on how the head consumes it — the encoder was not the weak component on RoboCasa, the head was. **Open:** no `QwenPI_v4` + causal-backbone arm exists, so head-quality and head-encoder-fit are not yet separated. |
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
| 1893392 (+retry1894445) | rc_piv4c_s42now | RoboCasa eval, **causal backbone + LayerwiseFM_v4**, seed42 @50k — the missing 2x2 cell | **PARTIAL, 13/17 tasks: 0.638.** Matched against encoder+layerwise (1858853) on the same 13 tasks: **encoder 0.631 vs causal 0.638, delta +0.006 = 0.2 SE — statistically identical.** 3 units (CloseToasterOvenDoor, OpenDrawer, PickPlaceDrawerToCounter) died on **GPU2 of jpbo-063-01** with `Fatal Python error: Aborted` after 3 attempts each — the degraded-GPU signature, all on one device; resubmitted as1894445. **This falsifies the encoder attribution**: on RoboCasa the +0.28 belongs to the layerwise head, not to the bidirectional backbone. Seed4 still training (1891593). |
| 1877720–1877727 | initial tr_g* grid segments | **CANCELLED intentionally after100–300 warmup steps** when the training allocation was corrected from2h to12h. No complete checkpoint was written; replacements are1878150–57. |
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

### Architecture × batch grid replacements1878150–1878157 and added pair1878254

All eight one-node smokes passed20 optimizer updates, trainer eval at10/20 and checkpoint save inside interactive allocation1873548. Training jobs use12h limits. The wrapper safely checkpoints the actual step plus full optimizer/scheduler/RNG state at the five-minute warning, submits a fresh12h resume segment if80k is incomplete, and submits2h exact4k evaluations whenever20k/40k/60k/80k appears. Generated IDs and parents are recorded in `slurm_logs/libero_grid_job_ledger.tsv`.

```bash
sbatch --parsable --job-name=tr_g32zt --time=12:00:00 train_libero_grid_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_full.yaml ervla_grid_tokfull_b32_s42 examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_camdrop030.yaml ervla_grid_tokcam_b32_s42 42  # 1878150

sbatch --parsable --job-name=tr_g32ct --time=12:00:00 train_libero_grid_pair_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_fullmem_noz.yaml ervla_grid_encfull_b32_s42 examples/LIBERO/train_files/ervla_causal_gr00t_fullmem_noz_b64.yaml ervla_grid_causal_b32_s42 42 --datasets.vla_data.per_device_batch_size 16 --trainer.max_train_steps 80000 --trainer.num_warmup_steps 8000 --trainer.save_interval 20000 --trainer.eval_interval 1000 --trainer.save_full_training_state true --trainer.save_total_limit 32 --trainer.full_state_save_total_limit 2  # 1878151

sbatch --parsable --job-name=tr_g64zf --nodes=1 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_full.yaml ervla_grid_tokfull_b64_s42 42  # 1878152
sbatch --parsable --job-name=tr_g64zc --nodes=1 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_camdrop030.yaml ervla_grid_tokcam_b64_s42 42  # 1878153
sbatch --parsable --job-name=tr_g128zc --nodes=2 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_camdrop030.yaml ervla_grid_tokcam_b128_s42 42  # 1878154
sbatch --parsable --job-name=tr_g128e --nodes=2 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_fullmem_noz.yaml ervla_grid_encfull_b128_s42 42  # 1878155
sbatch --parsable --job-name=tr_g256zc --nodes=4 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_camdrop030.yaml ervla_grid_tokcam_b256_s42 42  # 1878156
sbatch --parsable --job-name=tr_g256e --nodes=4 --time=12:00:00 train_libero_grid_slurm.sh examples/LIBERO/train_files/ervla_v5_gr00t_fullmem_noz.yaml ervla_grid_encfull_b256_s42 42  # 1878157
sbatch --parsable --job-name=tr_g32zp --time=12:00:00 train_libero_grid_pair_slurm.sh examples/LIBERO/train_files/ervla_gr00t_sharedz_v5_zmem4_tokenonly_zonly.yaml ervla_grid_zonly_b32_s42 examples/LIBERO/train_files/ervla_v5_piv4_actiononly_grid_b32.yaml ervla_grid_piv4_b32_s42 42  # 1878254
```

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

### RoboCasa 50k needs THREE segments, not two — budget for it

`save_interval: 10000` interacts badly with the 12h wall. A segment that dies at
step 28,036 resumes from **20,000**, not 28,036, so up to 9,999 steps are thrown
away per timeout, and the two seeds in a pair drift apart because they hit the
wall at different steps. On the causal arm: s4 lost 8,036 steps, s42 only 648,
and after one continuation s4 is 11,600 steps behind its pair-mate.

piv4 got to 50k in two segments (1858730 -> 1858851) by luck of where its
checkpoints fell. Assume three for anything at this step count, and chain the
continuations with `afterany` up front rather than reacting to each timeout.

A cheaper fix for future runs: `save_interval: 5000` halves the worst-case loss
at the cost of more checkpoint space (these checkpoints are 5.84 GB each because
`freeze_modules: ''` saves the whole backbone).

### RESULT: the RoboCasa 2x2 closes against the encoder hypothesis

Seed 42, 13 tasks evaluated on both arms with identical manifests:

| backbone | head | mean |
|---|---|---:|
| causal | DiT-B | 0.354 |
| v5 encoder | DiT-B | 0.379 |
| v5 encoder | LayerwiseFM_v4 | 0.631 |
| **causal** | **LayerwiseFM_v4** | **0.638** |

The two layerwise arms differ by 0.006 against an SE of 0.027. So the large
RoboCasa gain is a property of **how the action head reads the backbone**, not of
bidirectional encoding. Any paper claim of the form "bidirectional backbones
improve manipulation" is not supported by this benchmark and must be dropped.

What survives: (a) the embodied-reasoning result, where the bidirectional 2B
beats a matched causal 2B 0.690 vs 0.576 on 12 benchmarks and matches a 9B —
that is a backbone claim and is untouched by this; (b) the control claim
restated as an interface result, which is what the evidence actually supports.

Caveats before this goes in a paper: single seed (s4 still training under
1891593); 13 of 17 tasks, the other 4 lost to a bad GPU and being re-run as
1894445; and the per-task profiles differ sharply even though the means agree
(NavigateKitchen +0.31 for causal, CloseBlenderLid -0.21, PickPlaceCounterToStove
+0.25), which is worth a paragraph of its own.

## Planned / not yet launched

### RoboCasa PI-v4 head on causal backbone — LAUNCHED 1878149 -> 1878217 -> 1878219 -> 1878223/24

The missing cell of the 2x2. At step 50k:

| backbone | head | mean |
|---|---|---:|
| causal | DiT-B | 0.354 |
| v5 encoder | DiT-B | 0.379 |
| v5 encoder | LayerwiseFM_v4 | **0.661 / 0.659** |
| causal | LayerwiseFM_v4 | **this run** |

Near 0.66 means the layerwise head is simply better and the encoder claim
collapses; near 0.35 means the encoder does the work and only this head can read
it. No other single run separates those.

Submit from `/e/project1/m3/blank4/code/starVLA-upstream-merge` — RoboCasa is
worktree-only and the live tree fails with `Unable to open file`.

```bash
Y=examples/simBenchmarks/Robocasa_365/train_files/ervla_robocasa365_piv4_causal.yaml

sbatch --parsable --time=02:00:00 --job-name=sm_rc_piv4c \
  train_robocasa365_seed_pair_slurm.sh "$Y" ervla_robocasa365_piv4_causal_smoke 4 42 \
  --trainer.max_train_steps 20 --trainer.num_warmup_steps 2 \
  --trainer.eval_interval 10 --trainer.save_interval 20 --trainer.logging_frequency 1   # 1878149

sbatch --parsable --dependency=afterok:1878149 --job-name=tr_rc365_piv4c \
  train_robocasa365_seed_pair_slurm.sh "$Y" ervla_robocasa365_piv4_causal 4 42          # 1878217

sbatch --parsable --dependency=afterany:1878217 --job-name=tr_rc365_piv4c_rs \
  train_robocasa365_seed_pair_slurm.sh "$Y" ervla_robocasa365_piv4_causal 4 42 \
  --trainer.is_resume true                                                              # 1878219
```

Evals (from `/e/scratch/m3/blank4/rc365_smoke`), manifests generated from
`manifest_piv4_s{4,42}.txt` by swapping only the run id — verified identical
otherwise, so task list, n_envs=24 and 48 episodes/task all match the piv4 run:

```bash
sbatch --parsable -J rc_piv4c_s4 -t 03:00:00 --dependency=afterok:1878219 \
  -o /e/scratch/m3/blank4/rc365_smoke/rc_piv4c_s4_%j.out \
  -e /e/scratch/m3/blank4/rc365_smoke/rc_piv4c_s4_%j.out \
  --export=ALL,MANIFEST=/e/scratch/m3/blank4/rc365_smoke/manifest_piv4_causal_s4.txt,OUTDIR=robocasa365_piv4_causal_50k,SEED=42,VIDEOS=0,CKPT_STEP=50000 \
  /e/scratch/m3/blank4/rc365_smoke/rc365_units.sbatch   # 1878223  (s42 identical -> 1878224)
```

Watch in the smoke: this arm sets `freeze_modules: ''` (matching `pi_causal`,
since there is no enc-dec structure to freeze), so unlike piv4 it trains the
whole backbone — heavier, and a different gradient path than any previous
LayerwiseFM_v4 run.

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

## LIBERO piv4 image-dropout arms (pre-encoder modality dropout)

New mechanism: `framework.modality_dropout` in
`starVLA/model/modules/modality_dropout.py`, mixed into `Qwen_PI_v3` so `QwenPI_v4`
inherits it. Removes a row's images **before** the bidirectional encoder runs, which is
what LIT / APT / LA4VLA / action-priors all force and what the existing
`shared_z.memory_dropout_mode: camera` does **not** (that one masks image columns out of
the action head's memory *after* the encoder has already mixed vision into the language
tokens).

Two removal mechanisms, deliberately not interchangeable:
* `removal: pixels` zeroes `pixel_values` after the processor. Token count, sequence
  length, masks, mrope positions and FLOPs stay bit-identical to a sighted row, so rows at
  different rates stay comparable inside one batch — but a blind row costs a full sighted
  step. Same argument as `QWen3_Blind.py`.
* `removal: tokens` never hands the images to the processor: no image tokens, no vision
  tower, much shorter sequence, so a bigger batch fits. The instruction then sits at a
  different offset than in a sighted row, so this belongs in a *uniform* phase, never
  mixed with sighted rows.

`vision_rate` accepts a scalar or a `{start, end, start_step, end_step}` ramp with the
same semantics as `trainer.loss_scale` ramps. `Trainer._publish_train_step` pushes
`completed_steps` onto the framework every optimizer step so the schedule advances.

**Correction worth carrying forward:** the piv4 configs' `freeze_modules` does NOT freeze
the encoder. In `full_duplicate`, `language_model.layers` is the **decoder**
(`train_starvla.py:1477`), which `skip_decoder: true` never runs; the encoder lives in
`language_model.encoder_layers` and is absent from the list. A real enc-dec run logs
`4511.708M Total, 2127.532M Trainable`, i.e. one full stack trains. The "frozen-backbone
representation comparison" comment in the RoboCasa configs does not describe the config.
Only `ervla_v5_piv4_seq_stage1.yaml` actually freezes the VLM (`freeze_modules:
'qwen_vl_interface'`).

Control for every arm: `ervla_v5_piv4_actiononly.yaml`, seeds 42/43 at 20k (training
1858694, evals 1858835/36). Do not re-run it.

**All arms carry proprioception.** `include_state: true` / `state_dim: 8`, and
`Qwen_PI_v3._add_state_conditioning` quantises the normalised state into 256 uniform bins
over [-1,1] and appends `<instruction> [STATE] <bins> [ACTION]` (the pi-0.5 convention,
`share_tools.state2str_transform`), so state reaches the bidirectional encoder as ordinary
text tokens. A blinded row therefore carries instruction + state, which is LA4VLA's
vision-free stage-1 input exactly.

That forces a **new control**: `ervla_v5_piv4_state.yaml`, identical to
`ervla_v5_piv4_actiononly.yaml` apart from those two entries. The dropout arms are measured
against *it*, not against the older no-state run — adding state and adding dropout are two
changes. The no-state run (1858694, evals 1858835/36) stays useful as the separate "does
state help at all" reference.

```bash
cd /e/project1/m3/blank4/code/starVLA

# Mandatory smoke gate: 20 updates + eval 10/20 for every arm, four GPUs in parallel.
sbatch --parsable smoke_piv4_imgdrop_slurm.sh                                   # 1878940
```

### Launched 2026-09-18 19:02 (gate 1879430 passed)

Order matters: **`state` and `blind` run first, as a pair.** `blind` is the state-only
ceiling, and if it lands near its control then LIBERO success cannot discriminate between
visual representations and nothing in the other arms is interpretable. `QWen3_Blind.py`
records proprioception alone already matching the best VLM backbone on LIBERO at every
demo count, and `PROBE_INSIGHTS.md` line 72 records causal RoboCasa collapsing 100/144 →
0/144 when the state suffix is removed. This pair measures how much of LIBERO success
survives with state but no pixels.

```bash
cd /e/project1/m3/blank4/code/starVLA

# 1. the ceiling (run first, alone)
sbatch --parsable --job-name=tr_piv4_blind train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_blind.yaml \
  ervla_v5_piv4_blind 42 43

# 2. the headline arm: annealed image dropout, one run, no second checkpoint
sbatch --parsable --job-name=tr_piv4_anneal train_seed_pair_slurm.sh \
  examples/LIBERO/train_files/ervla_v5_piv4_imgdrop_anneal.yaml \
  ervla_v5_piv4_imgdrop_anneal 42 43

# 3. the staged comparison (one seed per node: stage 2's checkpoint path carries the seed)
#    stage 1: 4k x 128 global (32/device x 4 GPUs), whole VLM frozen, no image tokens
for S in 42 43; do
  sbatch --parsable --job-name=tr_piv4_seq1_s$S --gres=gpu:4 train_libero_slurm.sh \
    --config examples/LIBERO/train_files/ervla_v5_piv4_seq_stage1.yaml \
    --run_id ervla_v5_piv4_seq_stage1_s$S --seed $S
done
#    stage 2: 12k x 64 global (16/device x 4 GPUs), ordinary sighted training from it.
#    4,000x128 + 12,000x64 = 1.28M samples = the control's 20,000x64. Matched on samples,
#    not steps. Fill in the stage-1 job id for the dependency.
for S in 42 43; do
  sbatch --parsable --job-name=tr_piv4_seq2_s$S --gres=gpu:4 \
    --dependency=afterok:<STAGE1_JOBID_s$S> train_libero_slurm.sh \
    --config examples/LIBERO/train_files/ervla_v5_piv4_seq_stage2.yaml \
    --run_id ervla_v5_piv4_seq_stage2_s$S --seed $S \
    --trainer.pretrained_checkpoint \
      playground/Checkpoints/ervla_v5_piv4_seq_stage1_s$S/checkpoints/steps_4000_pytorch_model.pt
done
```

RoboCasa365 counterparts exist in the worktree
(`ervla_robocasa365_piv4_imgdrop_anneal.yaml`, `ervla_robocasa365_piv4_blind.yaml`,
schedule scaled to the 50k budget: blind to 5k, ramp to 25k, sighted thereafter) and are
**not** launched — LIBERO first. Their control is
`ervla_robocasa365_piv4.yaml`, seeds 4/42 at 50k: **0.6630 / 0.6544** (evals 1858852/53).
There, `include_state: true` and PI puts state in the prompt, so the blind arm is a
state-only ceiling rather than an instruction-only one.

### Operating notes

* A rate set on `enabled: false` is rejected at construction. That combination is a
  silent no-op arm, which is the one failure mode that yields a clean-looking duplicate
  of the control.
* `removal: tokens` + `mode: camera` is rejected: rows would differ both in length and in
  which camera sits at which offset.
* Watch `vision_keep_rate` (realised) against `vision_drop_rate` (scheduled) in wandb. A
  persistent divergence means `protect_state` is rescuing more rows than intended.
* Unit coverage: `test/test_modality_dropout.py`, 26 tests, mostly guarding the
  row→image→patch mapping — if that is wrong the arm trains identically to the control
  and reports a plausible, wrong "no effect".

### W&B

Group **`piv4_imgdrop_libero`** in `starvla_libero` (entity `niblank`). `.wandb_group` is
pre-seeded into every output dir by `launch_piv4_imgdrop_libero.sh`, because the trainer
reads that file and never writes it — a `--wandb_group` override alone is lost on a 12h-wall
resume.

`WANDB_MODE=offline` on compute nodes, so nothing appears online until it is synced:

```bash
cd /e/project1/m3/blank4/code/starVLA
nohup bash scripts/sync_piv4_imgdrop_wandb.sh --watch \
  1879710 1879711 1879712 1879719 1879720 1879723 1879724 \
  > slurm_logs/piv4_imgdrop_wandb_sync.log 2>&1 &     # running since 19:02, pid 2247095
```

It discovers run dirs by their `.wandb_group` file (so a later arm needs no edit), appends
every offline segment into the run's canonical `.wandb_run_id` (so a resumed arm stays one
W&B run rather than several unrelated ones), fixes the group server-side, loops every 15
min, and takes a final snapshot once all watched jobs end. Login node only — compute nodes
have no route to wandb.ai. One pass without `--watch`.

### What to read, in order

1. **`blind` vs `state`** (1879715/16 vs 1879713/14). This decides whether anything else is
   interpretable. `QWen3_Blind.py` records proprioception alone already matching the best
   VLM backbone on LIBERO at every demo count; `PROBE_INSIGHTS.md` line 72 records causal
   RoboCasa collapsing 100/144 → 0/144 without the state suffix. If blind lands near its
   control, stop and say so — that is a publishable negative and the control neither LIT,
   APT, LA4VLA nor the action-prior papers report.
2. **`anneal` vs `state`** — does the one-stage curriculum buy anything at all.
3. **`anneal` vs `seq_stage2`** — one-stage mixing against the staged version, at matched
   samples. LA4VLA is the only paper that A/B'd these (sequential 86.75 vs mixed 87.53 on
   MetaWorld; 96.28 vs 95.75 on LIBERO — a tie both ways), so a tie here reproduces theirs
   and the interesting outcome is a gap.

### Not launched

RoboCasa365 counterparts (`ervla_robocasa365_piv4_imgdrop_anneal.yaml`,
`ervla_robocasa365_piv4_blind.yaml` in the worktree, schedule scaled to 50k: blind to 5k,
ramp to 25k, sighted after) are written and validated but **not** submitted — LIBERO first.
Control there is `ervla_robocasa365_piv4.yaml` seeds 4/42 at 50k: **0.6630 / 0.6544**
(evals 1858852/53). RoboCasa already has `include_state: true`, so no config change is
needed to carry state.

## W&B groups and the sync watchers

`WANDB_MODE=offline` on compute nodes, so **nothing appears online until something syncs
it**. `scripts/sync_wandb_group.sh` is that something. It is group-agnostic: `WANDB_GROUP_`
selects which runs an instance owns, so one copy serves every family and several instances
watch different groups side by side. (It was `sync_piv4_imgdrop_wandb.sh`; stop a watcher
before renaming or editing its file — bash lazy-reads a running script by byte offset.)

| group | runs | watcher |
|---|---|---|
| `piv4_imgdrop_libero` | the 8 image-dropout runs (1879710/11/12/19/23 + the two stage-2 continuations) | `--watch 1879710 1879711 1879712 1879719 1879720 1879723 1879724` |
| `libero_batch_sweep` | the 12 global-batch runs above | `--watch 1878150 1878151 1878152 1878153 1878154 1878155 1878156 1878157 1878254` |

```bash
cd /e/project1/m3/blank4/code/starVLA
WANDB_GROUP_=piv4_imgdrop_libero nohup bash scripts/sync_wandb_group.sh \
  --watch 1879710 1879711 1879712 1879719 1879720 1879723 1879724 \
  > slurm_logs/wandb_sync_piv4_imgdrop.log 2>&1 &
WANDB_GROUP_=libero_batch_sweep nohup bash scripts/sync_wandb_group.sh \
  --watch 1878150 1878151 1878152 1878153 1878154 1878155 1878156 1878157 1878254 \
  > slurm_logs/wandb_sync_batch_sweep.log 2>&1 &
```

Operating notes for the watcher:

* Login node only — compute nodes have no route to wandb.ai.
* It appends every offline segment into the run's canonical `.wandb_run_id`. The sweep runs
  already carry **two** streams each (one per 12h segment), which without this would be two
  disconnected W&B runs per arm instead of one.
* `--append --include-synced --no-mark-synced` **does not duplicate**: verified on a run
  synced twice concurrently — 4 rows, 4 unique steps, 0 duplicates. It appends by offset,
  so re-sending a prefix is a no-op.
* Grouping an already-running experiment works **server-side**: a run past `wandb.init`
  cannot pick up a newly written `.wandb_group`, so `fix_groups` sets it through the API.
  The sweep runs went `vla-train` → `libero_batch_sweep` that way. Their next resumed
  segment inits with the group directly from the file.
* The project resolves case-insensitively: URLs may read `starVLA_Libero` while the config
  says `starvla_libero`. Cosmetic; the API finds the run either way.
* A run can show `state=finished` on the server while training continues — each sync pass
  finalises its upload. Read `squeue`, not the W&B state badge.

## Batch sweep v2 — why v1 was cancelled and what replaced it

**v1 (1878150-57, 1878254) was uninterpretable.** Every arm got
`max_train_steps 80000 / num_warmup_steps 8000` whatever its batch. Expressed in samples
that is wildly unequal:

| arm | warmup samples | % of the anchor's 2.56M budget | LR at 2.56M |
|---|---|---|---|
| b32 | 256k | 10% | schedule complete |
| b64 | 512k | 20% | ~50% of peak |
| b128 | 1.02M | 40% | ~85% of peak |
| b256 | **2.05M** | **80%** | **~100% of peak** |

At the sample-matched point b256 had spent 80% of its budget warming up and was still at
peak LR. `ENCODER_PUSH_PLAN.md` line 7 had in fact specified a schedule "parameterized by
samples"; v1 did not implement it.

**v2 schedule.** `num_warmup_steps 2000` for every arm — the absolute warmup the 20k runs
used (`ervla_zonly_gr00t_sharedz_ground_temporal_v5`, the 78.475% anchor, and its b64
control are both 2000/20000). Steps are sample-matched to 2.56M where that clears a 30,000
step floor, else the floor:

| arm | steps | samples | note |
|---|---|---|---|
| b32 | 80,000 | 2.56M | matches the anchor exactly |
| b64 | 40,000 | 2.56M | |
| b128 | 30,000 | 3.84M | floor |
| b256 | 30,000 | 7.68M | floor |

The floor is not optional: the b64/10k **mid-cosine** checkpoint scored 69.663% against
76.875% for the same run with its schedule complete — **-7.2pp from schedule position
alone**, against a 1.6pp batch effect. Every v2 arm completes its own cosine at its own
endpoint, so no arm is ever read mid-schedule.

**Stated limitation:** b128/b256 are *not* sample-matched; they see 1.5x and 3x the
anchor's data. That is the conservative direction — if b32 still wins on a third of the
samples the sample-efficiency claim is stronger; only a large-batch win would be
unattributable.

**Run ids are `ervla_gridv2_*`**, not reused, so v1's checkpoints, dataloader state and
offline W&B streams are neither resumed from nor appended to at colliding step numbers.

```bash
cd /e/project1/m3/blank4/code/starVLA
bash relaunch_batch_sweep_v2.sh          # emits every job id
WANDB_GROUP_=libero_batch_sweep_v2 nohup bash scripts/sync_wandb_group.sh \
  --watch 1881040 1881041 1881043 1881044 1881045 1881046 1881047 1881048 1881049 \
  > slurm_logs/wandb_sync_batch_sweep_v2.log 2>&1 &
```

### The anchor these arms are chasing

`ervla_zonly_gr00t_sharedz_ground_temporal_v5` — shared-z + GR00T + readout, **global batch
32**, 20k steps, warmup 2000. LIBERO-plus exact4k seeds 42/43 = 78.675% / 78.275%, mean
**78.475%** (SD 0.283pp). Matched b64 control = 76.875%, **-1.600pp** (s42 -3.075, s43 -0.125).

From `results_collected/batch32_vs_batch64_training_curves.csv`:

| comparison | b32 | b64 |
|---|---|---|
| action loss, steps 16k-20k (matched STEPS) | 0.04636 | **0.03061** |
| action loss, final 20k | 0.04377 | **0.02557** |
| action loss, 320k-640k examples (matched SAMPLES) | **0.05225** | 0.05939 |
| open-loop MSE, matched samples | **0.04844** | 0.05418 |
| mse_score RMSE-corrected, matched samples | **0.14075** | 0.15337 |
| mse_score RAW, matched samples | 0.00332 | 0.00256 ← **invalid** |

**Do not compare raw `mse_score` across per-device batch sizes.**
`train_starvla.py:1071` computes `np.linalg.norm(pred - gt) / np.prod(actions.shape)`,
i.e. **RMSE / sqrt(N)** with `N = eval_batch x action_horizon x action_dim`. It therefore
carries a spurious **1/sqrt(batch)** factor. Verified against the archived curves:

    b32 anchor   per_device=16  N=1792  raw*sqrt(N) = 0.14054   (csv 0.14075)
    b64 control  per_device=32  N=3584  raw*sqrt(N) = 0.15326   (csv 0.15337)
    raw ratio 0.7711  vs  predicted (1/sqrt2)*(RMSE ratio) 0.7705

Two things follow:

* It is **per-device** batch, not global: `eval_action_model` runs `predict_action` on the
  main process's local batch only. The anchor used per_device 16 and its b64 control 32 --
  that is the whole 1/sqrt(2). **Sweep v2 is unaffected**, because every arm holds
  `per_device_batch_size: 16` and takes its global batch from GPU count, so the eval batch
  is 16 everywhere and `mse_score` is directly comparable across the grid.
* `action_horizon` is in that same denominator. It is 16 in every sweep config, the anchor
  and the b64 control, so it is not a confound today -- but a horizon-8 row
  (`OPTIMIZATION_SWEEP_PLAN.md` line 64) would show a raw `mse_score` ~sqrt(2) "better" for
  free. Recover RMSE as `raw * sqrt(eval_batch * horizon * action_dim)` before comparing.

Also note `eval_action_model` draws from `self._get_next_batch()` -- the **training**
dataloader. `mse_score` is a fit diagnostic on a training batch, not validation;
`open_loop/mse_mean` on fixed trajectories is the held-out one.

b64 ends training with *lower* action loss and *worse* rollout. The supervised metrics do
not select the better checkpoint — screen on exact4k success, not on the loss curve.

## Chunk size: LIBERO is 8, and that is upstream's default

`action_horizon: 8` (with `future_action_window_size: 7`) is now set in all eleven
in-flight LIBERO configs — the five piv4 image-dropout arms and the six sweep arms.

**This restores the upstream convention rather than inventing one.** On
`upstream/starVLA_dev` (github.com/starVLA/starVLA):

| file | action_horizon |
|---|---|
| `examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml` | **8** |
| `examples/modelExtensions/CoTrainVLM/train_files/starvla_cotrain_libero.yaml` | **8** |
| `starVLA/config/training/starvla_cotrain_libero.yaml` | **8** |
| `QwenGR00T` dataclass default | **8** |
| `QwenPI` dataclass default | 16 |
| `openpi/pi05_libero_8gpu.yaml` / `pi0_libero_8gpu.yaml` | 10 / 50 (openpi recipes, not ours) |

Every native starVLA LIBERO config is 8. Our 16 came from `QwenPI`'s dataclass default by
inheritance, not by decision — the ervla configs descend from the PI lineage.

**Timing.** The eval env is `OffScreenRenderEnv` at robosuite's default `control_freq=20`
and `model2libero_interface.py:179` steps one action per env step, so **chunk 8 = 0.4 s**
at rollout; one second would be 20 steps. Note the LeRobot mirror declares `fps: 10.0`
(`libero_lerobot/libero/meta/info.json`) against that 20 Hz env — if the demos really were
subsampled 2x, every LIBERO run in this repo trains on 10 Hz motion and replays at 20 Hz.
Unverified and independent of the chunk change, but worth closing out.

**Comparability cost, stated plainly.** The 78.475% LIBERO-plus anchor
(`ervla_zonly_gr00t_sharedz_ground_temporal_v5`), its b64 control and every archived curve
in `results_collected/batch32_vs_batch64_training_curves.csv` were trained at horizon 16.
The chunk-8 runs are a different policy class, so "did we beat 78.475%" is no longer
like-for-like. The cheap way to keep that comparison is the execute-4/8/16 inference screen
on the existing horizon-16 checkpoint (`OPTIMIZATION_SWEEP_PLAN.md` line 64), which needs
no retraining.

Raw `mse_score` will also read ~sqrt(2) "better" at chunk 8 for free, because
`action_horizon` sits in its `N = eval_batch x horizon x action_dim` denominator. Corrected
RMSE is unaffected.

**Restart hygiene.** Relaunching reuses the same run ids, and the cancelled horizon-16 runs
had already written `.wandb_run_id`, offline W&B streams and (for `seq_stage1`) checkpoints
into those directories. Left alone, the new runs would have appended chunk-8 history onto
the chunk-16 W&B runs at colliding steps, and `seq_stage2` would have loaded a horizon-16
checkpoint into a horizon-8 head. Both families' stale directories were moved to
`playground/Checkpoints/_archive_h16_*` before any job initialised, and `.wandb_group` was
re-seeded into the fresh ones. **Archive before relaunching under a reused run id.**

## Sequential arm: 10k + 20k, and what it can claim

Stage 1 was 4,000 steps (total-sample-matched to the control). Too short for a prior to
form when the whole VLM is frozen and only the piv4 head trains. Now:

| | steps | global batch | samples |
|---|---|---|---|
| stage 1 (vision-free, VLM frozen) | 10,000 | 128 | 1.28M |
| stage 2 (sighted, full FT) | 20,000 | 64 | 1.28M |
| **total** | | | **2.56M** |
| control `ervla_v5_piv4_state` | 20,000 | 64 | 1.28M |

**Stage 2 now matches the control exactly** — same 20,000 steps, same warmup 2,000, same
64 global batch — so the only difference between them is the initialisation. That is the
LIT/APT convention (APT runs 100k stage-1 iterations before its stage 2) and it is the
framing the research question needs: *does a vision-free pretraining phase help?*

**It is NOT total-compute-matched.** The arm spends 2x the control's samples. A win
therefore supports "vision-free pretraining helps", **not** "staging is free". If the
stronger claim is wanted, the control has to be rerun at 40,000 steps (2.56M samples);
that arm does not currently exist.

Chain is fully automatic: `tr_seq1 -> tr_seq2 (afterok) -> exact4k (afterok)`. Nothing
needs launching by hand when stage 1 lands.

## Blind-phase length is now an ablation, not an assumption

Two annealed arms share every other setting:

| arm | blind through | ramp | fully sighted from | job |
|---|---|---|---|---|
| `imgdrop_anneal` | 2,000 | 2k → 10k | 10,000 | 1881719 |
| `imgdrop_anneal_late` | 10,000 | 10k → 20k | never (rate hits 0 at the last step) | 1885185 |

**Why the second one exists.** At step 14,000 the original arm is fully sighted yet its
loss still tracks the permanently blind arm rather than the sighted control — pooled over
both seeds, steps 12k+: `state` 0.0370, `blind` 0.0416, `anneal` **0.0415**, against a
seed spread of 0.0024. Four thousand fully sighted steps did not recover the gap. Either
vision is worth almost nothing on this loss (blind−state is only +0.0046, ~2x the seed
noise), or the blind phase durably suppressed visual reliance. The second is the mechanism
LIT/APT claim to induce; the pair of arms is what separates "the blind phase was too
short" from "the blind phase did something that does not wash out".

**Known tension in the late arm, accepted deliberately:** the ramp ends at
`max_train_steps`, so no row is ever trained with dropout fully off, while evaluation is
fully sighted (`inference_vision_rate: 0.0`). The policy meets an input distribution at
rollout it never saw in training. If it underperforms for that reason rather than on
merit, the fix is a third arm ending the ramp near 18,000 to leave a sighted tail.

It also makes the staging comparison clean: 10,000 blind steps is exactly the sequential
arm's stage 1 (`ervla_v5_piv4_seq_stage1.yaml`), so `anneal_late` vs `seq_stage2` is
one-stage mixing against two-stage curriculum at **equal blind budget** — the A/B LA4VLA
ran (sequential 86.75 vs mixed 87.53 on MetaWorld, 96.28 vs 95.75 on LIBERO; a tie both
ways).

## RESULTS — image-dropout family, LIBERO-plus exact4k

All eight training jobs COMPLETED clean (exit 0:0). Evals at n=4,000 unless marked partial.

| arm | horizon | state | success | note |
|---|---|---|---|---|
| `actiononly_s42` / `s43` (old control) | 16 | no | **76.12% / 75.20%** | the healthy reference |
| `state_s43` | 8 | yes | **53.02%** | s42 still running |
| `seq_stage2_s43` | 8 | yes | **1.88%** | 10k blind stage 1 → 20k fully sighted stage 2 |
| `imgdrop_anneal_s42` | 8 | yes | **0.22%** | fully sighted from step 10,000 |
| `blind_s43` | 8 | yes | **0.03%** | blind at train and eval |

**1. Vision is necessary on LIBERO-plus. The state-only ceiling is zero.** `blind` scores
0.03% (1/4,000). This is the control none of LIT/APT/LA4VLA report, and it comes out
decisively: proprioception plus instruction solves nothing on this suite. Note this does
**not** contradict `QWen3_Blind.py`, which found proprioception competitive on *vanilla*
LIBERO — LIBERO-plus perturbs backgrounds, camera pose, objects and language, and state
alone cannot track any of that. The two results together say the state shortcut is a
vanilla-LIBERO artifact that the plus suite removes.

**2. Pre-encoder image dropout is catastrophic, and it does not recover.** `anneal` is
fully sighted for its last 10,000 steps and still scores 0.22%, i.e. `blind`'s number, not
the control's. `seq_stage2` gets 20,000 *fully sighted* steps after its frozen vision-free
stage and reaches only 1.88%. A vision-free phase does not "prime" this encoder; it
destroys visual grounding durably enough that ordinary sighted training cannot repair it.
That is the opposite sign of the LIT/APT/LA4VLA claim, measured with the controls they omit.

**3. Training loss completely failed to predict this.** Pooled over both seeds at steps
12k+: `state` 0.0370, `blind` 0.0416, `anneal` 0.0415 — a 12% spread on the flow-matching
loss for a **53% versus 0.2%** behavioural difference. The supervised metric ranked these
arms as near-identical. This is the sharpest instance yet of the warning already in
`PROBE_INSIGHTS.md` and the batch32/64 record: **screen on rollout success, never on the
loss curve.**

**Open confound, do not skip.** `state` (53.02%) versus `actiononly` (76.12%/75.20%)
changes **two** things at once: action_horizon 16 → 8 and the addition of quantised state.
The 23pp drop cannot be attributed to either alone. Resolving it needs one arm at
horizon 8 without state, or horizon 16 with state. Until then "chunk 8 is the upstream
default" is a convention claim, not a performance claim.

## Compute audit, 2026-09-19 ~10:00

Two real problems found, both fixed; everything else is doing useful work.

**1. Infinite resubmit loop (fixed).** `train_libero_grid_slurm.sh` line 148 hardcodes
`steps_80000` as its completion test and line 78 only evaluates 20k/40k/60k/80k. Sweep v2
targets 40k (b64) and 30k (b128/b256), so those arms finished training, never produced a
`steps_80000` checkpoint, and resubmitted themselves forever — **six 14-minute segments in
90 minutes doing zero steps of work** (1888372/1888421/1888479/1888604/1888696/1888892,
all `COMPLETED 0:0`). The b64 arms were already done at 40,000 with their 20k and 40k evals
correctly auto-submitted, so only the loop was waste. Cancelled 1889234/1889295, wrote
`train_libero_grid_target_slurm.sh` (derives the target from `--trainer.max_train_steps`,
evaluates milestones *below* the target plus the target itself, and chains resumes onto
itself), and resumed the four 30k arms onto it from their checkpoints
(1889364/65/66/67). The b32 arms target 80,000, which matches the old hardcode, so they
were never affected and still run on the original script.

**2. Dead eval on a degraded render node (resubmitted).** `ep_seq2_s42` (1883889) ran
**4h21 with 213 permanently exhausted shards and zero completed episodes** — `rc=134`
(SIGABRT), the `mjr_readPixels` signature in CLAUDE.md. Cancelled and resubmitted as
1889391, which lands elsewhere.

**Partially degraded, left running:** grid evals 1888088 / 1888090 / 1888083 carry 12-31
permanently failed shards each and will return incomplete suites. Check their episode
counts before using; a suite short of 1,000 is not comparable with a full one.

**Worth knowing, not waste:** the auto-submitted b32 evals score `steps_20000` and
`steps_40000` of an **80,000-step cosine**. Those are mid-schedule checkpoints. They are
valid as a within-arm learning curve and **invalid as an endpoint comparison** — the
b64/10k diagnostic measured that error at -7.2pp. Do not put them in a batch-size table.

**How to spot the loop again:** `tail slurm_logs/libero_grid_job_ledger.tsv`. Resume
entries for the same run arriving every few minutes, with segments that `COMPLETED 0:0` in
~14 minutes, is the signature. A healthy segment runs to the 12h wall or to the target.

## BUG: the LIBERO eval never sent proprioceptive state — first eval round is VOID

`eval_libero.py` computed the 8-dim state and then dropped it when building the request:

```python
state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]))
observation = {..., "observation.state": np.expand_dims(state, axis=0), ...}
example_dict = {"image": [...], "lang": ...}      # state never forwarded
```

`Qwen_PI_v3.predict_action` then takes `state = ... if "state" in examples[0] else None`, so
every state-trained policy rolled out with **no `[STATE]` suffix and no state token**,
having had both in every training sample. The LIBERO client is the only one with this gap —
RoboCasa365, Franka, RoboTwin and RoboChallenge all send state. It was invisible because
every LIBERO config before 2026-09 used `state_dim: 0`.

**It explains the whole result pattern, monotonically in state-reliance:**

| arm | reliance on state | round-1 rollout |
|---|---|---|
| `actiononly` (no state) | none | 76.12% — unaffected |
| `state` | partial, vision compensates | 53.02% |
| `seq_stage2` (stage 1 = instruction+state only) | high | 1.88% |
| `anneal` | very high | 0.22% |
| `blind` (state is its only non-visual channel) | total | 0.03% |

and it explains the contradiction that exposed it: `seq2` finished with a **lower** action
loss than the control (0.01289 vs 0.01391), normal `mse_score` (0.00207) and normal
open-loop MSE (0.04045). Those are computed in the trainer, where state IS present. The
models are fine; only the rollout was broken. **No retraining is needed — re-evaluation is.**

### The fix

* `policy_wrapper.py` advertises `expects_state` in the handshake, derived from the
  checkpoint's own `framework.action_model.state_dim > 0`. Verified: state arms True,
  `actiononly` and every sweep arm False.
* `model2libero_interface.py` reads it (default False, the historically correct value).
* `eval_libero.py` forwards `observation.state` **only when the checkpoint expects it**.
  Sending state unconditionally would append a `[STATE]` suffix to models never trained
  with one and break the healthy 76% control.

**State is sent RAW, and that is deliberate.** `Libero4in1DataConfig.transform()`
normalises the ACTION keys only, so the training loader hands the framework raw state and
`state2str_transform` quantises that into 256 bins over [-1,1]. Passing raw state
reproduces the training quantiser exactly, saturation included (roll is ~pi and pins to bin
255 in training too). Normalising it here would be a different, equally silent bug.
Layout matches dim-for-dim: `modality.json` is `[x,y,z,roll,pitch,yaw,pad,gripper]` and
dims 6-7 hold the two `gripper_qpos` finger joints (observed range +/-0.04), exactly what
the client's concatenation puts there.

Round-2 evals: 1889568/69 state, 1889571/73 blind, 1889574/75 anneal, 1889576/77
anneal_late, 1889578/79 seq2.

### Also: widespread EGL shard failures

Running grid evals carry **36-60 permanently exhausted shards out of 128** each
(1888083/84/86/88/90), i.e. 28-47% of episodes missing. `rc=134` / `mjr_readPixels`, the
degraded-render-node signature. Those results are not merely noisier — a suite short of
1,000 episodes is **not comparable** with a full one and may be biased if the failures
correlate with particular tasks. Check `total_count == 1000` per suite before using any
number. 1883889 was the extreme case (213 exhausted, 0 episodes, 4h21) and was resubmitted.

## EGL aborts: cause, error handling, and the mitigation

**How errors are handled (they do NOT retry forever).** `eval_libero_plus_slurm.sh` sets
`SUITE_ATTEMPTS=3`. Each retry runs with `STARVLA_RESUME_EVAL=1`, so shards already on disk
are kept and only failures re-run; after 3 attempts the suite gives up and writes a failure
sentinel. Bounded — but on a bad node you still pay 3x before it stops, and the shards that
exhausted their attempts are simply **missing from the results**, which is what makes a
partial suite dangerous rather than merely noisy.

**Why it hits LIBERO-plus and not plain LIBERO.** The concurrency differs by 8x:

| | workers_per_gpu (actual) | servers_per_gpu |
|---|---|---|
| `eval_libero_slurm.sh` | **4** | 1 |
| `eval_libero_plus_slurm.sh` | **32** | 1 |

Each sim worker opens its own MuJoCo EGL context, so LIBERO-plus runs 32 concurrent EGL
contexts per GPU against plain LIBERO's 4. Note the LIBERO-plus **docstring is stale** — it
advertises `default: 8` and `servers_per_gpu default: 2` while lines 124-125 actually apply
32 and 1. The aborts land immediately after `Starting episode 1...`, i.e. at the first
render, and are `Aborted` (SIGABRT, rc=134) rather than a Python traceback.

This also explains the partial node-dependence recorded in CLAUDE.md. At 32 contexts a
healthy node copes and a marginal one does not, which is why two jobs showed 0 failures
while five showed 36-60 of 128 shards in the same hour. It is not purely "a bad node".

**Mitigation in flight:** the round-3 state-arm evals (1889613-1889622) run with
`--workers_per_gpu 12`. If failure counts drop to ~0, lower concurrency is the fix and the
default should be reduced (and the docstring corrected); if they do not, the cause is
elsewhere and node placement is the remaining lever. Compare permanently-exhausted shard
counts against round 2 before changing the default.

## Chunk size 8 is NOT the cause of the low sweep numbers

Retracted. `results_master.csv` has completed chunk-8 LIBERO-plus runs scoring well above
the sweep: `cot_trace_ours_v3_cotw01` **83.08%**, `cot_trace_det_v3_cotw01` 81.22%,
`gr00t_deeps` 77.17% — all `action_horizon: 8`, all n=4,000. Chunk 8 is fine.

`tokfull_b64` at 57.30% had a complete eval (n=4,000, not EGL-truncated), so that is a real
number for that arm. `ervla_gr00t_sharedz_v5_zmem4_tokenonly_full` is a **new variant that
has never completed a run before**; the ~76% z-family numbers it sits below are different
architectures, not the same one at a different chunk size. The simplest reading is that the
token-only zmem4 variant is weaker, not that anything is broken.

## Correction: the state switch already existed — it was unreachable, not missing

My earlier entry said the LIBERO eval had no state path and I added one. That was wrong in
two ways and is corrected here.

1. **I patched the wrong files, twice.** `examples/LIBERO/eval_files/` is the plain-LIBERO
   path. Every exact4k job runs **LIBERO-plus**, and specifically its *parallel* client
   `examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py`. Rounds 2 and 3 of
   the evals (1889568-79, 1889613-22) were therefore still stateless and were cancelled.

2. **The parallel client already supported state.** Line 896:

   ```python
   if os.environ.get("STARVLA_EVAL_INCLUDE_STATE", "0") == "1":
       if state.shape != (8,):
           raise RuntimeError(f"expected 8-D LIBERO state, got {state.shape}")
       example_dict["state"] = observation["observation.state"].astype(np.float32)
   ```

   It is off by default and carries the right 8-D shape guard. No new code was needed.

**The actual defect: the variable could never be set.** `eval_libero_in_one.sh` runs
`apptainer exec --cleanenv`, which strips the host environment apart from explicitly listed
`--env` flags, and `STARVLA_EVAL_INCLUDE_STATE` was not among them. Exporting it at any
level above the container had no effect, so the branch was unreachable dead code. Fixed by
adding the passthrough next to `STARVLA_PAIRED_ROLLOUT`:

```bash
--env "STARVLA_EVAL_INCLUDE_STATE=${STARVLA_EVAL_INCLUDE_STATE:-0}" \
```

**Round 4** (1889850-1889859) runs with `--export=ALL,STARVLA_EVAL_INCLUDE_STATE=1` and
`--workers_per_gpu 12`. Any future eval of a state-trained LIBERO checkpoint needs that
export or it silently measures a stateless policy.

Still outstanding: `policy_wrapper.metadata` now advertises `expects_state`, and the two
plain-LIBERO files auto-enable state from it, while the parallel path uses the env var.
Two mechanisms for one contract. The env var is authoritative for LIBERO-plus; the
handshake field is informational. Worth unifying — the auto-detecting version cannot be
forgotten at submit time, which is exactly how this round was lost.

## EGL mitigation looks effective

At `--workers_per_gpu 12`, all ten round-3 evals showed **0 permanently exhausted shards**
after 22 minutes (one had 28 retries, all of which succeeded). Round 2 at the default 32
workers showed 36-60 permanent failures per job in the same window. Evidence supports
lowering the default from 32; confirm on round 4 before changing it.

## Aggregator trap: re-running an eval into the same results dir MERGES with the old one

Round 4 of the image-dropout evals (state fixed, 12 workers) wrote into the same
`results/libero-plus/` directories as the stateless round 1 (32 workers). Shards are
named by their `[start,end)` episode range; a different worker count produces a different
sharding, so round-4 files did NOT overwrite round-1 files — they sat beside them, and
`aggregate_results.py` globs every `*.json` in the suite dir. Its own docstring (line 31)
warns about exactly this. Result: `state_s43` read **65.6%** from the JSON while its own
job log said **77.7%** — a blend of 53% stale and 77.7% fresh.

Fixed 2026-09-19 12:40: every file older than round 4's start was moved to
`results/libero-plus/_stale_round1_stateless/` in all ten dirs, per-suite and combined
`overall_results.json` regenerated, and each total verified against the per-job log count
(`grep -c '^INFO:dual_logger:Success: ' slurm_logs/eval_libero_plus_<jid>.err`). All ten
match to 0.1pp at n=4000.

**Rule:** when re-evaluating a checkpoint, either pass a fresh `output_dir` or clear the
old shards first. The job log is the ground truth; the JSON is only as clean as the dir.
Note also the harness logs every episode twice (`INFO -` and `INFO:dual_logger:`); count
one prefix only, or every episode figure is ~2x inflated (rates are unaffected).

## RESULTS (corrected) — image-dropout family, LIBERO-plus exact4k, n=4000, chunk 8, with state

| arm | s42 | s43 | mean |
|---|---|---|---|
| `state` (control) | 77.7% | 77.7% | **77.7%** |
| `seq_stage2` (10k vision-free frozen stage 1 → 20k sighted) | 61.4% | 61.1% | 61.3% |
| `imgdrop_anneal` (blind→2k, ramp→10k) | 40.9% | 37.2% | 39.1% |
| `imgdrop_anneal_late` (blind→10k, ramp→20k) | 34.5% | 38.1% | 36.3% |
| `blind` (state-only, train+eval) | 36.4% | 35.0% | 35.7% |

Supersedes the round-1 table above, which measured a stateless-eval bug, not the arms.

## Per-axis: bidirectional encoder vs matched causal, LIBERO -> LIBERO-plus zero-shot

Seed-averaged, n=8,000 per column (2 seeds x 4,000). `bidir+state` = QwenPI_v4, state_dim 8,
horizon 8. `bidir nostate` = QwenPI_v4, state_dim 0, horizon 16. `causal` = QwenPI_v3
decoder-only, state_dim 0, horizon 16. Note the head differs (v4 vs v3; v4 alone was worth
~+2.5pp in earlier matched seeds), so `bidir+state - causal` is NOT yet an architecture-only
number. The missing control is **causal + PI_v4 + state at horizon 8** — one training run.

| axis | bidir+state | bidir nostate | causal | bidir+state - causal |
|---|---|---|---|---|
| Camera Viewpoints | 63.7% | 55.3% | 52.1% | **+11.6** |
| Robot Initial States | 60.8% | 62.1% | 51.2% | **+9.5** |
| Objects Layout | 78.7% | 79.5% | 70.2% | **+8.5** |
| Light Conditions | 94.5% | 87.4% | 87.6% | +6.8 |
| Language Instructions | 83.9% | 84.1% | 80.2% | +3.7 |
| Background Textures | 92.9% | 92.5% | 92.3% | +0.6 |
| Sensor Noise | 78.9% | 77.7% | 81.0% | **-2.0** |
| **overall** | **77.7%** | 75.7% | 72.0% | **+5.7** |

Reading: the encoder's advantage is concentrated on the GEOMETRIC axes (camera, initial
state, layout) — the three LIBERO-Plus names as where SOTA VLAs are most brittle. It is flat
on background and loses on sensor noise. Adding state moved camera (+8.4) and light (+7.1)
but not initial-state or layout, which the architecture alone already carried.

**Operating note — a slow eval on a marginal node is not worth waiting out.** `--resume`
skips every shard already on disk, so cancelling a crawling eval and resubmitting into the
SAME `output_dir` loses no completed work and moves the remaining shards to a fresh node.
Signature: episode count far below sibling jobs launched at the same time, plus a nonzero
`exhausted 3 attempts` count. Recover the exact `output_dir` from
`sacct -j <id> -X -o SubmitLine%600` — a different dir makes `--resume` a no-op restart.

## Operating note: `--resume` silently re-runs every shard (found 2026-09-19)

`eval_libero_in_one.sh:408` builds the resume probe as
`${start}_to_${end}${result_tag}.json` where `result_tag` is only `_exact${N}`.
`eval_libero_model.py:703` appends `_shard${i}of${M}` on top of that when `num_shards > 1`.
So the probe looks for `0_to_2519_exact1000.json` while disk holds
`0_to_2519_exact1000_shard43of48.json` — **the test never matches and `--resume` is a no-op.**

Signature: a `--resume` job prints **zero** "already complete, skipping" lines despite a
populated `logs/<suite>/` directory, and runs for the full wall time.

Not a correctness bug as long as the re-run uses the **same worker count** as the files on
disk: the shard names then collide exactly and get overwritten in place. It becomes a
correctness bug the moment the worker count differs, because the old and new shards have
different names, coexist, and `aggregate_results.py` globs `*.json` over both — the same
merge trap that corrupted `state_s43` (65.6% JSON vs 77.7% log).

Fix is staged but NOT applied — `1893711` is executing this script and bash lazy-reads it by
byte offset. Apply after it drains:
`scratchpad/resume_shard_tag.patch` (guard: a `--resume` over a complete dir must print 48
skip lines and exit in <1 min).
