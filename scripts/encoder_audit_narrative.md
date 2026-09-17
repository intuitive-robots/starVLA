## Answers that determine the paper

**Encoder > causal is conditional on what “encoder” means.** Shared-z v5 averages77.375% (SD1.273 pp, n8,000) against causal71.658% (SD0.775 pp, n12,000): +5.717±0.623 pp using episode SE. Restricting both to seeds42/43 gives+5.375 pp; the paired seed gains are+3.975 and+6.775 pp. That is a reproducible internal-protocol gain, although two seeds give a very unstable small-sample confidence interval. The simpler old bidirectional control gains only+0.433±0.580 pp; v5+PI gains+1.467±0.644 pp; v5+GR00T gains+2.967±0.637 pp. The GR00T comparison changes the head as well as the backbone relative to causal+PI, so it is not a clean directionality effect.

On RoboCasa, causal580/1,632=35.539±1.185% versus v5 617/1,632=37.806±1.200% gives **+2.267±1.687 pp**, below two combined episode SE. Per-seed gains are+0.980 pp (seed4) and+3.554 pp (seed42); training-seed SDs are1.040 pp causal and0.780 pp v5. All17 declared tasks exist for both seeds; they do not constitute the official18-task atomic or50-task overall benchmark. On plain LIBERO, the same-arm n400 runs are smoke tests under the requested criterion: no matched full-protocol claim is established. LIBERO and its perturbation extension are also not two independent robot domains.

**Shared-z is the better-supported intervention; the encoder objective is not isolated.** Config.full.yaml comparisons, excluding run/output paths, show:

| comparison (all n4,000 each) | effect ± combined SE (pp) | what actually changes |
|---|---|---|
| zsup78.125 − zbase71.550 | +6.575 ±0.968 | shared-z module, supervision and temporal path together; not a one-factor objective test |
| zsup78.125 − zonly76.600 | +1.525 ±0.936 | only memory_dropout_rate0.15→1.0 |
| zsup78.125 − znodrop74.875 | +3.250 ±0.947 | only memory_dropout_rate0.15→0 |
| zsup78.125 − zshuf75.625 | +2.500 ±0.942 | only shuffle_targets false→true |
| MLM-control75.525 − W2-corrected74.700 | +0.825 ±0.967 | masked slots/prompt path; MLM **loss disabled** in the control |

The raw sources are the corresponding canonical rows above; config sources are `playground/Checkpoints/<run>/config.full.yaml`. Zonly is not “no supervision”: it has grounding, phase, trajectory and temporal losses, a128-dimensional z and4 learned queries; memory dropout1 forces the expert through z. Zbase has no shared_z block despite its name. “All bidirectional versus causal” changes checkpoint pretraining and frozen modules too. Causal+shared-z with the identical supervision is missing. The decisive experiment is encoder on/off × shared-z on/off at matched pretraining/data/compute, with a separate actual MLM-loss switch.

**Compute is not yet matched.** The seed-pair launcher allocates two GPUs per run, but shared-z v5 config.full.yaml uses per-device batch16 and the actual DeepSpeed log confirms train_batch_size32 (`slurm_logs/train_seed_pair_1836346_s42.log:334`). Causal and v5-PI configs use per-device batch32 on two GPUs (nominal global64). At20k steps this is approximately0.64M versus1.28M sampled examples before any sampler repeats. Shared-z head dropout is0.2, causal/v5-PI0.0, and v5-GR00T0.1. These are additional recipe factors; neither a matched-compute nor sample-efficiency claim follows from equal step counts. The launcher comment saying batch64 does not match the shared-z run.

**RoboCasa has task-specific reversals.** Pooling two seeds gives96 episodes per arm/task. Largest encoder gains: TurnOnMicrowave37/96 versus23/96 (+14.58 pp), OpenDrawer37/96 versus26/96 (+11.46), TurnOnElectricKettle66/96 versus56/96 (+10.42). Largest losses: CoffeeSetupMug38/96 versus52/96 (−14.58), SlideDishwasherRack43/96 versus51/96 (−8.33), TurnOnSinkFaucet39/96 versus46/96 (−7.29). These small task samples are descriptive; no multiplicity-adjusted win is asserted. Treating tasks as the sampling unit gives SE1.876 pp for the mean paired task difference, again leaving+2.267 pp below twoSE. Sources and individual n/SE are in the per-task table above.

**The LIBERO-plus advantage is primarily geometry/initial-state robustness.** Replicated shared-z versus causal gains14.67 pp in robot init,11.57 in object layout,6.72 in language,6.41 in camera, but loses2.51 in background and0.89 in sensor noise; lighting is+0.44. The last three gains/losses are small relative to their uncertainty and category multiplicity. W has the same geometry pattern, with sensor noise−1.20 pp. The shared-z seed43 headline is particularly fragile to seed selection: seed42 is1.8 pp lower. No full-protocol plain-LIBERO comparison establishes an in-distribution advantage.

**CoT’s82.6/82.2 are still not4,000-episode results.** Their surviving roots have1,153 episodes, while newer libero_10 suite files have1,000 each. The rest of the new full run has not been aggregated. For ours_v3, libero_10 alone changes209/256→833/1,000 (+1.659 pp); det_v3 changes208/256→787/1,000 (−2.550 pp). Never combine that new suite with old roots. The det_readout suite update is726/1,000. The ours_full649/1,000 suite file is historical (September12), not evidence that replacement1855180 landed; the scratch740/1,000 file is also from September12. Neither can be combined with new shards or treated as an overall score.

A descriptive estimate using **all25 runs** with a surviving four-suite small result and full result (one smallest-n old artifact per run; all32 transitions are tabulated) gives median full−small **−0.259 pp**, mean−0.055 pp,10th–90th empirical quantiles−2.784 to+1.940 pp, observed range−5.241 to+9.867 pp. This does not justify a blanket−3.6 pp haircut. Applied mechanically to the two CoT roots, the median projection is about82.3/81.9%, with an empirical sensitivity band roughly79.8–84.5/79.4–84.1%; this is **not a confidence interval or a benchmark result**. Different protocols and winner selection limit transportability. The claimed old-zonly0.812 small file no longer survives among its raw root artifacts, so that historical shrinkage cannot be verified from the current JSONs.

**No single raw success score collapses the observed shared-z gain**: seed42 still beats causal42 by3.975 pp if the seed43 result is discarded. The encoder-specific explanation is much less secure than those counts. Its most consequential missing result is a causal backbone with the same shared-z bottleneck, supervision and compute; if that reaches77–78%, the claim becomes a shared-latent training/readout contribution rather than an encoder-directionality win. Among existing files, the seed43 shared-z4k JSON is the key replication, audited as3,131/4,000 with matching suite totals, but independent policy/input contract verification remains essential. Confidence: high in the arithmetic, moderate in matched-protocol implementation, low in directionality as the identified cause.

## Discrepancies with ENCDEC_STATUS.md and docs/STATUS.md

| narrative | raw audit / correction |
|---|---|
| “best protocol-complete0.783,” seed42 pending | seed42 landed0.76475; report two-seed0.77375±0.01273 SD, not best-seed0.78275 |
| old shared-z0.776 treated as comparable to exact4k |0.77633165 has4,149 episodes; its exact4k sibling is0.766. Keep both; canonical exact4k by declared rule |
| “inside zonly backbone makes no difference” | point estimates are old0.766 versus v5 mean0.77375; no strong backbone conclusion from one old seed/two new seeds |
| status ranking leaves out objective/backbone siblings | W0.76425, MLM-control0.75525, W20.747, zsup0.78125, qwen08b-GR00T0.77175 and decoder-latent0.75025 all have4k artifacts |
| equal20k steps / launcher batch64 implies matched compute | shared-z v5 actual global batch32; causal/v5-PI nominal64, with different head dropout0.2/0.0/0.1. Equal steps are not equal sample or compute budget |
| qwen08b-GR00T or head comparison proves encoder advantage | different backbones/heads; no matched causal-GR00T control for the v5 head contrast |
| seed42 RoboCasa incomplete after15/34 retry units | union of surviving JSONs now has34/34 declared benchmark units, matching both17-task manifests; old failures are attempts, not final missing tasks |
| RoboCasa “0.363 vs0.373” | correct for seed4 (296/816 vs304/816), incomplete for current evidence: seed42 is284/816 vs313/816 |
| enc-dec hardly needs state, causal needs it totally | no-state6 cells reproduce, but suffix deletion confounds input information with prompt format; shuffled values preserve causal94/144 versus encoder87/144 |
| state-corruption follow-up not done |18/18 st_* files exist after1854695 fills5 missing units; zeros favor encoder, shuffle does not |
| more flow steps ruled out | small30k experiment only: existing files contain mixed24- and96-episode cells; these do not rule out another checkpoint/task dependence |
| training loss lower “at every bin” and quoted25k0.0387/0.0401 | re-binning raw logs uses explicit [25k,30k): seed42 v50.033593 vs causal0.034903; seed4 0.033526 vs0.034123. [45k,50k) seed4 0.021225 vs0.021804, seed42 0.021235 vs0.021688. Direction survives; exact quoted25k values are not this bin definition |
| paired open-loop cosine0.964 / gripper98% establishes “same actions” | these are diagnostic aggregate similarities, not equivalent policies; low-frequency decisive mistakes can change closed-loop success. Raw openloop JSONs are separate from rollout n |
| 17-task paired RC sweep takes40 min | successful seed4 log timestamps22:44:36→23:41:32 give56m56s; sacct allocation57m57s. The42m39s retry was partial work, not a complete fresh benchmark |
|929 rollouts/h/GPU ⇒ about3,700/h/node | individual-client throughput excludes launch/load costs. Complete1,632-rollout paired sweep delivered~1,690/h/node from start/end timestamps |
| CoT1850350–53 queued |1850350–52 running at audit;1850353 failed after8 exhausted shards; replacement1855180 running with multiple exhausted shards |
| shared-z+GR00T never combined | later status update is correct: training1851382 already running, evals1851383/84 dependent; no final result yet |
| q35 port/device blocker | historical repeated failures are real, but replacement1852305 is now RUNNING with dependent evals1852306/7; no completed benchmark proves resolution |
| docs/STATUS final dense still in flight, no continuation control needed | final dense and raw evaluations exist; full-epoch aggregate0.690442→0.671040. A no-head continuation remains needed to isolate auxiliary loss |
| dense is best / probes alone identify the strongest model | early aux-dense aggregate0.677429 recomputes; it is spatial/VQA, not policy success; final_action_head0.683575 beats final dense0.671040 among those continuations |
|0.6780 final_action_linear /0.6400 tracetime | recomputed0.677859 /0.639572; weight0.01 tracetime rerun now0.666865 and must remain a separate arm |
| all probe records durable and correctly indexed | collector overwrote generic starVLA patch-only identities, defaulting to Bridge. Fixed source identity, dataset inference and full measurement fingerprint; see PROBE_INSIGHTS.md |
| probe sweep missing two DROID-wrist cells | both now landed; the full-table launcher used wrist-only for33sim cells, previously mislabeled external; desired ext+wrist patch/token coverage is66/99, not99/99 |
| broader decodability means earlier action access or a useful proxy | no layer sweep exists; available policy-triplet readouts have Spearman−0.5; most11-readout policy cells are absent |

Existing bug descriptions about extra-camera indexing, RoboCasa target slices and ridge scaling are retained as historical constraints, not assumed to invalidate every corrected run. Mechanistic assertions about encoder-memory identity, upstream defaults and byte-identical refactors are code-validation claims, not measurable from success JSONs; this audit does not recertify them. Lower training loss is not benchmark evidence. Full loss-bin sources are in [training_loss_bins.json](results_collected/training_loss_bins.json).

## Raw hand checks and collector changes

Manually read these JSON count fields independently of collector output, then independently asserted **all1,954 row counts/rates** against raw payloads:

| source | direct arithmetic | result |
|---|---|---|
| `playground/Checkpoints/ervla_zonly_pi_sharedz_ground_temporal_v5_s43/results/libero-plus/overall_results.json` | (665+801+854+811)/4,000 |0.78275 |
| `playground/Checkpoints/ervla_zonly_pi_sharedz_ground_temporal_v5_s42/results/libero-plus/overall_results.json` |3,059/4,000 |0.76475 |
| `playground/Checkpoints/ervla_pi_causal_actiononly_pifix_nolatent_nodrop_2gpu_s42/results/libero-plus/overall_results.json` |2,900/4,000 |0.725 |
| `playground/Checkpoints/ervla_w_pi_encoder_actiononly/results/libero-plus-4k-exact-v1/overall_results.json` |3,057/4,000 |0.76425 |
| `playground/Checkpoints/ervla_pi_causal_actiononly_pifix_nolatent_nodrop_2gpu_s42/results/libero/overall_results.json` | plain top-level suites,381/400 |0.9525 |
| worktree `playground/Checkpoints/ervla_robocasa365_pi_causal_s4/checkpoints/steps_50000_pytorch_model.eval/robocasa_OpenStandMixerHead_seed42_envs24_eps48_h500_act8_native_s4_50k.json` |40 true values/48; payload seed42 is eval seed |0.8333333333; training seed4 |

`collect_results.py` already handled both LIBERO shapes. Fixed: training-seed attribution; RC task names from JSON env; full filename protocol in result_dir so legacy scaling/steps experiments cannot collapse into an empty tag; binary-outcome validation; task-manifest-based RC completion; nonempty failed_shards exclusion from full flags; shared-z control arm labeling. No stored success-rate disagreements were found. Changes affect grouping and eligibility, not underlying outcomes. Commands were rerun with filtered outputs directed to separate folders so the master remains unfiltered.

## External reference points — reported by authors, not computed from our files

Checked2026-09-17. No external paper provides per-episode outcomes here, so n and SE are **not available in the inspected summary tables** unless stated. Do not invent error bars or pool these with our JSONs. “Strongest verified here” is not an exhaustive certification of SOTA.

| benchmark | published/released method | reported average success | protocol relation and source |
|---|---|---|---|
| LIBERO | OpenVLA-OFT |97.1% | four suites; multi-image+state recipe, parallel chunks and continuous L1 prediction; [authors](https://openvla-oft.github.io/) |
| LIBERO | Discrete Diffusion VLA |96.3% | masked discrete actions, adaptive parallel decoding/remasking; [paper](https://arxiv.org/abs/2508.20072). This is an action-decoder method, not identical to our encoder+DiT |
| LIBERO | VLANeXt |97.4% | four-suite result,10k-step recipe study with batch256; [paper, Tables2–3](https://arxiv.org/html/2602.18532v1) |
| LIBERO | Cosmos Policy |98.5% | video-diffusion-derived control, much larger training budget; [authors’ ICLR2026 entry](https://moojink.com/) |
| LIBERO | InternVLA-A1.5 |98.9% | released LIBERO-tuned checkpoint; strongest plain-LIBERO result verified here; [model card](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero) |
| LIBERO-plus | π0 / π0-FAST |53.6% /61.6% | official zero-shot robustness comparison, not our PI-head causal control; [benchmark authors](https://github.com/sylvestf/LIBERO-plus) |
| LIBERO-plus | OpenVLA-OFT |69.6% | official robustness result; one trial per official perturbed task; [benchmark authors](https://github.com/sylvestf/LIBERO-plus) |
| LIBERO-plus | OpenVLA-OFT+ |79.6% | trained with LIBERO-plus augmentation, **not zero-shot**; [CVPR2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf) |
| LIBERO-plus | VLANeXt |80.1% | same LIBERO-trained checkpoints under unseen perturbations; [paper Table3](https://arxiv.org/html/2602.18532v1) |
| LIBERO-plus | SRPO |82.1% | official benchmark’s listed result; RL/post-training and data scope differ; [benchmark](https://github.com/sylvestf/LIBERO-plus), [SRPO paper](https://arxiv.org/abs/2511.15605) |
| LIBERO-plus | InternVLA-A1.5 |84.8% | released zero-shot benchmark result, same LIBERO-tuned checkpoint; strongest verified here; [model card](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero) |

Our shared-z mean77.375% is7.425 pp below the84.8% reference, but this is a **descriptive, protocol-unmatched difference**, not a statistical comparison. To claim a leaderboard win, run the released competitor and our frozen checkpoint through the same complete official task list, camera/state contract, action execution/horizon, checkpoint-selection rule and training-data declaration. The internal4k exact sampler must be published or replaced by the official task list. Plain n400 cannot certify a win over published97–99% results. Original LIBERO commonly evaluates50 episodes per each of40 tasks (2,000 total); the requested≥4,000 gate is a stricter internal rule, not the universal literature definition.

| RoboCasa365 official method | overall50 tasks | atomic-seen18 tasks | why our17-task score is not comparable |
|---|---|---|---|
| Xiaomi-Robotics-1 |57.4% |80.2% | stronger verified official leader, different pretraining and all50 target tasks |
| ABot-M0.6 |46.6% |79.4% | different training and evaluation scope |
| GR00T N1.5 |23.9% |50.7% | official re-evaluation uses RoboCasa1.0.1 and1.5× longer horizon |
| GR00T N1.6 |21.9% |51.1% | official policy, not just our GR00T-style action head |
| π0.5 |16.9% |39.6% | Human300 multi-task pretraining and official splits |
| π0 |14.8% |34.6% | same distinction;35.54/37.81% on our target-atomic training is not a win over official overall14.8% |

All six rows: [official leaderboard, updated2026-09-12](https://robocasa.ai/leaderboard.html); protocol/config details: [benchmark repository](https://github.com/robocasa-benchmark/leaderboard), [GR00T submission](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/gr00t_n1.5_2026_05_19.md). The official overall combines18 atomic-seen,16 composite-seen,16 composite-unseen tasks. A17-task horizon500 target-trained evaluation lacks one atomic task and all composite tasks. Reviewers need task lists, kitchen split/version, task-specific horizons, n/task, data budget, model size, full/LoRA training and seed variation. Our incomplete official scope cannot be repaired by comparing percentages alone.

Relevant design ideas: VLANeXt’s learned query buffer and frequency-domain action objective motivate small readout/frequency ablations; its state-to-VLM result cautions against assuming direct state-to-DiT injection will help. OFT motivates a deterministic parallel continuous-action baseline. Discrete Diffusion VLA motivates adaptive refinement only for a model trained for masked action reconstruction. InternVLA-A1.5’s latent foresight supervision is related to our shared-z temporal signal, but copying its large video teacher is post-deadline work. None of these reports establishes bidirectional-encoder superiority by itself.

## Operational evidence and job snapshot

No training or evaluation jobs were launched by this audit. Existing work is allowed to finish; no duplicate eval was submitted.

| job | observed status around14:00 CEST2026-09-17 | checkpoint/result evidence |
|---|---|---|
|1850294 | COMPLETED,1h28m06s | shared-z v5 seed42,20k checkpoint,3,059/4,000; fully landed |
|1836354 | COMPLETED,1h38m38s | shared-z v5 seed43,20k,3,131/4,000 |
|1850350 /1850351 /1850352 | RUNNING,~4h38m elapsed | CoT ours_v3/det_v3/det_readout; roots still small-n, new first suite landed; final_model checkpoints exist |
|1850353 | FAILED,2h36m19s | ours_full;8 exhausted first-suite shards, no new complete4k root |
|1855180 | RUNNING,~1h12m elapsed | replacement ours_full with16 shards; log already contains multiple exhausted shards; new result not landed |
|1851382 | RUNNING,~4h05m | shared-z+GR00T training42/43; existing job, no new training launched |
|1851383 /1851384 | PENDING dependency | dependent shared-z+GR00T evals; no4k root yet |
|1852305 | RUNNING,~3h29m | q35 pair replacement, previously absent from JOBS ledger; status is not proof of final success |
|1852306 /1852307 | PENDING dependency | q35 evals, no completed result yet |
|1850179 | COMPLETED,42m39s | partial retry attempt; union with1842872 now supplies seed42’s34/34 declared RC units |
|1851833 /1854695 | COMPLETED,2h18m06s /10m02s | corruption matrix plus5-unit repair; all18 JSONs now exist |
|1850693 | COMPLETED,2h50m08s | tracetime weight0.01; aggregate and6 pooled-probe dataset/camera records landed |
|1851073 /1855867 | COMPLETED / COMPLETED (25m01s) | both flagged DROID-wrist patch/token records landed;33sim wrist-only cells must not fill combined-camera slots |
|1855204 | RUNNING | existing final-action-linear weight0.1 work; separate from finished linear arm |

The complete read-only accounting snapshot is [jobs_accounting.txt](results_collected/jobs_accounting.txt). `scripts/jobs_status.sh` was read and run. CoT stdout/stderr inspected at `slurm_logs/eval_libero_plus_<jobid>.{out,err}`; raw snapshot precedes any later overwritten root. Failed-shard rows above list every nonempty sentinel. Scratch manifests/client logs were inspected; `failed_units_*` is absent, so explicit JSON-vs-manifest coverage is the completion criterion.

Healthy20k LIBERO-plus exact4k evals take1.3–1.7h on one4-GPU node (5.2–6.8GPU-h), not the CoT runtime. CoT is already>4.6h and only part-way through suites: budget8–12h plus retries,32–48GPU-h, with a hard checkpoint on shard completion. RC17-task paired sweep is57m fresh (about3.9GPU-h); a single policy gets roughly30–60m depending on scheduling. Per-task client logs show~3–5min sim time plus startup, not48 independent sequential GPU jobs. Allocate1.5–2h for a complete pair with retry margin. Existing LIBERO seed-pair training took7h31m on4GPUs; RC training required an initial12h allocation plus~7h38m resume for paired arms. Future cost estimates are planning ranges, not guaranteed runtimes.

Video inventory only: worktree `results/robocasa365_rollouts_30k/` has40 video files and `results/robocasa365_pnp_30k/` has96; neither directory exists in the live tree. Videos are not counted as additional outcomes. Open-loop diagnostic JSONs at `/e/scratch/m3/blank4/rc365_smoke/openloop_driver_{causal,v5}_1842595.json` show126 queries each, cosine0.963782/0.964206 and gripper agreement0.983383/0.987351,16 driving episodes each. They do not independently verify the17-task benchmark or prove policy equivalence.
