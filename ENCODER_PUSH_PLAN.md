# Encoder push plan — 2026-09-17

**Prioritize an identifiable shared-latent encoder contribution, then improve geometry robustness and closed-loop readout.** Do not optimize the probe score as the objective. We have a 5.72 pp shared-z gain over the internal causal baseline, but directionality alone is not isolated, RoboCasa’s2.27 pp pooled gap is below two episode SE, and the strongest verified external LIBERO-plus reference is 84.8% versus our 77.375% two-seed mean.

All proposed effects below are hypotheses, in absolute percentage points relative to the relevant matched control, not measured results or additive gains. Costs use GH200 GPU-hours and include validation evals where stated. Healthy 4k LIBERO-plus eval:5–7 GPU-h, 1.3–1.7 h on 4 GPUs. One LIBERO training seed: roughly 15 GPU-h; a two-seed pair used 30 GPU-h over 7.5 h. RoboCasa from-scratch causal+encoder pair at a training seed: approximately80 GPU-h/20 h on 4 GPUs, including resume; a 10k-step continuation pair is estimated16–24 GPU-h/4–6 h. These are extrapolations from logs, not reservations. Queue time and rendering failures are additional. Every Booster allocation must use all 4 GPUs, via independent jobs/shards as appropriate.

Use a fixed development task set for screens, then evaluate the selected recipe once on frozen4k/official task lists and all seeds. Report all confirmations, including failures. A “kill” below stops that intervention after the specified evidence; it never turns missing outcomes into zeros. No jobs were launched by this audit.

## Ranked experiments

### 1. Freeze protocols and finish matched evaluation — must have for the paper

**Hypothesis/evidence:** some apparent gains are sample/protocol artifacts: old-zonly0.77633/n 4,149 differs from its exact 0.766/n 4,000; CoT roots are still 1,153 episodes. **Change:** publish task IDs, category counts, cameras/state contract, checkpoint fingerprints, inference settings and environment versions; finish existing CoT jobs and their failed shards without merging old roots. Evaluate final shared-z and causal on plain LIBERO at both the standard 2,000-trial protocol and a predeclared 4,000-trial confirmation; supply all seeds. Complete missing RoboCasa atomic task on all four existing checkpoints; use official horizon/split for the external comparison.

**Expected effect:** no promised score gain; removes false wins. **Budget:**60–100 GPU-h for6–10 healthy 4k policy evals, plus existing CoT 32–48 GPU-h/run and retry reserve;1–2 days with 4–6 nodes. **Kill:** stop claiming a benchmark win if task sets/settings cannot be matched or the gain disappears; keep those results as explicitly partial evidence.

### 2. Encoder on/off × shared-z on/off — must have for the paper

**Hypothesis/evidence:** the shared-latent pathway, not generic bidirectionality, creates most of the gain. Old-bidir−causal is +0.43 pp; zsup−zbase is +6.575 pp. **Change:** four cells with the same base initialization/data and pretraining budget, action head, state format, global batch, optimizer, 20k steps and evaluation protocol. The causal+shared-z cell gets the identical targets, temporal supervision, query count and latent width. Compare direction masks using equally pretrained backbones; if this cannot be implemented fairly by day 3, state that pretraining and directionality are bundled rather than calling it a one-factor test.

**Expected effect:** encoder-specific interaction could be0–5 pp; it may collapse to zero. **Budget:**160–185 GPU-h for8 training runs (2 seeds/cell) plus 8 full evals;7.5 h training +2 h eval on 4 parallel nodes, allow2 days. **Kill:** no encoder attribution if causal+shared-z is within 1 pp and within 2combined SE of encoder+shared-z. This is the decisive missing result.

### 3. Replicate shared-z supervision/dropout ablations — must have for the paper

**Hypothesis/evidence:** aligned supervisory targets and blocking the expert’s visual-memory bypass matter. Existing one-factor zsup−shuffled is +2.50 pp; dropout 0.15−0 is +3.25 pp; dropout 1 is 1.525 pp below0.15. **Change:** first replicate zsup, zshuf and znodrop at seed 43. Then disable all z auxiliary losses while keeping the z-only path, and separately turn temporal supervision off while preserving grounding targets; use seeds 42/43. Keep dim 128/4 queries fixed until those results land.

**Expected effect:** best partial dropout could add0–2 pp over zonly; alignment effects may shrink with seeds. **Budget:**140–165 GPU-h for7new train+eval runs;1–2 days on 3–4 nodes. **Kill:** if a component changes success<1 pp in both seeds or reverses without a clear failure category, omit that component from the causal story. Do not call zonly an unsupervised control.

### 4. Test the actual encoder-objective switch — must have for the paper

**Hypothesis/evidence:** masked slots/readout rather than MLM loss may explain the “MLM control” result; its loss_enabled is false. **Change:** same checkpoint, prompt slots, mapping data and memory path, encoder_mlm.loss_enabled off/on at fixed low weight, plus the no-slot control; log masked-target coverage. Reuse only controls with identical configs. Two seeds, matched compute.

**Expected effect:**−2 to +2 pp; no current isolated positive evidence. **Budget:**85–130 GPU-h for4–6 train+eval runs if a slot control can be reused;1–2 days. **Kill:** discard the objective claim if added loss does not beat slot-only in both seeds or harms plain LIBERO by>1 pp. Keep objective/architecture terminology precise in the paper.

### 5. Finish shared-z+GR00T; add a causal+GR00T control — must have for the paper

**Hypothesis/evidence:** readout quality limits use of the encoder representation. V5-GR00T exceeds v5-PI by 1.50 pp across two seeds, while much higher probe R² often produces little rollout gain. **Change:** existing1851382 and dependent 1851383/84 already cover the encoder shared-z-GR00T arm; add equally trained causal-GR00T and causal-shared-z-GR00T if needed for an interaction claim. Do not count a different head as evidence for directionality.

**Expected effect:**0–2 pp over shared-z-PI; the two effects may not combine. **Budget:**existing pair~30 GPU-h +10–14evalGPU-h; two extra causal seeds 40–45 GPU-h, ~10 h/node. **Kill:** if mean gain<1 pp or one seed regresses>2 pp, keep PI as the main recipe. If causal benefits equally, report a head effect.

### 6. Geometry-targeted camera fusion, with background/noise guardrails — would strengthen the paper

**Hypothesis/evidence:** shared-z’s strongest gains are robot init/layout, but camera success is only 58.16%; background −2.51 pp and sensor noise −0.89 pp remain weaknesses. DROID ext+wrist dilution appears in all five pooled readouts, without a state-baseline/sample-count change. **Change:** per-camera query pooling/normalization, single-camera dropout 0.1, and separate moderate viewpoint/crop versus photometric/noise augmentation arms. Match augmentation for causal too. Do not simply increase all augmentation: existing shared-z augmentation averages76.325%, below v5 shared-z 77.375% and confounded by backbone differences.

**Expected effect:**+1–3 overall, +3–6camera pp if fusion is limiting; could hurt in-distribution precision. **Budget:**80–130 GPU-h for4–6 train+eval runs;1–2 days. **Kill:** no>2 pp camera gain on development tasks, or>1 pp loss in plain LIBERO/background on confirmation. Keep only one recipe for final seeds.

### 7. Format-preserving state training and vision controls on RoboCasa — must have for the paper

**Hypothesis/evidence:** causal 0/144 without state but 94/144 with shuffled state implicates prompt structure; encoder’s robust advantage is narrower than the no-state table suggests. **Change:** delimiters retained with missing-value markers; real/shuffle/random/zero matched across arms, seeds 4/42, and task sets selected before evaluation. Add encoder and causal continuations with state_dropout_rate0.1 and 0.3; maintain prompt structure at dropout. Include vision-zero/one-camera controls to distinguish state reliance from visual policy competence.

**Expected effect:** format repair may improve causal more; state dropout may give0–3 pp encoder clean success and 5–15 pp corrupted-state robustness. **Budget:**20–35 GPU-h for the expanded diagnostic matrix plus 40–70 GPU-h for paired continuation screens and two-seed confirmation;1–2 days. **Kill:** reject “encoder is less proprioception-dependent” if it does not persist in≥2 corruption types and both seeds; reject a training variant if clean success drops>1 pp. Direct continuous state into DiT is a separate later ablation, not an assumed fix.

### 8. Replan more often; limited inference refinement — would strengthen the paper

**Hypothesis/evidence:** the encoder fits actions better yet may accumulate closed-loop errors; n_action_steps8 executes half the 16-action prediction. **Change:** execute 4/8/16 predicted actions with fixed trained horizon and 4 flow steps, on the same episodes. Evaluate8 flow steps only for the best execution length. For mask-trained arms, average2–4mask-conditioned continuous predictions as a separate latency-costed ablation; do not average incompatible discrete gripper modes blindly.

**Expected effect:**−2 to +2 pp, especially recovery/long tasks. **Budget:**30–55 GPU-h for screens and paired 4k confirmations;8–16 h over 2 nodes. **Kill:**<1 pp confirmed gain or>2×latency without meaningful robustness improvement. Existing30k/96-episode flow-step results do not justify sweeping many integration counts. Keep train-time horizon unchanged; new horizons need retraining.

### 9. Small auxiliary readout and masking-ratio screen — would strengthen the paper

**Hypothesis/evidence:** action co-training spreads decodability, but final dense and tracetime hurt the spatial/VQA aggregate. No evidence localizes an earlier useful layer. **Change:** first probe layers 4/8/14/20/28 on matched policies, then at most one low-weight auxiliary action head (0.001/0.01) at a selected layer or learned text+vision query. Compare action masking ratios 0/0.15/0.3, with an equal-duration no-head continuation and fixed target handling. Do not train all combinations.

**Expected effect:**−2 to +1.5 pp; lower priority than shared-z because probe-to-success ordering is currently negative. **Budget:**4–8 GPU-h measurement plus 45–80 GPU-h for short continuations and two confirmations;1day if launched by day 4. **Kill:** probe improves without≥1 pp rollout gain, or spatial/VQA retention drops>1 pp. The 11-readout ranking cannot authorize this experiment on its own.

### 10. Frequency-domain loss on the action chunk — would strengthen the paper

**Hypothesis/evidence:** temporal supervision may improve usable trajectories rather than generic feature quality; [VLANeXt](https://arxiv.org/html/2602.18532v1) reports a successful frequency-domain action objective. **Change:** small DCT loss on predicted clean16-step action chunks, separate motion/gripper scaling, retaining the existing flow objective; one-seed continuation screen then2 seeds for a winner. Check whether current temporal-z loss already captures the benefit.

**Expected effect:**0–2 pp, especially motion consistency; untested here. **Budget:**35–65 GPU-h;1day. **Kill:** no confirmed success gain, worse gripper switching, or extra training cost without recovery benefit. This is a literature-inspired intervention, not an explanation already proved by our probes.

### 11. Official competitors and third behavior benchmark — must have for a broad/SOTA claim

**Hypothesis/evidence:** outperforming causal+PI does not establish the paper’s broad thesis; strongest verified released references are substantially stronger. **Change:** evaluate released OFT and InternVLA-A1.5 alongside our selected checkpoints on matched official LIBERO/plus protocols; state explicitly which training distributions differ. RoboCasa: include all 18 atomic tasks with official horizons; a 50-task SOTA claim additionally needs the prescribed training/splits. For a third behavior domain, prefer an already supported SimplerEnv task family with existing action data; audit embodiment/action compatibility before spending compute. Spatial/VQA scores and DROID probes do not count.

**Expected effect:**information, not guaranteed gains. **Budget:**40–100 GPU-h evals; third-domain training80–160 GPU-h as a rough unmeasured allowance, 2–4 days; official 50-task foundation-policy training is not credibly costed from current logs. **Kill:** if third-domain integration is not producing valid smoke rollouts by day 3, narrow the claim; never present an unvalidated adapter score as a benchmark. No unfamiliar full training campaign after day 5.

### 12. Larger latent/teacher or discrete masked-action architecture — post-deadline

**Hypothesis/evidence:** [Discrete Diffusion VLA](https://arxiv.org/abs/2508.20072) uses adaptive remasking; [InternVLA-A1.5](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero) supervises latent foresight with a video teacher. **Change:** z width/query-count scaling, richer future targets, teacher distillation, or a properly trained masked discrete action decoder with confidence-based refinement. This is not an inference flag on today’s continuous shared-z model.

**Expected effect:**unknown; could expand the attainable policy family. **Budget:**≥200–500 GPU-h plus integration, 3–7 days minimum and uncertain. **Kill:** no faithful small-scale reproduction or matched-compute gain. Do not destabilize the paper’s main model with a new architecture in the final week.

## Resource order and paper decision

Reserve the first capacity for ranks1–5 and 7. Run one cheap inference screen alongside them. Ranks 6/9/10 compete for the same discretionary budget; pick at most two after the first results. The minimal audit/ablation package is roughly 400–650additionalGPU-h depending on reusable controls, excluding existing jobs, expanded official external evaluation and a new third-domain training campaign. At 4 fully utilized nodes (16 GPUs), that is 25–41 h of ideal occupancy spread over 3–5calendar days; queue/failure allowance makes it6 days. With only one node, the broad deadline claim is not credible.

Predeclare the main claim by day 4, select the final recipe by day 5, and freeze all headline results by day 8 (September 24). A result that arrives later may enter an appendix if independently audited, but must not rewrite the thesis. If causal+shared-z closes the gap, pivot the claim to the shared-latent architecture and clearly name the encoder/pretraining bundle; if RoboCasa remains neutral, a “wins everywhere” encoder claim is unsupported.
