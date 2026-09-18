# LIBERO optimization sweep — September 18, 2026

## Decision

Prioritize an optimization-budget sweep, but run it on one fixed architecture first: the historical shared-z + GR00T/readout winner. Its two exact-4k seeds score 78.675% and 78.275% (mean 78.475%) at global batch 32 and 20k updates. The matched legacy batch-64 recipe scores 75.600% and 78.150% (mean 76.875%) at 20k updates. This 1.600-point mean change is large enough to affect the paper, but it does not identify a generic batch effect: batch, local batch composition and action-example exposure all changed.

The existing batch-64 checkpoints at 10k updates have now completed exact 4k at 2,776/4,000 = 69.400% (seed 42, SE 0.729pp) and 2,797/4,000 = 69.925% (seed 43, SE 0.725pp), mean 69.663% and seed SD 0.371pp. Their mean is 8.812 points below the batch-32/20k mean at the same 0.64M sampled examples and 7.212 points below their own batch-64/20k endpoints. This makes the long batch-32 curve the primary performance run and motivates adding batch16, where the same sample budget supplies still more optimizer updates. It does not isolate batch size because each 10k checkpoint is halfway through a 20k cosine schedule rather than the endpoint of a 10k schedule.

The released upstream PI recipe is not a reason to copy batch 128 and 100k updates directly. It combines Qwen3-VL-4B, the historical 36-layer all-cross action model, full tuning, horizon 8, global batch 128 and 100k updates (12.8M action samples). It reports 77.0%, below our current two-seed mean despite 20 times as many sampled examples as our batch-32 winner. The useful lesson is that our learning curve is under-measured, not that 12.8M samples is known to be optimal.

## Clean first sweep

Use the same legacy shared-z + GR00T/readout architecture, data, augmentation, loss weights and learning rates in every cell. Each scored endpoint gets its own complete cosine schedule and10% warmup. An intermediate checkpoint from an80k schedule is not a substitute for a separately scheduled20k or40k run.

Launch the most promising cells immediately:

| global batch | mapping | action samples | updates | warmup | seeds | purpose |
|---:|---|---:|---:|---:|---|---|
| 16 | 2 GPUs × 8/device | 0.64M | 40k | 4k | 42,43 | small-batch endpoint |
| 16 | 2 GPUs × 8/device | 1.28M | 80k | 8k | 42,43 | longer small-batch candidate |
| 16 | 2 GPUs × 8/device | 2.56M | 160k | 16k | 42,43 | maximum deadline budget |
| 32 | 2 GPUs × 16/device | 0.64M | 20k | 2k | reuse completed42,43 | current78.475% anchor |
| 32 | 2 GPUs × 16/device | 1.28M | 40k | 4k | 42,43 | primary longer-training candidate |
| 32 | 2 GPUs × 16/device | 2.56M | 80k | 8k | 42,43 | primary maximum-budget candidate |
| 64 | 4 GPUs × 16/device | 0.64M | 10k | 1k | 42,43 | properly decayed large-batch endpoint |
| 128 | 8 GPUs × 16/device | 0.64M | 5k | 500 | 42,43 | properly decayed upstream-scale endpoint |

The old batch64/10k69.663% pair cannot fill the fresh10k cell because it is halfway through a20k schedule. Promote batch64 or128 to separately scheduled1.28M and2.56M endpoints only if its two-seed0.64M mean is within1pp of batch32 or improves both seeds over the completed batch64/20k mean76.875%. This successive-halving rule preserves calendar time and evaluation attention even when GPU supply is ample.

Keeping 16 examples per GPU matters for this model. The distribution loss gathers z over the distributed batch, while the target-separation loss forms same-language pairs only inside each rank's local batch. The existing batch-64 control used 2 GPUs × 32/device and therefore changed the local pair distribution as well as global batch. The table holds16/device for batches32–128; batch16 necessarily uses8/device and is a performance candidate rather than a pure global-batch control. If two-node DDP is unstable or more than30% slower per sample, use4 GPUs ×16/device with two-step gradient accumulation for batch128 and record that the z distribution loss still sees microbatches of64.

Do not change learning rate in this sweep. The current VLM/action split of 1e-5/1e-4 is already a competitive StarVLA-style recipe, and adding LR would make the first matrix uninterpretable. Only test a square-root-scaled action-head LR after the batch-128 curve shows clear optimization underfitting at matched samples.

Every new configuration first runs 20 optimizer updates with ordinary in-training evaluation and writes a complete step-20 checkpoint. After that passes, use the first 100–200 production-style updates to measure throughput. This is a training/integration gate, not a simulator rollout smoke.

Run exact4k for every independently scheduled endpoint above; the score differences of interest are too small for a1,000-episode selector. Intermediate checkpoints may receive deterministic1,000-episode screens but remain explicitly partial. Both seeds start in every listed cell; do not select a batch or step budget from one seed.

If success improves by at least one point from1.28M to2.56M samples, extend only the winning batch to a separately scheduled5.12M endpoint. Do not jump directly to12.8M. Stop scaling when the exact4k mean fails to improve by one point, either seed regresses by more than two points, or training loss improves while rollout success falls.

## Architecture order

1. **Recipe selection:** run the batch/sample sweep only on the completed shared-z + GR00T/readout winner. It has the highest behavior score and existing batch-32/64 evidence.
   The sweep therefore uses the legacy AdaLN/self-attention z path, not the four-token corrected z-memory path. At matched batch64/20k, legacy dropout1 scores75.600%/78.150% (mean76.875%) and corrected z-memory dropout1 scores76.500%/75.950% (mean76.225%), a paired change of+0.900/−2.200pp and mean−0.650pp. The retained-memory variants are76.788% legacy versus75.825% corrected, mean−0.963pp, although that comparison also changes readout memory to the full hidden sequence. Do not run the full optimization grid on the lower-scoring corrected path.
   Run one bounded architectural diagnostic in parallel: corrected z-memory at global batch32,20k updates,2k warmup, seeds42/43. This matches the historical winner's batch and schedule and tests whether the corrected path was specifically hurt by batch64. Promote it to the long curve only if its two-seed mean is within1pp of legacy and neither seed regresses by more than2pp.
2. **Encoder claim:** apply the selected batch, sample budget and execution length to a plain full-memory bidirectional encoder + GR00T and a genuinely matched causal full-memory + GR00T control. Match language-stack trainability as well as data, initialization family, head and optimizer. This comparison, rather than PI-v4 versus an unlike causal head, tests directionality.
3. **Performance candidate:** combine shared z with QwenPI_v4 only after its information path is implemented and passes the 20-update gate. QwenPI_v4 action-only improved its matched v5 PI control by 2.538 points but is still 2.812 points below the shared-z winner. Do not run the full optimization grid again; use the selected recipe and two seeds.
4. **Causal confirmation:** run only the final selected recipe, not a causal hyperparameter sweep. Sweeping the causal baseline separately would give unequal tuning budgets and consume evaluation time without resolving the encoder mechanism.

## Chunk length

Raw policy-server metadata verifies that the shared-z winner predicts and executes 16 actions. First add a validated evaluation override and compare executing 4, 8 and 16 actions from the same horizon-16 checkpoint. This isolates replanning frequency and makes the 8-action comparison to upstream useful without retraining.

If execute-8 gains at least one point on the 1,000-episode screen, confirm it at exact 4k. Only after the optimization recipe is selected should one seed be retrained with prediction horizon 8 and execution length 8. Replicate horizon 8 only if it beats the horizon-16 model with execute-8 by at least one point. Changing prediction horizon inside the batch sweep would confound target length with optimization.

## Throughput and allocation

Measured production logs give about 1.05 s/update for the historical 2-GPU batch-32 run, or 30.5 action samples/s. The 2-GPU batch-64 controls take about 1.70–1.94 s/update, or 33–38 samples/s. Doubling the local batch therefore improved sample throughput by only roughly 10–25%, not 2×.

The batch16 160k trajectory is expected to take roughly30–45 hours and must use short resumable allocations; measure before projecting. Batch32 40k/80k should take roughly12/23–25 hours per run at historical throughput; two seeds of one endpoint fit on one4-GPU node. Fresh batch64/10k should take roughly3–4 hours on4 GPUs. Batch128/5k should take roughly2–3 hours on8 GPUs if cross-node communication is healthy. Use additional nodes for independent cells rather than enlarging local batch merely to occupy GPUs. The immediate scheduled grid uses eleven4-GPU nodes: three for the six batch16 runs, two for the four new batch32 runs, two for batch64 and four for batch128. The corrected-z pair and horizon screen use separate capacity.

## Daily schedule

| date | work and gate |
|---|---|
| Sep 18 | Finish the running batch-64/10k exact-4k diagnostic. Correct the execute-length record. Prepare sample-based configs and launcher diffs; do not submit new training without an explicit launch instruction. |
| Sep 19 | Run20-update + trainer-eval smokes for batch16/32/64/128 mappings, then100–200-update throughput probes. Launch the eleven-node independently scheduled grid: batch16 at0.64/1.28/2.56M, batch32 at1.28/2.56M, and batch64/128 at0.64M, all with seeds42/43. In parallel, smoke and launch the bounded corrected-z batch32/20k pair. Start the horizon16 execute4/8/16 rollout screen independently. |
| Sep 20 | Exact4k the completed independently scheduled endpoints while long batch16/32 trajectories continue. Promote batch64/128 to1.28/2.56M only if their two-seed0.64M endpoints meet the predeclared gate. Do not select from open-loop loss alone. |
| Sep 21 | Complete the batch16/32 maximum-budget curves and promoted large-batch endpoints. Select batch, budget and execution length from exact4k two-seed means. |
| Sep 22 | Start the matched plain-encoder versus causal full-memory control at the selected recipe. Run one horizon-8 training seed only if execute-8 passed. Smoke QwenPI_v4 + shared z if its path is ready. |
| Sep 23 | Exact-4k the matched directionality controls and promoted horizon/PI-v4 candidate. Release second seeds only for candidates within one point of the best recipe or needed for the headline comparison. |
| Sep 24 | Last safe arrival for headline results. Freeze the main tables, recipe and claims. Later results go to the appendix unless they repair a correctness problem. |
| Sep 25–26 | Recompute tables from raw files, finish writing, audit attribution/protocols and render the submission. No exploratory training. |
