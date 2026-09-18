# LIBERO optimization sweep — September 18, 2026

## Decision

Prioritize an optimization-budget sweep, but run it on one fixed architecture first: the historical shared-z + GR00T/readout winner. Its two exact-4k seeds score 78.675% and 78.275% (mean 78.475%) at global batch 32 and 20k updates. The matched legacy batch-64 recipe scores 75.600% and 78.150% (mean 76.875%) at 20k updates. This 1.600-point mean change is large enough to affect the paper, but it does not identify a generic batch effect: batch, local batch composition and action-example exposure all changed.

The existing batch-64 seed-42 checkpoint at 10k updates has now completed exact 4k at 2,776/4,000 = 69.400% (SE 0.729pp). It has the same 0.64M sampled examples as the batch-32/20k seed-42 winner, but is 9.275 points worse; its own batch-64/20k checkpoint is 75.600%, 6.200 points better. This makes the long batch-32 curve the primary performance run and batch 128 a boundary test. It does not isolate batch size because the 10k checkpoint is halfway through a 20k cosine schedule rather than the endpoint of a 10k schedule. Seed 43 is still pending and must be added from its raw JSON before using a two-seed mean.

The released upstream PI recipe is not a reason to copy batch 128 and 100k updates directly. It combines Qwen3-VL-4B, the historical 36-layer all-cross action model, full tuning, horizon 8, global batch 128 and 100k updates (12.8M action samples). It reports 77.0%, below our current two-seed mean despite 20 times as many sampled examples as our batch-32 winner. The useful lesson is that our learning curve is under-measured, not that 12.8M samples is known to be optimal.

## Clean first sweep

Use the same legacy shared-z + GR00T/readout architecture, data, augmentation, loss weights and learning rates in every cell. Parameterize warmup and cosine decay by processed action samples. Train every trajectory to 2.56M samples, with 5% warmup (128k samples), and save equal-sample checkpoints:

| global batch | preferred mapping | updates | warmup | checkpoints at 0.64M / 1.28M / 2.56M samples | first wave |
|---:|---|---:|---:|---|---|
| 32 | 2 GPUs × 16/device | 80k | 4k | 20k / 40k / 80k | seeds 42 and 43; two runs on one node |
| 64 | 4 GPUs × 16/device | 40k | 2k | 10k / 20k / 40k | seeds 42 and 43; one node per seed |
| 128 | 8 GPUs × 16/device | 20k | 1k | 5k / 10k / 20k | seed 42; two-node DDP boundary test |

Keeping 16 examples per GPU matters for this model. The distribution loss gathers z over the distributed batch, while the target-separation loss forms same-language pairs only inside each rank's local batch. The existing batch-64 control used 2 GPUs × 32/device and therefore changed the local pair distribution as well as global batch. The table above holds local composition fixed. If two-node DDP is unstable or more than 30% slower per sample, use 4 GPUs × 16/device with two-step gradient accumulation for the batch-128 boundary test and record that the z distribution loss still sees microbatches of 64.

Do not change learning rate in this sweep. The current VLM/action split of 1e-5/1e-4 is already a competitive StarVLA-style recipe, and adding LR would make the first matrix uninterpretable. Only test a square-root-scaled action-head LR after the batch-128 curve shows clear optimization underfitting at matched samples.

Every new configuration first runs 20 optimizer updates with ordinary in-training evaluation and writes a complete step-20 checkpoint. After that passes, use the first 100–200 production-style updates to measure throughput. This is a training/integration gate, not a simulator rollout smoke.

Run a deterministic 1,000-episode LIBERO-plus screen at each equal-sample checkpoint. Keep these rows explicitly partial. Run exact 4k on the best checkpoint from each batch trajectory and on any checkpoint within one point of the best screen. Replicate batch 128 at seed 43 only if seed 42 beats the best batch-32/64 screen by at least one point or remains within one point while being materially faster.

If success improves by at least one point from 1.28M to 2.56M samples, extend only the winning batch to 5.12M samples with the same sample-based schedule. Do not jump directly to 12.8M. Stop scaling when the exact-4k mean fails to improve by one point, either seed regresses by more than two points, or training loss improves while rollout success falls.

## Architecture order

1. **Recipe selection:** run the batch/sample sweep only on the completed shared-z + GR00T/readout winner. It has the highest behavior score and existing batch-32/64 evidence.
2. **Encoder claim:** apply the selected batch, sample budget and execution length to a plain full-memory bidirectional encoder + GR00T and a genuinely matched causal full-memory + GR00T control. Match language-stack trainability as well as data, initialization family, head and optimizer. This comparison, rather than PI-v4 versus an unlike causal head, tests directionality.
3. **Performance candidate:** combine shared z with QwenPI_v4 only after its information path is implemented and passes the 20-update gate. QwenPI_v4 action-only improved its matched v5 PI control by 2.538 points but is still 2.812 points below the shared-z winner. Do not run the full optimization grid again; use the selected recipe and two seeds.
4. **Causal confirmation:** run only the final selected recipe, not a causal hyperparameter sweep. Sweeping the causal baseline separately would give unequal tuning budgets and consume evaluation time without resolving the encoder mechanism.

## Chunk length

Raw policy-server metadata verifies that the shared-z winner predicts and executes 16 actions. First add a validated evaluation override and compare executing 4, 8 and 16 actions from the same horizon-16 checkpoint. This isolates replanning frequency and makes the 8-action comparison to upstream useful without retraining.

If execute-8 gains at least one point on the 1,000-episode screen, confirm it at exact 4k. Only after the optimization recipe is selected should one seed be retrained with prediction horizon 8 and execution length 8. Replicate horizon 8 only if it beats the horizon-16 model with execute-8 by at least one point. Changing prediction horizon inside the batch sweep would confound target length with optimization.

## Throughput and allocation

Measured production logs give about 1.05 s/update for the historical 2-GPU batch-32 run, or 30.5 action samples/s. The 2-GPU batch-64 controls take about 1.70–1.94 s/update, or 33–38 samples/s. Doubling the local batch therefore improved sample throughput by only roughly 10–25%, not 2×.

The batch-32 80k trajectory should take about 23–25 hours per run at historical throughput; two seeds fit on one 4-GPU node. A 4-GPU × 16 batch-64 trajectory should finish in roughly 12–15 hours if scaling is healthy, but this must be measured. An 8-GPU batch-128 trajectory should take roughly 6–9 hours if cross-node communication is healthy. Use additional nodes for independent seeds and configurations rather than enlarging local batch merely to occupy GPUs. The first wave requires five 4-GPU nodes: one for two batch-32 seeds, two for the batch-64 seeds, and two for the single batch-128 two-node run.

## Daily schedule

| date | work and gate |
|---|---|
| Sep 18 | Finish the running batch-64/10k exact-4k diagnostic. Correct the execute-length record. Prepare sample-based configs and launcher diffs; do not submit new training without an explicit launch instruction. |
| Sep 19 | Run 20-update + trainer-eval smokes for batch 32/64/128 mappings, then 100–200-update throughput probes. Launch the five-node first wave after the gates pass. Start the horizon-16 execute-4/8/16 rollout screen independently. |
| Sep 20 | Read 0.64M and 1.28M checkpoint screens while long trajectories continue. Kill only on integration failure or a predeclared severe regression; do not select from open-loop loss alone. |
| Sep 21 | Complete the 2.56M curves and exact-4k evaluations for the best checkpoints. Select batch, budget and execution length. Replicate batch 128 only if it meets its gate. |
| Sep 22 | Start the matched plain-encoder versus causal full-memory control at the selected recipe. Run one horizon-8 training seed only if execute-8 passed. Smoke QwenPI_v4 + shared z if its path is ready. |
| Sep 23 | Exact-4k the matched directionality controls and promoted horizon/PI-v4 candidate. Release second seeds only for candidates within one point of the best recipe or needed for the headline comparison. |
| Sep 24 | Last safe arrival for headline results. Freeze the main tables, recipe and claims. Later results go to the appendix unless they repair a correctness problem. |
| Sep 25–26 | Recompute tables from raw files, finish writing, audit attribution/protocols and render the submission. No exploratory training. |
