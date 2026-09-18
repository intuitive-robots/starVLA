# LIBERO optimization and architecture sweep — September 18, 2026

## Anchor and decision

The best eligible zero-shot LIBERO-plus model is the legacy shared-z+GR00T encoder:78.675% and78.275% over exact4k seeds42/43, mean78.475%±0.283pp seed SD. The83.075% causal CoT model is numerically higher but trained on LIBERO-plus and is excluded from the zero-shot comparison.

Do not train batch16 and do not spend new runs on the legacy z pathway. Keep the78.475% result as a fixed performance anchor. The new sweep uses corrected, interpretable memory paths and changes batch and architecture in parallel. Start seed42 for every cell; confirm only the best cells and claim-critical controls with seed43.

## Training schedule

Each run trains for80k updates with checkpoints at20k,40k,60k and80k. Use8k warmup and one80k cosine schedule. These checkpoints form a practical learning trajectory; they are not equivalent to independently scheduled20k/40k endpoints. The completed batch32/20k result remains the properly decayed short-schedule anchor.

| global batch | mapping | samples at20k / 40k / 60k / 80k |
|---:|---|---|
|32|2 GPUs ×16/device|0.64M /1.28M /1.92M /2.56M|
|64|4 GPUs ×16/device|1.28M /2.56M /3.84M /5.12M|
|128|8 GPUs ×16/device|2.56M /5.12M /7.68M /10.24M|
|256|16 GPUs ×16/device|5.12M /10.24M /15.36M /20.48M|

Keeping16 examples/GPU preserves the shared-z local pair distribution for batches32–256. Batch64 must therefore use4 GPUs rather than the historical2×32 mapping. Batch128/256 use two/four nodes per run. If multi-node throughput loses more than30% against the one-node projection, test gradient accumulation separately; do not silently treat a microbatch-64 z regularizer as identical to a true batch128/256 regularizer.

Hold the current LR split fixed for the first grid: VLM interface1e-5, action/shared-z1e-4. If batch128/256 clearly underfit at20k while smaller batches do not, branch a square-root-scaled action-head LR from the same initialization. Do not scale the trainable VLM LR automatically.

Every cell first runs20 optimizer updates, ordinary in-training evaluation and a complete step20 checkpoint on its intended GPU mapping. Use the next100–200 production-style updates to measure throughput. This is not a simulator rollout smoke.

Run exact4k at20k/40k/60k/80k for seed42. Selection across many checkpoints is exploratory; the chosen architecture×batch×step and the clean encoder/causal comparison require seed43 confirmation on the same exact task list.

## Architecture rows

Run four core GR00T rows at every batch. They use the same12-layer alternating cross/self action head, full final encoder memory representation where applicable, horizon16, data, augmentation and optimizer.

| row | action cross-memory | purpose |
|---|---|---|
| corrected encoder z-only | four always-visible projected z tokens; ordinary encoder memory masked (`memory_dropout_rate:1`) | clean shared-latent bottleneck |
| corrected encoder z+memory | four always-visible z tokens; full encoder memory retained on85% of training rows and all evaluation rows (`memory_dropout_rate:0.15`) | test whether detailed encoder memory complements z |
| encoder full-memory/no-z | full bidirectional encoder sequence, no z, no learned readout compressor | normal encoder control with the identical GR00T head |
| causal full-memory/no-z | full causal sequence, no z, identical GR00T head | causal control |

The causal row must match language-stack trainability. The completed causal control froze its causal language stack while the encoder's action-producing stack was trainable, so it is not sufficient for this matrix. Record that causal and encoder pretraining remain different unless a same-weight mask-only control is implemented.

Do not use PI-v4 instead of the GR00T controls: that would change the action head and destroy the directionality comparison. Add QwenPI-v4 action-only as a fifth performance row across the four batches if capacity is available. It is scientifically separate: QwenPI-v4 currently rejects `shared_z.enabled`, so it cannot represent the corrected-z rows until that path is implemented. Existing PI-v4 exact4k at batch64/20k is76.125%/75.200%, mean75.663%.

No new legacy-z rows are needed. Its batch32/20k two-seed result remains the score to beat.

## Resources and gates

The four-row core matrix has16 seed42 training runs. At16 examples/GPU it occupies30 four-GPU nodes if fully concurrent: two nodes for four batch32 runs, four for batch64, eight for batch128 and16 for batch256. PI-v4 adds approximately eight node equivalents. Queue supply, rather than GPU-hour budget, may limit concurrency.

At historical speed,80k batch32 is roughly23–25h. The larger distributed runs have the same update count and should be budgeted24–30h until the100–200-update profiles land. Use short resumable allocations with checkpoint validation instead of requesting speculative multi-day wall times.

Kill a cell for non-finite loss, missing gradients/information path, repeated resume failure, or throughput below70% of its nearest mapping. Do not kill from open-loop loss alone. Promote seed43 for the best two performance cells and all four rows at the selected batch/step needed for the paper's architecture comparison.

## Chunk length

The78.475% winner predicts and executes16 actions. Evaluate execute4/8/16 on that same checkpoint first. Horizon8 training remains one separate batch32 row after the inference screen; do not multiply the full batch×architecture matrix by horizon.

## RoboCasa parallel track

The existing QwenPI-v4 RoboCasa training job1858851 completed in9:00:18; its seed4/42 evaluation jobs1858852/53 are pending. Do not duplicate it.

The RoboCasa worktree has the older shared-z dropout path but does not yet contain `cross_memory_tokens` or `_augment_shared_z_cross_memory`. Port the verified corrected-z implementation, add an explicit mask/information-flow test, and run20-update in-training-eval smokes before any full job. After that gate, run one training seed at the native global batch64/50k recipe for the same four GR00T rows above. Chain the declared17-task×48-episode evaluation and promote seed4 only after valid seed42 results. This work is parallel and must not reuse the cancelled flawed-z jobs1863688/90/92.

## Daily schedule

| date | LIBERO-plus | parallel RoboCasa and paper work |
|---|---|---|
| Sep18 | Generate the four-batch × four-core-architecture seed42 configs plus optional PI-v4 row. Prepare resumable20/40/60/80 checkpoints and exact4k chains. No training submission without an explicit launch instruction. | Let PI-v4 eval jobs1858852/53 run. Port corrected z-memory tokens into the RoboCasa worktree and add the information-flow test. |
| Sep19 | Run all intended-mapping20-update/in-training-eval smokes and100–200-update throughput probes. Release healthy seed42 cells in parallel. Start execute4/8/16 screen. | Smoke the four corrected GR00T RoboCasa rows at global64. Release one training seed only after the port and state-conditioning tests pass. |
| Sep20 | Exact4k completed20k checkpoints while training continues. Do not compare the new mid-schedule20k checkpoints as if they used the historical20k cosine endpoint. | Evaluate completed RoboCasa PI-v4 and corrected-GR00T checkpoints; reconcile failed-unit manifests. |
| Sep21 | Exact4k40k/60k arrivals; select provisional top cells and prepare seed43 confirmations without stopping healthy80k runs. | Decide whether the cross-benchmark encoder result is positive, neutral or negative. |
| Sep22 | Finish80k and exact4k. Select architecture×batch×step from seed42, then launch seed43 for the top two and claim-critical encoder/causal controls. | Freeze RoboCasa architecture and task protocol; second training seed only for promoted rows. |
| Sep23 | Complete seed43 confirmations and final execution-length result. | Freeze main claims, tables and figures. |
| Sep24 | Last safe headline-result arrival. | Recompute all tables from raw JSON and audit provenance. |
| Sep25–26 | No exploratory training; writing, reproducibility, anonymization and final rendering. | Appendix-only corrections. |
