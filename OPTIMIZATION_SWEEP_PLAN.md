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

Do not reduce the claim-critical batch128/256 z rows to4×32 and8×32. The shared-z distribution regularizer gathers across ranks, but its separation loss searches same-language pairs only inside each rank's local batch. A32/device run therefore changes both hardware mapping and the supervision received by z. It also optimizes more slowly in wall time per update. If queue supply blocks batch256 at16/device, deprioritize batch256 before changing its local batch; a32/device run may be kept as a separately labelled performance/resource arm, not merged into the clean batch sweep.

Historical SLURM accounting supports that decision:

| local batch | representative jobs | global batch / GPUs per run |20k loop time | charged GPU-h/run | GPU-h per1M samples |
|---:|---|---|---|---:|---:|
|16|1851382, legacy shared-z|32 /2|5:50–5:52|11.84|18.49|
|32|1861658, legacy shared-z dropout1|64 /2|9:26–10:45|21.65|16.92|
|32|1861661, legacy shared-z dropout0.15|64 /2|9:21–10:41|21.49|16.79|
|32|1863852/53, corrected z-memory|64 /2|9:52–11:02|22.05–22.23|17.23–17.37|

Thus32/device is about6–9% cheaper per sample, but a20k-update run costs about1.8× as many GPU-hours because it sees twice as many samples, and steps take roughly1.6–1.9× longer. For equal global batch,4×32 or8×32 would use about half the GPUs but should be expected to land substantially later than8×16 or16×16. Compute is not the current constraint, so preserve16/device and minimize result latency and objective confounding.

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

The corrected implementation already places z at the front of the GR00T cross-memory: `[z_1, z_2, z_3, z_4, h_1, ..., h_L]` in `starVLA/model/modules/shared_z.py::_augment_shared_z_cross_memory`. These tokens are produced after the encoder has run. GR00T cross-attention has no causal mask over its key/value memory and applies no new position-dependent encoding to these projected z tokens, so moving them between the front and end would not make the path more bidirectional. Making z participate in the encoder's bidirectional self-attention would require a distinct architecture: learned latent queries in the encoder, or a second encoder pass after z is computed. Do not conflate that loopback arm with the corrected cross-memory sweep.

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
| Sep19 | Run all16/device intended-mapping20-update/in-training-eval smokes and100–200-update throughput probes. Confirm the logged memory order is four z tokens followed by encoder memory. Release healthy seed42 cells in parallel. Start execute4/8/16 screen. Deprioritize batch256 rather than silently remapping it to32/device if nodes are unavailable. | Smoke the four corrected GR00T RoboCasa rows at global64. Release one training seed only after the port and state-conditioning tests pass. |
| Sep20 | Exact4k completed20k checkpoints while training continues. Do not compare the new mid-schedule20k checkpoints as if they used the historical20k cosine endpoint. | Evaluate completed RoboCasa PI-v4 and corrected-GR00T checkpoints; reconcile failed-unit manifests. |
| Sep21 | Exact4k40k/60k arrivals; select provisional top cells and prepare seed43 confirmations without stopping healthy80k runs. | Decide whether the cross-benchmark encoder result is positive, neutral or negative. |
| Sep22 | Finish80k and exact4k. Select architecture×batch×step from seed42, then launch seed43 for the top two and claim-critical encoder/causal controls. | Freeze RoboCasa architecture and task protocol; second training seed only for promoted rows. |
| Sep23 | Complete seed43 confirmations and final execution-length result. | Freeze main claims, tables and figures. |
| Sep24 | Last safe headline-result arrival. | Recompute all tables from raw JSON and audit provenance. |
| Sep25–26 | No exploratory training; writing, reproducibility, anonymization and final rendering. | Appendix-only corrections. |
