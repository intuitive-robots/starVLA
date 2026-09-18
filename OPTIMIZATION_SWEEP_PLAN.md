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

## Executed architecture rows

The submitted grid is deliberately sparse. It avoids rerunning completed batch64 controls and spends the large-batch cells on the two most informative encoder variants.

| global batch | fresh seed42 rows | GPUs/run | initial job |
|---:|---|---:|---|
|32|token-only z+full memory; token-only z+camera dropout; encoder full-memory/no-z; causal full-memory/no-z|2 (two runs/node)|1877720,1877721|
|64|token-only z+full memory; token-only z+camera dropout|4|1877722,1877723|
|128|token-only z+camera dropout; encoder full-memory/no-z|8|1877724,1877725|
|256|token-only z+camera dropout; encoder full-memory/no-z|16|1877726,1877727|

“Token-only z” means four z-derived prefix tokens are concatenated before ordinary encoder memory for GR00T cross-attention, while z-AdaLN is disabled. This removes the previous double conditioning. The full-memory arm retains the complete encoder sequence. The structured dropout arm selects30% of training rows, masks exactly one verified image-token span per selected row, and preserves language, the other camera and all four z tokens. Since the two cameras have similar token counts, the observed memory keep rate is about0.86. Evaluation keeps both cameras. The four z queries still pool the joint encoder sequence; camera-specific z queries remain a later ablation if structured dropout is positive.

All six one-node smoke runs completed20 steps with trainer eval at10/20, finite losses, nonzero gradients and complete checkpoints. The batch128/256 jobs reuse those tested configs, but their multi-node startup and throughput remain a production gate. The camera-dropout smokes logged partial keep rates, establishing that the camera mask is active. No simulator rollout was used as a smoke.

The completed batch64 encoder full-memory/no-z reference is `ervla_v5_gr00t_traceloop_1pass_noz_all` (exact4k mean74.125%). It has no readout compressor and no z, but does include the trace auxiliary head; record that caveat instead of treating it as a pure no-aux control. The completed causal batch64 full-memory/no-z reference is63.288% over two seeds. These rows are reused because the user explicitly ruled out duplicate batch64 training.

Each initial job has a2h limit. This site rejects `--requeue` and reports `Requeue=0`, so a clean time-limit checkpoint submits a new2h resume job. Each training segment scans for20k/40k/60k/80k checkpoints and submits the optimized exact4k evaluator immediately:32 workers/GPU, one policy server/GPU, dynamic client-count batch ceiling, zero batching wait and resume enabled. Every generated successor/eval ID is appended to `slurm_logs/libero_grid_job_ledger.tsv`.

Kill a cell for non-finite loss, missing gradients/information path, repeated resume failure, or multi-node throughput below70% of the one-node projection. Do not select from training MSE: historical loss ordering failed to predict rollout ordering. Promote seed43 for the best two rollout cells and the matched encoder/causal comparison at the selected batch/step.

## Chunk length

The78.475% winner predicts and executes16 actions. Evaluate execute4/8/16 on that same checkpoint first. Horizon8 training remains one separate batch32 row after the inference screen; do not multiply the full batch×architecture matrix by horizon.

## RoboCasa parallel track

The existing QwenPI-v4 RoboCasa training job1858851 completed in9:00:18; its seed4/42 evaluation jobs1858852/53 are pending. Do not duplicate it.

The RoboCasa worktree has the older shared-z dropout path but does not yet contain `cross_memory_tokens` or `_augment_shared_z_cross_memory`. Port the verified corrected-z implementation, add an explicit mask/information-flow test, and run20-update in-training-eval smokes before any full job. After that gate, run one training seed at the native global batch64/50k recipe for the same four GR00T rows above. Chain the declared17-task×48-episode evaluation and promote seed4 only after valid seed42 results. This work is parallel and must not reuse the cancelled flawed-z jobs1863688/90/92.

## Daily schedule

| date | LIBERO-plus | parallel benchmark and paper work |
|---|---|---|
| Sep18 | **Done:** implement token-only z and structured camera dropout; pass six20-step/in-training-eval smokes; submit jobs1877720–1877727. Monitor allocation/startup and verify first production losses plus throughput. | Record final RoboCasa PI-v4 result (seed4/42:66.30/65.44% over816 episodes each) and update the cross-benchmark framing. |
| Sep19 | Monitor auto-resume ledger. Validate multi-node batch128/256 scaling. Run execute4/8/16 inference screen on the78.475% anchor without retraining. Do not add PI-v4+z until at least the20k token-only rollout result. | Recompute readiness/result tables from raw files; freeze exact task lists and config hashes. |
| Sep20 | Analyze the first exact4k20k results as mid-schedule checkpoints. Continue healthy runs toward40k; kill only on rollout failure plus no learning-curve evidence, or operational criteria above. | Prepare seed43 configs, but launch only for promoted cells. No new RoboCasa shared-z branch before the LIBERO token/camera decision. |
| Sep21 | Compare token-full versus camera-dropout versus encoder control at matched available batches. Choose provisional batch/architecture and launch seed43 confirmations. | Scientific gate: decide whether the paper supports an encoder win, an encoder-head win, or only a recipe result. |
| Sep22 | Continue40k/60k evaluations; stop dominated large-batch cells. | Freeze main ablation structure and claims. |
| Sep23 | Land seed43 confirmation for the selected endpoint and matched control where feasible. | Draft final tables/figures with uncertainty and protocol caveats. |
| Sep24 | Last safe headline-result arrival. Recompute every table from raw JSON. | Provenance/config audit and writing. |
| Sep25–26 | No exploratory training; only failed-eval recovery and claim-critical confirmation. | Reproducibility, anonymization and final rendering. |
