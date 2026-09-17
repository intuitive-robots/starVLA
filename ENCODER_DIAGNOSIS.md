# What the encoder evidence says, and how to make it matter in rollouts

This is an interpretation of raw probe JSONL, rollout JSON, and a **targeted configuration audit**—not a claim that a probe is a policy metric. It distinguishes observations, inferences, and tests. The repository contains 135 LIBERO training-YAML paths; this document does not treat them as 135 controlled experiments.

## Bottom line

The encoder is producing more action-relevant visual information than the causal control, but the existing policy heads extract it inconsistently. The strongest closed-loop result is not generic enc-dec: it is the **structured, temporally supervised, shared-z bottleneck**. The practical goal is therefore not “maximize mean linear-probe R2”; it is to build a view-aware, task-conditioned action readout around the shared representation and prove which component causes the gain.

## Five observations that matter

| observation | raw evidence | interpretation |
|---|---|---|
| Shared-z is much larger than plain enc-dec | zonly-v5 .774 ± .005 (n=8,000) vs causal .717 ± .004 (n=12,000); v5 PI .731 ± .005 (n=8,000) | The valuable signal is structured latent control, not merely a bidirectional encoder. |
| The current head leaves representation value on the table | Same v5 backbone: PI .731 versus GR00T .746 (both n=8,000). Direct RoboCasa patch probe: causal .0382 versus enc-dec .2699 (n=960 each), but rollout only .355 versus .378 (n=1,632 each). | A readout/control bottleneck is at least as plausible as a representation bottleneck. |
| High action R2 does not predict behavior | Tracetime has Bridge/DROID R2 .2097/.1238 but aggregate .6400; full epoch has .1511/.0967 and aggregate .6904. | Frozen ridge R2 should screen hypotheses, never select the policy. It can reward task/instruction shortcuts. |
| Extra views are valuable only when the representation/readout can use them | Across 12 LIBERO-plus arms, wrist raises mean R2 roughly .08–.15 (n=960); on RoboCasa it is about .03–.07. DROID wrist commonly lowers R2 by about .01. | Naive multi-view pooling is not reliably fused. Camera-specific tokenization and query routing are needed. |
| V5 carries a visual fallback but still uses state | Three-task no-state: causal 0/144, v5 85/144=.590 ± .041. Zero state still drops v5 to .306 ± .038. | The encoder can support vision-first control; it does not make proprioception irrelevant. |

The LIBERO-plus perturbation result reinforces this: zonly-v5’s gain is concentrated in robot initialization (+.153), object layout (+.125), and camera viewpoint (+.083), while backgrounds are worse (-.033). This is consistent with learning a task/object/trajectory representation rather than a uniformly better image encoder.

## Critical caveat: shared-z is currently a package, not an isolated variable

The two YAMLs compared side-by-side here are `examples/LIBERO/train_files/ervla_v5_pi_actiononly_pifix.yaml` and `ervla_zonly_pi_sharedz_ground_temporal_v5.yaml`, because they underlie the key full-protocol v5 PI ↔ zonly-v5 comparison. The remaining YAMLs include older-backbone, decoder, CoT, GR00T, and real-robot variants; many change several dimensions simultaneously or lack a matched full-protocol result, so they cannot identify a shared-z main effect.

| setting | v5 PI | zonly-v5 | why it matters |
|---|---:|---:|---|
| encoder memory into action DiT | layerwise encoder states | dropped with probability 1; DiT sees only z | tests compression/invariance, not simply encoder type |
| learned representation | none | 128-d z from four learned queries | candidate causal mechanism |
| supervision | action only | box, relation, visibility, 3D trajectory, phase, future offset 8 | candidate causal mechanism |
| cross-attention adapters | frozen | not in `freeze_modules`, therefore trainable | confound likely large enough to matter |
| action DiT dropout | 0 / final false | .2 / final true | regularization confound |
| VLA batch/device | 32 | 16 | optimization/noise confound |

Thus `.774 - .731 = .043` is an excellent *recipe* effect, but not yet a shared-z-only effect. Conversely, `memory_dropout_rate: 1` is scientifically useful: if the effect survives a compressed 128-d action interface, the representation is sufficient for robust action selection.

## Recommended model direction

Use **z as a task-conditioned controller state, not as the only information bottleneck forever**:

1. Pool each camera with a small number of view-labeled queries; condition queries on instruction and z.
2. Predict z from structured trajectory/grounding targets as now, but pass both z and selected local visual tokens to the DiT.
3. Train with stochastic memory availability (for example p=1, .5, 0) rather than only `memory_dropout_rate=1`. This asks z to be sufficient under shifts while preserving precise visual servoing when memory is reliable.
4. Make the action head predict/attend to an object/trajectory query before diffusion. GR00T’s 32 readout queries are the strongest clue that learned extraction, not raw layerwise memory, helps.

This architecture directly targets the observed failure mode: global/linear probes find action information, but a plain PI DiT does not reliably select the right local, camera-specific information.

## Ranked experiments (do not launch from this document)

| priority | experiment | result that confirms it | result that kills it |
|---:|---|---|---|
| 0 | Finish and full-evaluate the already-running zonly+GR00T pair (`1851382`, dependent evals `1851383/4`) | two complete 4k runs beat both zonly-PI (.774 pooled) and v5-GR00T (.746 pooled) by >2 combined SE | no gain over zonly-PI: do not spend time on a fancier readout |
| 1 | Recipe-matched 2x2: shared-z off/on x adapter-trainable/frozen; hold PI, dropout, batch, schedule fixed | positive z main effect in both seeds after adapter control | effect belongs to adapter tuning or vanishes |
| 2 | Within the matched z arm: structured targets off/on; then temporal future target off/on | geometry/trajectory losses selectively recover camera/layout/init categories | no category or rollout effect despite better auxiliary loss |
| 3 | z-conditioned, camera-labeled readout queries; compare memory dropout 1, .5, 0 | native rollout non-inferior and camera/layout improves; no-state remains strong | either native precision or no-state robustness collapses |
| 4 | State-dropout training and full 17-task state intervention | v5’s no-state advantage repeats without a >2-SE native penalty | improvement is limited to the hand-picked three tasks |

Priority 1 is the paper-critical test. It costs four 20k trainings (roughly 120 GPU-hours at observed throughput) plus four 4k evaluations (roughly 24 GPU-hours) for two seeds. Priorities 2–4 should be sequential: use the category metric to decide whether the next training is justified.

## Probe protocol changes before using probes to steer training

1. Report two measures together: visual-only action R2 and the **increment over a state-only baseline**. On Bridge, state-only is .141; on DROID, .089. A high absolute value can be state/task leakage.
2. Add an instruction-shuffle and view-shuffle control. Existing token-class evidence says instruction tokens are frequently most decodable; existing patch maps often peak at borders on real data. Both can inflate action R2 without improving visual grounding.
3. Keep camera labels and use per-view queries. The wrist result is strongly dataset-dependent, so mean pooling across views is an avoidable information bottleneck.
4. Gate any claimed probe/behavior relationship through a small calibration set of full rollouts. The present correlation is not reliable even in sign across the action-auxiliary arms.

## Paper-safe claim if the above does not land

The data already supports: *structured shared-latent supervision yields a compressed visual controller that is more robust to camera/layout/initial-state shifts and can act without proprioception.* It does not yet support: *bidirectional encoders universally beat causal action models* or *linear action decodability predicts robot behavior*.
