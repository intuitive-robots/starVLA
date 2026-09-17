# ICLR readiness — 17 September 2026

## Verdict: conditional no-go for the stated thesis

Do not submit in ten days claiming “encoder action models are best for behavior generation across benchmarks.” Full LIBERO-plus supports a narrower result: two-seed shared-z-v5 beats the three-seed causal baseline by +.057 ± .006 (8,000 versus 12,000 episodes). Matched two-seed RoboCasa is only +.023 ± .017 (1.34 SE), and plain LIBERO is saturated. Encoder-only evidence is mostly single-seed and readout-confounded.

A conditional submission is viable if the headline becomes: **a shared-latent encoder-style action model improves robustness under LIBERO-plus perturbations and retains behavior under missing proprioception.**

## Sufficient now

- Causal LIBERO-plus baseline: three seeds, 12,000 full-protocol episodes.
- Shared-z-v5: two seeds, 8,000 full-protocol episodes; old-backbone zonly (.776, n=4,149) and zsup (.781, n=4,000) corroborate it.
- Mechanistic decomposition: biggest zonly gains are robot-init +.153, layout +.125, and camera +.083; backgrounds are -.033.
- Complete two-seed, 17-task RoboCasa table and a three-task state intervention. The latter shows v5 visual fallback, not a broad win.

## Blocking gaps

- RoboCasa fails the two-SE bar. A reviewer can reject “across benchmarks” on this basis.
- CoT .826/.822 are only n=1,153; no valid 4k re-evaluation has landed. Job 1855180 is losing shards; 1850350–52 are running.
- No third benchmark or matched-compute accounting. GR00T-vs-PI head effects confound the encoder objective.
- `w`, `w2`, MLM, and zsup are mostly one seed. There is no two-seed factorial causal/encoder-objective x shared-z-on/off comparison with the same head and checkpoint rule.
- State corruption is one seed and three tasks. Probe R2 is not validated as a rollout surrogate.

The result that would collapse the story if wrong is the shared-z-v5 improvement. Arithmetic confidence is high—raw 1,000/suite JSONs give 3,059/4,000 and 3,131/4,000—but protocol-equivalence confidence is only moderate until configs, checkpoint selection, training tokens, and wall time are audited side-by-side.

## Ten-day plan

Historical complete LIBERO-plus runs take 1:13–1:39 wall time, about 5–7 GPU-hours/run on four GPUs. The full 17-task RoboCasa pair took 0:58 wall time, about 4 GPU-hours/pair. Reserve retry slack for queue and rendering failures. No job was launched by this audit.

| day | must have | nice to have |
|---|---|---|
| Sep 17 | babysit 1850350–52; do not accept 1855180 until its shard fault is fixed; re-submit a clean CoT 4k eval only with the ledger command and existing checkpoint | zonly-v5 seed 3; equal-compute causal rerun |
| Sep 18 | verify every CoT suite is 1,000 episodes before merge | RoboCasa seed-43 17-task pair |
| Sep 19 | launch factorial causal/encoder-objective x shared-z off/on, fixed PI head, two seeds | camera-conditioned/augmented zonly pilot |
| Sep 20 | first factorial 4k evaluations; audit configs/parameters/tokens/wall time | full 17-task state matrix |
| Sep 21 | finish factorial first seeds; decide whether causal story survives | third-benchmark smoke then full if aligned |
| Sep 22 | second seed for promising factorial cells | DINO/probe control |
| Sep 23 | experiment freeze; regenerate master CSV, CIs, and figures | extra CoT variant |
| Sep 24 | write only complete results | videos |
| Sep 25 | last safe full-evaluation landing; table lock | — |
| Sep 26–27 | writing, reproducibility audit, submit | no new dependency |

If a clean CoT 4k result is not complete by Sep 19, omit CoT from the main story.

## Experiments that could sharpen the paper

| proposal | confirmation / kill criterion | estimated cost |
|---|---|---:|
| camera conditioning + camera augmentation for zonly | confirm camera +>.04 without material robot/layout loss; kill if the 4k camera CI overlaps baseline | train pair ~30 GPU-h + two evals ~12 GPU-h |
| shared-z factorial, fixed PI head | confirm a positive shared-z main effect in both seeds and >2 combined SE; kill if it tracks head/checkpoint | four trains ~120 GPU-h + evals ~24 GPU-h |
| encoder objective factorial without shared-z | confirm a reproducible objective main effect after head control; kill if <2 SE or signs disagree | combined with above |
| state dropout, then 17-task no-state test | confirm no-state gain and native non-inferiority within 2 SE; kill if native degrades | ~30 GPU-h + ~4 eval GPU-h |
| behavior-calibrated readout objective | confirm an arm improving R2 and 4k rollout; kill if they remain uncorrelated | ~30 GPU-h + ~6 eval GPU-h |

The honest publishable framing today is perturbation robustness and modality reliance with closed-loop evidence centered on LIBERO-plus. It is not universal encoder superiority.
