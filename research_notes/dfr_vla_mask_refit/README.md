# Deep Feature Reweighting for VLA Robustness

**Date:** 2026-09-04  
**Question:** Does *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations* explain our mask-refit result, and is group-balanced head refitting a promising way to improve VLA robustness on LIBERO-Plus?

## Bottom line

We ran the cheap experiment. The configured DFR-style action refits do **not**
improve robustness, although an optimization-controlled follow-up remains
worthwhile before rejecting the mechanism entirely.

A newer, stricter successful-trajectory study now sharpens that conclusion.
For each of four policy families, it pairs exact simulator states from that
policy's own successful clean rollouts with held-out background, lighting,
camera, and sensor-noise renders across all four LIBERO suites. P-causal and
PI-encoder fail the offline gate. GR00T-causal and GR00T-encoder show strong
late-DiT flow-loss recovery offline, but the corresponding every-step residual
correction **reduces** closed-loop success: causal changes 30.6% to 25.6% on
160 perturbations and 77.5% to 62.5% clean; encoder changes 37.5% to 15.4% on
104 perturbations and 76.9% to 19.2% clean. The negative result remains after
restricting to cases where the source successful actions replay in the Plus
fork. See
[`SUCCESSFUL_PAIRED_STUDY.md`](../../experiments/dfr_causal_poc/SUCCESSFUL_PAIRED_STUDY.md)
and the compact
[`paired_success_comparison_v3.md`](../../experiments/dfr_causal_poc/outputs/paired_success_comparison_v3.md).

Our mask-refit result is good evidence for a **representation/readout mismatch on the mask task**: after a spatial-suite mask tail is frozen, its original mask head almost completely fails off-suite, while a fresh mask probe recovers high IoU. That is closely analogous to the diagnostic intuition behind Deep Feature Reweighting (DFR): useful information can be present in a representation even when the learned readout uses it badly.

It is **not** yet evidence that a small action-head refit will remove spurious correlations from a VLA. The mask experiment differs from DFR in its objective, head, data construction, and metric. More importantly, our downstream policy controls currently say that real mask supervision is no better than shuffled masks and that the strongest unfrozen no-mask policy scores **73.4%** on LIBERO-Plus versus **65.6%** for the mask-fine-tuned policy. There is a serious input-rotation confound in that comparison, so it is not a clean rejection of mask shaping, but it prevents us from using the mask result as positive policy evidence.

The completed falsification study freezes P-causal and refits either its final
7k-parameter affine or full 1.06M-parameter velocity decoder using
action-labelled clean/background/light/noise examples. On matched closed-loop
evaluation, baseline scores 94/140 LIBERO+ and 29/30 clean. A fresh final affine
scores 85/140 and 28/30; a fresh balanced decoder scores 55/140 and 20/30; a
warm balanced decoder scores 95/140 and 28/30. The warm one-success difference
is indistinguishable from noise, while both fresh decoders have not recovered
the clean source policy. Full protocol and per-category results are in
[`experiments/dfr_causal_poc/RESULTS.md`](../../experiments/dfr_causal_poc/RESULTS.md).

The best next step is therefore narrower: make the final affine recover the
source clean loss/success before comparing balanced and clean-only refits over
several seeds. Background, lighting, and sensor noise remain valid
label-preserving targets. Target-object position and robot initial pose are not
nuisances: they change the correct action and require new successful
trajectories rather than erasure or unchanged action labels.

My confidence-weighted verdict:

- **Good diagnostic hypothesis:** the VLA represents more task-relevant information than its policy readout uses.
- **Unproven narrow method:** the tested balanced refit did not improve
  appearance robustness; optimization-controlled refits could still test the
  representation/readout hypothesis.
- **Weak broad claim:** “last-layer retraining solves LIBERO-Plus” is unlikely, especially for camera geometry and robot-state shifts.
- **Novelty:** DFR-for-VLAs alone is probably not enough for a strong paper in 2026. A publishable contribution would need a causal definition of VLA shortcut groups, a principled refit-locus study, failures as well as wins, and generalization across policy families.

## 1. What the DFR paper actually establishes

[Kirichenko, Izmailov, and Wilson (ICLR 2023)](https://arxiv.org/abs/2204.02937) study supervised classification with groups defined by the Cartesian product of target label and a spurious attribute—for example, bird class × background type. Their procedure is:

1. Train an ordinary ERM classifier on the imbalanced training distribution.
2. Freeze its feature extractor and discard the classifier.
3. Fit a new **linear** classifier from scratch on a small, held-out, group-balanced reweighting set.

The surprising result is not that balanced data helps. It is that the ERM representation often still contains linearly accessible core features, even when the original classifier heavily weights the shortcut. On Waterbirds, DFR changes worst-group/mean accuracy from **74.9/98.1** to **92.9/94.2**; on CelebA it changes **46.9/95.3** to **88.3/91.3**. The mean-accuracy losses are important: DFR moves performance between groups rather than creating a free improvement everywhere.

Several details are load-bearing:

- The reweighting data contains equal numbers from each target × nuisance group and is held out from base training.
- The classifier is reinitialized, not merely fine-tuned.
- The implementation standardizes frozen features, uses strong L1 regularization, and averages several independently refitted heads.
- Hyperparameters are selected using group-aware held-out validation.
- A literal linear classifier is sufficient in their setting. Retraining more layers actually hurts their Waterbirds/CelebA experiments.

The paper does **not** prove that useful core features are always retained. It finds that they often are, except under sufficiently extreme simplicity bias. Nor does DFR guarantee removal of the shortcut from the representation; it changes the readout weights.

A critical follow-up, [*Is Last Layer Re-Training Truly Sufficient for Robustness to Spurious Correlations?*](https://arxiv.org/abs/2308.00473), reproduces useful worst-group gains but shows that DFR can retain mixed core/spurious neurons and sacrifice previously strong groups. On ISIC, for example, benign/no-patch accuracy falls from **94.29** to **77.72**, even while the worst group improves. That is a warning for robotics, where a compensating loss can be unsafe even if an aggregate or worst-group score rises.

## 2. What our mask-refit result says

The relevant internal protocol is documented in [`RESULTS_TIER3.md`](../../../encdec-vlm/train/encoder_decoder_training/enc_dec_seg/RESULTS_TIER3.md). A frozen encoder is evaluated on unseen episodes with either its jointly trained mask head or a fresh probe trained for six epochs across all four LIBERO suites.

### The striking result

For the spatial-suite tail, `manip` IoU is:

| Suite | Original head | Fresh refit | No-tail encoder + fresh refit |
|---|---:|---:|---:|
| Spatial | 0.934 | 0.900 | 0.476 |
| Object | 0.001 | 0.715 | 0.548 |
| Goal | 0.047 | 0.576 | 0.615 |
| Long-horizon | 0.002 | 0.740 | 0.557 |

Held-out referring accuracy also rises from **0.444** for the no-mask-FT encoder to **0.960** for the spatial-tail encoder with a fresh readout.

The defensible conclusion is:

> Spatial mask training made off-suite task-object information substantially more recoverable, while the jointly learned spatial mask head did not transfer that information off-suite.

This is a valuable result. It separates “the representation lacks the information” from “the existing readout fails to use the information.” It also suggests a useful diagnostic template for the action policy.

### Why this is DFR-like, but not DFR

| Property | DFR paper | Our mask refit |
|---|---|---|
| Base objective | Ordinary ERM classification | Explicit dense mask fine-tuning |
| Frozen representation | Penultimate classification features | Encoder patch/token features |
| Refit module | Linear logistic classifier | Learned patch projection plus role-query MLP and bilinear scoring; instruction query was zeroed for the enc-dec result |
| Refit data | Held-out, target × nuisance group-balanced subset | Held-out suites/episodes, not a nuisance-balanced action set |
| Target | Class label | Dense role mask |
| Main metric | Worst-group classification accuracy | Mask IoU / referring accuracy |
| Behavioral evidence | Direct task output | Auxiliary representation only |

The probe in [`phase0a_probe.py`](../../../encdec-vlm/train/encoder_decoder_training/enc_dec_seg/phase0a_probe.py) is not a single linear classifier. It learns a patch projection and an instruction/role-conditioned query network, then scores them bilinearly. It is lightweight, but expressive enough that “linear core information was already present” is too strong a description.

The off-suite collapse of the original spatial head also has alternatives to “spurious correlation”: suite-specific role priors, calibration, target-name distributions, or incompatible head semantics can all produce the same pattern. To attribute it to a shortcut, we need controlled counterfactual groups in which the nuisance changes while the desired target/action is held fixed.

## 2a. Why CoT-direction removal and mask refitting looked positive

These two earlier results are relevant, but they are positives for different
claims. Neither is a previous demonstration that DFR improves a VLA policy.

### CoT direction removal: a narrow behavioral positive

The CoT probe did substantially more than ask whether an arbitrary attribute
was decodable. It fitted a reduced-rank ridge map from the exact layer states
consumed by PI to **instruction-residualized assistant-token TF-IDF**, using a
trajectory-aware 333/147 train/test split. The true held-out directional cosine
was far above shuffled labels: in raw 2048-D states the median was 0.514 versus
-0.008 for K and 0.478 versus -0.022 for U. The intervention then removed the
top 32 fitted directions at every even PI input layer. Held-out R2 was near
zero or negative, so this was directional decodability, not faithful recovery
of the full CoT text.

The complete behavioral record is:

| Intervention | K | U |
|---|---:|---:|
| Projected-space removal, standard LIBERO | 387/400 vs 384/400 | 389/400 vs 392/400 |
| Raw pre-projector removal, standard LIBERO | 395/400 vs 384/400 | 395/400 vs 392/400 |
| Raw pre-projector removal, exact LIBERO+ | **2897/4000 vs 2859/4000 (+0.95pp)** | **3039/4000 vs 3039/4000 (tie)** |

Thus “CoT removal worked” is too broad. The cleaner projected-space comparison
gave symmetric +3/-3 episode changes, which is a null. Raw removal produced a
small exact-4k gain for K but did not replicate on U. We have no paired
significance test with coupled flow noise for that result. The defensible claim
is that removing the fitted linear CoT subspace is safe and **possibly mildly
helpful for K**, not that it reliably removes a shortcut.

There are good mechanistic reasons why this edit was safer than the action-head
refit:

1. **The concept was known and deliberately injected.** CoT supervision created
   a labelled auxiliary signal from which the removal basis could be fitted.
   Lighting or an unnamed LIBERO shortcut does not provide such a clean target.
2. **CoT and action used different parts of the pathway.** K's action sensitivity
   is early/middle-layer-heavy, while CoT gradients become enormous only in late
   layers where PI barely uses the state. At layer 26, within-task state
   shuffling changed actions by only about 0.6%, and CoT/action activation-gradient
   cosine was approximately 0.0003. The fitted signal was present but largely
   orthogonal to useful action computation.
3. **The intervention was low rank and preserved the trained policy.** It removed
   32 directions from 2048-D raw states and changed held-out open-loop actions by
   only 2.99% relative RMSE for K and 2.02% for U. It did not reinitialize the
   action generator. In the DFR PoC, resetting the final velocity map changes
   every step of flow integration; resetting the full decoder destroyed clean
   recovery.
4. **K had a plausible harmful training artifact to remove.** Its auxiliary CoT
   objective pushed late features in a direction almost orthogonal to action.
   Deleting part of that drift can act as regularization. U's exact tie is
   consistent with its action pathway already ignoring the same information.

Raw removal occurs before an input-dependent LayerNorm and linear projector, so
it changes more than a perfectly isolated semantic coordinate. This is another
reason not to attribute K's +0.95 points uniquely to “removing CoT.” The result
is closer to a targeted LEACE-style activation edit than to DFR: it edits the
representation at inference, whereas DFR leaves features intact and refits a
supervised readout.

The full source record is the “PI activation pathways and causal CoT-subspace
test” section of
[`HANDOVER_ERVLA.md`](../../../encdec-vlm/train/encoder_decoder_training/enc_dec_cot/HANDOVER_ERVLA.md).

### Mask FT: a strong representation positive, not a policy positive

Mask fine-tuning genuinely worked for the **mask representation task**. Dense
per-patch supervision directly trained which pixels belong to the instructed
manipulated object, and the target-swap construction prevented the scene alone
from identifying the correct same-category object. In the bidirectional
encoder, image tokens can see the instruction, so training can store the
instruction-object binding in the patch states themselves. That is why an
instruction-free readout can subsequently succeed: “query-free” zeros the
instruction only in the probe query; the encoder had already received and bound
the instruction.

This mechanism has strong controls. Training all 28 enc-dec layers moved
query-free referring accuracy from 0.452 to 0.986 +/- 0.004, while the causal
control with the same mask recipe remained near chance at 0.419 because its
earlier image tokens cannot attend to the later instruction. With a mismatched
instruction, accuracy fell from 0.988 to 0.456 and probability mass moved onto
the same-category distractor (0.008 to 0.471). Therefore the gain was real
language-conditioned binding, not merely learning to segment “a bowl.”

The separate off-suite refit result answers a readout question. The spatial
tail's own head was trained only on the spatial-suite distribution and collapsed
on other suites, but a fresh all-suite, query-free role probe recovered
`manip` IoU of 0.715 on Object, 0.576 on Goal, and 0.740 on LIBERO-10. The most
plausible reading is that mask supervision put reusable object/role information
in the frozen representation, while the original mask head retained
suite-specific calibration or role priors. A newly supervised, query-free but
role-conditioned probe could remap those features to the mask labels.

That is DFR-like evidence for a representation/readout mismatch, but there are
three crucial limitations:

- The new probe receives direct dense mask labels and is not merely a linear
  classifier; it has a patch projection, instruction/role query MLP, and
  bilinear scorer.
- Its reported output is mask IoU/referring accuracy—the same task used to shape
  the representation—not robot success.
- The apparent early policy benefit was not caused by mask content. Once the
  backbone, prompt, freeze regime, and action framework were controlled, real
  masks were indistinguishable from shuffled masks and no-mask adaptation.
  Unfreezing the plain encoder was the dominant lever (73.4% LIBERO+), and the
  mask-fine-tuned version was lower (65.6%). The 180-degree mask/policy input
  mismatch weakens that negative transfer comparison, but it does not turn the
  mask probe into positive action evidence.

### Why these results do not contradict the failed action refit

| Experiment | What was changed | What the positive establishes | What it does not establish |
|---|---|---|---|
| CoT rank-32 removal | Known auxiliary subspace deleted; trained action policy retained | Low-rank deletion is behaviorally safe; K may contain mildly harmful auxiliary drift | A natural LIBERO shortcut was found; DFR works |
| Mask FT + fresh probe | Representation shaped with dense labels; new mask decoder trained | Task-object binding is present and a mismatched mask readout can be replaced | The action policy can use it; robustness improves |
| DFR-style action refit | Final affine or velocity MLP reset/refitted on synthetic domains | Direct test of clean recovery and closed-loop robustness | Positive result—the configured refits failed |

The common encouraging thread is only that VLA representations are modular
enough to contain linearly or cheaply accessible information that the action
path may ignore. The missing step is showing that a **small action readout can
use invariant task information while preserving the learned flow policy**. Our
600-step fresh refits did not pass that prerequisite. This is why the next test
must first match clean loss and clean rollouts, then ask whether balanced data
improves held-out nuisance groups.

## 3. The downstream policy evidence is currently sobering

The later policy decomposition in the same results file is more relevant to the proposed VLA claim than the mask IoU result:

| Policy regime | Mask signal | Vanilla | LIBERO-Plus |
|---|---|---:|---:|
| Frozen, bare prompt | None | 90.8 | 34.8 |
| Frozen, bare prompt | Shuffled | 87.2 | 33.6 |
| Frozen, bare prompt | Real | 86.8 | 37.1 |
| Unfrozen, bare prompt | None | **97.5** | **73.4** |
| Unfrozen, bare prompt | Real | 96.2 | 65.6 |

Within the frozen controls, real masks are indistinguishable from shuffled or no masks. Unfreezing the plain encoder is the dominant lever: **+38.6 points** on LIBERO-Plus. In the unfrozen comparison, mask FT is lower on six of seven axes and loses 7.8 aggregate points.

There is an important caveat: the mask-FT images were later found to be rotated 180° relative to policy train/eval images. Thus the honest conclusion is only that a tail shaped on the wrong image orientation did not help the upright policy. This substantially weakens the negative mask-transfer result. It does **not** explain real-mask ≈ shuffled-mask within the mask-FT arms because both share the same rotation.

So the current evidence ledger is:

- **Supported:** mask representations can transfer much better than their original mask readouts.
- **Supported:** policy architecture/training state matters much more than the tested mask objective.
- **Not supported:** real dense masks currently improve policy robustness.
- **Unknown:** a correctly constructed, balanced action-readout refit improves worst-group control.

## 4. LIBERO-Plus is not one spurious-correlation problem

[LIBERO-Plus](https://arxiv.org/abs/2510.13626) contains 10,030 evaluation tasks over seven perturbation dimensions. The benchmark calls them generalization factors, not all “spurious correlations.” That distinction matters because DFR assumes the spurious attribute is irrelevant to the true target.

| Factor or subfactor | Does the correct action stay fixed for the same underlying state/task? | DFR fit | Main risk |
|---|---|---|---|
| Background texture/material | Usually yes | High | Feature entanglement with object appearance |
| Lighting | Usually yes | High | Severe lighting can change observability, not just style |
| Sensor noise | Yes, before observation corruption | High | The representation may discard information under strong corruption |
| Language paraphrase/style | Yes | Medium–high | Robustness can be faked by ignoring language |
| Added distractor objects | Usually yes | High | Occlusion can make the problem genuinely harder |
| Familiar object × location pairing | Yes when tested with matched difficulty | High | Requires explicit held-out pairings, not generic layout bins |
| Camera viewpoint/FOV | World action can remain fixed | Medium | Requires geometric equivariance and visibility; a final head may not recover missing geometry |
| Target-object pose | **No** | Low / invalid nuisance | The target motion and grasp trajectory must change |
| Robot initial state | **No** | Low / invalid nuisance | Correct action depends on kinematics and proprioception |
| Instruction semantics/goal identity | **No** | Invalid nuisance | Erasing it creates language-blind behavior |

LIBERO-Plus itself supports this split. OpenVLA-OFT drops from 97.1 clean to **59.7 camera** and **37.2 robot-initial-state**, while background remains **92.4**. Its generalized post-training baseline uses more than 20,000 successful trajectories and reaches **79.5 overall** and **92.8 camera**. That is the baseline a refit method must compare against, not only the original brittle policy.

The generalized-data construction also exposes a practical boundary: it could replay existing state/action sequences under backgrounds, lights, camera changes, language rewrites, noise, and added distractors, but excluded pose-changing variants because replayed trajectories were unreliable. This is exactly the causal distinction above. For appearance/style changes, paired data can preserve the action label. For target and robot pose changes, new expert trajectories are needed.

## 5. What is the “last layer” of this VLA?

There is no unique analogue of a classification layer in the current policy.

For `QwenGR00T`, the path is approximately:

```text
final Qwen encoder tokens
    -> optional 2-layer, 32-query cross-attention readout projector
    -> 12-layer diffusion transformer conditioned on state, noisy action, and time
    -> 2-layer action-decoder MLP
    -> iterative flow-matching action chunk
```

See [`QwenGR00T.py`](../../starVLA/model/framework/VLM4A/QwenGR00T.py), [`readout.py`](../../starVLA/model/modules/projector/readout.py), and [`GR00T_ActionHeader.py`](../../starVLA/model/modules/action_model/GR00T_ActionHeader.py).

The literal final affine layer only maps an already context-mixed diffusion state to velocity/action dimensions. A shortcut based on object location, camera, or language can enter through cross-attention and all 12 DiT blocks. Refitting only the final affine layer may therefore be too weak. Conversely, refitting the whole action expert is much more expensive and is no longer the elegant “logistic regression on cached features” intervention from DFR.

The right experiment is a **refit-locus ladder**, with all selected modules reinitialized:

1. Final output affine layer only.
2. Full two-layer action-decoder MLP.
3. Readout projector + action decoder.
4. Adapters or the last *K* DiT blocks + action decoder.
5. Entire action expert.
6. Full-policy fine-tuning as an upper bound.

The `QwenPI`/layerwise path should be treated separately because it conditions the action model on several VLM layers and has no equivalent single readout bottleneck. A result on one head should not be generalized to all VLAs.

## 6. LEACE is a diagnostic, not automatically a complementary fix

[LEACE](https://arxiv.org/abs/2306.03819) applies a closed-form affine projection that makes a labeled concept inaccessible to every linear predictor while minimizing a least-squares representation edit. This is different from DFR:

- DFR changes which existing features the output uses; it does not guarantee erasure.
- LEACE removes linearly decodable information from the representation; a nonlinear decoder may still recover it.

A DFR × LEACE factorial experiment could reveal whether a refitted policy still depends on a labeled nuisance. But applying LEACE blindly is dangerous in control. Camera pose, target layout, and robot state encode causally necessary geometry. Erasing them can make correct actions impossible. Even background and lighting are entangled with object identity and boundaries.

If used, concept removal should be restricted to demonstrably label-preserving factors and checked for retained task-state decodability. The key metric is behavior under paired counterfactuals, not merely whether a linear nuisance probe falls to chance.

## 7. A minimal decisive experiment

### 7.1 Hypothesis

A frozen VLA trained on biased LIBERO data contains enough task-causal information to act successfully in minority environments, but its learned action readout overweights environment shortcuts. A small, held-out, group-balanced refit can improve worst-group success without materially reducing clean success.

### 7.2 Start with label-preserving factors

Use paired renderings/replays of the same underlying demonstrations for:

- clean versus novel background;
- clean versus lighting shift;
- clean versus sensor noise;
- canonical versus paraphrased instruction;
- no distractor versus added distractor;
- familiar versus held-out object-location pairing, with matched reach/grasp difficulty.

Do not initially mix in target-pose or robot-initial-state shifts. Camera should be reported separately because it is label-preserving at the world-action level but much more representation-demanding.

Define groups as `(task identity or action stratum, nuisance value)`, not merely “clean/OOD.” For continuous actions, balance by task and intervention cell, then audit action-distribution matching. Otherwise a model can learn a new action prior rather than a less spurious policy.

### 7.3 Controls

For each equal-size adaptation set, compare:

- original frozen policy;
- continued fine-tuning of the existing module;
- from-scratch balanced refit at every locus in the ladder above;
- an equal-data unbalanced refit;
- ordinary augmentation/post-training with the same data and trainable parameter count where possible;
- full expert and full-policy fine-tuning upper bounds.

Use held-out task seeds, simulator states, textures, instructions, and nuisance levels for test. Do not refit on the LIBERO-Plus evaluation instances. Use at least three training seeds and paired rollout manifests.

### 7.4 Metrics that prevent a false win

- Per-group and worst-group closed-loop success, with confidence intervals.
- Clean success and the largest negative group delta.
- Paired recovery/loss counts on identical underlying episodes.
- Counterfactual action consistency for label-preserving renderings.
- Goal-swap and contradictory-instruction sensitivity, so language blindness cannot masquerade as paraphrase robustness.
- Core-state and nuisance decodability with both linear and small nonlinear probes.
- Data, trainable-parameter, wall-clock, and inference-cost comparisons.

Imitation MSE is useful for debugging but cannot replace rollout success: small action errors compound, and an averaged “robust” action can be unsafe.

### 7.5 Pre-registered expectations

- Background/light/noise may improve with a small readout, but current scores are already high, so ceiling effects limit aggregate gains.
- Object-location counterfactuals are the strongest clean test of a shortcut-reweighting story.
- Literal output-layer refitting will probably be insufficient for camera and language because the relevant conditioning is mixed upstream.
- Camera gains will likely require the readout projector, DiT adapters, or explicit crop/view augmentation.
- Robot-initial-state and target-pose performance should not improve through nuisance erasure; they require better state/geometry conditioning and appropriate trajectories.
- If the full action expert must be retrained and equal-data augmentation matches it, the result is domain adaptation—not a DFR-style last-layer finding.

### 7.6 Go/no-go criterion

Continue toward a paper only if a refit smaller than the full action expert:

- recovers at least 30% of the available robustness gap on two or more label-preserving factors;
- loses no more than 2 percentage points of clean success;
- improves worst-group and paired recovery consistently across three seeds;
- beats an equal-data augmentation/post-training control; and
- preserves sensitivity to genuine goal/instruction changes.

Stop or reframe the project if gains require full-policy fine-tuning, disappear under task-balanced controls, trade one fragile group for another, or coincide with reduced language/goal sensitivity.

## 8. Research positioning in 2026

The motivation is real but the space is no longer empty:

- [*Robust Skills, Brittle Grounding*](https://arxiv.org/abs/2602.24143) directly documents object-location shortcuts in SmolVLA and Pi-0.5 and separates primitive execution from instruction-conditioned success.
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626) already provides the seven-axis diagnosis and a strong generalized post-training baseline.
- [*Restoring the Right Stream*](https://doi.org/10.3390/e28090975) uses frozen-policy, per-stream test-time restoration on LIBERO-Plus and reports a +5.3 point aggregate gain, including robot-initial-state improvement from 20% to 38% in its original paired evaluation.
- [*Restoring Linguistic Grounding via Train-Free Attention Recalibration*](https://arxiv.org/abs/2603.06001) targets LIBERO language blindness without retraining.

Therefore, “VLAs have spurious correlations and readout changes help” is not a sufficient novelty claim. The sharper contribution would be:

> **Where is the smallest intervention locus at which a VLA’s latent task knowledge becomes behaviorally usable under a causally defined shortcut break?**

That framing makes the mask-refit finding a motivating observation, not the conclusion. It also makes negative results useful: different shifts may require output reweighting, representation adaptation, input-stream restoration, or new causal data.

## Recommendation

Run the compact refit-locus study, beginning with object-location pairings plus background/light counterfactuals on the strongest no-mask policy. Do not begin with LEACE, the mask-fine-tuned policy, or a seven-axis aggregate. If a small reinitialized readout beats equal-data augmentation on held-out worst groups while preserving clean and goal sensitivity, this becomes a credible new VLA direction. If not, the mask-refit result should remain a representation diagnostic rather than a robustness method.

## 9. Empirical no-reset alignment follow-up

We ran the smaller intervention suggested by the failed fresh-decoder refit. One
rank-1 projected-state direction per layer was fit separately from synthetic
background, lighting, and sensor-noise pairs. Eighty-four suppression gates
start at zero, and selected arms tune only the pretrained final action affine.
Nothing in the flow generator is reinitialized. The fit used 3,000 underlying
demonstration steps / 6,000 clean-plus-perturbed presentations; official
LIBERO+ remained evaluation-only.

The best arm—learned gates plus warm final affine—retained 30/30 clean and
scored 97/140 LIBERO+, versus 29/30 and 94/140 for the source. This is not yet
evidence of shortcut removal. The matched overall test is `p=0.629`, and the
three trained groups total 47/57 for both learned and random gate controls.
Learned gates favor lighting (14/15 versus 10/15 random) while the random gates
favor sensor noise (20/25 versus 17/25 learned). Directly against the unchanged
source, learned lighting is only 14/15 versus 13/15.

Thus the no-reset construction is promising enough for a nuisance-specific
replication because it preserves the policy, unlike fresh decoder fitting. It
does **not** validate broad DFR for VLAs. The next acceptable result must repeat
over training seeds and show that a lighting-only learned basis beats both the
source and equal-rank random intervention without trading away noise or clean
success. Full protocol, matched tests, and artifacts are in
[`experiments/dfr_causal_poc/RESULTS.md`](../../experiments/dfr_causal_poc/RESULTS.md#zero-initialized-low-rank-alignment-follow-up).

## Primary sources

- Kirichenko, Izmailov, Wilson. [*Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*](https://arxiv.org/abs/2204.02937), ICLR 2023.
- Le et al. [*Is Last Layer Re-Training Truly Sufficient for Robustness to Spurious Correlations?*](https://arxiv.org/abs/2308.00473), 2023.
- Belrose et al. [*LEACE: Perfect Linear Concept Erasure in Closed Form*](https://arxiv.org/abs/2306.03819), NeurIPS 2023.
- He et al. [*LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*](https://arxiv.org/abs/2510.13626), 2025.
- Emukpere, Deffayet, Renders. [*Robust Skills, Brittle Grounding*](https://arxiv.org/abs/2602.24143), 2026.
- Yan, Shen. [*Restoring the Right Stream: Training-Free OOD Robustness for Vision–Language–Action Policies*](https://doi.org/10.3390/e28090975), Entropy 2026.
- Zhang et al. [*Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration*](https://arxiv.org/abs/2603.06001), 2026.
