# Beyond DFR: cross-domain lessons for robust VLA refitting

Deep-research report, 7 September 2026

## Executive conclusion

The fresh experiment is **negative for our capped late residual**, not for the
larger idea that useful action information survives in the representation.
The earlier GR00T-causal validation gain (+9.375 points) did not reproduce on a
fresh trajectory and unseen perturbation variant: it became -1.25 points. The
four-policy confirmation therefore gives no evidence that the current gate
improves robustness.

This also was not standard DFR. [DFR](https://openreview.net/forum?id=Zb6c8A-Fghk)
freezes a feature extractor, throws away a *static linear classifier*, and fits
a new classifier once on an independent group-balanced set. We instead added a
residual to a velocity field repeatedly during flow integration. The first
correction changes the next latent visited, so later corrections run off the
teacher-forced training distribution. A lower one-step flow loss is therefore
not enough.

The strongest direction is still promising, but it should now be framed as
**action-equivalent counterfactual consistency**, not “DFR for VLAs” and not
latent erasure:

1. generate exact same-state clean/perturbed observation pairs;
2. warm-fine-tune a small upstream visual/fusion adapter and, if needed, the
   last one or two action blocks;
3. supervise both views with the same successful flow target and explicitly
   align their velocity predictions at shared flow coordinates;
4. preserve the clean policy with a clean teacher/parameter anchor;
5. evaluate fresh closed-loop success, with plain balanced mixed fine-tuning as
   the hard baseline.

This is directly supported by the accepted
[RobustVLA ICLR 2026 paper](https://openreview.net/forum?id=cS6xizdYD5) and,
even more specifically for camera changes, the very recent but unreviewed
[Cross-View Action Consistency preprint](https://arxiv.org/abs/2608.06965).

Newest evidence sharpens that recommendation: a sampler-aware rank-4 PI
projector adapter replicated its offline successful-action endpoint gain on a
sealed split, yet produced 96/140 LIBERO-Plus successes versus 97/140 for the
frozen policy, with identical clean performance (28/30). Exact
counterfactuals and backpropagation through the short sampler are therefore
still insufficient when training states come only from clean successful
rollouts. Any continuation should add perturbed on-policy recovery states and
a behavioral training signal; simply scaling this offline loss is not the
next justified step.

## What we actually established

### New explicit-alternating PI result

The earlier PI rows below were later found to use QwenPI_v3's legacy all-cross
routing. We therefore repeated the upstream flow-alignment diagnostic with
`ervla_w2_pi_encoder_actiononly_pifix_nolatent`, whose saved config explicitly
uses 14 alternating cross-attention and 14 self-attention Action-DiT blocks.
Fresh capture produced 316/320 clean successes and 7,680 exact-state
clean/appearance pairs across all 40 tasks.

On 256 held-out validation pairs, a 172,032-parameter LoRA update to the 14
VLM-to-action projection layers reduces perturbed-to-clean velocity-field MSE
by 10.44% (task-bootstrap 95% interval 9.24--11.75%) with clean drift equal to
0.049% of teacher-field energy. Q/K LoRA reaches 5.72%, V/output LoRA 6.50%,
and a 224-parameter head gate 3.97%. This favors correcting the projection over
changing cross-attention routing in the tested parameterizations.

The result still fails the preregistered behavioral-proxy gate: projection
LoRA improves the four-step Euler endpoint-to-clean error by only 3.45%, not
the required 10%, and the lighting endpoint worsens. No arm was selected for
the sealed test or closed-loop evaluation. Thus we now have a real-alternating
PI mechanism signal at the projector, but not evidence of recovered policy
success. Full details are in
[`PI_FLOW_FIELD_STUDY.md`](../../experiments/dfr_causal_poc/PI_FLOW_FIELD_STUDY.md).

A subsequent sampler-aware study optimized the rank-4 projector LoRA through
all four Euler steps instead of fitting isolated velocity targets. That passed
the validation screen (+10.16% endpoint-to-successful-action improvement) and
replicated on a sealed test (+9.60%, 95% CI [6.04%, 12.88%]). Its full matched
closed-loop result was still null: 96/140 versus the frozen policy's 97/140,
paired CI [-7.86, +7.14] points and McNemar p=1.0; clean was 28/30 for both.
Camera was +2/24 and robot initial state +1/21, but background and language
were each -2, so no nuisance-specific claim is warranted. See
[`PI_UNROLLED_ENDPOINT_STUDY.md`](../../experiments/dfr_causal_poc/PI_UNROLLED_ENDPOINT_STUDY.md).

The paired dataset holds MuJoCo state and proprioception fixed, changes only
the rendering, and uses the source policy's action chunk from a clean
successful trajectory. This is a good counterfactual construction for
background, lighting, sensor noise, and camera appearance.

The offline screen found the following:

| Frozen policy | Best late readout result on held-out perturbed pairs | What it means |
|---|---:|---|
| GR00T causal | residual MSE -27.14%; paired-data advantage +8.22 pp | useful action signal is recoverable in the sampled late representation |
| GR00T encoder | residual MSE -25.18%; paired-data advantage +5.10 pp | same, on fewer eligible tasks |
| PI causal | residual MSE -3.56% | did not pass the offline gate |
| PI encoder | residual MSE -3.37% | did not pass the offline gate |

Important architecture qualification: “GR00T causal” in this table is the
`libero_gr00t_arm_base` checkpoint with the optional two-layer, 32-query
`ReadoutProjector` enabled. Its path is `VLM tokens -> learned readout-query
self/cross-attention -> Action DiT cross-attention`. This is not an inherent
GR00T requirement: the no-readout branch in `QwenGR00T` sends the full final VLM
token sequence directly to the Action DiT. The GR00T recoverability result is
therefore scoped to this readout-bottleneck checkpoint, rather than evidence
for a generic bare `VLM tokens -> projection -> cross-attention` architecture.

“Recoverable” is the right word. It does **not** mean:

- the recoverable direction is the causal feature used by the deployed policy;
- the late head alone is the bottleneck;
- a nuisance was erased;
- or the corrected vector field will remain stable over an action rollout.

This distinction is well established outside robotics. In NLP,
[Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/) found that probe
accuracy need not track behavioral importance. A probe shows information is
available to a chosen decoder; only a controlled intervention plus behavioral
evaluation shows causal use.

## Fresh confirmation: the current method failed

| Policy | Clean success | Appearance-perturbed success | Perturbed change, task-bootstrap 95% interval |
|---|---:|---:|---:|
| GR00T causal | 31/40 -> 30/40 | 63/160 -> 61/160 | -1.25 pp [-8.75, +6.25] |
| GR00T encoder | 15/23 -> 18/23 | 37/92 -> 39/92 | +2.17 pp [-9.78, +14.13] |
| PI causal | 38/40 -> 38/40 | 120/160 -> 120/160 | 0 pp [-5, +5] |
| PI encoder | 39/40 -> 40/40 | 123/160 -> 121/160 | -1.25 pp [-6.875, +3.75] |

All jobs completed cleanly; this is not another LLVM/container failure. All
four miss the predeclared success criteria. The exact result files are:

- [GR00T causal](../../experiments/dfr_causal_poc/outputs/paired_success_groot_causal_v3/confirmation_v1/closed_loop/summary.json)
- [GR00T encoder](../../experiments/dfr_causal_poc/outputs/paired_success_groot_encoder_v3/confirmation_v1/closed_loop/summary.json)
- [PI causal](../../experiments/dfr_causal_poc/outputs/paired_success_p_causal_v3/confirmation_v1/closed_loop/summary.json)
- [PI encoder](../../experiments/dfr_causal_poc/outputs/paired_success_pi_encoder_v3/confirmation_v1/closed_loop/summary.json)

Why did GR00T causal look convincingly positive before? Four settings were
compared on the same validation set and the best clean-noninferior candidate
was selected. Its bootstrap interval was computed for that selected candidate,
not adjusted for selecting the best of four. Closed-loop success is also a
discontinuous outcome: small action changes can flip a marginal grasp. The
fresh confirmation is exactly the safeguard against this selection optimism.
The validation effect was real on that split, but it was not a replicated
generalization effect.

## What counts as a spurious correlation in a policy?

Let the physical task state be `s`, observation be `o = R(s,n)`, nuisance be
`n`, and successful action distribution be \(\pi^*(a\mid s,c)\) for command
`c`. For an appearance intervention `T_n` that leaves the physical state
and command unchanged, the desired condition is

\[
\pi(a\mid o,c) \approx \pi(a\mid T_n(o),c).
\]

This does not require naming the model's internal shortcut. The simulator
intervention operationally defines what should not matter. The nuisance may be
a texture, a combination of lighting and camera, or a distributed latent
direction.

For a physical transformation \(h:s\mapsto s'\), such as moving the object or
changing the robot start, the action will generally change. The right condition
is equivariance or relabelling,

\[
\pi(a'\mid R(s',n),c), \qquad a'=\psi_h(a),
\]

when a known action transform \(\psi_h\) exists; otherwise we need a new expert
or successful rollout. Treating these cases as same-action pairs would train
the wrong behavior. They may still expose memorization or causal confusion,
but they are not appearance invariances.

Purely observational data cannot tell us which unnamed latent factor should be
discarded without assumptions. This is the broader identifiability lesson from
[unsupervised disentanglement](https://arxiv.org/abs/1811.12359). Multiple
environments, counterfactual interventions, failure feedback, or human priors
provide the missing structure.

## Cross-domain evidence: what transfers and what does not

| Family | Evidence outside this experiment | Useful transfer | Critical limit |
|---|---|---|---|
| DFR / AFR / SELF | Last-layer refits work on image and text classifiers; [AFR](https://proceedings.mlr.press/v202/qiu23c.html) and [SELF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/265bee74aee86df77e8e36d25e786ab5-Abstract.html) reduce dependence on group labels | cheap screen when a frozen representation already linearly exposes the core feature | static classification head, direct label, no recurrent/flow dynamics |
| GroupDRO / balancing | [GroupDRO](https://openreview.net/forum?id=ryxGuJrFvS) optimizes worst predefined group loss; simple [group balancing](https://proceedings.mlr.press/v177/idrissi22a.html) is often surprisingly strong | LIBERO-Plus supplies free task/perturbation/severity groups | worst flow loss is not worst rollout success; GroupDRO needs strong regularization and can overfit |
| JTT / AFR / SELF | errors, confidence, or disagreement enrich underperforming groups without train-time group labels | use paired action sensitivity or model disagreement to select trajectories | generic action loss also selects ambiguous states and label noise, not only shortcuts |
| EIIL / XRM / GEORGE | [EIIL](https://proceedings.mlr.press/v139/creager21a.html), [XRM](https://proceedings.mlr.press/v235/pezeshki24a.html), and subclass clustering infer latent groups | cluster trajectory-level failure/sensitivity signatures when the shortcut is unnamed | inferred groups may be difficulty, task phase, or noise rather than a causal nuisance |
| Targeted counterfactual augmentation | [Targeted Augmentations](https://proceedings.mlr.press/v202/gao23g.html) and [counterfactual invariance](https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html) support changing only irrelevant factors | exact same-state LIBERO render pairs are unusually clean supervision | indiscriminate invariance removes useful domain-dependent information |
| Contrastive representation tuning | [Correct-N-Contrast](https://proceedings.mlr.press/v162/zhang22z.html) changes the representation, not only the classifier | pull same-action counterfactuals together and keep different-action states apart | positive-only alignment can collapse action-relevant geometry |
| LEACE / INLP | [LEACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) removes a labelled concept from all linear decoders with minimum affine distortion | good layerwise causal diagnostic with rank-matched random controls | needs concept labels; only linear erasure; repeated surgery shifts activations; no robustness guarantee |
| CLIP/VLM debiasing | language-described nuisance contrastive tuning improves CLIP worst-group accuracy; [RaVL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/95c6ae3f3393786203a4b6dcb9df1036-Abstract-Conference.html) discovers region-level shortcuts | intervene upstream at visual tokens/projector; language can specify known nuisance hypotheses | still single-shot classification; localizable regions are easier than distributed control shortcuts |
| LLM activation steering | methods such as [SteerFair](https://arxiv.org/abs/2406.03631) infer directions from semantically equivalent prompt variants | supports the idea of using transformation pairs rather than explicit attribute labels | QA/generation evidence; steering can trade off capabilities and has no sequential stability guarantee |
| Causal imitation / visual RL | [Causal Confusion](https://arxiv.org/abs/1905.11979) shows low imitation loss can coexist with poor deployed reward; task-informed/world-model abstractions suppress distractors | use interventions, transitions, reward, and recovery labels—not representation correlation alone | substantially more supervision/compute; history and appearance may be genuinely task relevant |
| World models | [Denoised MDPs](https://arxiv.org/abs/2206.15477) separate controllable/reward-relevant state from exogenous noise; recent WAMs add video-dynamics priors | temporal prediction can favor objects, motion, and controllable structure over pixels | a comparative 2026 preprint finds WAMs strong but not uniformly better than VLAs; architecture, data, and compute are confounded |

The broad lesson is consistent: methods work when their supervision specifies
which differences should be ignored and which task signal must be preserved.
“Find a decodable direction and subtract it” is rarely sufficient.

## Lessons from CV, VLMs, LLMs, and general deep learning

### Computer vision and standard deep networks

DFR is stronger than “a trick for simple CNNs”: its original study includes
ResNet-50 and an appendix ViT-B/16 result. But all of these are static
classification problems. The reliable recurring pattern across DFR, CnC, and
targeted augmentation is that ERM may retain core and nuisance features, while
balanced or counterfactual supervision changes which features the predictor
uses.

There is no universal algorithmic winner. [DomainBed](https://openreview.net/forum?id=lQdXeXDoWtI)
found carefully tuned ERM competitive with or better than a large collection of
domain-generalization methods under matched evaluation. Later group-robustness
work likewise finds rankings depend on class/group imbalance and stopping
time. Therefore our necessary baselines are ordinary balanced mixed training,
strong regularization, and a genuinely held-out model-selection protocol.

Pretrained-model adaptation adds another warning. Full fine-tuning can distort
useful pretrained features; [LP-FT](https://arxiv.org/abs/2202.10054) and
[WiSE-FT](https://arxiv.org/abs/2109.01903) improve shifted classification by
warm-starting the head and/or interpolating with the pretrained weights. For a
VLA this does not guarantee interpolated success—the policy is nonlinear and
closed loop—but it motivates identity initialization, clean distillation, and
a post-training base/adapted weight interpolation sweep.

### Vision-language models

VLM results support intervening at visual tokens or cross-modal alignment,
rather than only at the final output. A CLIP study uses language descriptions
of known spurious relations plus a multimodal contrastive loss and reports
large Waterbirds worst-group gains on both ResNet and ViT backbones
([Yang et al.](https://arxiv.org/abs/2304.03916)).
[RaVL](https://proceedings.neurips.cc/paper_files/paper/2024/hash/95c6ae3f3393786203a4b6dcb9df1036-Abstract-Conference.html)
instead clusters region-level features to discover local shortcuts and then
fine-tunes with a region-aware loss. These are valuable mechanisms, but their
labels are class logits and their shortcuts are often localizable image
regions; neither result establishes continuous-action robustness.

For us, the analogue of “same class, different shortcut” is “same successful
action at the same physical state, different rendering.” The analogue of the
contrastive negative is a genuinely different physical state or task phase
requiring a different action. This negative/geometry-preserving term matters:
pulling all appearances together without preserving action differences can
collapse useful state information.

### Language models

LLM work separates three questions that our initial experiment blurred:

1. **Detection:** can a probe decode a concept or bias?
2. **Intervention:** can that information be selectively changed?
3. **Behavior:** does the change improve the desired output without collateral
   damage?

INLP and LEACE address the second question for named, linearly represented
concepts. Activation-steering methods such as
[SteerFair](https://arxiv.org/abs/2406.03631) use semantically equivalent prompt
variants to infer nuisance directions without labelled examples. This is
conceptually close to our paired renders: the transformation family supplies
weak supervision even when the internal latent correlation is unnamed. But
LLM evidence is mainly QA, classification, and text generation. It offers no
reason to expect a fixed steering vector to remain stable when its output is
recurrently integrated into the next policy input.

The practical import is to keep probe/LEACE studies as diagnostics, and learn
small adapters with the actual task objective. A convincing erasure study would
need nuisance probe AUC at chance, a matched random-direction intervention,
unchanged clean success, and improved fresh perturbed rollouts.

### World models and sequential learning

World-model work offers a different route: define useful state by dynamics,
controllability, and reward, not by pixel reconstruction. Task-Informed
Abstractions and [Denoised MDPs](https://arxiv.org/abs/2206.15477) explicitly
factor reward-relevant/controllable state from exogenous distractors. In
imitation learning, [Causal Confusion](https://arxiv.org/abs/1905.11979) shows
why this matters: held-out behavioral-cloning loss can be good while deployed
reward is poor, and targeted environment interventions or expert queries are
needed to identify the right mechanism.

This argues for future-state/action/reward auxiliaries if paired supervised
tuning fails. It does not imply that an image-generating world model is
automatically robust. Current WAM comparisons mix different pretraining data,
model sizes, action heads, and inference budgets; they are evidence that video
priors can help, not a causal isolation of why.

## The strongest VLA-specific evidence

### 1. RobustVLA: closest accepted method

[RobustVLA](https://arxiv.org/abs/2510.00037), published at ICLR 2026, combines:

- input consistency under task-semantics-preserving perturbations;
- robust flow matching against worst-case action noise; and
- UCB selection of the currently harmful perturbation family.

It reports +12.6 absolute points on pi0 and +10.4 on OpenVLA across 17
perturbations. This is much closer to our setting than DFR because it optimizes
the flow policy during training and evaluates closed loop. We should copy the
input-consistency and perturbation-selection pieces first. Its output
adversarial training is a separate ablation, not necessary for the initial
appearance POC.

### 2. Cross-view consistency: closest minimal experiment

The August 2026 [Cross-View Action Consistency](https://arxiv.org/abs/2608.06965)
preprint resets LIBERO demonstrations to identical MuJoCo states, renders two
camera views, gives both the same action-flow target, and aligns predicted
velocities at the same sampled flow coordinate. On the authors' LIBERO-Plus
camera track it reports 79.8% for flow-matching only on the paired data versus
87.2% with explicit consistency; shuffled pairs collapse to 25.8%. The matched
pair-exposure control is especially important: it separates the value of more
data from the value of correct pairing.

This is almost exactly the experiment we should implement. It is, however, a
new unreviewed preprint and camera-only evidence, so we should reproduce it
rather than treat it as established.

The adapter location is also literature-backed. The accepted CVPR 2026 paper
[VLA Models Are More Generalizable Than You Think](https://openaccess.thecvf.com/content/CVPR2026/html/Li_VLA_Models_Are_More_Generalizable_Than_You_Think_Revisiting_Physical_CVPR_2026_paper.html)
reports camera success increasing from 48.5% to 87.1% with only about 4K
visual-token affine parameters; a low-rank ViT adapter reaches 90.8%. It uses
target-view adaptation rather than our paired-invariance protocol, but it is
strong evidence that changing spatial/visual features upstream is more
plausible than repeatedly correcting a late action vector field.

### 3. LIBERO-Plus mixed fine-tuning: mandatory data baseline

The accepted [LIBERO-Plus CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html)
uses more than 20,000 successful trajectories for mixed fine-tuning and reports
79.6% aggregate success, including large camera and noise gains. It also finds
that joint perturbations are not decomposable. Any proposed invariant method
must compare against ordinary clean+perturbed mixed training and eventually
test unseen compositions.

### 4. QuoVLA: conceptually close, empirically much larger

[QuoVLA](https://arxiv.org/abs/2605.24890) explicitly argues that pretrained
VLM latents are action-sufficient but overcomplete, and tries to collapse
action-equivalent inputs into a quotient representation. That is closely
aligned with our hypothesis. But QuoVLA inserts a learned prefix bottleneck
before the action expert, quantizes it, trains dual action branches, and trains
the system on the task data. It is not a tiny post-hoc late correction.

As of 7 September 2026, QuoVLA is an arXiv preprint; I found no official
conference acceptance. Its ideal quotient is defined by equality of the
unknown optimal action law, while 8-bit quantization is a learned heuristic
proxy. It is a valuable architectural direction, not confirmation of our gate.

### 5. World action models: useful signal, not a magic solution

A recent comparative [WAM robustness study](https://arxiv.org/abs/2603.22078)
reports Cosmos-Policy at 82.2% on LIBERO-Plus, but pi0.5 at 85.7%. This suggests
video dynamics priors can help, especially with visual noise, but WAMs are not
uniformly more robust and are more expensive. The clean scientific route is to
first add a future/transition-aware auxiliary objective or action-conditioned
temporal consistency—not replace the entire policy with a WAM based on one
benchmark comparison.

## Representation change versus attention failure

The robust-action hypothesis should **not** require clean and perturbed visual
tokens to be equal. Lighting, texture, and sensor changes will alter token
values; a camera change may also move object evidence between spatial tokens.
The useful requirement is weaker:

> Conditioned on the perturbed tokens, the frozen policy pathway still contains
> enough physical-state information for a small action-side mechanism to recover
> the successful clean action distribution.

There are at least three distinct failure modes:

1. **Encoding failure:** task-relevant geometry is absent or too corrupted in
   the VLM tokens.
2. **Routing failure:** useful information is present, but action queries attend
   to or weight the wrong tokens or heads under perturbation.
3. **Late readout/dynamics failure:** information reaches the Action DiT, but its
   velocity readout or iterative flow integration uses it incorrectly.

Our current spectral gate intervenes before PI cross-attention on every
layerwise projected VLM state,

\[
M'_l=M_l-U_lD_lU_l^\top(M_l-\mu_l).
\]

It therefore changes the keys and values seen by the action blocks and changes
attention *indirectly*. It does not learn action-query routing, key/query
projections, per-head selection, or attention logits. Its negative four-seed
confirmation rejects this fixed nuisance-subspace shrinkage rule, not the
broader attention-routing hypothesis. The earlier conditional tokenwise gate
also lost substantial clean success, showing that a flexible input-dependent
gate can be destructive without a strong clean constraint.

### What the experiments currently support

- Nuisance probes show that background, lighting, and sensor-noise information
  is present in several frozen P-causal layers. This establishes nuisance
  decodability, not preservation of the correct physical-state feature.
- The rank-one sensor-noise deletion moved noisy actions about 14% toward clean
  actions above a random-direction control. This is narrow causal evidence that
  one representation direction affects action usefully; background and lighting
  deletion moved actions in the wrong direction.
- Late GR00T states support linear residual recovery under teacher forcing:
  27.14% and 25.18% perturbed residual-MSE reductions for causal and encoder
  GR00T. The paired-data advantages of 8.22 and 5.10 points make a purely generic
  action prior less likely, but do not prove that the same semantic feature is
  preserved or that attention is wrong.
- PI causal and PI encoder did not pass the same offline screen. We therefore
  currently have no corresponding representation-sufficiency evidence for PI.
- Every closed-loop confirmation is negative: the capped late residual fails on
  all four policies, and the larger static-gate replication is below warm-only,
  random-gate, and source controls on mean appearance success.
- The mask-probe result shows off-suite object information is recoverable for
  the mask objective. Shuffled-mask/action controls do not show that this object
  information improves the deployed action policy.

Thus the supported statement is narrow: **some perturbation and action-residual
information is decodable in frozen representations, especially late GR00T
states.** We have not shown that the correct action-sufficient representation
survives every perturbation, that attention is the mechanism of failure, or that
any tested gate improves robustness.

### Causal attention diagnostic before training another gate

For exact same-state pairs, use the same noisy action and flow time and inspect
one cross-attention block at a time. Write its output as `A V`, with clean and
perturbed quantities `A_c`, `V_c`, `A_p`, and `V_p`. Compare the normal
perturbed output `A_p V_p` against two hybrid interventions:

- `A_c V_p`: clean routing with perturbed value content;
- `A_p V_c`: perturbed routing with clean value content.

Measure reduction in clean-versus-perturbed velocity disagreement over flow
time and, for promising layers only, final denoised-action discrepancy. If
`A_c V_p` helps substantially while `A_p V_c` does not, routing is implicated;
the reverse implicates token/value content. If only `A_c V_c` works, the two
effects interact and an attention-only adapter is unlikely to suffice.

Direct equality of attention maps is inappropriate for camera changes because
correct attention may move to a different image location. Run the hybrid test
first on lighting, texture, and sensor noise, where token correspondence is
meaningful; use only action/velocity consistency for camera shifts.

Only after this diagnostic supports routing should we train an attention-side
adapter. The cleanest test freezes the VLM and Action DiT and updates separate
tiny LoRA modules on cross-attention Q/K (routing) versus V/O (content), or
per-head identity-initialized residual gates. Train with paired velocity-field
consistency along the clean flow path, clean-policy distillation, and an
identity/parameter anchor. A per-head scalar gate alone is risky: it can appear
robust merely by suppressing vision and replaying task priors, so evaluation
must include visually similar states requiring different actions.

## Recommended next POC

### Hypothesis

For exact same-state appearance pairs, training the *existing policy* to emit
the same velocity field will improve fresh closed-loop robustness beyond the
gain from merely exposing it to both renderings.

### Model and scope

Start with **GR00T causal, camera viewpoint only**:

- it had the strongest offline recoverability signal;
- fresh camera success is only 7/40 (17.5%), leaving clear headroom;
- and restricting to one nuisance makes the result interpretable.

If this passes, extend to all four appearance groups and then PI causal. Do not
start with a full four-model grid.

Train a warm-start, identity-initialized adapter in the visual-token/projector
or VLM-to-action fusion path. A small LoRA/affine visual-token modulation is the
first choice. Unfreeze the final action projection and at most the last one or
two action blocks only if the upstream-only adapter cannot fit. Do not train an
action head or diffusion model from scratch.

### Objective

For clean observation `o`, appearance transform \(T_g(o)\), successful action
chunk `a`, shared noisy action \(x_t\), and shared flow time `t`:

\[
\begin{aligned}
L ={}& L_{FM}(v_\theta(x_t,t,o),u)
 + L_{FM}(v_\theta(x_t,t,T_g(o)),u) \\
&+ \lambda\,\|v_\theta(x_t,t,T_g(o))
       - \operatorname{sg}[v_{\theta_0}(x_t,t,o)]\|_2^2 \\
&+ \beta\,\|v_\theta(x_t,t,o)
       - v_{\theta_0}(x_t,t,o)\|_2^2
 + \mu\|\theta-\theta_0\|_2^2 .
\end{aligned}
\]

The first two terms are ordinary paired mixed fine-tuning. The third transfers
the base policy's clean successful behavior to the perturbed view. The fourth
and fifth protect clean behavior. Use the same flow noise/time within each
pair; otherwise the consistency term contains sampling noise.

For later multi-group training, either sample groups uniformly or use a UCB /
GroupDRO weight based on paired excess loss

\[
\Delta_g = L_{FM}^{g,perturbed}-L_{FM}^{paired,clean}.
\]

Keep clean as its own protected group. The optimizer may use this surrogate,
but model selection remains closed-loop success.

### Data reality

The current GR00T-causal capture has 7,680 pair records total: 2,560 train,
2,560 validation, and 2,560 test. Each record contains a 16-step action chunk,
so the training split has 40,960 action-token targets. Those tokens are not
40,960 independent state-action examples, and this is not an 80k-demonstration
dataset. It was adequate for a 7k--21k-parameter affine residual; it is small
for broad action-expert fine-tuning. That is another reason to use a compact
adapter first.

For the minimal camera POC, use the existing train trajectories and generate
more render variants from the same states if cheap. More renderings improve
nuisance coverage but do not increase physical-state diversity. If the small
adapter overfits, capture more successful source trajectories before increasing
the trainable parameter count.

### Required arms

1. frozen base policy;
2. paired mixed flow fine-tuning, no explicit consistency;
3. the same model/data with explicit paired consistency;
4. shuffled-pair negative control for an offline/small-rollout sanity check.

Only arm 3 beating arm 2 on fresh closed-loop camera success supports the
pairing/invariance mechanism. Both 2 and 3 beating the base only shows that
perturbation exposure helps. The shuffled arm verifies that arbitrary feature
collapse is not the source of a gain.

### Split and decision rule

- Split by source trajectory before generating render pairs.
- Hold out at least one camera recipe/severity and all corresponding render
  seeds for confirmation.
- Use validation only for adapter location, \(\lambda\), and stopping.
- Run confirmation once on unused successful source trajectories.
- Primary endpoint: paired closed-loop perturbed success, clustered by task.
- Safety endpoint: clean success no worse than -5 points.
- Suggested pass: at least +5 perturbed points over the mixed-fine-tune control,
  task-cluster 95% lower bound above zero, and clean noninferiority.

Every Booster node must run four one-GPU shards, with global batch size and
optimization semantics unchanged.

## What to do for layout and robot-start failures

Do not reuse the same action labels. Use one of:

- simulator relabelling or fresh successful rollouts;
- DAgger/DART-style recovery collection;
- a known geometric action transform when coordinate equivariance is exact;
- or, after the supervised POC, closed-loop policy optimization such as
  [DPPO](https://arxiv.org/abs/2409.00588) with a clean-policy constraint.

These data ask a different question: not “ignore appearance,” but “recompute
the action from the changed physical state.” Failure trajectories help identify
where to collect or upweight data, but without a correct action or reward they
do not by themselves teach recovery.

## Should we continue this direction?

Yes, for **one redesigned POC**. Stop the late residual/gate line as a
performance method. Keep it as a negative result and diagnostic evidence.

The go/no-go sequence should be:

1. reproduce the camera paired-consistency effect with a small upstream adapter;
2. require improvement over matched paired mixed fine-tuning, not only over the
   frozen policy;
3. if positive, add background/light/noise with balancing or UCB and test unseen
   compositions;
4. if only offline flow loss improves again, stop supervised latent-invariance
   work and move to recovery-labelled data or closed-loop optimization;
5. consider QuoVLA-style bottlenecks or world-model auxiliaries only as the
   larger follow-up, not the next minimal experiment.

The project-worthy claim is no longer “last-layer DFR fixes VLA shortcuts.” A
defensible claim would be: **exact action-equivalent counterfactual pairing plus
selective policy adaptation improves closed-loop robustness, and the benefit is
separable from simple augmentation exposure.**

## Active PI flow-field locus screen (2026-09-07)

The next diagnostic now uses the intended PI architecture rather than the
GR00T checkpoint's optional readout-token bottleneck. It compares a direct
projector LoRA against three genuinely different Action-DiT cross-attention
interventions: Q/K LoRA (routing), V/output LoRA (transported content), and a
448-parameter per-head gate. The frozen clean PI field is the target at the
same `x_t` and `t` on the bridge to an action chunk taken from a successful
clean trajectory. Clean preservation is explicit and the test split is sealed.

The exact hypothesis, arms, data counts, metrics, and decision rule are in
[PI_FLOW_FIELD_STUDY.md](../../experiments/dfr_causal_poc/PI_FLOW_FIELD_STUDY.md).
