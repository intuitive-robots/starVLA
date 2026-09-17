# Probe insights for the encoder paper — 2026-09-17

**Use probes to choose interventions, not as a substitute for policy success. No readout is currently validated as a cheap success proxy.** The only matched LIBERO-plus policy triplet gives negative rank correlation for every measured readout. RoboCasa has only two checkpoint-matched policies with these readouts.

The regenerated record contains 469 raw records. Each quoted number below is recomputed from that JSONL or its identified raw ACTPROBE source, not the markdown or HTML. The intended external/ext+wrist 11-arm × 9 dataset/camera grid has 66/99 cells with both patch and token readouts. The missing 33 are LIBERO10/LIBERO-plus/RoboCasa ext+wrist cells: the completed full-table jobs actually used wrist-only cameras in those slots. Those 33 wrist-only cells exist but cannot fill ext+wrist blanks. The two flagged DROID ext+wrist cells have now landed (job 1855867 COMPLETED, 25m01s); all 99 actually requested launcher cells have results. Thus the historical 82/90 statement is stale and uses a different denominator. The weight 0.01 tracetime arm is separate and has pooled probes but not a complete 11-readout grid.

## Matched policy probes and actual benchmark scores

Final_model and named evaluation checkpoints have matching tensor-storage CRC/size inventories for the three LIBERO-plus policies (20k) and two RoboCasa seed 4 policies (50k). This is a strong content check without loading weights onto a GPU; it is not a cryptographic hash of every tensor. RoboCasa seed 42 probes used **40k**, while benchmark JSONs use **50k**, so that seed is excluded from checkpoint-matched correlation. [Join audit](results_collected/probe_policy_linkage.csv). All these probes use the external camera only and omit proprioception (build_qwenvl_inputs receives only images/instructions), while RoboCasa rollouts use external+wrist+state. Thus checkpoints match, inference inputs do not. n_test=960 frames from held-out episodes; do not treat frames as independent episodes.

| probe / policy | held-out n | pooled standardized R² | mean patch R² | best patch R² | top16 R² | rollout % ± SE; n | raw probe |
|---|---|---|---|---|---|---|---|
| starvla_lplus_causal | 960 | 0.3673 | 0.2410 | 0.2653 | 0.4042 | 72.500 ± 0.706; 4000 | [svlap_lplus_causal.log](/e/project1/m3/blank4/code/train_downstream/probe_logs/svlap_lplus_causal.log) |
| starvla_lplus_encdec | 960 | 0.4523 | 0.4208 | 0.4277 | 0.4709 | 72.025 ± 0.710; 4000 | [svlap_lplus_encdec.log](/e/project1/m3/blank4/code/train_downstream/probe_logs/svlap_lplus_encdec.log) |
| starvla_lplus_encoder | 960 | 0.4103 | 0.3750 | 0.393 | 0.4366 | 76.425 ± 0.671; 4000 | [svlap_lplus_encoder.log](/e/project1/m3/blank4/code/train_downstream/probe_logs/svlap_lplus_encoder.log) |
| starvla_casa_causal_s4 | 960 | 0.0423 | 0.0229 | 0.0712 | 0.1286 | 36.275 ± 1.683; 816 | [svlap_casa_causal_s4.log](/e/project1/m3/blank4/code/train_downstream/probe_logs/svlap_casa_causal_s4.log) |
| starvla_casa_encdec_s4 | 960 | 0.28 | 0.1321 | 0.2562 | 0.3477 | 37.255 ± 1.693; 816 | [svlap_casa_encdec_s4.log](/e/project1/m3/blank4/code/train_downstream/probe_logs/svlap_casa_encdec_s4.log) |

## The 11 readouts ranked against benchmark success

The four observed LIBERO-plus readouts tie at Spearman −0.5 (3 arms); the other seven are unmeasured on the matched policies and cannot be ranked. Pearson is supplied only as a descriptive check. An exact two-sided permutation test on three ranks cannot reject the null here (p=1). At two RoboCasa arms, correlation ±1 is algebraic and supplies no validation. The original `enc` readout is unstandardized and must not be silently substituted for `patch_meanref`; the latter is the standardized mean column used here. Hyperparameter details may differ from pooling-sweep `mean`, so the prospective proxy study must rerun one frozen probe protocol.

| readout | LP rank / coverage | LP Pearson r | LP Spearman ρ | RC coverage / r, ρ |
|---|---|---|---|---|
| enc | unrankable / 0 | NA | NA | 0 / NA |
| mean | tie / 3 | -0.092 | -0.500 | 2 / 1.000,1.000 (uninformative) |
| per-view | unrankable / 0 | NA | NA | 0 / NA |
| grid | unrankable / 0 | NA | NA | 0 / NA |
| attn | unrankable / 0 | NA | NA | 0 / NA |
| patch mean | tie / 3 | 0.177 | -0.500 | 2 / 1.000,1.000 (uninformative) |
| patch best | tie / 3 | 0.219 | -0.500 | 2 / 1.000,1.000 (uninformative) |
| patch top16 | tie / 3 | -0.115 | -0.500 | 2 / 1.000,1.000 (uninformative) |
| state tok | unrankable / 0 | NA | NA | 0 / NA |
| instruction | unrankable / 0 | NA | NA | 0 / NA |
| first patch | unrankable / 0 | NA | NA | 0 / NA |


Do not attach the same backbone probe to all its fine-tuned policy heads: those are different final weights and repeated copies would inflate sample size. The full_epoch/dense/trace/final_action grid has spatial/VQA evaluations, not matching LIBERO-plus or RoboCasa rollouts for every cell. [Correlation CSV](results_collected/probe_benchmark_correlations.csv). To find a usable proxy: freeze the 11 readouts and held-out episodes; probe ≥8 final policy variants including shared-z, withhold entire training seeds for validation; require positive leave-one-family-out rank correlation and prospective ordering of two new policies. Until then screen benchmark success, not R².

## Where action is decodable, and what to change

**Depth is not measured.** Neither the JSONL nor the patch page contains a layer sweep. `scripts/starvla_patch_probe.py` explicitly reads hidden_states[-1] (encoder output for enc-dec, last layer for causal). These files cannot establish “available earlier.” A frozen layer 4/8/14/20/28 readout sweep on the matched policies is the needed measurement; only if a middle layer predicts rollout improvements should an auxiliary head be attached there.

**Spatial availability is broader in the encoder, but broad availability alone does not improve control.** The matched LIBERO-plus bidirectional arm has mean-patch R² 0.4208 versus causal 0.2410, and best/pooled ratio 0.946 versus 0.722, yet its policy is 0.475 pp worse in seed 42. Mechanism hypothesis: bidirectional mixing broadcasts a global state/action correlate to many token positions, while the policy head fails to extract additional control-relevant information. Experiment: compare shared-z query readout against whole-memory and equal-capacity pooled readout on the same frozen encoder; predict gains in robot-init/layout, not a uniform gain in every category.

**RoboCasa’s visual signal is much stronger in the encoder despite near-equal success.** Seed4 pooled R² 0.2800 versus 0.0423 and mean-patch0.1321 versus 0.0229 imply that a linear head can extract substantially more action-related information. Seed42 at 40k repeats the direction (0.2699 versus 0.0382) but is not a 50k matched-success comparison. Mechanism hypothesis: useful visual encoding is present but the readout/training distribution is limiting closed-loop recovery. Experiment: shared-z or learned query readout on RoboCasa, with held-out task-level recovery tests; predict a larger gain on mobile/layout-recovery tasks than on already easy articulation tasks.

**The strongest-looking causal CoT arm has an action-conditioning path that our encoder reasoning arms lack.** In `libero_plus_qwen08b_gr00t_cot_trace_ours_v3_cotw01`, training feeds teacher-forced trace-token hidden states to GR00T and rollout generates a trace, re-encodes it, and feeds the resulting token sequence to the action head. Its learned GR00T readout projector is disabled. By contrast, the encoder-decoder U/V arms compute decoder CoT loss but `QWen3_EncDec.forward()` returns only encoder states to PI; the action head never sees decoder reasoning. Their75.975%/75.875% full-protocol scores therefore do not test explicit plan conditioning. The partial CoT reevaluation is still running and cannot yet be called a4k result. Mechanism hypothesis: the encoder already represents the scene broadly, but an ordered decoder plan supplies the task-specific interface that the action head fails to extract. Experiment: first compare clean/prompt-only/wrong-trace inference on the causal checkpoint; if trace content matters, concatenate encoder memory with generated decoder-trace states in a matched encoder-decoder GR00T arm. Predict the largest gain in spatial/object-layout categories. A null wrong-trace effect would identify the apparent mechanism as an artifact of a second pass or extra tokens.

**Token position is a readout opportunity, not modality attribution.**

| raw record | n_test | visual mean R² | first patch | state tokens | instruction |
|---|---|---|---|---|---|
| full_epoch_tclass_bridge | 2400 | 0.1305 | 0.1141 | -0.0011 | 0.1814 |
| full_epoch_tclass_droid | 2399 | 0.103 | 0.0983 | -0.0018 | 0.1201 |
| final_action_dec_tclass_bridge | 2400 | 0.1718 | 0.1696 | 0.1997 | 0.1844 |
| trace_reverse_tclass_bridge | 2400 | 0.2035 | 0.1989 | 0.2235 | 0.2103 |


All records above are in docs/probe_results.jsonl with state_in_prompt=true, state_mode=tokens. Bridge n_train5592/n_test2400; DROID 5600/2399, different K8 versusK15. Mechanism hypothesis: text positions offer a useful integrated summary and learned state tokens become usable when trained, but contextualized “state tokens” also contain visual information. Experiment: a small trainable query pooling text+vision, compared to vision-only under equal parameter count, with a format-preserving state ablation; predict modest gains if information routing is limiting. A state-token R² above raw-state R² does not demonstrate extra proprioceptive information.

**DROID wrist addition weakens all five pooled readouts for the full-epoch model.**

| record | n_test | enc | mean | per-view | grid | attn | raw state |
|---|---|---|---|---|---|---|---|
| full_epoch_poolsim_droid_ext | 2399 | 0.0967 | 0.1022 | 0.1022 | 0.0701 | 0.0631 | 0.089 |
| full_epoch_poolsim_droid_wrist | 2399 | 0.0701 | 0.0757 | 0.0683 | 0.0538 | 0.0427 | 0.089 |
| final_action_dec_qsim_droid_ext | 2399 | 0.1213 | 0.1365 | 0.1365 | 0.1173 | 0.1111 | 0.089 |
| final_action_dec_qsim_droid_wrist | 2399 | 0.113 | 0.1239 | 0.1147 | 0.1042 | 0.1038 | 0.089 |


The state baseline stays0.0890 and sample counts match, unlike the old missing-camera-index bug. Full-epoch standardized mean falls0.1022→0.0757; final_action_dec falls0.1365→0.1239. Mechanism hypothesis: camera fusion or sample-limited regularization dilutes the external signal; these are ext+wr versus ext, not wrist-only measurements, so they do not prove the wrist is useless. Experiment: camera-specific normalization/query tokens and train-time single-camera dropout, then ext/wrist/ext+wr policy evaluation under camera perturbations. Keep the wrist: the published OFT benchmark reports a substantial wrist benefit, and our DROID probes do not transfer automatically to LIBERO.

**The state diagnostic supports format sensitivity, with a narrower robustness benefit.** On the same three RoboCasa tasks, causal clean 100/144 (69.44%)→no-state 0/144; encoder 105/144 (72.92%)→85/144 (59.03%). With format retained, shuffled values leave causal 94/144 versus encoder 87/144; random 76/144 versus 90/144; zero 22/144 versus 44/144. Mechanism hypothesis: deleting the suffix disrupts causal token routing; both policies also depend on state values, especially off-manifold zero values. Experiment: separately retain [STATE]/[ACTION] delimiters with a missing-value marker, train modest state dropout in both arms, and serialize actual prompts. Predict that much of causal’s no-state collapse disappears with format repair; require a remaining encoder advantage across ≥2 seeds before claiming proprioception robustness.

**Patch brightness is not localized causal use.** DROID full_epoch mean patch0.0100→final_action_dec0.0850, best patch0.1055→0.1252, standardized mean 0.1030→0.1367. Mechanism hypothesis: action co-training distributes a global summary rather than making object pixels uniquely informative. Experiment: keep readout tokens fixed and intervene on original pixels (camera shift, border occlusion versus equal-area object occlusion), then assess task success. Do not train a border-suppression loss from decodability alone. The HTML’s claim that empty-border decodability proves a visual shortcut is unsupported by a contextualized ViT/encoder map.

**More decodable features can coincide with worse non-policy downstream scores.** The raw evaluation files confirm the following spatial/VQA aggregate (mean of 12 score components, accuracy or exp(−DFD)); this is neither action success nor a third behavior-generation benchmark.

| model | recomputed aggregate | coverage | source |
|---|---|---|---|
| enc_dec_2b_v5_tb9216_final | 0.690442 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_tb9216/2026-09-13_04-43-47_job1770825/eval/final/all_benchmarks_enc_dec_2b_v5_tb9216_final.json) |
| enc_dec_2b_v5_final_aux_dense | 0.671040 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_aux_dense/2026-09-15_15-14-53_job1808280/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_aux_dense.json) |
| enc_dec_2b_v5_aux_dense | 0.677429 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_aux_dense/2026-09-13_00-05-26_job1772529/eval/all_benchmarks_enc_dec_2b_v5_aux_dense.json) |
| enc_dec_2b_v5_aux_base | 0.673392 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_aux_base/2026-09-13_00-05-23_job1772526/eval/all_benchmarks_enc_dec_2b_v5_aux_base.json) |
| enc_dec_2b_v5_aux_trace_reverse | 0.611335 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_aux_trace_reverse/2026-09-15_19-16-21_job1817045/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_aux_trace_reverse.json) |
| enc_dec_2b_v5_final_action_dec | 0.679779 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_action_dec/2026-09-16_11-33-18_job1826988/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_action_dec.json) |
| enc_dec_2b_v5_final_action_head | 0.683575 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_action_head/2026-09-16_11-33-18_job1826989/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_action_head.json) |
| enc_dec_2b_v5_final_action_linear | 0.677859 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_action_linear/2026-09-16_17-07-28_job1830054/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_action_linear.json) |
| enc_dec_2b_v5_final_action_tracetime | 0.639572 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_action_tracetime/2026-09-16_17-18-30_job1837546/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_action_tracetime.json) |
| enc_dec_2b_v5_final_action_tracetime_w001 | 0.666865 | 12/12 | [raw JSON](/e/project1/m3/blank4/code/train_downstream/train/outputs/enc_dec_2b_v5_final_action_tracetime_w001/2026-09-17_09-34-40_job1850693/eval/samples-500k/all_benchmarks_enc_dec_2b_v5_final_action_tracetime_w001.json) |


Final dense continuation decreases the full-epoch aggregate0.690442→0.671040. Tracetime weight 0.01 improves over weight 0.1 (0.666865 versus 0.639572) but still trails the starting checkpoint. Mechanism hypothesis: aggressive auxiliary objectives redirect capacity away from general visual-language behavior. Experiment: bounded auxiliary weights0.001/0.01 with no-head continued-training control, and a matched rollout evaluation; kill if probe improves but policy success does not. A no-head continuation is necessary even when data were previously seen: optimization time and repeated exposure remain confounds.

## Measurement and narrative discrepancies

- Fixed collector identity collisions: historical `starVLA` patch-only records overwrote one another, and absent dataset defaulted incorrectly to Bridge. Recover from source-log basename and cache; use a full measurement fingerprint. The repaired record preserves differing historical targets rather than pretending they are replicates.
- Fixed camera grouping: single wrist-only views were incorrectly classified as external. Rows now retain camera identity, checkpoint, state mode, target dimensions/K and sample counts; prefix matching no longer folds dense_w10 into dense. The requested combined-camera grid remains incomplete even though all launcher cells completed.
- Removed the hard-coded `bench` column from build_full_table.py: it was a spatial/VQA aggregate, not LIBERO-plus/RoboCasa success. Split weight 0.01 tracetime from weight 0.1; prefix matching previously mixed them. Table cells remain an inventory, not permission to combine state modes, targets or checkpoints.
- The HTML labels pooled `patch_meanref` as “mean patch”; true mean patch is the mean of R2_act_patch_map and is much smaller in several arms. The page’s rollout values0.769/0.732/0.728 disagree with canonical raw values0.76425/0.725/0.72025.
- docs/STATUS.md “only enc-dec clears proprioception” holds only for the stated frozen, untrained-causal comparisons and original text-state protocol. It is not a conclusion about trained causal policies, and the quantized-token protocol gives different results.
- docs/STATUS.md final dense run and trace continuation are no longer pending; full raw evaluations exist. Dense’s early0.677429 does recompute, but calling dense the best aux policy is unsupported because that score is a VQA/spatial aggregate.
- The state-baseline, action-slice and ridge-standardization bugs described in STATUS.md remain real caveats. Historical records without explicit dimensions/state mode cannot be silently joined to corrected runs. RoboCasa n_test960 with the corrected7D end-effector/gripper target must be checked independently from 12D loader order.
- starvla_patch_probe.py normalizes action targets with full-data mean/std before splitting. Per-dimension R² is affine-invariant, but any shared hyperparameter selection using normalized MSE should be rerun with train-only target statistics. This is a measurement caveat, not proof that all scores are invalid.
- No per-layer results, no matched 11-readout shared-z policy sweep, and no matching50k seed 42 policy probes exist in the requested record. These block earlier-layer claims, a full correlation ranking, and seed-robust proxy validation.
- The full-table patch grids for Bridge/DROID are stored as [300,1]/[576,1], not trustworthy 2D coordinates. Geometry needs original image_grid_thw and camera token offsets; the HTML heatmap is insufficient evidence for exact patch coordinates.

The probes prioritize shared-latent/readout ablations, modest auxiliary weights, camera fusion and prompt-preserving state tests. They do not independently support the encoder-paper claim.
