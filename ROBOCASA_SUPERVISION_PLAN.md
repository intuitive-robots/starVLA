# RoboCasa supervision — September 17, 2026

LIBERO supervision is available; equivalent RoboCasa labels are not yet generated. Reuse the target interface and derive labels from simulator replay before spending training time on the supervised transfer. Free-form reasoning prose is not required by shared-z. The two-pass experiment specifically requires a predicted five-point trace that becomes an action input. To test the winning causal mechanism faithfully, use its remaining-motion **object** trace; a chunk-local gripper trace is a different target and must be a named ablation.

## Verified inputs

Computed from local files, not previous summaries:

| input | verified status |
|---|---|
| LIBERO `cot_d_cam3d.jsonl` | 273,062 rows; target point249,321, object box229,040, five-point2D trace271,318, five-point3D trace271,372, phase273,062 |
| RoboCasa local LeRobot | 9,126 episodes;2,231,347 frames;18 task sources; three recorded cameras; raw state16 and action12 |
| Source provenance | All9,126 `(source_prefix,source_episode_index)` pairs unique; all18 source prefixes match the official archive registry |
| Local replay extras | Absent from the consolidated training dataset |
| Downloaded pilot | CloseBlenderLid archive691,466,240 bytes; first episode361 frames; action, state, frame index and timestamp match local episode0 exactly (maximum absolute difference0) |
| Pilot simulator metadata | `states.npz`:361×217; model XML, camera configurations, object/fixture metadata and environment/controller versions present. Rendering and semantic alignment remain unverified. |

Audit artifacts: [audit.json](results_collected/robocasa_supervision/audit.json), [episode provenance](results_collected/robocasa_supervision/episode_sources.csv), [pilot comparison](results_collected/robocasa_supervision/pilot_alignment.json), [LIBERO target counts](results_collected/robocasa_supervision/libero_targets.json). Source data: `/e/scratch/m3/datasets/lerobot_3_0/robocasa365_target_atomic`. Pilot and archive: `/e/scratch/m3/blank4/rc365_supervision/`. The registry snapshot is in the artifact directory; rerun `scripts/audit_robocasa_supervision.py --help` for its explicit inputs.

LIBERO source: `/e/project1/m3/blank4/code/encdec-vlm/train/encoder_decoder_training/enc_dec_cot/mappings/cot_d_cam3d.jsonl`; generator `gen_cot_libero.py` alongside it. The generator uses simulator geometry/masks and demo alignment, not an LLM teacher. The selected GR00T+z YAML supervises point, box, point-minus-box-center relation, visibility,3D trajectory and phase, plus future-latent consistency at offset8. It does not currently select the2D trajectory head. Its five-point trajectory represents the **gripper**, which must not be conflated with an annotated manipulated-object trace used by other reasoning datasets.

Official [dataset documentation](https://raw.githubusercontent.com/robocasa/robocasa/main/docs/datasets/using_datasets.md) describes replay extras containing the environment metadata, per-episode model and states. The [official archive registry](https://raw.githubusercontent.com/robocasa/robocasa/main/robocasa/models/assets/box_links/box_links_ds.json) supplies the download locations. Availability and the pilot match are verified above; successful rendered reconstruction is not yet established.

## What to generate

| target | derivation and required check |
|---|---|
| Five-point2D object trace for causal-matched two-pass | Selected manipulated-object track from the current frame to the end of its annotated subtask/window, simplified and sampled along spatial arc length. Preserve the causal target semantics; do not substitute a16-step gripper trace. Validate target-box endpoint snapping separately. |
| Optional five-point2D gripper trace | Replay/projection or matching gripper masks; horizon16 and boundary handling as in the shared-z label generator. Explicitly distinguish end-effector projection from visible-mask centroid. This is a target ablation, not the causal-matched reasoning target. |
| Five-point3D gripper displacement | Replay world gripper poses, transformed into the **current** wrist-camera frame; signed cm, current gripper as origin. Do not sum commanded actions or use a different future camera frame for each point. |
| Object box, target point, relation | Instance masks plus task/fixture references. Articulated fixtures need handle/knob/door semantics, not an arbitrary whole-appliance box. Navigation needs a defined goal representation or an explicit masked target. |
| Gripper phase | Validate raw gripper command and measured qpos/contact convention. This is gripper state, not a complete subtask label for appliance operation. |
| Visibility and validity | Distinguish offscreen/occluded from unavailable annotation. Current LIBERO parser infers visibility from tag presence; blindly omitting unsupported RoboCasa labels would teach false invisibility. |
| Temporal latent target | Future RGB/instruction at offset8 with episode-boundary checks; available without reasoning annotations, but this alone is a different, temporal-only recipe. |

Recompute transforms for RoboCasa: mobile base, wrist extrinsics, image orientation and augmentation/crop alignment. Do not copy LIBERO's image flip. Raw RoboCasa action order is base4, mode1, end-effector translation3, rotation3, gripper1; the model loader reorders it. Local LeRobot v3 consolidates episodes into files, so join by source IDs and episode-relative frame, never video/parquet filename alone. Human task text is already present, but does not supply boxes or future coordinates. Any future-derived targets are training-only; policy evaluation must use predictions.

## Next work and gates

1. **Day1:** provenance audit and first source episode match completed. Next replay that episode on a compute node, compare saved RGB and project gripper traces. Extend to pick/place, an articulated fixture, a switch and navigation before bulk conversion. This is annotation validation, not the trainer smoke or policy rollout evaluation.
2. **Day2:** recover the remaining source extras, implement deterministic target mapping, report valid-label fractions per task/camera and inspect overlays. Require correct frame alignment and geometry in each task family. Port GR00T+z to the RoboCasa worktree in parallel. If geometry is unresolved by end of Day2, do not promise supervised RoboCasa results at the Sep21 gate; prioritize the existing v4 comparison and treat temporal-only z as a separately named optional experiment.
3. **After labels pass:** present the final layouts, then run matched no-z+state, z+state with zero visual memory, and z+state with15% memory dropout. Preserve32D model-state contract,12D action, two-camera order,50k steps and seeds4/42. Use matched global64 (32/device ×2GPUs/run), two independent runs per4-GPU node. Each new configuration must complete20 optimizer updates, normal trainer eval and checkpoint inspection before full training. New RoboCasa jobs have not been submitted.
4. **Day3–4 if gates pass:** train independent matched pairs and chain the declared task-set evaluations; reserve roughly20h/pair from previous RoboCasa logs plus1.5–2h evaluation, but GR00T+z throughput remains unmeasured. Label/replay time is also unmeasured. No guaranteed completion date follows from archive availability alone.

Keep the existing LIBERO memory and reasoning tracks running independently. The critical path here is trustworthy labels and data integration, not another auxiliary-head sweep. RoboCasa remains a conditional supporting benchmark until matched rollout evidence lands; the Sep21 scientific decision and Sep24 headline freeze do not move.

Causal trace audit: saved `playground/Checkpoints/libero_plus_qwen08b_gr00t_cot_trace_ours_v3_cotw01/config.full.yaml` selects `data/cot_mappings/libero_plus_full_ours_trace.jsonl`, with action horizon8. `scripts/create_cot_mapping.py` takes the remaining object track (`trace[start_idx:]`), applies simplification and five-point arc-length sampling, and can snap the endpoint to the selected target box. It does not truncate to the action chunk. The first raw trajectory has126 per-frame entries, with changing starts/interior points and the same target endpoint in inspected samples. The scope is the annotation window/subtask, not necessarily the entire multi-subtask episode.

## Generation execution update

User requested BOTH full-subtask object and full-subtask gripper paths, plus every shared-z target. Implemented in the RoboCasa worktree at `scripts/robocasa_supervision/` (commitfd7d801); dense paths remain separate by entity/subtask, with additional remaining-path five-point summaries and independent16-frame z trajectories. Restored missing external cameras from episode metadata and disabled multisample antialiasing for integer segmentation IDs. All18 source archives recovered (9.167GB); pilot covers18 tasks/4,282 frames. All54 initial task/camera views compared with source videos: meanMAE3.807/255,max5.746, with overlays inspected. Final coordinate/phase-convention pilot:4 episodes/1,182frames, zero invariant errors. Full generation array1862967_[0-3] is submitted on4nodes/16GPUs/64workers, output `/e/scratch/m3/blank4/rc365_supervision/labels_v1`. This is generation in progress, not a claim that9,126 episodes are finished.

Object points are rigid geometry/handle centers; static microwave button uses its official interaction region and depth visibility. Terminal object position defines the hindsight target point, rather than a claimed receptacle-mask centroid. Full-subtask gripper/object paths and optional summaries are distinct from chunk-local z motion. Phase uses LIBERO commanded-gripper current/end semantics. Multi-door subtask boundaries are inferred from motion order and hand transition, marked for review; overlapping intervals fail explicitly. Navigation retains gripper/temporal labels with object/grounding masks false. Native shared-z exports include explicit validity masks and still require loader/augmentation integration before training. See the worktree README for the exact schema.
