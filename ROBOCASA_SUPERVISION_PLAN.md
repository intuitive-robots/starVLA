# RoboCasa supervision — completion and next steps

Complete: **9,126 episodes,9,591 subtasks,2,231,347 frames**,18 atomic task families.
All source actions, robot states and frame indices matched the consolidated training data.
Both full-subtask object and full-subtask gripper paths are retained. No training was launched.

| artifact | contents |
|---|---|
| `labels_v1/episode_XXXXXX/targets.npz` | Dense object/gripper world and image trajectories in all3 cameras; full-subtask slices; remaining2D/3D summaries; grounding, phase, chunk3D and future-frame targets |
| `labels_v1/episode_XXXXXX/COMPLETE.json` | Subtask boundaries, separate entity IDs, camera/target definitions, validity coverage and source/code provenance |
| `mappings_v1/subtasks.jsonl` | One row per subtask; dense-array references and full-subtask five-point summaries |
| `mappings_v1/object_trace.jsonl` | Five-point remaining object-path supervision |
| `mappings_v1/gripper_trace.jsonl` | Five-point remaining gripper-path supervision |
| `mappings_v1/object_trace_full_subtask.jsonl` | Whole-subtask object path, beginning to end |
| `mappings_v1/gripper_trace_full_subtask.jsonl` | Whole-subtask gripper path, beginning to end |
| `mappings_v1/shared_z_targets.jsonl` | Per-frame z targets and explicit masks, including phase, chunk-local3D and future-image links |

Five-point text mappings use the left camera and omit invalid/out-of-view paths. Dense labels retain all3 cameras and their validity. A whole-subtask mapping includes earlier motion at later frames; use the remaining-path version to match causal action planning.

Shared-z includes point, box, relation, visibility, gripper phase,16-frame3D motion (clipped to subtask boundaries), and offset8 future-frame links. Full-subtask trajectories are separate fields and are not truncated to16frames.3D axes are image-right/down/camera-forward; chunk coordinates use rounded cm divided by20. Gripper phase matches the LIBERO commanded-current/chunk-end convention.

Validity and limitations: **465 multi-part episodes /930 subtasks /210,283 frames** have inferred boundaries, flagged explicitly. The main training recipe includes these programmatic annotations because their object/gripper paths were visually checked and targets are bounded to the selected door/part. A conservative `include_unreviewed_boundaries:false` run remains available as a boundary-quality ablation; it is not the default. Navigation has no manipulated object: object/grounding losses are masked. Object traces track rigid geometry/handle centers; static microwave buttons use the official interaction region and a depth visibility check. The target point is the terminal object point projected into the current view, not a claimed receptacle-mask centroid. The target visibility channel denotes in-frame validity, not occlusion of a future object.

Validation: all9,126 episodes passed geometry/boundary/shape audits.270 replay-versus-recorded frame/view comparisons across18 tasks had mean absolute RGB error3.719/255,max7.509; overlays were inspected. The focused subtask review grids for CloseFridge episode502 and OpenCabinet episode2529 show the complete object/gripper paths on both sides of their inferred switches.189 initial failures were corrected and re-audited; `unresolved_failures.json` is empty. Initial failed-shard logs remain as history, not missing/zero-valued data.

Generation and loader code: `/e/project1/m3/blank4/code/starVLA-upstream-merge/scripts/robocasa_supervision/` and `starVLA/dataloader/robocasa_supervision.py`. Native validity-mask and augmentation integration is complete. It was exercised through both the single-dataset and random mixture sampling paths on the real two-camera dataset: images decoded at256², actions were16×12, state was1×32, and the six shared-z fields reached `cot_structured_targets`. The frame221 fridge check removed the offset8 future image because it crosses the frame222 door boundary. Five focused mask/geometry tests pass. Cold random label lookup measured22.9ms/sample and warm same-episode lookup0.048ms/sample; video decoding still dominates.

The integration exposed a recipe discrepancy: existing RoboCasa YAMLs said `augmentation: crop_photometric`, but the PandaOmron data config ignored that field. New matched arms must set `robocasa_joint_augmentation: true`; old runs keep the switch off so their actual recipe is not silently changed. Cropped-out points remain invalid and out of range rather than being clipped into edge targets. Partially cropped boxes are conservatively masked.

Dataset package: `/e/scratch/m3/blank4/rc365_supervision/`. Machine-readable reports are in `results_collected/robocasa_supervision/generation_audit.json` and `export_audit.json`.

Schedule: Day1 generation and native loader/augmentation integration completed. Day2 port the verified GR00T+z path into the RoboCasa worktree, prepare the matched no-z, z-only and z+retained-memory configs with the joint-augmentation switch enabled in every arm, and run the intended two-run,2GPUs/run,global64 smoke. It must complete20 updates plus trainer evaluation before any full training. No new RoboCasa training has been submitted. Keep the Sep21 scientific gate and Sep24 headline freeze.
