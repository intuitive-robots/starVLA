# RoboCasa supervision generation

Owning tree: `starVLA-upstream-merge`. Output root:
`/e/scratch/m3/blank4/rc365_supervision`.

This generates **full-subtask object and gripper trajectories**, plus the shared-z
labels. It uses the official episode XML, state trajectory and camera metadata.
It never steps a policy or treats missing results as zero labels.

## Reproducible stages

1. `recover_sources.py --manifest <episode_sources.csv> --root <root> --workers 4`
   recovers the exact18 source archives; extracts only extras, metadata and low-dimensional
   data. Original source IDs map to consolidated LeRobot episodes.
2. `preflight.py` audits task/entity selection over every episode without loading
   renderer assets. It writes `entity_preflight.json`; unknown tasks/entities fail explicitly.
3. `run_slurm.sh pilot 4 <output>` generates one episode per task, one worker/GPU.
   Use inside an allocation via `srun --jobid=<id> ...`, or submit with `sbatch`.
   `EPISODE_IDS=0,502,...` narrows the pilot. Full mode processes all episodes.
   `SHARD_GROUP`, `SHARD_GROUPS` distribute disjoint episode shards across nodes;
   each node still runs workers on all4 GPUs.
4. Inspect RGB overlays and compare replay RGB against the source videos. Inspect
   full-subtask boundaries for multi-part tasks. Do not promote failed pilots.
5. `export.py --labels <labels> --manifest <manifest> --output <mappings>` writes
   separate object/gripper five-point CoT mappings and native shared-z targets.
   Missing episodes raise by default. Unreviewed multi-entity boundaries are excluded
   by default and counted; `--include-unreviewed-boundaries` is an explicit override,
   not a recommended training default. Dense full paths remain available regardless.

Run geometry checks in the RoboCasa container:
`python scripts/robocasa_supervision/test_geometry.py`.

## Per-episode output

`COMPLETE.json` contains exact source provenance, entities, subtask frame intervals
`[start,end)`, boundary method/review flags, coordinate definitions and coverage.
`targets.npz` contains:

- Dense world-space object and gripper paths, with separate entity axes. Slice by
  the recorded subtask interval to obtain the **entire** subtask trajectory.
- Dense image paths in all three recorded cameras, plus per-frame in-view flags.
  These are projections of a rigid object point/handle center and the end-effector
  site, not CoTracker predictions or mask-centroid trajectories.
- Five arc-length samples of the remaining subtask for **both** object and gripper,
  in2D and3D. Full3D summaries are current-wrist-frame displacements in cm.
- `trajectory3d_chunk`: the separate z target, five time samples up to15 future
  frames, truncated before the next subtask. Camera axes are image-right,
  image-down, camera-forward; integer cm divided by20, matching the LIBERO parser's
  scaling. No LIBERO180-degree image flip is applied to RoboCasa.
- Per-frame object and gripper boxes/visibility from geometry segmentation. Integer
  segmentation uses no multisample antialiasing. The static microwave button has
  no separate visible mesh: its official interaction-region box is projected and
  checked against rendered depth; this exception is explicitly recorded.
- Target point: terminal rigid object point for that subtask projected in the current
  camera. This is a hindsight goal, **not** a claimed receptacle-mask centroid.
  Relation is target point minus current visible-box center.
- `ground_visibility` is target-in-frame and object-visible; validity is separate.
  Target occlusion is not inferred from its terminal location. Navigation has no
  manipulated object: its object/grounding labels are masked, never fabricated.
- Gripper phase matches LIBERO's commanded-current versus commanded-chunk-end rule:
  keep-open0, close1, keep-closed2, open3. It is not a subtask completion signal.
- Future-frame links at offset8 and explicit validity, bounded by the subtask.

Out-of-frame coordinates remain unclipped in dense paths. A CoT mapping is emitted
only when its remaining image path is in view. The native z file carries explicit
validity masks; using LIBERO's tag-presence visibility parser directly would lose
that distinction. Loader integration and augmentation alignment remain a separate
training gate; no new training is launched by these scripts.

## Multi-entity episodes

Task metadata identifies manipulated entities, excluding distractors. For multiple
moving parts (e.g. left/right fridge doors), joint-motion order and the gripper's
nearest-entity transition propose separate subtask boundaries. Those boundaries
are marked `boundary_needs_review`; they are not human-ground-truth annotations.
Overlapping manipulation intervals raise rather than inventing a boundary. The
single-entity atomic tasks use their entire episode as the semantic interaction.
Composite tasks are not present in the current18-task training set and require
additional semantic rules before use.

The saved-state replay restores external camera definitions from `ep_meta.json`:
the source XML itself often contains only the wrist camera. Asset paths are
remapped into the pinned local RoboCasa container without changing scene geometry.
Every episode's action/state/frame-index arrays are compared against the local
training copy before any output is accepted.
