"""Native RoboCasa subtask targets, with geometry and missingness kept separate.

No simulation, text parsing, or global multi-GB JSONL scan in a training worker.
A bounded episode cache reads the audited dense bundles on demand.
"""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


class RoboCasaSupervision:
    def __init__(self, config, dataset_name, video_keys, *, horizon=16, future_offset=0):
        if dataset_name != 'robocasa365_target_atomic':
            raise ValueError('Native labels currently cover robocasa365_target_atomic only')
        configured_roots = config.get('labels_roots')
        if configured_roots is None:
            configured_roots = [config['labels_root']]
        self.roots = [Path(root) for root in configured_roots]
        missing_roots = [root for root in self.roots if not root.is_dir()]
        if missing_roots:
            raise FileNotFoundError(missing_roots)
        self.cameras = [key.removeprefix('video.') for key in video_keys]
        if self.cameras[0] != 'robot0_agentview_left':
            raise ValueError('Native cam1 grounding requires left agentview first')
        if 'robot0_eye_in_hand' not in self.cameras:
            raise ValueError('Native camera-frame 3D targets require the wrist input')
        if horizon != 16 or future_offset not in (0, 8):
            raise ValueError('Native v1 chunk targets require horizon16 and future offset0 or8')
        # Multi-part boundaries are deterministic annotations from joint motion and
        # hand proximity. Keep them by default; callers can request the conservative
        # masked ablation explicitly without discarding 210k useful training frames.
        self.include_unreviewed = bool(config.get('include_unreviewed_boundaries', True))
        self.cache_size = int(config.get('episode_cache_size', 8))
        if self.cache_size < 1:
            raise ValueError('episode_cache_size must be positive')
        self.trace_subject = config.get('trace_subject', 'gripper')
        self.trace_span = config.get('trace_span', 'remaining')
        if self.trace_subject not in ('gripper', 'object') or self.trace_span not in ('remaining', 'full_subtask'):
            raise ValueError('trace_subject must be gripper/object and trace_span remaining/full_subtask')
        self.cache = OrderedDict()

    def _episode(self, episode):
        if episode not in self.cache:
            candidates = [root / f'episode_{episode:06d}' for root in self.roots]
            complete = [root for root in candidates if (root / 'COMPLETE.json').is_file()]
            if not complete:
                raise FileNotFoundError(
                    f'No completed supervision bundle for episode {episode} in {candidates}'
                )
            # Roots are ordered by precedence. This permits a canonical merged root
            # followed by retry/archive roots without making identical copies ambiguous.
            root = complete[0]
            meta = json.loads((root / 'COMPLETE.json').read_text())
            if meta['version'] != 'rc365-subtask-traces-v1' or meta['episode_index'] != episode:
                raise ValueError(f'Unexpected native supervision identity: {root}')
            # Materialize once: repeatedly indexing compressed NPZ re-inflates each array.
            with np.load(root / 'targets.npz', allow_pickle=False) as archive:
                keys = ('subtask_id', 'future_frame_index', 'future_frame_valid', 'phase',
                        'trajectory3d_chunk', 'trajectory3d_chunk_valid', 'target_point',
                        'target_point_valid', 'object_box', 'object_visible', 'ground_visibility',
                        'ground_visibility_valid', 'object_uv', 'object_in_frame',
                        'gripper_uv', 'gripper_in_frame', 'gripper_trace_remaining_2d',
                        'object_trace_remaining_2d', 'gripper_trace_valid', 'object_trace_valid')
                arrays = {key: archive[key] for key in keys}
            self.cache[episode] = root, meta, arrays
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode]

    def resolve(self, episode, frame):
        root, meta, a = self._episode(int(episode))
        if not 0 <= frame < meta['n_frames']:
            raise IndexError((episode, frame, meta['n_frames']))
        sub = meta['subtasks'][int(a['subtask_id'][frame])]
        start, end, entity = sub['start'], sub['end'], sub['entity_index']
        if not start <= frame < end:
            raise ValueError('Frame/subtask identity mismatch')
        accepted = self.include_unreviewed or not sub['boundary_needs_review']
        camera_ids = [meta['cameras'].index(c) for c in self.cameras]
        traces, paths, trace_valid = {}, {}, {}
        for subject in ('object', 'gripper'):
            for span, first in [('remaining', frame), ('full_subtask', start)]:
                key = f'{subject}_{span}'
                traces[key] = a[f'{subject}_trace_remaining_2d'][first, camera_ids].copy()
                trace_valid[key] = a[f'{subject}_trace_valid'][first, camera_ids].copy() & accepted
                if subject == 'object':
                    paths[key] = (a['object_uv'][first:end, entity][:, camera_ids].copy()
                                  if entity >= 0 else np.zeros((end-first, len(camera_ids), 2)))
                else:
                    paths[key] = a['gripper_uv'][first:end][:, camera_ids].copy()
        box = a['object_box'][frame, entity, 0].copy() if entity >= 0 else np.zeros(4)
        box_valid = bool(entity >= 0 and a['object_visible'][frame, entity, 0] and accepted)
        point = a['target_point'][frame, 0].copy()
        point_valid = bool(a['target_point_valid'][frame, 0] and accepted)
        targets = dict(trajectory3d=a['trajectory3d_chunk'][frame].reshape(-1).copy(),
                       phase=int(a['phase'][frame]), target_point=point, object_box=box,
                       ground_relation=point-(box[:2]+box[2:])/2,
                       ground_visibility=a['ground_visibility'][frame, 0].copy())
        valid = dict(trajectory3d=bool(a['trajectory3d_chunk_valid'][frame] and accepted),
                     phase=accepted, target_point=point_valid, object_box=box_valid,
                     ground_relation=point_valid and box_valid,
                     ground_visibility=a['ground_visibility_valid'][frame, 0].copy() & accepted)
        return dict(targets=targets, valid=valid, traces=traces, trace_valid=trace_valid,
                    _trace_paths=paths, cameras=self.cameras, trace_subject=self.trace_subject,
                    trace_span=self.trace_span, frame_index=int(frame), episode_index=int(episode),
                    subtask_id=sub['subtask_id'], subtask_start=start, subtask_end=end,
                    entity=sub['entity'], boundary_source=sub['boundary_source'],
                    boundary_needs_review=sub['boundary_needs_review'], boundary_accepted=accepted,
                    future_frame_index=int(a['future_frame_index'][frame]),
                    future_frame_valid=bool(a['future_frame_valid'][frame] and accepted),
                    source_path=str(root / 'targets.npz'))


def augment_native_targets(record, *, left, top, crop_width, crop_height,
                           image_width, image_height, angle_degrees=0.):
    """Use the exact image crop/rotation; never turn offscreen points into edge labels.

    A partially cropped object box cannot determine remaining visible segmentation.
    Mask that box and its visibility channel conservatively, rather than inventing it.
    """
    result = deepcopy(record)
    target, valid = result['targets'], result['valid']
    angle = np.deg2rad(angle_degrees)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    size = np.array([image_width, image_height])
    origin, extent = np.array([left, top]), np.array([crop_width, crop_height])

    def transform(points):
        cropped = (np.asarray(points)*size - origin)/extent
        pixels = cropped*size
        final = ((pixels-size/2) @ rot + size/2)/size
        keep = (np.isfinite(cropped).all(axis=-1) & (cropped >= 0).all(axis=-1)
                & (cropped <= 1).all(axis=-1) & np.isfinite(final).all(axis=-1)
                & (final >= 0).all(axis=-1) & (final <= 1).all(axis=-1))
        return final.astype(np.float32), keep

    point, inside = transform(target['target_point'])
    target['target_point'] = point
    valid['target_point'] = bool(valid['target_point'] and inside)
    x1, y1, x2, y2 = target['object_box']
    corners, retained = transform([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
    original_box_valid = valid['object_box']
    valid['object_box'] = bool(original_box_valid and retained.all())
    target['object_box'] = np.r_[corners.min(axis=0), corners.max(axis=0)]
    target['ground_relation'] = point - (target['object_box'][:2]+target['object_box'][2:])/2
    valid['ground_relation'] = valid['target_point'] and valid['object_box']
    target['ground_visibility'][0] = float(valid['target_point'])
    if original_box_valid:
        # AABB disjoint from either the crop or output proves no pixels remain.
        lo, hi = np.array([x1,y1])*size, np.array([x2,y2])*size
        outside = ((hi < origin).any() or (lo > origin+extent).any()
                   or (corners.max(axis=0) < 0).any() or (corners.min(axis=0) > 1).any())
        target['ground_visibility'][1] = float(not outside)
        valid['ground_visibility'][1] = bool(retained.all() or outside)
    # Cropping doesn't change metric camera axes. In-plane rotation changes x/y only.
    xyz = np.asarray(target['trajectory3d']).reshape(5, 3).copy()
    xyz[:, :2] = xyz[:, :2] @ rot
    target['trajectory3d'] = xyz.reshape(-1)
    for key, points in result['traces'].items():
        result['traces'][key], _ = transform(points)
        paths, keep = transform(result['_trace_paths'][key])
        result['_trace_paths'][key] = paths
        result['trace_valid'][key] &= keep.all(axis=0)
    result['augmentation'] = dict(left=left, top=top, crop_width=crop_width,
                                  crop_height=crop_height, image_width=image_width,
                                  image_height=image_height, angle_degrees=angle_degrees)
    return result


def pack_native_supervision(sample, record, video_keys):
    if [key.removeprefix('video.') for key in video_keys] != record['cameras']:
        raise ValueError('Native labels require fixed camera selection/order')
    # Existing heads mask absent fields. Keep per-channel validity in native metadata;
    # conservatively omit a whole head if any of its channels is unknown.
    structured = {'_native_robocasa': True}
    for key, value in record['targets'].items():
        if np.asarray(record['valid'][key]).all():
            structured[key] = value.tolist() if isinstance(value, np.ndarray) else value
    key = f"{record['trace_subject']}_{record['trace_span']}"
    if record['trace_valid'][key][0]:
        structured['trajectory2d'] = record['traces'][key][0].reshape(-1).tolist()
    sample['cot_structured_targets'] = structured
    sample['native_supervision'] = {key: value for key, value in record.items() if not key.startswith('_')}
    sample['future_frame_valid'] = record['future_frame_valid']
    if not record['future_frame_valid']:
        sample.pop('future_image', None)
