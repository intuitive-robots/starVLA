"""
Create a phase-based CoT mapping JSONL from an annotations.jsonl file.

Each annotation entry may contain multiple phases (via phase_annotations).
This script emits one output entry per phase, covering that phase's frame range.

Output format (one JSON per line):
    {
        "trajectory_name": "cylinder_cube_full/0/1",
        "start_frame":     0,
        "end_frame":       123,
        "cot_text":        "What is the next subtask? ... <|point|>{...}<|/point|>"
    }

Point format (default)
──────────────────────
Points are emitted in Qwen3.5VL grounding format:
    <|point|>{"point_2d": [x, y], "label": "object name"}<|/point|>

With --box_format, boxes are used instead:
    <|box|>{"bbox_2d": [x1, y1, x2, y2], "label": "..."}<|/box|>

With --trace_format, centroid traces are used instead:
    <|trace|>{"trace_2d": [[x, y], ...]}<|/trace|>

Phase → spatial target mapping:
    grasp              → selected initial object, label = task_obj_info.object
    interact / release → selected target object/location,
                         label = task_obj_info.target_location

Selection modes:
    ours      → use initial_object_box and the lightweight winner trace
    detection → use lightweight candidate confidences for the initial box/trace

Both legacy RoboG rows and compact robog.annotation v2 rows are supported.
For v2, detection mode reads the published baselines.detector record.

Target selection:
    mode           → match selection_mode: annotation target for ours, detection-confidence target for detection
    annotation     → use target_object_box
    detection      → use target candidate confidences
    trace_endpoint → use the target candidate that contains the selected trace endpoint

Trajectories with only an "interact" phase (no grasp) are handled correctly.

Usage
─────
    python scripts/create_cot_mapping.py \\
        --annotations /path/to/annotations.jsonl \\
        --output      data/cot_mappings/real_robot_ours_ours_cot.jsonl \\
        [--box_format]
        [--trace_format]
        [--selection_mode {ours,detection}]
        [--target_selection {mode,annotation,detection,trace_endpoint}]
        [--all]
        [--holdout_fraction 0.1]
        [--holdout_seed 42]
        [--verbose]
"""

import argparse
import base64
import io
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Spatial formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

_GRASP_PHASES = {"grasp"}
_PLACE_PHASES = {"interact", "release", "transport", "carry"}
DEFAULT_COORD_SIZE = (256.0, 256.0)
DEFAULT_HUMAN_PROMPT = "Your task is {instruction}. What is the next subtask? Point to the target location."
DEFAULT_TRACE_HUMAN_PROMPT = (
    "Your task is {instruction}. Generate the 2D trajectory the object should follow to complete the task. "
    "Output exactly 5 points."
)


def _is_publication_v2(row: dict) -> bool:
    schema = row.get("schema")
    return (
        isinstance(schema, dict)
        and schema.get("name") == "robog.annotation"
        and str(schema.get("version")) == "2.0"
    )


def _publication_v2_to_legacy_view(row: dict) -> dict:
    """Expose publication-v2 fields through the legacy names used below.

    The adapter is intentionally read-only with respect to the input row and
    only reconstructs fields that are present in the compact public schema.
    Rejected detector candidates are not part of v2 and therefore cannot be
    reconstructed here.
    """
    if not _is_publication_v2(row):
        return row

    source = row.get("source") or {}
    window = row.get("window") or {}
    task = row.get("task") or {}
    obj = row.get("object") or {}
    selection = row.get("selection") or {}
    detector_baseline = ((row.get("baselines") or {}).get("detector") or {})
    detector_initial = detector_baseline.get("initial") or {}
    detector_target = detector_baseline.get("target") or {}
    detector_track = detector_baseline.get("track") or {}
    initial = obj.get("initial") or {}
    target = obj.get("target") or {}
    track = obj.get("track") or {}

    trajectory_name = source.get("trajectory_id")
    if trajectory_name is None:
        raise ValueError("publication-v2 annotation is missing source.trajectory_id")
    if not obj and row.get("arms"):
        raise ValueError(
            f"publication-v2 bimanual annotation {trajectory_name!r} is not supported "
            "by the single-object CoT mapping format"
        )

    start_frame = int(window.get("start_frame", 0))
    phase_annotations = []
    for phase in task.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        phase_annotations.append({
            "phase": phase.get("type", ""),
            "start_frame": phase.get("start_frame"),
            "end_frame": phase.get("end_frame_inclusive"),
            "description": phase.get("description"),
        })

    point_meta = dict(track.get("point_selection") or {})
    absolute_frame_indices = track.get("frame_indices")
    if absolute_frame_indices is not None:
        point_meta["frame_indices_relative_to_window"] = [
            int(frame) - start_frame for frame in absolute_frame_indices
        ]
    elif track.get("sampling") == "tracking_timeline" and track.get("xy_pixels"):
        # RoboG's tracking timeline is constructed with np.linspace from the
        # detection frame through the final frame. Early v2 files named that
        # sampling rule but omitted the explicit indices, so reconstruct the
        # deterministic timeline for phase-aligned trace tails.
        track_start = max(0, int(initial.get("frame_index", start_frame)) - start_frame)
        track_end = max(
            track_start,
            int(window.get("end_frame_exclusive", start_frame + 1)) - start_frame - 1,
        )
        point_meta["frame_indices_relative_to_window"] = np.linspace(
            track_start,
            track_end,
            len(track["xy_pixels"]),
            dtype=int,
        ).tolist()
        point_meta["frame_indices_source"] = "reconstructed_tracking_timeline"

    lightweight = {}
    arrays = row.get("arrays") or {}
    if arrays.get("blob"):
        lightweight["arrays_blob"] = arrays["blob"]
    if selection.get("selected_candidate_index") is not None:
        lightweight["winner_idx"] = int(selection["selected_candidate_index"])
    if detector_initial.get("box_xyxy_pixels") is not None:
        lightweight["candidate_boxes"] = [detector_initial["box_xyxy_pixels"]]
        lightweight["candidate_confidences"] = [detector_initial.get("detector_score")]
    if detector_target.get("box_xyxy_pixels") is not None:
        lightweight["target_candidate_boxes"] = [detector_target["box_xyxy_pixels"]]
        lightweight["target_candidate_confidences"] = [detector_target.get("detector_score")]

    detector_point_meta = {}
    detector_frame_indices = detector_track.get("frame_indices")
    if detector_frame_indices is not None:
        detector_point_meta["frame_indices_relative_to_window"] = [
            int(frame) - start_frame for frame in detector_frame_indices
        ]

    image_size = window.get("image_size_hw")
    legacy = dict(row)
    legacy.update({
        "trajectory_name": str(trajectory_name),
        "dataset_name": source.get("dataset_id"),
        "subtask_index": source.get("subtask_index"),
        "language_instruction": task.get("instruction"),
        "start_frame": start_frame,
        "end_frame": window.get("end_frame_exclusive", start_frame),
        "num_frames": window.get("trajectory_frame_count"),
        "phase_annotations": phase_annotations,
        "movement_type": task.get("action"),
        "task_obj_info": {
            "action": task.get("action"),
            "object": obj.get("label"),
            "object_instance": obj.get("instance_id"),
            "instance_relation_to_previous": obj.get("relation_to_previous"),
            "start_location": task.get("start_location"),
            "target_location": task.get("target_location"),
        },
        "initial_object_box": initial.get("box_xyxy_pixels"),
        "initial_object_box_detection_confidence": initial.get("detector_score"),
        "initial_object_box_selection_score": selection.get("score"),
        "initial_object_box_selection_score_name": selection.get("method"),
        "target_object_box": target.get("box_xyxy_pixels"),
        "target_object_box_confidence": target.get("detector_score"),
        "verified_target": target.get("verified"),
        "obj_traces_cotracker_point": track.get("xy_pixels"),
        "obj_traces_cotracker_point_meta": point_meta,
        "detector_baseline_trace": detector_track.get("xy_pixels"),
        "detector_baseline_trace_meta": detector_point_meta,
        "lightweight": lightweight,
        "_publication_v2": True,
        "_publication_detector_baseline": bool(detector_initial),
    })
    if initial.get("frame_index") is not None:
        legacy["detection_frame_index"] = int(initial["frame_index"]) - start_frame
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        legacy["image_height"] = image_size[0]
        legacy["image_width"] = image_size[1]
    return legacy


def _load_annotations(path: Path) -> list[dict]:
    annotations = []
    with open(path) as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                annotations.append(_publication_v2_to_legacy_view(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Could not load annotation line {line_number}: {exc}") from exc
    return annotations


def _box_center(box: list) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _box_center_float(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _normalise_coord(value: float, size: float) -> int:
    if size <= 0.0:
        size = 1000.0
    return max(0, min(1000, int(round(float(value) * 1000.0 / size))))


def _normalise_point(x: float, y: float, coord_size: tuple[float, float]) -> list[int]:
    width, height = coord_size
    return [_normalise_coord(x, width), _normalise_coord(y, height)]


def _normalise_box(box: list, coord_size: tuple[float, float]) -> list[int]:
    x1, y1, x2, y2 = box
    nx1, ny1 = _normalise_point(x1, y1, coord_size)
    nx2, ny2 = _normalise_point(x2, y2, coord_size)
    return [nx1, ny1, nx2, ny2]


def _qwen_point(x: int, y: int, label: str, coord_size: tuple[float, float]) -> str:
    payload = json.dumps({"point_2d": _normalise_point(x, y, coord_size), "label": label}, ensure_ascii=False)
    return f"<|point|>{payload}<|/point|>"


def _qwen_box(box: list, label: str, coord_size: tuple[float, float]) -> str:
    payload = json.dumps({"bbox_2d": _normalise_box(box, coord_size), "label": label}, ensure_ascii=False)
    return f"<|box|>{payload}<|/box|>"


def _qwen_trace(points: list[list[float]], coord_size: tuple[float, float]) -> str:
    payload = json.dumps(
        {"trace_2d": [_normalise_point(x, y, coord_size) for x, y in points]},
        ensure_ascii=False,
    )
    return f"<|trace|>{payload}<|/trace|>"


def _format_box_or_point(box: list, label: str, box_format: bool, coord_size: tuple[float, float]) -> str:
    if box_format:
        return _qwen_box(box, label, coord_size)
    cx, cy = _box_center(box)
    return _qwen_point(cx, cy, label, coord_size)


def _obj_label(task_obj_info) -> str:
    if isinstance(task_obj_info, dict):
        return task_obj_info.get("object") or "object"
    return str(task_obj_info) if task_obj_info else "object"


def _target_label(task_obj_info) -> str:
    if isinstance(task_obj_info, dict):
        return task_obj_info.get("target_location") or "target location"
    return "target location"


def _lightweight_payload(row: dict) -> dict:
    for key in ("lightweight", "lightweight_payload", "payload"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    confidence_breakdown = row.get("confidence_breakdown")
    if isinstance(confidence_breakdown, dict):
        value = confidence_breakdown.get("lightweight")
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    return row


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                score = item.get("confidence", item.get("score", item.get("conf")))
                if score is not None:
                    out.append(float(score))
            elif isinstance(item, (int, float)):
                out.append(float(item))
        return out
    return []


def _candidate_confidences(ann: dict, prefix: str) -> list[float]:
    lightweight = _lightweight_payload(ann)
    keys = (
        f"{prefix}_box_confidences",
        f"{prefix}_box_confidence",
        f"{prefix}_confidences",
        f"{prefix}_confidence",
        f"{prefix}_box_scores",
        f"{prefix}_box_score",
        f"{prefix}_scores",
        f"{prefix}_score",
        f"{prefix}_detection_confidences",
        f"{prefix}_detection_scores",
    )
    for source in (ann, lightweight):
        for key in keys:
            vals = _as_float_list(source.get(key))
            if vals:
                return vals

    detections_key = f"{prefix}_detections"
    for source in (ann, lightweight):
        vals = _as_float_list(source.get(detections_key))
        if vals:
            return vals
    return []


def _normalise_boxes(boxes: Any) -> list:
    if not boxes:
        return []
    if not isinstance(boxes, list):
        return []
    if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
        return [boxes]
    return boxes


def _select_box(boxes: Any, confidences: list[float], use_confidence: bool) -> tuple[Optional[list], Optional[int]]:
    boxes = _normalise_boxes(boxes)
    if not boxes:
        return None, None

    idx = 0
    if use_confidence and confidences:
        limit = min(len(boxes), len(confidences))
        if limit > 0:
            idx = max(range(limit), key=lambda i: confidences[i])
    return boxes[idx], idx


def _box_iou(box_a: list, box_b: list) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _lightweight_candidate_boxes(row: dict) -> list:
    boxes = _lightweight_payload(row).get("candidate_boxes")
    return _normalise_boxes(boxes)


def _lightweight_target_candidate_boxes(row: dict) -> list:
    boxes = _lightweight_payload(row).get("target_candidate_boxes")
    return _normalise_boxes(boxes)


def _target_candidate_boxes(row: dict) -> list:
    return _lightweight_target_candidate_boxes(row) or _normalise_boxes(row.get("target_candidate_boxes"))


def _target_candidate_confidences(row: dict) -> list[float]:
    keys = (
        "target_candidate_box_confidences",
        "target_candidate_box_confidence",
        "target_candidate_confidences",
        "target_candidate_confidence",
        "target_candidate_box_scores",
        "target_candidate_box_score",
        "target_candidate_scores",
        "target_candidate_score",
    )
    lightweight = _lightweight_payload(row)
    for source in (lightweight, row):
        for key in keys:
            vals = _as_float_list(source.get(key))
            if vals:
                return vals
    return []


def _select_lightweight_box(
    row: dict,
    boxes_key: str,
    confidences_key: str,
    use_confidence: bool = True,
) -> tuple[Optional[list], Optional[int]]:
    lightweight = _lightweight_payload(row)
    return _select_box(
        lightweight.get(boxes_key),
        _as_float_list(lightweight.get(confidences_key)),
        use_confidence=use_confidence,
    )


def _select_annotation_target_box(row: dict) -> tuple[Optional[list], Optional[int]]:
    box, idx = _select_box(row.get("target_object_box"), [], use_confidence=False)
    if box is not None:
        return box, idx
    boxes = _target_candidate_boxes(row)
    box, idx = _select_box(boxes, [], use_confidence=False)
    return box, idx


def _select_detection_target_box(row: dict) -> tuple[Optional[list], Optional[int]]:
    boxes = _target_candidate_boxes(row)
    confidences = _target_candidate_confidences(row)
    return _select_box(boxes, confidences, use_confidence=True)


def _select_density_target_box(row: dict, winner_start_idx: Optional[int], start_box: Optional[list] = None) -> tuple[Optional[list], Optional[int]]:
    """Select target candidate using track-point density penalized by box area.

    Scores = target_point_score_ungated[winner_start_idx] / sqrt(area / max_area).
    Candidates smaller than 0.5× start_box area are excluded.
    Falls back to winner_pick.best_target_idx, then annotation target_object_box.
    """
    lightweight = _lightweight_payload(row)
    target_boxes = [
        [float(v) for v in b[:4]]
        for b in (lightweight.get("target_candidate_boxes") or [])
        if isinstance(b, (list, tuple)) and len(b) >= 4
    ]

    def _fallback() -> tuple[Optional[list], Optional[int]]:
        wp = lightweight.get("winner_pick") or {}
        idx = wp.get("best_target_idx")
        if idx is not None and 0 <= int(idx) < len(target_boxes):
            return target_boxes[int(idx)], int(idx)
        ann_box, ann_idx = _select_annotation_target_box(row)
        return ann_box, ann_idx

    if not target_boxes or winner_start_idx is None:
        return _fallback()

    tp_scores = lightweight.get("target_point_score_ungated")
    if not tp_scores or winner_start_idx >= len(tp_scores):
        return _fallback()
    scores_for_winner = tp_scores[winner_start_idx]
    if len(scores_for_winner) != len(target_boxes):
        return _fallback()

    areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in target_boxes]
    max_area = max(areas) if areas else 0.0
    if max_area == 0.0:
        return _fallback()

    min_area = 0.0
    if start_box and len(start_box) >= 4:
        start_area = max(0.0, start_box[2] - start_box[0]) * max(0.0, start_box[3] - start_box[1])
        min_area = 0.5 * start_area

    density_scores = [
        float(pt) / (area / max_area) ** 0.5 if area >= min_area and area > 0 and float(pt) > 0 else 0.0
        for pt, area in zip(scores_for_winner, areas)
    ]
    if max(density_scores) == 0.0:
        return _fallback()

    best_idx = int(max(range(len(density_scores)), key=lambda i: density_scores[i]))
    return target_boxes[best_idx], best_idx


def _lightweight_winner_idx(row: dict) -> Optional[int]:
    lightweight = _lightweight_payload(row)
    for key in ("winner_idx", "new_winner_idx"):
        value = lightweight.get(key)
        if isinstance(value, int):
            return value

    override = row.get("lightweight_winner_override") or lightweight.get("lightweight_winner_override")
    if isinstance(override, dict):
        value = override.get("new_winner_idx", override.get("winner_idx"))
        if isinstance(value, int):
            return value

    winner_pick = lightweight.get("winner_pick")
    if isinstance(winner_pick, dict):
        value = winner_pick.get("best_start_idx", winner_pick.get("winner_idx"))
        if isinstance(value, int):
            return value
    return None


def _centroid_traces_count(row: dict) -> int:
    """Number of per-candidate traces in the arrays blob (0 when absent).

    Cached on the row: the blob is base64 + npz and decoding it per lookup is
    expensive on multi-million-entry mappings.
    """
    cached = row.get("_centroid_traces_count")
    if cached is not None:
        return int(cached)
    count = 0
    blob = _lightweight_payload(row).get("arrays_blob")
    if blob:
        try:
            raw = base64.b64decode(str(blob).encode("ascii"))
            with np.load(io.BytesIO(raw)) as npz:
                if "centroid_traces" in npz.files:
                    traces = np.asarray(npz["centroid_traces"])
                    if traces.ndim == 3:
                        count = int(traces.shape[0])
        except Exception:
            count = 0
    row["_centroid_traces_count"] = count
    return count


def _trace_idx_for_box(row: dict, box: list | None, preferred_idx: int | None = None) -> Optional[int]:
    candidate_boxes = _lightweight_candidate_boxes(row)
    # publication-v2 rows expose only the detector's box in candidate_boxes (a
    # 1-element list), while winner_idx and centroid_traces are both indexed
    # against the full original candidate array. Validating preferred_idx only
    # against candidate_boxes therefore rejected every nonzero winner and fell
    # through to index 0 -- silently returning candidate 0's trace instead of
    # the selected winner's. Accept the index if EITHER array can serve it.
    limit = max(len(candidate_boxes), _centroid_traces_count(row))
    if preferred_idx is not None and 0 <= int(preferred_idx) < limit:
        return int(preferred_idx)
    if box is None or not candidate_boxes:
        return None
    return max(range(len(candidate_boxes)), key=lambda i: _box_iou(candidate_boxes[i], box))


def _centroid_trace_from_arrays_blob(row: dict, candidate_idx: int | None) -> list[list[float]] | None:
    lightweight = _lightweight_payload(row)
    blob = lightweight.get("arrays_blob")
    if not blob or candidate_idx is None:
        return None
    try:
        raw = base64.b64decode(str(blob).encode("ascii"))
        with np.load(io.BytesIO(raw)) as npz:
            if "centroid_traces" not in npz.files:
                return None
            centroid = np.asarray(npz["centroid_traces"], dtype=float)
    except Exception:
        return None
    if centroid.ndim != 3 or centroid.shape[-1] < 2:
        return None
    if not (0 <= int(candidate_idx) < centroid.shape[0]):
        return None
    trace = centroid[int(candidate_idx)]
    return [[float(x), float(y)] for x, y in trace[:, :2]]


def _tracking_frame_indices_from_arrays_blob(
    row: dict,
    point_meta_key: str = "obj_traces_cotracker_point_meta",
) -> list[int] | None:
    point_meta = row.get(point_meta_key)
    if isinstance(point_meta, dict):
        indices = point_meta.get("frame_indices_relative_to_window")
        if isinstance(indices, (list, tuple)):
            return [int(i) for i in indices]

    lightweight = _lightweight_payload(row)
    blob = lightweight.get("arrays_blob")
    if not blob:
        return None
    try:
        raw = base64.b64decode(str(blob).encode("ascii"))
        with np.load(io.BytesIO(raw)) as npz:
            if "gripper_rep_tracking_frame_indices" not in npz.files:
                return None
            indices = np.asarray(npz["gripper_rep_tracking_frame_indices"], dtype=int)
    except Exception:
        return None
    if indices.ndim != 1:
        return None
    return [int(i) for i in indices.tolist()]


def _coord_size_from_arrays_blob(row: dict) -> tuple[float, float] | None:
    lightweight = _lightweight_payload(row)
    blob = lightweight.get("arrays_blob")
    if not blob:
        return None
    try:
        raw = base64.b64decode(str(blob).encode("ascii"))
        with np.load(io.BytesIO(raw)) as npz:
            for key in ("candidate_masks", "target_candidate_masks"):
                if key not in npz.files:
                    continue
                masks = np.asarray(npz[key])
                if masks.ndim >= 2:
                    height, width = masks.shape[-2:]
                    return float(width), float(height)
    except Exception:
        return None
    return None


def _coord_size_from_annotation(row: dict) -> tuple[float, float]:
    explicit_pairs = (
        ("image_width", "image_height"),
        ("width", "height"),
        ("frame_width", "frame_height"),
        ("video_width", "video_height"),
    )
    lightweight = _lightweight_payload(row)
    for source in (row, lightweight):
        for width_key, height_key in explicit_pairs:
            width = source.get(width_key)
            height = source.get(height_key)
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                return float(width), float(height)
        for key in ("image_size", "frame_size", "resolution"):
            value = source.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return float(value[0]), float(value[1])

    return _coord_size_from_arrays_blob(row) or DEFAULT_COORD_SIZE


def _trace_from_tracker(row: dict, preferred_keys: tuple[str, ...]) -> list[list[float]] | None:
    for key in preferred_keys:
        trace = row.get(key)
        if trace is None:
            continue
        arr = np.asarray(trace, dtype=float)
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            visible = np.isfinite(arr[:, :, 0]) & np.isfinite(arr[:, :, 1])
            points = []
            for t in range(arr.shape[0]):
                if not visible[t].any():
                    continue
                med = np.median(arr[t, visible[t], :2], axis=0)
                points.append([float(med[0]), float(med[1])])
            if len(points) >= 2:
                return points
        if arr.ndim == 2 and arr.shape[-1] >= 2:
            points = [[float(x), float(y)] for x, y in arr[:, :2] if np.isfinite(x) and np.isfinite(y)]
            if len(points) >= 2:
                return points
    return None


def _candidate_trace(row: dict, candidate_idx: int | None, fallback_keys: tuple[str, ...]) -> list[list[float]] | None:
    return _centroid_trace_from_arrays_blob(row, candidate_idx) or _trace_from_tracker(row, fallback_keys)


def _trace_tail_from_frame(
    trace: list[list[float]] | None,
    frame_indices: list[int] | None,
    frame: int,
) -> list[list[float]] | None:
    if not trace:
        return None
    if not frame_indices:
        start_idx = min(max(int(frame), 0), len(trace) - 1)
    else:
        limit = min(len(trace), len(frame_indices))
        if limit <= 0:
            return None
        start_idx = int(np.searchsorted(np.asarray(frame_indices[:limit], dtype=int), int(frame), side="left"))
        start_idx = min(max(start_idx, 0), limit - 1)
    return trace[start_idx:]


def _perpendicular_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    p = np.asarray(point, dtype=float)
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    denom = float(np.linalg.norm(e - s))
    if denom == 0.0:
        return float(np.linalg.norm(p - s))
    line = e - s
    rel = s - p
    return float(abs(line[0] * rel[1] - line[1] * rel[0]) / denom)


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    distances = [_perpendicular_distance(pt, points[0], points[-1]) for pt in points[1:-1]]
    if not distances:
        return points
    max_offset = int(np.argmax(distances)) + 1
    if distances[max_offset - 1] <= epsilon:
        return [points[0], points[-1]]
    left = _rdp(points[: max_offset + 1], epsilon)
    right = _rdp(points[max_offset:], epsilon)
    return left[:-1] + right


def _arc_length_sample(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    if n <= 0 or len(points) <= 1:
        return points
    if n == 1:
        return [points[-1]]

    pts = np.asarray(points, dtype=float)
    seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(seg_lens.sum())
    if total == 0.0:
        return [points[0] for _ in range(n)]

    targets = np.linspace(0.0, total, n)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    sampled: list[tuple[float, float]] = []
    for target in targets:
        seg_idx = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(seg_lens) - 1)
        seg_start = cumulative[seg_idx]
        seg_len = seg_lens[seg_idx]
        ratio = 0.0 if seg_len == 0.0 else (target - seg_start) / seg_len
        point = pts[seg_idx] + ratio * (pts[seg_idx + 1] - pts[seg_idx])
        sampled.append((float(point[0]), float(point[1])))
    return sampled


def _trace_arc(points: list[tuple[float, float]]) -> float:
    return sum(float(np.linalg.norm(np.asarray(points[i + 1]) - np.asarray(points[i]))) for i in range(len(points) - 1))


def _raw_trace_arc(trace: list[list[float]] | None) -> float:
    if not trace:
        return 0.0
    points = [(float(x), float(y)) for x, y, *_ in trace if np.isfinite(x) and np.isfinite(y)]
    if len(points) < 2:
        return 0.0
    return _trace_arc(points)


def _postprocess_trace(
    trace: list[list[float]] | None,
    snap_end_to_box: list | None = None,
    rdp_epsilon: float = 2.0,
    max_trace_pts: int | None = 8,
    arc_sample_pts: int = 5,
    snap_min_arc_px: float = 20.0,
) -> list[list[float]] | None:
    if not trace:
        return None
    points = [(float(x), float(y)) for x, y, *_ in trace if np.isfinite(x) and np.isfinite(y)]
    raw_arc = _trace_arc(points)
    if len(points) == 1:
        points = [points[0], points[0]]
    elif len(points) < 1:
        return None

    if rdp_epsilon > 0.0 and len(points) > 2:
        points = _rdp(points, rdp_epsilon)
    if max_trace_pts is not None and len(points) > max_trace_pts:
        if max_trace_pts < 2:
            return None
        indices = [int(round(i * (len(points) - 1) / (max_trace_pts - 1))) for i in range(max_trace_pts)]
        points = [points[i] for i in indices]
    if len(points) < 2:
        points = [points[0], points[0]]

    if arc_sample_pts and arc_sample_pts > 1:
        points = _arc_length_sample(points, arc_sample_pts)

    if snap_end_to_box is not None and raw_arc >= snap_min_arc_px:
        bx1, by1, bx2, by2 = (float(v) for v in snap_end_to_box)
        lx, ly = points[-1]
        if not (bx1 <= lx <= bx2 and by1 <= ly <= by2):
            points = points[:-1] + [((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)]

    return [[float(x), float(y)] for x, y in points]


# ──────────────────────────────────────────────────────────────────────────────
# Phase entry builder
# ──────────────────────────────────────────────────────────────────────────────

def build_phase_entries(
    ann: dict,
    box_format: bool = False,
    trace_format: bool = False,
    selection_mode: str = "ours",
    target_selection: str = "mode",
    coord_size: tuple[float, float] | None = None,
    human_prompt: str = "",
    trace_rdp_epsilon: float = 2.0,
    max_trace_pts: int | None = 8,
    trace_arc_sample_pts: int = 5,
    trace_snap_min_arc_px: float = 20.0,
    trace_min_arc_px: float = 20.0,
    prefer_mean_trace: bool = False,
    stats: dict | None = None,
) -> list[dict]:
    """
    Returns one dict per phase in ann['phase_annotations'].

    Each dict: {trajectory_name, start_frame, end_frame, conversations}
    conversations follows ShareGPT format:
        [{"from": "human", "value": <prompt template>},
         {"from": "gpt",   "value": <phase description + point>}]
    The human value may contain {instruction} which is filled in at training time.

    prefer_mean_trace: when True, swaps the _candidate_trace fallback order to
    prefer "obj_traces" (the per-frame mean of visible CoTracker points,
    already collapsed in trajectory_annotator.py) over "obj_traces_cotracker_point"
    (a single verified keypoint track). Only affects trajectories where the
    normally-preferred source is actually populated -- for annotation batches
    where "obj_traces_cotracker_point" is present (as it is for LIBERO-plus),
    this is the only way to get the mean-based trace instead of the point track.
    """
    phase_anns = ann.get("phase_annotations") or []
    if not phase_anns:
        return []

    traj_name = ann["trajectory_name"]
    task_obj_info = ann.get("task_obj_info", "")
    obj_lbl = _obj_label(task_obj_info)
    tgt_lbl = _target_label(task_obj_info)
    coord_size = coord_size or _coord_size_from_annotation(ann)
    trace_keys = (
        ("obj_traces", "obj_traces_cotracker_point")
        if prefer_mean_trace
        else ("obj_traces_cotracker_point", "obj_traces")
    )

    if selection_mode not in {"ours", "detection"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    init_box, _ = _select_box(ann.get("initial_object_box"), [], use_confidence=False)
    our_winner_idx = _lightweight_winner_idx(ann)

    if selection_mode == "detection":
        init_box, init_trace_idx = _select_lightweight_box(
            ann,
            "candidate_boxes",
            "candidate_confidences",
            use_confidence=True,
        )
        if init_box is None:
            init_box, _ = _select_box(ann.get("initial_object_box"), [], use_confidence=False)
            init_trace_idx = _trace_idx_for_box(ann, init_box)

        # naive max-confidence target selection
        cand_box, cand_idx = _select_detection_target_box(ann)
        if cand_box is None:
            cand_box, _ = _select_box(ann.get("target_candidate_boxes"), [], use_confidence=False)
        if cand_box is None:
            cand_box, cand_idx = _select_annotation_target_box(ann)
        cand_trace_idx = init_trace_idx
    else:
        # density-penalized track-point score target selection
        init_trace_idx = _trace_idx_for_box(ann, init_box, our_winner_idx)
        cand_box, cand_idx = _select_density_target_box(ann, our_winner_idx, init_box)
        cand_trace_idx = _trace_idx_for_box(ann, cand_box, our_winner_idx)

    entries = []
    if trace_format:
        if selection_mode == "detection" and ann.get("detector_baseline_trace"):
            full_trace = _trace_from_tracker(ann, ("detector_baseline_trace",))
            frame_indices = _tracking_frame_indices_from_arrays_blob(
                ann,
                point_meta_key="detector_baseline_trace_meta",
            )
        else:
            full_trace = _candidate_trace(
                ann,
                init_trace_idx,
                trace_keys,
            )
            frame_indices = _tracking_frame_indices_from_arrays_blob(ann)
        snap_box = cand_box if cand_box is not None else init_box
        if _raw_trace_arc(full_trace) < trace_min_arc_px:
            if stats is not None:
                stats["filtered_static_trajectories"] = stats.get("filtered_static_trajectories", 0) + 1
                stats["filtered_static_traces"] = stats.get("filtered_static_traces", 0) + sum(
                    int(ph["end_frame"]) - int(ph["start_frame"]) + 1
                    for ph in phase_anns
                    if ph.get("phase", "") in _GRASP_PHASES or ph.get("phase", "") in _PLACE_PHASES
                )
            return entries
        for ph in phase_anns:
            phase = ph.get("phase", "")
            if phase not in _GRASP_PHASES and phase not in _PLACE_PHASES:
                continue
            description = (ph.get("description") or "").strip()
            start = int(ph["start_frame"])
            end = int(ph["end_frame"])
            for frame in range(start, end + 1):
                trace_frame = frame - int(ann.get("start_frame", 0))
                trace_tail = _trace_tail_from_frame(full_trace, frame_indices, trace_frame)
                trace = _postprocess_trace(
                    trace_tail,
                    snap_end_to_box=snap_box,
                    rdp_epsilon=trace_rdp_epsilon,
                    max_trace_pts=max_trace_pts,
                    arc_sample_pts=trace_arc_sample_pts,
                    snap_min_arc_px=trace_snap_min_arc_px,
                )
                spatial = (
                    _qwen_trace(trace, coord_size)
                    if trace
                    else (
                        _format_box_or_point(snap_box or init_box, tgt_lbl, box_format, coord_size)
                        if (snap_box or init_box) is not None
                        else None
                    )
                )
                gpt_value = spatial or ""
                entries.append({
                    "trajectory_name": traj_name,
                    "start_frame": frame,
                    "end_frame": frame,
                    "conversations": [
                        {"from": "human", "value": human_prompt},
                        {"from": "gpt", "value": gpt_value},
                    ],
                })
        return entries

    for ph in phase_anns:
        phase = ph.get("phase", "")
        description = (ph.get("description") or "").strip()
        start = int(ph["start_frame"])
        end = int(ph["end_frame"])

        if phase in _GRASP_PHASES and init_box is not None:
            if trace_format:
                trace = _postprocess_trace(
                    _candidate_trace(ann, init_trace_idx, trace_keys),
                    snap_end_to_box=init_box,
                    rdp_epsilon=trace_rdp_epsilon,
                    max_trace_pts=max_trace_pts,
                    arc_sample_pts=trace_arc_sample_pts,
                    snap_min_arc_px=trace_snap_min_arc_px,
                )
                spatial = (
                    _qwen_trace(trace, obj_lbl, coord_size)
                    if trace
                    else _format_box_or_point(init_box, obj_lbl, box_format, coord_size)
                )
            elif box_format:
                spatial = _qwen_box(init_box, obj_lbl, coord_size)
            else:
                cx, cy = _box_center(init_box)
                spatial = _qwen_point(cx, cy, obj_lbl, coord_size)
        elif phase in _PLACE_PHASES and cand_box is not None:
            if trace_format:
                trace = _postprocess_trace(
                    _candidate_trace(ann, cand_trace_idx, trace_keys),
                    snap_end_to_box=cand_box,
                    rdp_epsilon=trace_rdp_epsilon,
                    max_trace_pts=max_trace_pts,
                    arc_sample_pts=trace_arc_sample_pts,
                    snap_min_arc_px=trace_snap_min_arc_px,
                )
                spatial = (
                    _qwen_trace(trace, tgt_lbl, coord_size)
                    if trace
                    else _format_box_or_point(cand_box, tgt_lbl, box_format, coord_size)
                )
            elif box_format:
                spatial = _qwen_box(cand_box, tgt_lbl, coord_size)
            else:
                cx, cy = _box_center(cand_box)
                spatial = _qwen_point(cx, cy, tgt_lbl, coord_size)
        else:
            spatial = None

        gpt_value = f"{description} {spatial}" if spatial else description

        entries.append({
            "trajectory_name": traj_name,
            "start_frame": start,
            "end_frame": end,
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt",   "value": gpt_value},
            ],
        })

    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _output_path_for_mode(out_path: Path, selection_mode: str, trace_format: bool) -> Path:
    parts = out_path.stem.split("_")
    if parts and parts[-1] == "trace":
        parts = parts[:-1]
    if "cot" in parts:
        cot_idx = len(parts) - 1 - parts[::-1].index("cot")
        parts.insert(cot_idx, selection_mode)
        if trace_format:
            parts.append("trace")
    else:
        if not parts or parts[-1] not in {"ours", "detection"}:
            parts.append(selection_mode)
        else:
            parts[-1] = selection_mode
        if trace_format:
            parts.append("trace")
    return out_path.with_name("_".join(parts) + out_path.suffix)


def _write_mapping(
    annotations: list[dict],
    out_path: Path,
    holdout: set[str],
    args: argparse.Namespace,
    selection_mode: str,
    trace_format: bool,
    coord_size_override: tuple[float, float] | None,
    max_trace_pts: int | None,
) -> tuple[int, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_holdout = 0
    skipped_no_phases = 0
    filtered_static = 0
    filtered_static_trajectories = 0
    seen_datasets: set[str] = set()
    samples: list[dict] = []

    with open(out_path, "w") as f_out:
        for ann in annotations:
            traj_name = ann["trajectory_name"]

            if traj_name in holdout:
                skipped_holdout += 1
                continue

            human_prompt = args.human_prompt or (DEFAULT_TRACE_HUMAN_PROMPT if trace_format else DEFAULT_HUMAN_PROMPT)
            stats: dict = {}
            entries = build_phase_entries(
                ann,
                box_format=args.box_format,
                trace_format=trace_format,
                selection_mode=selection_mode,
                coord_size=coord_size_override,
                human_prompt=human_prompt,
                trace_rdp_epsilon=args.trace_rdp_epsilon,
                max_trace_pts=max_trace_pts,
                trace_arc_sample_pts=args.trace_arc_sample_pts,
                trace_snap_min_arc_px=args.trace_snap_min_arc_px,
                trace_min_arc_px=args.trace_min_arc_px,
                prefer_mean_trace=args.prefer_mean_trace,
                stats=stats,
            )
            filtered_static += stats.get("filtered_static_traces", 0)
            filtered_static_trajectories += stats.get("filtered_static_trajectories", 0)
            if not entries:
                skipped_no_phases += 1
                continue

            for entry in entries:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
                if len(samples) < 5:
                    samples.append(entry)

                if args.verbose:
                    ds = traj_name.split("/")[0]
                    if ds not in seen_datasets:
                        seen_datasets.add(ds)
                        print(f"\n[{ds}] example trajectory ({len(entries)} phase entries):")
                        for e in entries:
                            human = e["conversations"][0]["value"]
                            gpt = e["conversations"][1]["value"]
                            print(f"  frames {e['start_frame']:>4} → {e['end_frame']:>4}")
                            print(f"    human: {human}")
                            print(f"    gpt:   {gpt[:120]}")

    print(f"\nDone. Wrote {written} entries to {out_path}")
    if skipped_holdout:
        print(f"  Skipped {skipped_holdout} entries (holdout trajectories).")
    if skipped_no_phases:
        print(f"  Skipped {skipped_no_phases} annotations with no phase_annotations.")
    if trace_format:
        print(
            f"  Filtered {filtered_static_trajectories} static trajectories "
            f"({filtered_static} frame entries) with < {args.trace_min_arc_px:g}px total motion."
        )

    print(f"\n{'─'*70}")
    print(f"Sample entries ({len(samples)} of {written}):")
    print(f"{'─'*70}")
    for s in samples:
        print(f"\n  trajectory : {s['trajectory_name']}")
        print(f"  frames     : {s['start_frame']} → {s['end_frame']}")
        print(f"  human      : {s['conversations'][0]['value']}")
        print(f"  gpt        : {s['conversations'][1]['value']}")
    print(f"{'─'*70}")

    return written, skipped_holdout, skipped_no_phases


def main():
    parser = argparse.ArgumentParser(description="Build phase-based CoT mapping JSONL from annotations.")
    parser.add_argument("--annotations", required=True, help="Path to annotations.jsonl")
    parser.add_argument("--output",      required=True, help="Output JSONL path")
    parser.add_argument(
        "--box_format",
        action="store_true",
        help="Use Qwen3.5VL box format instead of center points. Also used as trace fallback.",
    )
    parser.add_argument(
        "--trace_format",
        action="store_true",
        help="Use centroid traces from arrays_blob when available; falls back to point/box format otherwise.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=("ours", "detection"),
        default="ours",
        help=(
            "Box/trace selection mode. 'ours' uses annotation-selected boxes and lightweight winner trace; "
            "'detection' uses lightweight candidate confidences and target-candidate confidences."
        ),
    )
    parser.add_argument(
        "--best_box_by_confidence",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write all selection/trace combinations: ours/detection x point-or-box/trace.",
    )
    parser.add_argument(
        "--trace_rdp_epsilon",
        type=float,
        default=2.0,
        help="RDP simplification threshold for trace mode in pixels (0 disables simplification).",
    )
    parser.add_argument(
        "--max_trace_pts",
        type=int,
        default=8,
        help="Maximum trace points after RDP. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--trace_arc_sample_pts",
        type=int,
        default=5,
        help="Number of uniform arc-length keypoints to emit in trace mode. Use 0 to keep simplified points.",
    )
    parser.add_argument(
        "--trace_snap_min_arc_px",
        type=float,
        default=8.0,
        help="Only snap trace endpoint to the selected box center when total trace arc is at least this many pixels.",
    )
    parser.add_argument(
        "--trace_min_arc_px",
        type=float,
        default=10.0,
        help="Filter per-frame trace entries with less than this many pixels of remaining raw trajectory motion.",
    )
    parser.add_argument(
        "--prefer_mean_trace",
        action="store_true",
        help=(
            "Swap the trace-source fallback order to prefer 'obj_traces' (per-frame mean "
            "of visible CoTracker points) over 'obj_traces_cotracker_point' (a single "
            "verified keypoint track). Default (off) matches existing mapping files."
        ),
    )
    parser.add_argument(
        "--coord_width",
        type=float,
        default=None,
        help="Source coordinate width for 0-1000 normalization. Defaults to annotation metadata, NPZ mask width, then 256.",
    )
    parser.add_argument(
        "--coord_height",
        type=float,
        default=None,
        help="Source coordinate height for 0-1000 normalization. Defaults to annotation metadata, NPZ mask height, then 256.",
    )
    parser.add_argument(
        "--holdout_fraction",
        type=float,
        default=0.0,
        help="Fraction of unique trajectories to withhold from the mapping (0.0 = no holdout).",
    )
    parser.add_argument("--holdout_seed", type=int, default=42)
    parser.add_argument(
        "--human_prompt",
        default=None,
        help="Template for the human turn. Use {instruction} as placeholder. Defaults differ for trace and point/box modes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one example entry per dataset.")
    args = parser.parse_args()
    max_trace_pts = None if args.max_trace_pts == 0 else args.max_trace_pts
    selection_mode = "detection" if args.best_box_by_confidence else args.selection_mode
    coord_size_override = None
    if args.coord_width is not None or args.coord_height is not None:
        if args.coord_width is None or args.coord_height is None:
            parser.error("--coord_width and --coord_height must be passed together.")
        coord_size_override = (float(args.coord_width), float(args.coord_height))

    ann_path = Path(args.annotations)
    base_out_path = Path(args.output)
    base_out_path.parent.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(ann_path)
    print(f"Loaded {len(annotations)} annotation entries from {ann_path}")
    publication_v2_count = sum(bool(annotation.get("_publication_v2")) for annotation in annotations)
    missing_detector_baseline_count = sum(
        bool(annotation.get("_publication_v2"))
        and not annotation.get("_publication_detector_baseline")
        for annotation in annotations
    )
    if missing_detector_baseline_count and (args.all or selection_mode == "detection"):
        print(
            f"Note: {missing_detector_baseline_count} of {publication_v2_count} publication-v2 rows "
            "contain only the selected boxes/track. "
            "Their detection-mode outputs fall back to those published selections because rejected "
            "detector candidates are not available in those files."
        )

    # Holdout
    all_trajs = sorted({a["trajectory_name"] for a in annotations})
    holdout: set[str] = set()
    if args.holdout_fraction > 0.0:
        rng = random.Random(args.holdout_seed)
        n_holdout = max(1, int(len(all_trajs) * args.holdout_fraction))
        holdout = set(rng.sample(all_trajs, n_holdout))
        print(f"Holding out {len(holdout)} / {len(all_trajs)} trajectories (seed={args.holdout_seed})")
        holdout_path = base_out_path.with_suffix(".holdout_trajectories.txt")
        holdout_path.write_text("\n".join(sorted(holdout)) + "\n")
        print(f"Holdout list written to {holdout_path}")

    jobs = []
    if args.all:
        jobs = [(mode, trace) for mode in ("ours", "detection") for trace in (False, True)]
    else:
        jobs = [(selection_mode, args.trace_format)]

    for job_selection_mode, job_trace_format in jobs:
        out_path = _output_path_for_mode(base_out_path, job_selection_mode, job_trace_format)
        print(f"\nWriting selection_mode={job_selection_mode}, trace_format={job_trace_format} → {out_path}")
        _write_mapping(
            annotations,
            out_path,
            holdout,
            args,
            job_selection_mode,
            job_trace_format,
            coord_size_override,
            max_trace_pts,
        )


if __name__ == "__main__":
    main()
