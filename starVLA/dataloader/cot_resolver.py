"""
CoT (Chain-of-Thought) resolver for step-based annotation lookup.

Maps (trajectory_name, frame_index) → conversations using a pre-built JSONL mapping.
Trajectory names follow the lerobot convention: "{dataset_name}/{chunk_idx}/{file_idx}".
The mapping is built from annotations.jsonl via scripts/create_cot_mapping.py.

Return value of resolve() is a ShareGPT-style conversations list:
    [{"from": "human", "value": "<prompt with {instruction} placeholder>"},
     {"from": "gpt",   "value": "<assistant response with point/box>"}]

Backward compat: entries with legacy "cot_text" field are wrapped automatically.
"""

import json
import logging
import random
import re
import sqlite3
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_CAM_BLOCK = re.compile(r"<(?P<cam>cam1|cam2)>(?P<body>.*?)</(?P=cam)>", re.DOTALL)
_POINT = re.compile(r"<point>\s*\((-?\d+),\s*(-?\d+)\)\s*</point>")
_BOX = re.compile(
    r"<box>\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s*</box>"
)
_TRAJECTORY2D = re.compile(r"<trajectory(?:\s+[^>]*)?>(.*?)</trajectory>", re.DOTALL)
_TRAJECTORY3D = re.compile(r"<trajectory3d(?:\s+[^>]*)?>(.*?)</trajectory3d>", re.DOTALL)
_XY = re.compile(r"\((-?\d+),\s*(-?\d+)\)")
_XYZ = re.compile(r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)")


def extract_structured_cot_targets(conversation: Optional[list]) -> dict[str, list[float]]:
    """Parse augmentation-aligned regression targets from a LIBERO CoT answer.

    The worker calls this only after ``CoTVideoAugment`` has rewritten the assistant
    coordinates. Values are normalized here so the regression losses have comparable
    scales: image coordinates are stored in [0, 1000], and cam2 trajectories are signed
    centimeter deltas whose practical range is about [-20, 20]. Missing or malformed
    fields are simply absent; downstream losses mask them rather than inventing labels.
    """
    if not conversation:
        return {}
    assistant = next(
        (turn.get("value", "") for turn in conversation if turn.get("from") == "gpt"),
        "",
    )
    if not isinstance(assistant, str) or not assistant:
        return {}
    cameras = {m.group("cam"): m.group("body") for m in _CAM_BLOCK.finditer(assistant)}
    cam1, cam2 = cameras.get("cam1", ""), cameras.get("cam2", "")
    result: dict[str, list[float]] = {}

    point = _POINT.search(cam1)
    if point:
        result["target_point"] = [float(point.group(i)) / 1000.0 for i in (1, 2)]
    box = _BOX.search(cam1)
    if box:
        result["object_box"] = [float(box.group(i)) / 1000.0 for i in range(1, 5)]

    traj2d = _TRAJECTORY2D.search(cam1)
    if traj2d:
        points = _XY.findall(traj2d.group(1))
        if len(points) == 5:
            result["trajectory2d"] = [float(v) / 1000.0 for point in points for v in point]

    traj3d = _TRAJECTORY3D.search(cam2)
    if traj3d:
        points = _XYZ.findall(traj3d.group(1))
        if len(points) == 5:
            result["trajectory3d"] = [float(v) / 20.0 for point in points for v in point]
    return result


class CoTResolver:
    def resolve(
        self,
        trajectory_name: str,
        frame_index: int,
        *,
        camera_view: str | None = None,
    ) -> Optional[list]:
        raise NotImplementedError


class NullCoTResolver(CoTResolver):
    """Always returns None — no CoT supervision (backward-compatible default)."""

    def resolve(
        self,
        trajectory_name: str,
        frame_index: int,
        *,
        camera_view: str | None = None,
    ) -> Optional[list]:
        return None


class MappingCoTResolver(CoTResolver):
    """
    Step-based CoT lookup from a JSONL mapping file.

    Each line in the mapping file must have:
        trajectory_name  str        e.g. "cylinder_cube_full/0/1"
        start_frame      int        inclusive
        end_frame        int        inclusive
        conversations    list[dict] ShareGPT format: [{from, value}, {from, value}]
                         — OR legacy "cot_text" str (auto-wrapped for backward compat)

    For a given (trajectory_name, frame_index), returns the conversations list of the
    first interval [start_frame, end_frame] that contains frame_index, or None if no
    interval matches (holdout / unannotated steps → no CoT loss).
    """

    def __init__(self, mapping_path: str):
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"CoT mapping file not found: {mapping_path}")

        # index: trajectory_name → sorted list of (start_frame, end_frame, conversations)
        self._index: dict[str, list[tuple[int, int, list]]] = {}
        # Fast path for PER-FRAME annotations (start == end), where the interval list
        # degenerates to one entry per frame and a linear scan is O(episode length) per
        # sample. The simulator-derived LIBERO mappings are per-frame and ~273k lines, and
        # this object is rebuilt in EVERY dataloader worker, so both the scan and the
        # duplicated strings matter.
        self._exact: dict[tuple[str, int], list] = {}
        self._all_exact = True
        # Human turns are identical across every line of a mapping; interning collapses
        # 273k copies of the prompt to one.
        _intern: dict[str, str] = {}
        with open(mapping_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = entry["trajectory_name"]

                if "conversations" in entry:
                    conversations = entry["conversations"]
                else:
                    # Backward compat: wrap legacy cot_text as gpt-only conversation
                    conversations = [
                        {"from": "human", "value": "{instruction}"},
                        {"from": "gpt",   "value": entry["cot_text"]},
                    ]

                for turn in conversations:
                    v = turn.get("value")
                    if isinstance(v, str):
                        turn["value"] = _intern.setdefault(v, v)

                start, end = int(entry["start_frame"]), int(entry["end_frame"])
                self._index.setdefault(key, []).append((start, end, conversations))
                if start == end:
                    self._exact[(key, start)] = conversations
                else:
                    self._all_exact = False

        for key in self._index:
            self._index[key].sort(key=lambda x: x[0])

        total_intervals = sum(len(v) for v in self._index.values())
        if self._all_exact:
            self._index.clear()          # fully covered by the exact map
        print(
            f"[CoTResolver] Loaded {total_intervals} intervals across "
            f"{len(self._exact) and len({k[0] for k in self._exact}) or len(self._index)} "
            f"trajectories from {mapping_path}"
            f"{' (per-frame, exact lookup)' if self._all_exact else ''}"
        )

    @property
    def human_prompt_template(self) -> Optional[str]:
        """Return the human turn template from the first mapping entry, or None."""
        for intervals in self._index.values():
            if intervals:
                convs = intervals[0][2]
                if convs and convs[0].get("from") == "human":
                    return convs[0]["value"]
        for convs in self._exact.values():        # per-frame mappings clear _index
            if convs and convs[0].get("from") == "human":
                return convs[0]["value"]
        return None

    def resolve(
        self,
        trajectory_name: str,
        frame_index: int,
        *,
        camera_view: str | None = None,
    ) -> Optional[list]:
        hit = self._exact.get((trajectory_name, frame_index))
        if hit is not None:
            return hit
        intervals = self._index.get(trajectory_name)
        if not intervals:
            return None
        for start, end, conversations in intervals:
            if start <= frame_index <= end:
                return conversations
        return None

    @property
    def num_trajectories(self) -> int:
        return len(self._index) or len({k[0] for k in self._exact})


class SparcSQLiteCoTResolver(CoTResolver):
    """Indexed worker-side renderer for compact SPARC DROID mappings."""

    def __init__(self, mapping_path: str, *, camera_view: str | None = None,
                 variant_mode: str = "random", seed: int = 42,
                 require_rewrite: bool = False, motion_cache_size: int = 16,
                 min_selection_score: float | None = None,
                 immutable: bool = True):
        self.mapping_path = str(Path(mapping_path).resolve())
        if not Path(self.mapping_path).exists():
            raise FileNotFoundError(f"SPARC CoT SQLite mapping not found: {self.mapping_path}")
        self.camera_view = camera_view
        self.variant_mode = str(variant_mode).lower()
        if self.variant_mode not in {"random", "hash", "first"}:
            raise ValueError("variant_mode must be one of: random, hash, first")
        self.seed = int(seed)
        self.require_rewrite = bool(require_rewrite)
        self.motion_cache_size = max(1, int(motion_cache_size))
        self.min_selection_score = (
            None if min_selection_score is None else float(min_selection_score)
        )
        self.immutable = bool(immutable)
        self._connection = None
        self._pid = None
        self._rng = random.Random(self.seed)
        self._motion_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray] | None] = OrderedDict()

        connection = self._connect()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        schema = metadata.get("schema_version")
        if schema != "sparc-cot-sqlite-v1":
            raise ValueError(f"unsupported SPARC CoT mapping schema: {schema!r}")
        self._human_prompt_template = metadata.get("human_prompt")
        self.action_horizon = int(metadata.get("action_horizon", 20))
        self.waypoints = int(metadata.get("waypoints", 5))
        self.camera_views = tuple(row[0] for row in connection.execute(
            "SELECT DISTINCT camera_view FROM annotations ORDER BY camera_view"))
        if self.camera_view is None and len(self.camera_views) == 1:
            self.camera_view = self.camera_views[0]
        if self.camera_view is not None and self.camera_view not in self.camera_views:
            raise ValueError(
                f"SPARC camera {self.camera_view!r} is absent from the mapping; "
                f"available cameras: {list(self.camera_views)}"
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_pid"] = None
        state["_motion_cache"] = OrderedDict()
        return state

    def _connect(self) -> sqlite3.Connection:
        import os
        pid = os.getpid()
        if self._connection is None or self._pid != pid:
            if self._connection is not None:
                self._connection.close()
            uri = f"file:{self.mapping_path}?mode=ro"
            if self.immutable:
                # The generated SPARC mapping is finalized before training. Immutable mode
                # avoids SQLite trying to create WAL shared-memory sidecars from every
                # dataloader worker (and still works when the project quota is full).
                uri += "&immutable=1"
            self._connection = sqlite3.connect(
                uri, uri=True, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only=ON")
            self._connection.execute("PRAGMA mmap_size=268435456")
            self._pid = pid
            self._motion_cache = OrderedDict()
            self._rng = random.Random(self.seed ^ pid)
        return self._connection

    @property
    def human_prompt_template(self) -> Optional[str]:
        return self._human_prompt_template

    @property
    def num_trajectories(self) -> int:
        if self.min_selection_score is None:
            sql = """SELECT count(DISTINCT a.source_uuid)
                       FROM annotations a
                       JOIN annotation_rewrites r ON r.annotation_id=a.annotation_id
                      WHERE r.status='success'"""
            parameters = ()
        else:
            sql = """SELECT count(DISTINCT a.source_uuid)
                       FROM annotations a
                       JOIN annotation_rewrites r ON r.annotation_id=a.annotation_id
                      WHERE r.status='success' AND a.selection_score>?"""
            parameters = (self.min_selection_score,)
        return int(self._connect().execute(sql, parameters).fetchone()[0])

    @staticmethod
    def _decode_blob(blob: bytes | None, dtype: str, width: int = 1) -> np.ndarray:
        if blob is None:
            shape = (0, width) if width > 1 else (0,)
            return np.empty(shape, dtype=dtype)
        value = np.frombuffer(zlib.decompress(blob), dtype=dtype)
        return value.reshape(-1, width) if width > 1 else value

    @staticmethod
    def _norm_xy(xy: np.ndarray, width: int, height: int) -> list[tuple[int, int]]:
        result = np.rint(xy * np.asarray([1000.0 / width, 1000.0 / height])).astype(int)
        result = np.clip(result, 0, 1000)
        return [(int(x), int(y)) for x, y in result]

    @staticmethod
    def _interpolate(frames: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        unique_frames, unique_indices = np.unique(frames, return_index=True)
        unique_values = values[unique_indices]
        return np.stack([
            np.interp(query, unique_frames, unique_values[:, axis])
            for axis in range(unique_values.shape[1])], axis=1)

    def _variant(self, row: sqlite3.Row, frame_index: int) -> dict[str, str]:
        phase_groups = (
            json.loads(row["variants_by_phase_json"])
            if row["variants_by_phase_json"] else []
        )
        phase_index = int(row["phase_index"])
        variants = next(
            (
                item.get("variants", [])
                for item in phase_groups
                if int(item.get("phase_index", -1)) == phase_index
            ),
            [],
        )
        if variants:
            if self.variant_mode == "random":
                return variants[self._rng.randrange(len(variants))]
            if self.variant_mode == "hash":
                import hashlib
                # Keep one stable wording throughout a phase at evaluation time.
                value = f"{self.seed}\0{row['rewrite_key']}".encode("utf-8")
                index = int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")
                return variants[index % len(variants)]
            return variants[0]
        if self.require_rewrite:
            raise RuntimeError(
                f"annotation {row['annotation_id']} phase {phase_index} "
                "has no successful VLM rewrite"
            )
        description = str(row["description"] or row["phase_type"])
        obj, target = str(row["object_label"] or "object"), str(row["target_label"] or "")
        reasoning = (f"Continue the {description} phase with the {obj} toward the {target}."
                     if target else
                     f"Continue the {description} phase while controlling the {obj}.")
        return {"subtask": description, "reasoning": reasoning}

    def _motion(self, source_uuid: str) -> tuple[np.ndarray, np.ndarray] | None:
        if source_uuid in self._motion_cache:
            cached = self._motion_cache.pop(source_uuid)
            self._motion_cache[source_uuid] = cached
            return cached
        row = self._connect().execute(
            "SELECT xyz_zlib,gripper_zlib FROM episode_motion WHERE source_uuid=?",
            (source_uuid,)).fetchone()
        if row is None:
            return None
        cached = (self._decode_blob(row["xyz_zlib"], "<f4", 3),
                  self._decode_blob(row["gripper_zlib"], "<f4"))
        self._motion_cache[source_uuid] = cached
        while len(self._motion_cache) > self.motion_cache_size:
            self._motion_cache.popitem(last=False)
        return cached

    def _movement_text(self, row: sqlite3.Row, frame_index: int, end_frame: int) -> str:
        motion = self._motion(row["source_uuid"])
        if motion is None or not len(motion[0]):
            return f"continue the {row['phase_type']} motion"
        xyz, gripper = motion
        start = min(max(0, frame_index), len(xyz) - 1)
        end = min(max(start, end_frame), len(xyz) - 1)
        delta_cm = (xyz[end] - xyz[start]) * 100.0
        parts = []
        for value, positive, negative in (
            (delta_cm[0], "forward", "backward"),
            (delta_cm[1], "left", "right"),
            (delta_cm[2], "up", "down"),
        ):
            if abs(float(value)) >= 1.0:
                parts.append(f"move {positive if value > 0 else negative} {abs(float(value)):.0f} cm")
        if len(gripper):
            g0 = float(gripper[min(start, len(gripper) - 1)])
            g1 = float(gripper[min(end, len(gripper) - 1)])
            if g1 - g0 > 0.2:
                parts.append("close the gripper")
            elif g0 - g1 > 0.2:
                parts.append("open the gripper")
            elif g1 > 0.5:
                parts.append("keep the gripper closed")
            else:
                parts.append("keep the gripper open")
        return ", ".join(parts) if parts else "hold the current pose"

    def resolve(
        self,
        trajectory_name: str,
        frame_index: int,
        *,
        camera_view: str | None = None,
    ) -> Optional[list]:
        selected_camera = camera_view or self.camera_view
        if selected_camera is None:
            raise ValueError(
                "SPARC mapping contains multiple cameras; resolve() requires the "
                "camera_view selected for this sample"
            )
        if selected_camera not in self.camera_views:
            return None
        row = self._connect().execute(
            """SELECT a.*,p.phase_index,p.phase_type,p.start_frame AS phase_start,
                      p.end_frame AS phase_end,p.description,p.rewrite_key,
                      r.variants_by_phase_json
                 FROM annotations a
                 JOIN phases p ON p.annotation_id=a.annotation_id
                 JOIN annotation_rewrites r ON r.annotation_id=a.annotation_id
                WHERE a.source_uuid=? AND a.camera_view=?
                  AND r.status='success'
                  AND a.start_frame<=? AND a.end_frame>=?
                  AND p.start_frame<=? AND p.end_frame>=?
                  AND (? IS NULL OR a.selection_score>?)
                ORDER BY a.start_frame DESC,p.phase_index LIMIT 1""",
            (trajectory_name, selected_camera, frame_index, frame_index,
             frame_index, frame_index,
             self.min_selection_score, self.min_selection_score)).fetchone()
        if row is None:
            return None

        variant = self._variant(row, frame_index)
        horizon_end = min(frame_index + self.action_horizon - 1,
                          int(row["phase_end"]), int(row["end_frame"]))
        lines = [f"<subtask>{variant['subtask']}</subtask>", "<cam1>"]
        trace_frames = self._decode_blob(row["trace_frames_blob"], "<i4")
        trace_xy = self._decode_blob(row["trace_xy_blob"], "<f4", 2)
        anchors = json.loads(row["anchor_boxes_json"] or "[]")
        if anchors:
            frames = np.asarray([item[0] for item in anchors], dtype=np.int32)
            boxes = np.asarray([item[1] for item in anchors], dtype=np.float32)
            box = self._interpolate(frames, boxes, np.asarray([frame_index]))[0]
            # SPARC's selected visible-point centroid is the identity-consistent
            # object track. Some sparse intermediate box anchors drift to another
            # candidate, so use anchors for size but the selected track for center.
            if len(trace_frames) and len(trace_xy):
                center = self._interpolate(
                    trace_frames, trace_xy, np.asarray([frame_index]))[0]
                half_size = (box[2:] - box[:2]) * 0.5
                box = np.concatenate([center - half_size, center + half_size])
            x1, y1 = self._norm_xy(box[None, :2], row["image_w"], row["image_h"])[0]
            x2, y2 = self._norm_xy(box[None, 2:], row["image_w"], row["image_h"])[0]
            lines.append(f"<object>{row['object_label']} <box>({x1},{y1},{x2},{y2})</box></object>")
        if row["target_box_json"]:
            box = np.asarray(json.loads(row["target_box_json"]), dtype=np.float32)
            x, y = self._norm_xy(((box[:2] + box[2:]) * 0.5)[None, :],
                                 row["image_w"], row["image_h"])[0]
            lines.append(f"<target>{row['target_label'] or 'target'} <point>({x},{y})</point></target>")
        if len(trace_frames) and len(trace_xy):
            points = self._interpolate(
                trace_frames, trace_xy,
                np.linspace(frame_index, horizon_end, self.waypoints))
            body = " ".join(
                f"({x},{y})" for x, y in self._norm_xy(
                    points, row["image_w"], row["image_h"]))
            lines.append(f"<trajectory type=\"object\">{body}</trajectory>")
        lines.extend([
            "</cam1>",
            f"<movement>{self._movement_text(row, frame_index, horizon_end)}</movement>",
            f"<reasoning>{variant['reasoning']}</reasoning>",
        ])
        return [
            {"from": "human", "value": self._human_prompt_template},
            {"from": "gpt", "value": "\n".join(lines)},
        ]


def assert_cot_prompt_consistent(resolver: CoTResolver, cfg) -> None:
    """
    Assert that the human prompt template stored in the mapping matches the
    CoT_prompt configured in datasets.vla_data.  Mismatches mean the model
    would see a different user question at inference (two-pass CoT generation)
    vs. what was used during mapping creation.

    Only runs for resolvers with a recoverable template.
    """
    mapping_template = getattr(resolver, "human_prompt_template", None)
    if mapping_template is None:
        return

    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    if vla_data is None:
        return
    config_prompt = vla_data.get("CoT_prompt") if hasattr(vla_data, "get") else getattr(vla_data, "CoT_prompt", None)
    if config_prompt is None:
        return

    assert mapping_template == config_prompt, (
        f"\n[CoTResolver] Human prompt template mismatch!\n"
        f"  mapping file : {mapping_template!r}\n"
        f"  CoT_prompt   : {config_prompt!r}\n"
        f"Regenerate the CoT mapping with --human_prompt matching CoT_prompt in the YAML, "
        f"or update CoT_prompt in the YAML to match."
    )


def build_cot_resolver(cfg, is_inference: bool = False) -> CoTResolver:
    """
    Factory — reads datasets.vla_data.cot config block and returns
    the appropriate resolver. Falls back to NullCoTResolver when the
    block is absent or source is 'none'.

    Config schema (all optional):
        datasets:
          vla_data:
            cot:
              source: mapping      # 'mapping' | 'sparc_sqlite' | 'none'
              mapping_path: path/to/cot_mapping.jsonl
              dropout_enabled: true # training-only; sampled in dataloader workers
              dropout_rate: 0.5     # ERVLA default
              loss_scale: 0.1      # consumed in train_starvla.py, not here
    """
    cot_cfg = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    cot_cfg = getattr(cot_cfg, "cot", None) if cot_cfg is not None else None

    if cot_cfg is None:
        return NullCoTResolver()

    source = cot_cfg.get("source", "none") if hasattr(cot_cfg, "get") else getattr(cot_cfg, "source", "none")
    if source == "none" or source is None:
        return NullCoTResolver()

    if source == "mapping":
        mapping_path = cot_cfg.get("mapping_path") if hasattr(cot_cfg, "get") else getattr(cot_cfg, "mapping_path")
        if not mapping_path:
            if is_inference:
                logger.warning(
                    "[CoTResolver] Inference mode requested CoT mapping source but no mapping_path was set. "
                    "Falling back to NullCoTResolver."
                )
                return NullCoTResolver()
            raise ValueError("cot.source='mapping' requires cot.mapping_path to be set")
        if is_inference and not Path(mapping_path).exists():
            logger.warning(
                "[CoTResolver] Inference mode could not find mapping file at %s. "
                "Falling back to NullCoTResolver.",
                mapping_path,
            )
            return NullCoTResolver()
        return MappingCoTResolver(mapping_path)

    if source == "sparc_sqlite":
        get = cot_cfg.get if hasattr(cot_cfg, "get") else lambda key, default=None: getattr(cot_cfg, key, default)
        mapping_path = get("mapping_path")
        if not mapping_path:
            raise ValueError("cot.source='sparc_sqlite' requires cot.mapping_path to be set")
        if is_inference and not Path(mapping_path).exists():
            logger.warning(
                "[CoTResolver] Inference mode could not find SPARC SQLite mapping at %s. "
                "Falling back to NullCoTResolver.", mapping_path)
            return NullCoTResolver()
        return SparcSQLiteCoTResolver(
            mapping_path,
            camera_view=get("camera_view"),
            variant_mode="hash" if is_inference else get("variant_mode", "random"),
            seed=int(get("variant_seed", 42)),
            require_rewrite=bool(get("require_rewrite", False)),
            motion_cache_size=int(get("motion_cache_size", 16)),
            min_selection_score=get("min_selection_score"),
            immutable=bool(get("immutable", True)),
        )

    raise ValueError(
        f"Unknown cot.source: '{source}'. Expected 'mapping', 'sparc_sqlite', or 'none'.")
