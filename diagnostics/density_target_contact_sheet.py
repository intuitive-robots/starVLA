"""
Debug contact sheet: density-heuristic target selection vs annotation target.

Black background, 256×256 canvas per panel:
  green  = initial object box
  red    = annotation target_object_box
  cyan   = density-heuristic selected target
  yellow = raw trace (numbered dots + connecting lines)
  pink   = trace endpoint dot

Title: "<idx> <traj> | density=HIT/MISS ann=HIT/MISS"
"""

import json, math, textwrap
from pathlib import Path

import cv2
import numpy as np

from scripts.create_cot_mapping import (
    _select_box, _lightweight_winner_idx, _trace_idx_for_box,
    _candidate_trace, _raw_trace_arc, _lightweight_payload,
    _normalise_boxes, _target_candidate_boxes, _postprocess_trace,
)

ANN_PATH = Path("/e/home/jusers/blank4/jupiter/blank4/annotations/reverted_changes/ann_run_470269/ours_real_robot/annotations.jsonl")
OUT_DIR  = Path("/e/project1/m3/blank4/code/starVLA/diagnostics/density_target_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS   = 256
COLS     = 3
N_PANELS = 36   # how many trajectories to render

# ── geometry helpers ──────────────────────────────────────────────────────────

def pt_in_box(pt, box):
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1 <= x <= x2 and y1 <= y <= y2

def box_area(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def dist_pt_box(pt, box):
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = (float(v) for v in box)
    return math.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2))

# ── density heuristic ─────────────────────────────────────────────────────────

def select_density_target(row, lw_winner, init_box=None):
    lw = _lightweight_payload(row)
    raw_tb = lw.get("target_candidate_boxes") or []
    target_boxes = [
        [float(v) for v in b[:4]]
        for b in raw_tb if isinstance(b, (list, tuple)) and len(b) >= 4
    ]
    def fallback():
        orig, _ = _select_box(row.get("target_object_box"), [], use_confidence=False)
        if orig is None:
            orig, _ = _select_box(row.get("target_candidate_boxes"), [], use_confidence=False)
        return orig

    if not target_boxes or lw_winner is None:
        return fallback()
    tp_scores = lw.get("target_point_score_ungated")
    if not tp_scores or lw_winner >= len(tp_scores):
        return fallback()
    scores = tp_scores[lw_winner]
    if len(scores) != len(target_boxes):
        return fallback()
    areas = [box_area(b) for b in target_boxes]
    max_area = max(areas) if areas else 0.0
    if max_area == 0.0:
        return fallback()
    min_area = 0.5 * box_area(init_box) if init_box else 0.0
    density_scores = [
        float(pt) / (area / max_area) ** 0.5
        if area >= min_area and area > 0 and float(pt) > 0 else 0.0
        for pt, area in zip(scores, areas)
    ]
    if max(density_scores) == 0.0:
        return fallback()
    best_idx = max(range(len(density_scores)), key=lambda i: density_scores[i])
    return target_boxes[best_idx]

# ── drawing helpers ───────────────────────────────────────────────────────────

def draw_box(img, box, color, thickness=1, label=None):
    if box is None:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 3, 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

def draw_trace(img, trace, color=(0, 220, 220), dot_color=(180, 0, 220)):
    if not trace or len(trace) < 1:
        return
    pts = [(int(round(p[0])), int(round(p[1]))) for p in trace]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (pt[0] + 4, pt[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)
    # endpoint: pink
    cv2.circle(img, pts[-1], 4, dot_color, -1, cv2.LINE_AA)

def add_title(img, text, color=(200, 200, 200)):
    lines = textwrap.wrap(text, width=36)
    y = 9
    for line in lines[:2]:
        cv2.putText(img, line, (2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, color, 1, cv2.LINE_AA)
        y += 9

# ── main ──────────────────────────────────────────────────────────────────────

rows = [json.loads(l) for l in ANN_PATH.open() if l.strip()]

panels = []
panel_idx = 0

for ann in rows:
    if panel_idx >= N_PANELS:
        break

    init_box, _ = _select_box(ann.get("initial_object_box"), [], use_confidence=False)
    lw_winner   = _lightweight_payload(ann).get("winner_idx")
    tr_idx      = _trace_idx_for_box(ann, init_box, lw_winner)
    raw_trace   = _candidate_trace(ann, tr_idx, ("obj_traces_cotracker_point", "obj_traces"))

    if _raw_trace_arc(raw_trace) < 20:
        continue

    final = raw_trace[-1]

    ann_box, _ = _select_box(ann.get("target_object_box"), [], use_confidence=False)
    if ann_box is None:
        ann_box, _ = _select_box(ann.get("target_candidate_boxes"), [], use_confidence=False)

    density_box = select_density_target(ann, lw_winner, init_box)

    density_hit = density_box is not None and pt_in_box(final, density_box)
    ann_hit     = ann_box is not None and pt_in_box(final, ann_box)

    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)

    # all target candidates: dim gray
    for cand in (_target_candidate_boxes(ann) or []):
        draw_box(img, cand, (40, 40, 40), thickness=1)

    # annotation target: red
    draw_box(img, ann_box,     (0, 50, 200), thickness=1, label="ann")
    # density target: cyan
    draw_box(img, density_box, (200, 200, 0), thickness=2, label="density")
    # initial box: green
    draw_box(img, init_box,    (0, 200, 0),  thickness=1, label="init")

    # postprocess to the same 5-point arc-sampled trace that gets exported
    tr = _postprocess_trace(
        raw_trace,
        snap_end_to_box=density_box,
        rdp_epsilon=2.0,
        max_trace_pts=8,
        arc_sample_pts=5,
        snap_min_arc_px=20.0,
    )
    draw_trace(img, tr)

    traj_short = "/".join(ann["trajectory_name"].split("/")[-2:])
    d_str = "HIT" if density_hit else "MISS"
    a_str = "HIT" if ann_hit else "MISS"
    title = f"{panel_idx:02d} {traj_short} d={d_str} a={a_str}"
    title_color = (0, 200, 0) if density_hit else (0, 0, 220)
    add_title(img, title, title_color)

    panels.append(img)
    panel_idx += 1

# assemble contact sheet
rows_count = math.ceil(len(panels) / COLS)
sheet_h = rows_count * CANVAS
sheet_w = COLS * CANVAS
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
for i, panel in enumerate(panels):
    r, c = divmod(i, COLS)
    sheet[r * CANVAS:(r + 1) * CANVAS, c * CANVAS:(c + 1) * CANVAS] = panel

out_path = OUT_DIR / "density_target_contact_sheet.png"
cv2.imwrite(str(out_path), sheet)
print(f"Wrote {out_path}  ({sheet_w}×{sheet_h}px, {len(panels)} panels)")
