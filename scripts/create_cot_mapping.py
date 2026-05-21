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

Phase → spatial target mapping:
    grasp              → center of initial_object_box[0],   label = task_obj_info.object
    interact / release → center of target_candidate_boxes[0] (winner box),
                         label = task_obj_info.target_location

Trajectories with only an "interact" phase (no grasp) are handled correctly.

Usage
─────
    python scripts/create_cot_mapping.py \\
        --annotations /path/to/annotations.jsonl \\
        --output      data/cot_mappings/real_robot_cot.jsonl \\
        [--box_format]
        [--holdout_fraction 0.1]
        [--holdout_seed 42]
        [--verbose]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Spatial formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

_GRASP_PHASES = {"grasp"}
_PLACE_PHASES = {"interact", "release", "transport", "carry"}


def _box_center(box: list) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _qwen_point(x: int, y: int, label: str) -> str:
    payload = json.dumps({"point_2d": [x, y], "label": label}, ensure_ascii=False)
    return f"<|point|>{payload}<|/point|>"


def _qwen_box(box: list, label: str) -> str:
    x1, y1, x2, y2 = (int(v) for v in box)
    payload = json.dumps({"bbox_2d": [x1, y1, x2, y2], "label": label}, ensure_ascii=False)
    return f"<|box|>{payload}<|/box|>"


def _obj_label(task_obj_info) -> str:
    if isinstance(task_obj_info, dict):
        return task_obj_info.get("object") or "object"
    return str(task_obj_info) if task_obj_info else "object"


def _target_label(task_obj_info) -> str:
    if isinstance(task_obj_info, dict):
        return task_obj_info.get("target_location") or "target location"
    return "target location"


# ──────────────────────────────────────────────────────────────────────────────
# Phase entry builder
# ──────────────────────────────────────────────────────────────────────────────

def build_phase_entries(ann: dict, box_format: bool = False, human_prompt: str = "") -> list[dict]:
    """
    Returns one dict per phase in ann['phase_annotations'].

    Each dict: {trajectory_name, start_frame, end_frame, conversations}
    conversations follows ShareGPT format:
        [{"from": "human", "value": <prompt template>},
         {"from": "gpt",   "value": <phase description + point>}]
    The human value may contain {instruction} which is filled in at training time.
    """
    phase_anns = ann.get("phase_annotations") or []
    if not phase_anns:
        return []

    traj_name = ann["trajectory_name"]
    task_obj_info = ann.get("task_obj_info", "")
    obj_lbl = _obj_label(task_obj_info)
    tgt_lbl = _target_label(task_obj_info)

    # initial_object_box is [[x1,y1,x2,y2]]; unwrap to [x1,y1,x2,y2]
    raw_init = ann.get("initial_object_box")
    init_box: Optional[list] = raw_init[0] if raw_init else None

    # target_candidate_boxes is [[x1,y1,x2,y2], ...]; winner = index 0
    raw_cands = ann.get("target_candidate_boxes")
    cand_box: Optional[list] = raw_cands[0] if raw_cands else None

    entries = []
    for ph in phase_anns:
        phase = ph.get("phase", "")
        description = (ph.get("description") or "").strip()
        start = int(ph["start_frame"])
        end = int(ph["end_frame"])

        if phase in _GRASP_PHASES and init_box is not None:
            if box_format:
                spatial = _qwen_box(init_box, obj_lbl)
            else:
                cx, cy = _box_center(init_box)
                spatial = _qwen_point(cx, cy, obj_lbl)
        elif phase in _PLACE_PHASES and cand_box is not None:
            if box_format:
                spatial = _qwen_box(cand_box, tgt_lbl)
            else:
                cx, cy = _box_center(cand_box)
                spatial = _qwen_point(cx, cy, tgt_lbl)
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

def main():
    parser = argparse.ArgumentParser(description="Build phase-based CoT mapping JSONL from annotations.")
    parser.add_argument("--annotations", required=True, help="Path to annotations.jsonl")
    parser.add_argument("--output",      required=True, help="Output JSONL path")
    parser.add_argument(
        "--box_format",
        action="store_true",
        help="Use Qwen3.5VL box format instead of center points.",
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
        default="Your task is {instruction}. What is the next subtask? Point to the target location.",
        help="Template for the human turn. Use {instruction} as placeholder for the task instruction.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one example entry per dataset.")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        annotations = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(annotations)} annotation entries from {ann_path}")

    # Holdout
    all_trajs = sorted({a["trajectory_name"] for a in annotations})
    holdout: set[str] = set()
    if args.holdout_fraction > 0.0:
        rng = random.Random(args.holdout_seed)
        n_holdout = max(1, int(len(all_trajs) * args.holdout_fraction))
        holdout = set(rng.sample(all_trajs, n_holdout))
        print(f"Holding out {len(holdout)} / {len(all_trajs)} trajectories (seed={args.holdout_seed})")
        holdout_path = out_path.with_suffix(".holdout_trajectories.txt")
        holdout_path.write_text("\n".join(sorted(holdout)) + "\n")
        print(f"Holdout list written to {holdout_path}")

    written = 0
    skipped_holdout = 0
    skipped_no_phases = 0
    seen_datasets: set[str] = set()
    samples: list[dict] = []  # collect up to 5 entries for post-run display

    with open(out_path, "w") as f_out:
        for ann in annotations:
            traj_name = ann["trajectory_name"]

            if traj_name in holdout:
                skipped_holdout += 1
                continue

            entries = build_phase_entries(ann, box_format=args.box_format, human_prompt=args.human_prompt)
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
                            gpt   = e["conversations"][1]["value"]
                            print(f"  frames {e['start_frame']:>4} → {e['end_frame']:>4}")
                            print(f"    human: {human}")
                            print(f"    gpt:   {gpt[:120]}")

    print(f"\nDone. Wrote {written} entries to {out_path}")
    if skipped_holdout:
        print(f"  Skipped {skipped_holdout} entries (holdout trajectories).")
    if skipped_no_phases:
        print(f"  Skipped {skipped_no_phases} annotations with no phase_annotations.")

    print(f"\n{'─'*70}")
    print(f"Sample entries ({len(samples)} of {written}):")
    print(f"{'─'*70}")
    for s in samples:
        print(f"\n  trajectory : {s['trajectory_name']}")
        print(f"  frames     : {s['start_frame']} → {s['end_frame']}")
        print(f"  human      : {s['conversations'][0]['value']}")
        print(f"  gpt        : {s['conversations'][1]['value']}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
