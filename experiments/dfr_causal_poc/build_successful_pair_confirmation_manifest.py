#!/usr/bin/env python3
"""Build a fresh closed-loop confirmation split without rendering frame pairs.

The split uses the first successful source trajectory not present in the
train/validation/test manifest and the hardest LIBERO-Plus appearance variant
not present there.  Closed-loop evaluation renders observations online.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from render_successful_counterfactuals import CATEGORIES, canonical_stem


def _difficulty(row: dict) -> int:
    value = row.get("difficulty_level")
    return int(value) if value is not None else 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tasks", type=int, default=20)
    parser.add_argument("--minimum-tasks-per-suite", type=int, default=3)
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path(os.environ.get("LIBERO_HOME", "/app/LIBERO-plus"))
        / "libero/libero/benchmark/task_classification.json",
    )
    args = parser.parse_args()

    existing = [
        json.loads(line)
        for line in args.existing_manifest.read_text().splitlines()
        if line.strip()
    ]
    eligible_tasks = sorted({(row["suite"], row["base_task"]) for row in existing})
    used_episodes: dict[tuple[str, str], set[int]] = defaultdict(set)
    used_variants: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in existing:
        task = (row["suite"], row["base_task"])
        used_episodes[task].add(int(row["episode_idx"]))
        used_variants[(*task, row["category"])].add(int(row["variant_task_id"]))

    trajectories: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(args.trajectory_root.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("format") != "starvla_successful_libero_trajectory_v1":
            continue
        npz = Path(row.get("npz", path.with_suffix(".npz"))).resolve()
        if not npz.is_file():
            raise FileNotFoundError(npz)
        row["npz"] = str(npz)
        trajectories[(row["suite"], row["task_name"])].append(row)
    fresh_trajectories = {}
    for task in eligible_tasks:
        candidates = sorted(trajectories[task], key=lambda row: int(row["episode_idx"]))
        candidates = [
            row for row in candidates if int(row["episode_idx"]) not in used_episodes[task]
        ]
        if candidates:
            fresh_trajectories[task] = candidates[0]

    classification = json.loads(args.classification.read_text())
    variants: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for suite, suite_rows in classification.items():
        for row in suite_rows:
            category = str(row["category"])
            if category not in CATEGORIES:
                continue
            base_task = canonical_stem(str(row["name"]), category)
            variants[(suite, base_task, category)].append(
                dict(row, task_id=int(row["id"]) - 1)
            )

    records = []
    missing_variants = []
    for task, trajectory in sorted(fresh_trajectories.items()):
        suite, base_task = task
        for category in CATEGORIES:
            key = (suite, base_task, category)
            candidates = [
                row
                for row in variants[key]
                if int(row["task_id"]) not in used_variants[key]
            ]
            if not candidates:
                missing_variants.append(key)
                continue
            # This deterministic choice is fixed before confirmation outcomes:
            # hardest unused variant, then lowest numeric task id.
            selected = sorted(
                candidates, key=lambda row: (-_difficulty(row), int(row["task_id"]))
            )[0]
            category_stem = category.lower().replace(" ", "_")
            records.append(
                {
                    "format": "starvla_successful_counterfactual_pair_v1",
                    "pair_id": (
                        f"{suite}_task{int(trajectory['task_id']):02d}_"
                        f"ep{int(trajectory['episode_idx']):02d}_step0000:"
                        f"{category_stem}:confirmation"
                    ),
                    "suite": suite,
                    "task_id": int(trajectory["task_id"]),
                    "base_task": base_task,
                    "episode_idx": int(trajectory["episode_idx"]),
                    "anchor_step": 0,
                    "split": "confirmation",
                    "category": category,
                    "variant_role": "confirmation",
                    "variant_task_id": int(selected["task_id"]),
                    "variant_name": str(selected["name"]),
                    "difficulty_level": selected.get("difficulty_level"),
                    "language": str(trajectory["task_description"]),
                    "source_trajectory": trajectory["npz"],
                }
            )

    if missing_variants:
        raise RuntimeError(f"no unused appearance variant for: {missing_variants}")
    task_counts = Counter((row["suite"], row["base_task"]) for row in records)
    if task_counts and set(task_counts.values()) != {len(CATEGORIES)}:
        raise RuntimeError("incomplete category coverage")
    tasks = set(task_counts)
    if len(tasks) < args.minimum_tasks:
        raise RuntimeError(f"only {len(tasks)} fresh tasks; require {args.minimum_tasks}")
    suite_counts = Counter(suite for suite, _ in tasks)
    sparse = {
        suite: suite_counts[suite]
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        if suite_counts[suite] < args.minimum_tasks_per_suite
    }
    if sparse:
        raise RuntimeError(f"insufficient fresh suite coverage: {sparse}")

    records.sort(key=lambda row: row["pair_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = {
        "format": "starvla_successful_pair_confirmation_manifest_v1",
        "selection": "first unused successful trajectory; hardest unused variant",
        "eligible_original_tasks": len(eligible_tasks),
        "confirmation_tasks": len(tasks),
        "excluded_without_fourth_success": len(eligible_tasks) - len(tasks),
        "suite_tasks": dict(sorted(suite_counts.items())),
        "perturbed_cases": len(records),
        "clean_cases": len(tasks),
        "manifest": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
