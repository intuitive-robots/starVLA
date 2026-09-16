#!/usr/bin/env python3
"""Validate and merge four successful-counterfactual render shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--expected-tasks", type=int, default=40)
    parser.add_argument("--minimum-tasks", type=int, default=20)
    parser.add_argument("--minimum-tasks-per-suite", type=int, default=3)
    parser.add_argument("--expected-categories", type=int, default=4)
    args = parser.parse_args()
    rows = []
    summaries = []
    for shard in range(args.num_shards):
        directory = args.root / "shards" / str(shard)
        summaries.append(json.loads((directory / "capture_summary.json").read_text()))
        rows.extend(
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
    rows.sort(key=lambda row: row["pair_id"])
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate pair ids across shards")
    tasks = {(row["suite"], int(row["task_id"])) for row in rows}
    categories = sorted({row["category"] for row in rows})
    if args.expected_tasks > 0 and len(tasks) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, got {len(tasks)}")
    if len(tasks) < args.minimum_tasks:
        raise RuntimeError(f"expected at least {args.minimum_tasks} tasks, got {len(tasks)}")
    suite_counts = Counter(suite for suite, _ in tasks)
    sparse = {
        suite: suite_counts[suite]
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        if suite_counts[suite] < args.minimum_tasks_per_suite
    }
    if sparse:
        raise RuntimeError(
            f"insufficient suite diversity (minimum {args.minimum_tasks_per_suite} tasks/suite): {sparse}"
        )
    if len(categories) != args.expected_categories:
        raise RuntimeError(f"expected {args.expected_categories} categories, got {categories}")
    coverage = Counter((row["suite"], int(row["task_id"]), row["split"], row["category"]) for row in rows)
    expected = len(tasks) * 3 * args.expected_categories
    if len(coverage) != expected or min(coverage.values()) < 1:
        raise RuntimeError(f"incomplete task/split/category coverage: {len(coverage)} of {expected}")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "format": "starvla_successful_counterfactual_capture_v1",
        "pairs": len(rows),
        "tasks": len(tasks),
        "suites": sorted({row["suite"] for row in rows}),
        "categories": categories,
        "split_pairs": dict(Counter(row["split"] for row in rows)),
        "trajectories": len({row["source_trajectory"] for row in rows}),
        "max_state_diff": max(float(row["state_max_abs_diff"]) for row in rows),
        "max_policy_state_diff": max(float(row["policy_state_max_abs_diff"]) for row in rows),
        "max_source_restore_policy_state_diff": max(
            float(summary["max_source_restore_policy_state_diff"]) for summary in summaries
        ),
        "min_pixel_mae": min(
            max(float(row["external_pixel_mae"]), float(row["wrist_pixel_mae"])) for row in rows
        ),
        "shards": summaries,
        "manifest": str(manifest.resolve()),
    }
    (args.root / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
