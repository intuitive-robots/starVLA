#!/usr/bin/env python3
"""Merge RoboTwin shard summaries while rejecting omissions and duplicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument(
        "--allow-trial-splits",
        action="store_true",
        help="sum duplicate task rows from disjoint seed groups",
    )
    parser.add_argument(
        "--expected-seeds",
        help="comma-separated seed groups required once per task (for example 0,1,2,3,4)",
    )
    parser.add_argument(
        "--mode", choices=("demo_clean", "demo_randomized"), required=True
    )
    args = parser.parse_args()

    if len(args.summaries) != args.expected_shards:
        raise SystemExit(
            f"received {len(args.summaries)} shard summaries, "
            f"expected {args.expected_shards}"
        )

    rows: list[dict[str, object]] = []
    shard_records: list[dict[str, object]] = []
    seen_tasks: dict[str, Path] = {}
    expected_seeds = (
        {int(value) for value in args.expected_seeds.split(",")}
        if args.expected_seeds
        else None
    )
    seen_task_seeds: dict[str, set[int]] = {}
    for path in args.summaries:
        if not path.is_file():
            raise SystemExit(f"missing shard summary: {path}")
        summary = json.loads(path.read_text())
        if summary.get("mode") != args.mode:
            raise SystemExit(
                f"mode mismatch in {path}: {summary.get('mode')!r} != {args.mode!r}"
            )
        shard_trials = int(summary.get("trials_per_task", -1))
        if shard_trials <= 0:
            raise SystemExit(f"invalid trial count in {path}: {shard_trials}")
        if not args.allow_trial_splits and shard_trials != args.expected_trials:
            raise SystemExit(
                f"trial-count mismatch in {path}: "
                f"{summary.get('trials_per_task')} != {args.expected_trials}"
            )
        seed_group = summary.get("seed_group")
        if expected_seeds is not None:
            if seed_group is None or int(seed_group) not in expected_seeds:
                raise SystemExit(
                    f"unexpected or missing seed_group in {path}: {seed_group!r}; "
                    f"expected one of {sorted(expected_seeds)}"
                )
            seed_group = int(seed_group)
        shard_rows = summary.get("per_task")
        if not isinstance(shard_rows, list) or not shard_rows:
            raise SystemExit(f"missing per-task rows in {path}")
        for row in shard_rows:
            task = str(row["task"])
            if task in seen_tasks and not args.allow_trial_splits:
                raise SystemExit(
                    f"duplicate task {task!r} in {path} and {seen_tasks[task]}"
                )
            if int(row["trials"]) != shard_trials:
                raise SystemExit(
                    f"wrong trials for {task!r} in {path}: {row['trials']} "
                    f"!= shard declaration {shard_trials}"
                )
            if expected_seeds is not None:
                task_seeds = seen_task_seeds.setdefault(task, set())
                if seed_group in task_seeds:
                    raise SystemExit(
                        f"duplicate task/seed fragment for {task!r}, seed {seed_group}, "
                        f"in {path}"
                    )
                task_seeds.add(seed_group)
            seen_tasks[task] = path
            rows.append(row)
        shard_records.append(
            {
                "summary": str(path.resolve()),
                "num_tasks": len(shard_rows),
                "successes": int(summary["successes"]),
                "trials": int(summary["trials"]),
                "seed_group": seed_group,
            }
        )

    unique_task_count = len(seen_tasks)
    if unique_task_count != args.expected_tasks:
        raise SystemExit(
            f"merged {unique_task_count} unique tasks, expected {args.expected_tasks}"
        )
    if expected_seeds is not None:
        for task in seen_tasks:
            if seen_task_seeds.get(task, set()) != expected_seeds:
                raise SystemExit(
                    f"task {task!r} has seed groups "
                    f"{sorted(seen_task_seeds.get(task, set()))}, expected "
                    f"{sorted(expected_seeds)}"
                )
    if args.allow_trial_splits:
        combined: dict[str, dict[str, object]] = {}
        for row in rows:
            task = str(row["task"])
            target = combined.setdefault(
                task,
                {"task": task, "successes": 0, "trials": 0, "parts": []},
            )
            target["successes"] = int(target["successes"]) + int(row["successes"])
            target["trials"] = int(target["trials"]) + int(row["trials"])
            target["parts"].append(row)
        rows = list(combined.values())
        for row in rows:
            if int(row["trials"]) != args.expected_trials:
                raise SystemExit(
                    f"task {row['task']!r} merged {row['trials']} trials, "
                    f"expected {args.expected_trials}"
                )
            row["success_rate"] = int(row["successes"]) / int(row["trials"])

    rows.sort(key=lambda row: str(row["task"]))
    total_successes = sum(int(row["successes"]) for row in rows)
    total_trials = sum(int(row["trials"]) for row in rows)
    expected_total = args.expected_tasks * args.expected_trials
    if total_trials != expected_total:
        raise SystemExit(f"merged {total_trials} trials, expected {expected_total}")

    merged = {
        "mode": args.mode,
        "num_shards": len(args.summaries),
        "num_tasks": unique_task_count,
        "trials_per_task": args.expected_trials,
        "successes": total_successes,
        "trials": total_trials,
        "success_rate": total_successes / total_trials,
        "shards": shard_records,
        "per_task": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(
        f"[MERGE] {total_successes}/{total_trials} = "
        f"{100.0 * total_successes / total_trials:.3f}% across "
        f"{unique_task_count} tasks / {len(args.summaries)} shards"
    )


if __name__ == "__main__":
    main()
