#!/usr/bin/env python3
"""Aggregate per-task VLABench evaluator outputs and enforce the full denominator."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--num-episodes", type=int, required=True)
    args = p.parse_args()
    rows = []
    for task in args.tasks:
        detail = args.root / task / task / "detail_info.json"
        data = json.loads(detail.read_text())
        if len(data) != args.num_episodes:
            raise RuntimeError(f"partial VLABench result for {task}: {len(data)}/{args.num_episodes}")
        rows.extend(data)
    summary = {
        "total_count": len(rows),
        "success_count": sum(bool(x["success"]) for x in rows),
        "success_rate": sum(bool(x["success"]) for x in rows) / len(rows),
        "mean_intention_score": sum(float(x["intention_score"]) for x in rows) / len(rows),
        "mean_progress_score": sum(float(x["progress_score"]) for x in rows) / len(rows),
        "per_task": {
            task: json.loads((args.root / task / "evaluation_result.json").read_text())[task]
            for task in args.tasks
        },
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

