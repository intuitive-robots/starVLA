#!/usr/bin/env python3
"""Fail-closed aggregation of one RoboTwin Slurm evaluation log directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RATE_RE = re.compile(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%")
RESULT_RE = re.compile(r"Data has been saved to\s+(.+?/_result\.txt)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("mode", choices=("demo_clean", "demo_randomized"))
    parser.add_argument("expected_trials", type=int)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logs = sorted(args.log_root.glob("*_eval.log"))
    if not logs:
        raise SystemExit(f"no eval logs under {args.log_root}")

    rows: list[dict[str, object]] = []
    suffix = f"_{args.mode}_slot"
    for log in logs:
        text = ANSI_RE.sub("", log.read_text(errors="replace").replace("\r", "\n"))
        rates = RATE_RE.findall(text)
        if not rates:
            raise SystemExit(f"missing final success rate: {log}")
        successes, trials, percent = rates[-1]
        successes_i, trials_i = int(successes), int(trials)
        if trials_i != args.expected_trials:
            raise SystemExit(
                f"wrong trial count in {log}: {trials_i}, expected {args.expected_trials}"
            )
        if suffix not in log.name:
            raise SystemExit(f"cannot parse task from log filename: {log.name}")
        task = log.name.split(suffix, 1)[0]
        result_paths = RESULT_RE.findall(text)
        rows.append(
            {
                "task": task,
                "successes": successes_i,
                "trials": trials_i,
                "success_rate": successes_i / trials_i,
                "reported_percent": float(percent),
                "eval_log": str(log.resolve()),
                "result_file": result_paths[-1] if result_paths else None,
            }
        )

    total_successes = sum(int(row["successes"]) for row in rows)
    total_trials = sum(int(row["trials"]) for row in rows)
    summary = {
        "mode": args.mode,
        "seed_group": args.seed,
        "num_tasks": len(rows),
        "trials_per_task": args.expected_trials,
        "successes": total_successes,
        "trials": total_trials,
        "success_rate": total_successes / total_trials,
        "per_task": rows,
    }
    (args.log_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.log_root / "summary.tsv").open("w") as handle:
        handle.write("task\tsuccesses\ttrials\tsuccess_rate\n")
        for row in rows:
            handle.write(
                f'{row["task"]}\t{row["successes"]}\t{row["trials"]}\t'
                f'{float(row["success_rate"]):.6f}\n'
            )
    print(
        f"[AGGREGATE] {total_successes}/{total_trials} = "
        f"{100.0 * total_successes / total_trials:.3f}% across {len(rows)} tasks"
    )


if __name__ == "__main__":
    main()
