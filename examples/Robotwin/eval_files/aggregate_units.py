#!/usr/bin/env python3
"""Fail-closed aggregation of a unit-based RoboTwin evaluation (start_eval.sh `task@mode@seed` units).

Every expected unit (task x mode x seed group) must have exactly one eval log whose last
"Success rate: s/n" line has n == trials_per_unit. Seed groups are summed per (task, mode);
RoboTwin seed group g draws its episodes from seeds 100000 * (1 + g) onward (expert-checked).

  aggregate_units.py LOG_ROOT --tasks a,b --modes demo_clean,demo_randomized \
      --seed-groups 4 --trials-per-unit 25
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RATE_RE = re.compile(r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%")
LOG_RE = re.compile(r"^(?P<task>.+?)__(?P<mode>demo_clean|demo_randomized)__seed(?P<seed>\d+)__slot\d+_gpu\d+_port\d+_eval\.log$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_roots", type=Path, nargs="+",
                        help="one log directory, or several shard directories of the same checkpoint and phase(s)")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for summary_units.{json,tsv} (default: the first log directory)")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--seed-groups", type=int, required=True)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--trials-per-unit", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="report what exists instead of failing on missing units (canaries / triage)")
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    modes = [m for m in args.modes.split(",") if m]
    seeds = list(range(args.first_seed, args.first_seed + args.seed_groups))
    expected = {(t, m, s) for t in tasks for m in modes for s in seeds}

    found: dict[tuple[str, str, int], dict] = {}
    problems: list[str] = []
    logs = sorted(log for root in args.log_roots for log in root.glob("*_eval.log"))
    for log in logs:
        match = LOG_RE.match(log.name)
        if not match:
            continue
        key = (match["task"], match["mode"], int(match["seed"]))
        text = ANSI_RE.sub("", log.read_text(errors="replace").replace("\r", "\n"))
        rates = RATE_RE.findall(text)
        setup_failures = text.count("Unexpected setup failure")
        render_errors = text.count("Render Error")
        if key in found:
            problems.append(f"duplicate log for {key}: {log.name}")
            continue
        if not rates:
            problems.append(f"no success-rate line: {log.name} (render errors {render_errors}, setup failures {setup_failures})")
            continue
        successes, trials, _ = rates[-1]
        if int(trials) != args.trials_per_unit:
            problems.append(f"{log.name}: {trials} trials, expected {args.trials_per_unit}")
        found[key] = {
            "task": key[0], "mode": key[1], "seed_group": key[2],
            "successes": int(successes), "trials": int(trials),
            "unexpected_setup_failures": setup_failures, "render_errors": render_errors,
            "eval_log": str(log.resolve()),
        }
    missing = sorted(expected - set(found))
    if missing:
        problems.append(f"missing units: {missing}")
    if problems and not args.allow_partial:
        raise SystemExit("[AGGREGATE FAIL]\n  " + "\n  ".join(problems))

    per = defaultdict(lambda: {"successes": 0, "trials": 0, "units": 0})
    for row in found.values():
        cell = per[(row["task"], row["mode"])]
        cell["successes"] += row["successes"]
        cell["trials"] += row["trials"]
        cell["units"] += 1
    by_mode = {}
    for mode in modes:
        cells = [per[(t, mode)] for t in tasks if (t, mode) in per]
        s = sum(c["successes"] for c in cells)
        n = sum(c["trials"] for c in cells)
        by_mode[mode] = {
            "successes": s, "trials": n, "success_rate": (s / n) if n else None,
            "mean_task_success_rate": (sum(c["successes"] / c["trials"] for c in cells) / len(cells)) if cells else None,
            "per_task": {t: per[(t, mode)] | {"success_rate": per[(t, mode)]["successes"] / per[(t, mode)]["trials"]}
                         for t in tasks if (t, mode) in per},
        }
    summary = {
        "log_roots": [str(r) for r in args.log_roots],
        "tasks": tasks, "modes": modes, "seed_groups": seeds, "trials_per_unit": args.trials_per_unit,
        "complete": not problems, "problems": problems, "by_mode": by_mode,
        "units": sorted(found.values(), key=lambda r: (r["mode"], r["task"], r["seed_group"])),
    }
    out = args.out or args.log_roots[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_units.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "summary_units.tsv").open("w") as handle:
        handle.write("mode\ttask\tsuccesses\ttrials\tsuccess_rate\n")
        for mode in modes:
            for task, cell in by_mode[mode]["per_task"].items():
                handle.write(f"{mode}\t{task}\t{cell['successes']}\t{cell['trials']}\t{cell['success_rate']:.4f}\n")
    for mode in modes:
        m = by_mode[mode]
        if m["trials"]:
            print(f"[AGGREGATE] {mode}: {m['successes']}/{m['trials']} = {100.0 * m['success_rate']:.2f}% "
                  f"(mean of {len(m['per_task'])} task rates {100.0 * m['mean_task_success_rate']:.2f}%)")
    if problems:
        print("[AGGREGATE PARTIAL]\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
