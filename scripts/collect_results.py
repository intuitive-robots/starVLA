#!/usr/bin/env python3
"""Collect every eval result in the repo into one table.

Results live in three different shapes across two git trees, which is how the same run
ends up quoted at two different numbers:

  LIBERO-plus   <ckpt>/results/libero-plus/overall_results.json
                {suite: {"overall": {total_count, success_count, ...}, "<perturbation>": {...}}}
  LIBERO        <ckpt>/results/libero/overall_results.json
                {suite: {total_count, success_count, ..., "per_task": {...}}}
  RoboCasa365   <ckpt>/checkpoints/<step>.eval/robocasa_<Task>_<protocol>_<tag>.json
                {env, success_rate, successes: [...], seed, n_envs, ...}   (worktree only)

Episode count is carried through everything: a 0.81 from 256 episodes and a 0.77 from 4000
are not the same measurement, and sorting a leaderboard without that column is how the
zonly arm got quoted at 0.812 before it settled at 0.776.

  python scripts/collect_results.py                      # summary + write both CSVs
  python scripts/collect_results.py --benchmark libero-plus --min-episodes 4000
  python scripts/collect_results.py --run-filter zonly
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

TREES = [
    pathlib.Path("/e/project1/m3/blank4/code/starVLA"),
    pathlib.Path("/e/project1/m3/blank4/code/starVLA-upstream-merge"),
]


def _counts(node: dict) -> tuple[int, int] | None:
    """Pull (successes, episodes) from either JSON shape."""
    if not isinstance(node, dict):
        return None
    inner = node.get("overall") if isinstance(node.get("overall"), dict) else node
    if "total_count" in inner and "success_count" in inner:
        return int(inner["success_count"]), int(inner["total_count"])
    return None


def collect_libero(tree: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(tree.glob("playground/Checkpoints/*/results/*/overall_results.json")):
        run, benchmark = path.parts[-4], path.parts[-2]
        try:
            blob = json.load(path.open())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! unreadable {path}: {exc}", file=sys.stderr)
            continue
        for suite, node in blob.items():
            counts = _counts(node)
            if counts:
                rows.append({
                    "tree": tree.name, "benchmark": benchmark, "run": run, "unit": suite,
                    "successes": counts[0], "episodes": counts[1],
                    "success_rate": round(counts[0] / counts[1], 4) if counts[1] else None,
                    "source": str(path),
                })
    return rows


ROBOCASA_NAME = re.compile(r"robocasa_(?P<task>[A-Za-z0-9]+)_(?P<protocol>.*?)(?:_(?P<tag>[^_]*))?$")


def collect_robocasa(tree: pathlib.Path) -> list[dict]:
    rows = []
    for path in sorted(tree.glob("playground/Checkpoints/*/checkpoints/*.eval/*.json")):
        run = path.parts[-4]
        step = path.parts[-2].replace("_pytorch_model.eval", "")
        try:
            blob = json.load(path.open())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! unreadable {path}: {exc}", file=sys.stderr)
            continue
        successes = blob.get("successes")
        if successes is None:
            continue
        stem = path.stem
        task = stem.split("_")[1] if "_" in stem else stem
        # everything after the protocol block is the variant tag (nostate, st_zero, ...)
        tag = stem.split("_native_")[-1] if "_native_" in stem else ""
        rows.append({
            "tree": tree.name, "benchmark": "robocasa365", "run": f"{run}@{step}",
            "unit": f"{task}|{tag}" if tag else task,
            "successes": int(sum(bool(s) for s in successes)), "episodes": len(successes),
            "success_rate": round(blob.get("success_rate", 0.0), 4),
            "source": str(path),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="substring filter, e.g. libero-plus / libero / robocasa")
    ap.add_argument("--run-filter", help="substring filter on the run name")
    ap.add_argument("--min-episodes", type=int, default=0, help="hide runs measured on fewer")
    ap.add_argument("--out-dir", default="results_collected")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows: list[dict] = []
    for tree in TREES:
        if not tree.exists():
            print(f"  ! missing tree {tree}", file=sys.stderr)
            continue
        rows += collect_libero(tree)
        rows += collect_robocasa(tree)

    if args.benchmark:
        rows = [r for r in rows if args.benchmark in r["benchmark"]]
    if args.run_filter:
        rows = [r for r in rows if args.run_filter in r["run"]]

    out = pathlib.Path(args.out_dir)
    out.mkdir(exist_ok=True)
    fields = ["tree", "benchmark", "run", "unit", "successes", "episodes", "success_rate", "source"]
    with (out / "results_all.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    pooled: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        agg = pooled[(r["benchmark"], r["run"])]
        agg[0] += r["successes"]; agg[1] += r["episodes"]; agg[2] += 1
    summary = []
    for (benchmark, run), (suc, eps, units) in pooled.items():
        if eps < args.min_episodes:
            continue
        summary.append({
            "benchmark": benchmark, "run": run, "units": units,
            "episodes": eps, "successes": suc,
            "success_rate": round(suc / eps, 4) if eps else None,
            # binomial SE: the column that decides whether two runs actually differ
            "se": round((suc / eps * (1 - suc / eps) / eps) ** 0.5, 4) if eps else None,
        })
    summary.sort(key=lambda r: (r["benchmark"], -(r["success_rate"] or 0)))
    with (out / "results_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["benchmark", "run", "units", "episodes",
                                                "successes", "success_rate", "se"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"{len(rows)} rows -> {out/'results_all.csv'}")
    print(f"{len(summary)} run/benchmark pairs -> {out/'results_summary.csv'}\n")
    current = None
    shown = 0
    for r in summary:
        if r["benchmark"] != current:
            current, shown = r["benchmark"], 0
            print(f"\n{r['benchmark']}")
            print(f"  {'run':54s} {'rate':>6s} {'±se':>6s} {'eps':>6s} {'units':>5s}")
        if shown < args.top:
            print(f"  {r['run'][:54]:54s} {r['success_rate']:6.3f} {r['se']:6.3f} "
                  f"{r['episodes']:6d} {r['units']:5d}")
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
