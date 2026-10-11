#!/usr/bin/env python3
"""Aggregate ABC-Sim client shards: per-task success rate (ABC success_rate = ever_success), mean max progress
(ABC mean_max_progress), episodes found vs expected. Writes <out>/summary.json. Plain python3 + stdlib."""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--num_episodes", type=int, default=0)
    a = ap.parse_args()
    eps = defaultdict(dict)
    for path in glob.glob(os.path.join(a.out, "episodes_*_shard*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            eps[r["task"]][int(r["episode"])] = r
    tasks = [t for t in a.tasks.split(",") if t] or sorted(eps)
    summary = {"out": a.out, "tasks": {}}
    tot_k = tot_n = 0
    print(f"{'task':32s} {'succ':>9s} {'rate':>6s} {'95% CI':>13s} {'max_prog':>8s} {'steps':>6s} {'wall_s':>6s}")
    for t in tasks:
        rs = list(eps.get(t, {}).values())
        n = len(rs)
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, n)
        mp = sum(r["max_progress"] for r in rs) / n if n else float("nan")
        st = sum(r["sim_steps"] for r in rs) / n if n else float("nan")
        wall = sum(r["wall_s"] for r in rs) / n if n else float("nan")
        summary["tasks"][t] = dict(n=n, expected=a.num_episodes or None, successes=k, success_rate=k / n if n else None,
                                   ci95=[lo, hi], mean_max_progress=mp, mean_sim_steps=st, mean_wall_s=wall,
                                   missing=sorted(set(range(a.num_episodes)) - set(eps.get(t, {}))) if a.num_episodes else None)
        tot_k += k
        tot_n += n
        print(f"{t:32s} {k:4d}/{n:<4d} {100 * k / max(n, 1):6.1f} [{100 * lo:4.1f},{100 * hi:5.1f}] {mp:8.3f} {st:6.0f} {wall:6.0f}")
    summary["overall"] = dict(n=tot_n, successes=tot_k, success_rate=tot_k / tot_n if tot_n else None,
                              task_mean=sum(v["success_rate"] or 0 for v in summary["tasks"].values()) / max(len(tasks), 1))
    print(f"{'ALL':32s} {tot_k:4d}/{tot_n:<4d} {100 * tot_k / max(tot_n, 1):6.1f}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
