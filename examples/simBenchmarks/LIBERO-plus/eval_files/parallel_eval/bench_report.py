"""Report wall time, shard balance and per-episode cost for a benchmark arm.

Reads the artifacts every eval shard already writes -- the per-shard
``*_episodes.jsonl`` and the matching ``.log`` whose "# episodes completed so
far: N" lines are timestamped -- so it needs no instrumentation in the eval
itself and can be pointed at a finished production run just as well as at
``bench_libero_plus_sharding_slurm.sh`` output.
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re
import statistics

EPISODE_DONE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - INFO - # episodes completed so far:")


def shard_timings(log_dir: str):
    """Per-shard (duration, episodes) plus per-episode (category, seconds)."""
    shards, episodes = [], []
    for result in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        if result.endswith("overall_results.json"):
            continue
        records = [json.loads(l) for l in open(result.replace(".json", "_episodes.jsonl")) if l.strip()]
        # The worker names its .log with "_" where the result json uses "_to_".
        log = os.path.join(log_dir, os.path.basename(result)[: -len(".json")].replace("_to_", "_") + ".log")
        if not os.path.exists(log) or not records:
            shards.append((0.0, len(records)))
            continue
        starts, ends = None, []
        for line in open(log, errors="ignore"):
            if starts is None and " - INFO - " in line:
                starts = datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            m = EPISODE_DONE.match(line)
            if m:
                ends.append(datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        if len(ends) != len(records):
            shards.append(((ends[-1] - starts).total_seconds() if ends else 0.0, len(records)))
            continue
        prev = starts
        for end, record in zip(ends, records):
            episodes.append((record["category"], (end - prev).total_seconds()))
            prev = end
        shards.append(((ends[-1] - starts).total_seconds(), len(records)))
    return shards, episodes


def report(label: str, log_dir: str) -> dict:
    shards, episodes = shard_timings(log_dir)
    if not shards:
        print(f"\n=== {label}: no shards found in {log_dir}")
        return {}
    durations = [d for d, _ in shards]
    work = sum(durations)
    makespan = max(durations)
    per_cat = collections.defaultdict(list)
    for category, seconds in episodes:
        per_cat[category].append(seconds)
    print(f"\n=== {label}")
    print(f"  shards            : {len(shards)}  ({sum(n for _, n in shards)} episodes)")
    print(f"  makespan (slowest): {makespan / 60:7.1f} min")
    print(f"  median shard      : {statistics.median(durations) / 60:7.1f} min")
    print(f"  fastest shard     : {min(durations) / 60:7.1f} min")
    print(f"  total work        : {work / 60:7.1f} worker-min  -> perfect balance {work / 60 / len(shards):.1f} min")
    print(f"  balance efficiency: {100 * (work / len(shards)) / makespan if makespan else 0:6.1f}%  (100% = every worker busy until the end)")
    if episodes:
        print(f"  mean episode      : {statistics.mean([s for _, s in episodes]):7.1f} s")
        for category, values in sorted(per_cat.items(), key=lambda kv: -statistics.mean(kv[1])):
            print(f"      {category:24s} n={len(values):4d}  mean={statistics.mean(values):6.1f} s")
    return {
        "makespan_s": makespan,
        "work_s": work,
        "shards": len(shards),
        "episodes": sum(n for _, n in shards),
        "episode_mean_s": statistics.mean([s for _, s in episodes]) if episodes else None,
        "category_mean_s": {c: statistics.mean(v) for c, v in per_cat.items()},
    }


def success(root: str, suite: str):
    path = os.path.join(root, suite, "overall_results.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d.get("overall", {}).get("success_count"), d.get("overall", {}).get("total_count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_root", required=True, help="directory holding one subdirectory per arm")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--arm", action="append", default=None, help="explicit arm subdirectory (repeatable)")
    args = parser.parse_args()

    arms = args.arm or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.bench_root, "*")) if os.path.isdir(p)
    )
    stats = {}
    for arm in arms:
        root = os.path.join(args.bench_root, arm)
        s = report(arm, os.path.join(root, "logs", args.suite))
        if s:
            s["success"] = success(root, args.suite)
            stats[arm] = s

    if len(stats) > 1:
        first = list(stats)[0]
        print(f"\n=== speedup vs {first}")
        for arm, s in stats.items():
            rel = stats[first]["makespan_s"] / s["makespan_s"] if s["makespan_s"] else float("nan")
            ep = f"{s['episode_mean_s']:.1f}s/ep" if s["episode_mean_s"] else "-"
            succ = f"{s['success'][0]}/{s['success'][1]}" if s.get("success") else "?"
            print(f"  {arm:22s} {s['makespan_s'] / 60:6.1f} min  {rel:5.2f}x  {ep:>10}  success={succ}")
        # The arms must evaluate the same task set; differing episode counts mean
        # the comparison is not apples to apples.
        counts = {s["episodes"] for s in stats.values()}
        if len(counts) > 1:
            print(f"  [WARN] arms evaluated different episode counts: {counts}")


if __name__ == "__main__":
    main()
