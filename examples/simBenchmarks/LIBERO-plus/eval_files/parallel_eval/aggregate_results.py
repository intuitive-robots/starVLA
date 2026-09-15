"""Merge per-shard LIBERO-plus result JSONs into a success-rate summary.

Two modes, selected by ``--task_suite_name``:

- Given: scan only that one suite's ``logs/<suite>/`` shards and write the result
  to ``<root_path>/<suite>/overall_results.json`` -- a file private to that suite,
  so aggregating one suite can never race with / overwrite another suite's results
  (the bug this replaced: a single shared ``overall_results.json`` recomputed from
  whatever suites happened to be on disk at call time, so calling this once per
  suite -- sequentially or concurrently -- clobbered earlier suites' snapshots).
- Omitted (default): scan all 4 standard suites and write the combined view to
  ``<root_path>/overall_results.json``, unchanged from the original behavior. Call
  this ONCE, after every suite's per-suite aggregation has already run.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ALL_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def _aggregate_from_episodes(json_files: list) -> dict:
    """Count each (task_id, episode_idx) once, from the per-shard episode records.

    Summing the per-shard summary JSONs instead is only correct while every file
    in the directory belongs to the same shard layout: the shard set is whatever
    happens to match the glob, so a leftover file from an earlier run with a
    different worker count (or contiguous vs. round-robin sharding) overlaps the
    current shards and is silently added to the totals rather than rejected.
    Episode identity makes that impossible -- an overlapping shard re-reports the
    same episodes, and they collapse.
    """
    episodes = {}
    for file in json_files:
        ep_file = file[: -len(".json")] + "_episodes.jsonl"
        with open(ep_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                episodes[(record["task_id"], record["episode_idx"])] = record

    suite_results = {"overall": {"total_count": 0, "success_count": 0}}
    for record in episodes.values():
        success = 1 if record["success"] else 0
        for key in ("overall", record["category"]):
            bucket = suite_results.setdefault(key, {"total_count": 0, "success_count": 0})
            bucket["total_count"] += 1
            bucket["success_count"] += success
    return suite_results


def _aggregate_from_summaries(json_files: list) -> dict:
    """Legacy path: sum the per-shard summary JSONs (see _aggregate_from_episodes)."""
    suite_results = {"overall": {"total_count": 0, "success_count": 0}}
    for file in json_files:
        with open(file, encoding="utf-8") as f:
            results = json.load(f)
        for item, r in results.items():
            suite_results["overall"]["total_count"] += r["total_count"]
            suite_results["overall"]["success_count"] += r["success_count"]
            if item not in suite_results:
                suite_results[item] = dict(r)
            else:
                suite_results[item]["total_count"] += r["total_count"]
                suite_results[item]["success_count"] += r["success_count"]
    return suite_results


def aggregate_suite(root_path: str, task_suite: str) -> dict:
    cur_root = os.path.join(root_path, "logs", task_suite)
    json_files = [
        f for f in glob.glob(os.path.join(cur_root, "*.json"))
        if not f.endswith("overall_results.json")
    ]

    missing = [f for f in json_files if not os.path.exists(f[: -len(".json")] + "_episodes.jsonl")]
    if missing:
        print(
            f"[WARN] {task_suite}: {len(missing)}/{len(json_files)} shards have no episode records "
            f"(e.g. {os.path.basename(missing[0])}); falling back to summing shard summaries, which "
            f"double-counts any shards left over from a run with a different worker count.",
            file=sys.stderr,
        )
        suite_results = _aggregate_from_summaries(json_files)
    else:
        suite_results = _aggregate_from_episodes(json_files)

    for category, r in suite_results.items():
        total = r["total_count"]
        r["success_rate"] = float(r["success_count"]) / float(total) if total > 0 else 0.0

    return suite_results


def main() -> None:
    parser = argparse.ArgumentParser(description="aggregate results")
    parser.add_argument("--root_path", required=True, help="output_dir passed to eval_libero_model.py")
    parser.add_argument(
        "--task_suite_name",
        default=None,
        help="Limit to one suite and write a suite-scoped <root_path>/<suite>/overall_results.json "
        "instead of the combined <root_path>/overall_results.json (default: process all 4 suites).",
    )
    args = parser.parse_args()

    if args.task_suite_name:
        suite_results = aggregate_suite(args.root_path, args.task_suite_name)
        suite_out_dir = Path(args.root_path) / args.task_suite_name
        suite_out_dir.mkdir(parents=True, exist_ok=True)
        out = suite_out_dir / "overall_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(suite_results, f, indent=2)
        print(f"Wrote {out}")
        print(json.dumps(suite_results, indent=2))
        return

    combined = {suite: aggregate_suite(args.root_path, suite) for suite in ALL_SUITES}
    out = Path(args.root_path) / "overall_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
