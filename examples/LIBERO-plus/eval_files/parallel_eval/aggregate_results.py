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
from pathlib import Path

ALL_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def aggregate_suite(root_path: str, task_suite: str) -> dict:
    cur_root = os.path.join(root_path, "logs", task_suite)
    json_files = glob.glob(os.path.join(cur_root, "*.json"))

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
