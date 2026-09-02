"""Merge per-shard LIBERO result JSONs into one summary.

Reads ``<root>/logs/<suite>/<start>_to_<end>.json`` written by
``eval_libero_shard.py`` and prints / writes overall + per-task success rates.
"""

import argparse
import json
import pathlib
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, help="output_dir passed to eval_libero_shard.py")
    parser.add_argument("--task_suite_name", default=None, help="Limit to one suite (default: all found)")
    args = parser.parse_args()

    root = pathlib.Path(args.root_path)
    logs_root = root / "logs"
    if not logs_root.is_dir():
        raise SystemExit(f"No logs/ directory under {root}")

    suites = [logs_root / args.task_suite_name] if args.task_suite_name else sorted(p for p in logs_root.iterdir() if p.is_dir())

    overall = {}
    for suite_dir in suites:
        shard_files = sorted(suite_dir.glob("*_to_*.json"))
        if not shard_files:
            continue

        total, success = 0, 0
        per_task = defaultdict(lambda: {"task_description": "", "total_count": 0, "success_count": 0})
        covered = []

        for f in shard_files:
            with open(f, encoding="utf-8") as fh:
                try:
                    d = json.load(fh)
                except json.JSONDecodeError:
                    # Stray/empty/truncated file (e.g. debris from a killed
                    # prior run, or a still-in-flight write) -- skip it rather
                    # than crashing the whole aggregation over one shard.
                    print(f"  [WARN] {suite_dir.name}: unreadable shard {f.name}, skipping")
                    continue
            total += d["total_count"]
            success += d["success_count"]
            covered.append((d["start_idx"], d["end_idx"]))
            for task_id, r in d.get("per_task", {}).items():
                per_task[task_id]["task_description"] = r["task_description"]
                per_task[task_id]["total_count"] += r["total_count"]
                per_task[task_id]["success_count"] += r["success_count"]

        # Warn on gaps / overlaps between shards.
        covered.sort()
        cursor = covered[0][0] if covered else 0
        for lo, hi in covered:
            if lo > cursor:
                print(f"  [WARN] {suite_dir.name}: gap in episode coverage [{cursor}, {lo})")
            elif lo < cursor:
                print(f"  [WARN] {suite_dir.name}: overlapping shards near episode {lo}")
            cursor = max(cursor, hi)

        rate = success / total if total else 0.0
        overall[suite_dir.name] = {
            "total_count": total,
            "success_count": success,
            "success_rate": rate,
            "num_shards": len(shard_files),
            "per_task": {
                k: {**v, "success_rate": v["success_count"] / v["total_count"] if v["total_count"] else 0.0}
                for k, v in sorted(per_task.items(), key=lambda kv: int(kv[0]))
            },
        }

        print(f"\n=== {suite_dir.name} ({len(shard_files)} shards) ===")
        for task_id, v in overall[suite_dir.name]["per_task"].items():
            print(f"  task {task_id:>2}  {v['success_count']:>3}/{v['total_count']:<3} "
                  f"{v['success_rate'] * 100:5.1f}%  {v['task_description']}")
        print(f"  TOTAL     {success:>3}/{total:<3} {rate * 100:5.1f}%")

    if overall:
        grand_total = sum(v["total_count"] for v in overall.values())
        grand_success = sum(v["success_count"] for v in overall.values())
        print(f"\n=== ALL SUITES: {grand_success}/{grand_total} "
              f"{grand_success / grand_total * 100:.1f}% ===" if grand_total else "")

    out = root / "overall_results.json"
    merged = {}
    if out.is_file():
        with open(out, encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(overall)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
