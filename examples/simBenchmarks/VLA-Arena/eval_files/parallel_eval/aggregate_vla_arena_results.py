"""Merge VLA-Arena eval shards into one overall_results.json.

Each worker in the parallel harness writes shards/<partition>_<suite>_L<level>.json
containing {"<suite>/L<level>": {success_rate, avg_cost, num_episodes,
num_successes, task_level}}. This merges them into per-suite, per-level, and
per-category rollups plus a single episode-weighted overall number.

Success rates are recomputed from num_successes/num_episodes rather than
averaged, so suites with different episode counts are weighted correctly.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORIES = {
    "safety": "safety",
    "distractor": "distractor",
    "extrapolation": "extrapolation",
    "long_horizon": "long_horizon",
}


def _category(suite: str) -> str:
    for prefix, name in CATEGORIES.items():
        if suite.startswith(prefix):
            return name
    return "other"


def _rollup(rows):
    """Episode-weighted rollup over a list of shard result dicts."""
    eps = sum(int(r.get("num_episodes", 0)) for r in rows)
    suc = sum(int(r.get("num_successes", 0)) for r in rows)
    # avg_cost is a per-episode mean; weight it by episodes to recombine.
    cost_num = sum(float(r.get("avg_cost", 0.0)) * int(r.get("num_episodes", 0)) for r in rows)
    return {
        "success_rate": (suc / eps) if eps else 0.0,
        "avg_cost": (cost_num / eps) if eps else 0.0,
        "num_episodes": eps,
        "num_successes": suc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", required=True, help="output_dir containing shards/")
    ap.add_argument("--out", default=None, help="defaults to <root_path>/overall_results.json")
    args = ap.parse_args()

    root = Path(args.root_path)
    shard_dir = root / "shards"
    shards = sorted(shard_dir.glob("*.json"))
    if not shards:
        raise SystemExit(f"no shards found in {shard_dir}")

    by_key = {}
    for path in shards:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] skipping unreadable shard {path.name}: {exc}")
            continue
        for key, res in data.items():
            if key in by_key:
                print(f"[WARN] duplicate unit {key} (shard {path.name}); keeping first")
                continue
            by_key[key] = res

    per_suite = defaultdict(list)
    per_level = defaultdict(list)
    per_cat = defaultdict(list)
    per_suite_level = defaultdict(list)
    for key, res in by_key.items():
        # "<suite>/L<level>" or, when a worker ran a task slice,
        # "<suite>/L<level>/T<start>-<end>". Several slices roll up into one
        # (suite, level) entry, so key off the first two segments only.
        parts = key.split("/")
        suite, level = parts[0], parts[1].lstrip("L")
        per_suite[suite].append(res)
        per_level[f"L{level}"].append(res)
        per_cat[_category(suite)].append(res)
        per_suite_level[f"{suite}/L{level}"].append(res)

    out = {
        "overall": _rollup(list(by_key.values())),
        "by_category": {k: _rollup(v) for k, v in sorted(per_cat.items())},
        "by_level": {k: _rollup(v) for k, v in sorted(per_level.items())},
        "by_suite": {k: _rollup(v) for k, v in sorted(per_suite.items())},
        "by_suite_level": {k: _rollup(v) for k, v in sorted(per_suite_level.items())},
        "by_unit": dict(sorted(by_key.items())),
        "num_units": len(by_key),
        "num_shards": len(shards),
    }

    out_path = Path(args.out) if args.out else root / "overall_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    o = out["overall"]
    print(f"Merged {len(shards)} shards / {len(by_key)} units -> {out_path}")
    print(f"OVERALL  SR={o['success_rate']:.4f}  cost={o['avg_cost']:.4f}  "
          f"({o['num_successes']}/{o['num_episodes']} episodes)")
    for cat, r in out["by_category"].items():
        print(f"  {cat:<16} SR={r['success_rate']:.4f}  cost={r['avg_cost']:.4f}  n={r['num_episodes']}")


if __name__ == "__main__":
    main()
