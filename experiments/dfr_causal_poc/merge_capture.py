"""Validate and merge independently captured pair manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    rows = []
    summaries = []
    for shard in range(args.num_shards):
        directory = args.root / "shards" / str(shard)
        manifest = directory / "manifest.jsonl"
        summary = directory / "capture_summary.json"
        if not manifest.is_file() or not summary.is_file():
            raise FileNotFoundError(f"shard {shard} is incomplete: {directory}")
        rows.extend(json.loads(line) for line in manifest.read_text().splitlines() if line.strip())
        summaries.append(json.loads(summary.read_text()))
    rows.sort(key=lambda row: int(row["pair_id"]))
    pair_ids = [int(row["pair_id"]) for row in rows]
    expected = list(range(args.expected_pairs))
    if pair_ids != expected:
        raise RuntimeError(f"pair coverage mismatch: got {pair_ids}, expected {expected}")

    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    merged = {
        "format": "starvla_paired_libero_plus_v1",
        "pairs": len(rows),
        "base_tasks": sorted({row["base_task"] for row in rows}),
        "categories": sorted({row["category"] for row in rows}),
        "max_observed_state_diff": max(float(row["state_max_abs_diff"]) for row in rows),
        "num_shards": args.num_shards,
        "shard_summaries": summaries,
        "manifest": str(manifest.resolve()),
    }
    (args.root / "capture_summary.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged {len(rows)} pairs into {manifest}")


if __name__ == "__main__":
    main()
