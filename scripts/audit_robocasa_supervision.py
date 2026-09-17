"""Audit local RoboCasa metadata and map episodes to official source archives.

This produces provenance, not supervision. It never treats an archive URL as
proof that replay, camera alignment, or semantic labels have been validated.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset.resolve()
    registry_bytes = args.registry.read_bytes()
    registry = json.loads(registry_bytes)
    info = json.loads((root / "meta/info.json").read_text())
    columns = ["episode_index", "source_prefix", "source_episode_index", "length",
               "dataset_from_index", "dataset_to_index"]
    paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
    if not paths:
        raise ValueError("No episode metadata found")
    rows = [row for path in paths for row in pq.read_table(path, columns=columns).to_pylist()]
    keys = [(row["source_prefix"], row["source_episode_index"]) for row in rows]
    assert len(keys) == len(set(keys)), "Source episode IDs are not unique"
    assert len(rows) == info["total_episodes"], "Episode count mismatch"
    assert sum(row["length"] for row in rows) == info["total_frames"], "Frame count mismatch"
    assert all(row["dataset_to_index"] - row["dataset_from_index"] == row["length"] for row in rows)
    for row in rows:
        key = row["source_prefix"] + "/lerobot.tar"
        row["archive_key"] = key
        row["archive_url"] = registry.get(key, "")
    counts = collections.Counter(row["source_prefix"] for row in rows)
    summary = {
        "dataset": str(root), "episodes": len(rows),
        "frames": sum(row["length"] for row in rows), "task_source_counts": dict(sorted(counts.items())),
        "unique_source_episode_keys": len(set(keys)),
        "episodes_with_registry_match": sum(bool(row["archive_url"]) for row in rows),
        "local_extras_present": (root / "extras").exists(),
        "features": info["features"], "metadata_sources": [str(p) for p in paths],
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "status": "Provenance only; replay and supervision alignment not validated",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "episode_sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + ["archive_key", "archive_url"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["episode_index"]))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("features", "metadata_sources")}, indent=2))


if __name__ == "__main__":
    main()
