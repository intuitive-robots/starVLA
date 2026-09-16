#!/usr/bin/env python3
"""Merge selected-adapter test shards without touching training/validation."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fit_pi_flow_field_adapters import CATEGORIES


def summarize(records):
    keys = (
        "base_pair_endpoint_mse", "adapted_pair_endpoint_mse",
        "base_perturbed_successful_mse", "adapted_perturbed_successful_mse",
        "base_clean_successful_mse", "adapted_clean_successful_mse",
        "clean_endpoint_drift_mse", "successful_action_energy",
    )
    mean = {key: float(np.mean([row[key] for row in records])) for key in keys}
    mean.update(
        pair_endpoint_improvement=1 - mean["adapted_pair_endpoint_mse"] / mean["base_pair_endpoint_mse"],
        successful_endpoint_improvement=1 - mean["adapted_perturbed_successful_mse"] / mean["base_perturbed_successful_mse"],
        clean_successful_endpoint_improvement=1 - mean["adapted_clean_successful_mse"] / mean["base_clean_successful_mse"],
        clean_drift_over_successful_action_energy=mean["clean_endpoint_drift_mse"] / mean["successful_action_energy"],
    )
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for shard in range(4):
        payload = json.loads((args.root / f"shard_{shard}.json").read_text())
        records.extend(payload["records"])
    if len(records) != 256 or len({row["pair_id"] for row in records}) != 256:
        raise RuntimeError("expected 256 unique sealed-test pairs")
    overall = summarize(records)
    by_category = {category: summarize([row for row in records if row["category"] == category]) for category in CATEGORIES}
    clusters = defaultdict(list)
    for record in records:
        clusters[(record["suite"], record["task_id"])].append(record)
    keys = sorted(clusters)
    rng = np.random.default_rng(20263933)
    pair_values, success_values = [], []
    for _ in range(5000):
        sampled = rng.choice(len(keys), len(keys), replace=True)
        chosen = [row for index in sampled for row in clusters[keys[int(index)]]]
        metrics = summarize(chosen)
        pair_values.append(metrics["pair_endpoint_improvement"])
        success_values.append(metrics["successful_endpoint_improvement"])
    result = {
        "format": "starvla_pi_endpoint_sealed_test_v1", "status": "complete",
        "pairs": len(records), "tasks": len(keys), "overall": overall,
        "by_category": by_category,
        "task_cluster_bootstrap": {
            "replicates": 5000,
            "pair_endpoint_improvement_95pct": np.quantile(pair_values, (.025, .975)).tolist(),
            "successful_endpoint_improvement_95pct": np.quantile(success_values, (.025, .975)).tolist(),
        },
    }
    (args.root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
