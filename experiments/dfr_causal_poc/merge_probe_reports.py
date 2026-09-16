"""Merge four deterministic, rank-sharded causal probe reports."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ranks", default="1,4,8,16")
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",") if value.strip()]
    if not ranks:
        raise ValueError("no ranks supplied")

    reports = []
    for rank in ranks:
        path = args.root / "shards" / f"rank_{rank}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(json.loads(path.read_text()))

    merged = reports[0]
    invariant_keys = (
        "checkpoint",
        "manifest",
        "pairs",
        "activation_shape",
        "train_tasks",
        "test_tasks",
        "seed",
    )
    for report in reports[1:]:
        for key in invariant_keys:
            if report[key] != merged[key]:
                raise RuntimeError(f"rank-sharded reports disagree on {key}")

    combined_actions = {}
    for category in merged["action_interventions"]:
        baseline = reports[0]["action_interventions"][category]
        combined = {key: value for key, value in baseline.items() if key != "interventions"}
        combined["interventions"] = {}
        for report in reports:
            current = report["action_interventions"][category]
            for key, value in combined.items():
                if key == "interventions":
                    continue
                same = (
                    math.isclose(float(current[key]), float(value), rel_tol=1e-6, abs_tol=1e-10)
                    if isinstance(value, float)
                    else current[key] == value
                )
                if not same:
                    raise RuntimeError(f"rank reports disagree on {category}/{key}")
            overlap = combined["interventions"].keys() & current["interventions"].keys()
            if overlap:
                raise RuntimeError(f"duplicate interventions for {category}: {sorted(overlap)}")
            combined["interventions"].update(current["interventions"])
        combined_actions[category] = combined
    merged["action_interventions"] = combined_actions
    merged["ranks"] = ranks
    merged["rank_shards"] = [str((args.root / "shards" / f"rank_{rank}").resolve()) for rank in ranks]

    args.root.mkdir(parents=True, exist_ok=True)
    first = args.root / "shards" / f"rank_{ranks[0]}"
    shutil.copy2(first / "pooled_activations.npz", args.root / "pooled_activations.npz")
    shutil.copy2(first / "paired_subspaces.pt", args.root / "paired_subspaces.pt")
    (args.root / "report.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged ranks {ranks} into {args.root / 'report.json'}")


if __name__ == "__main__":
    main()
