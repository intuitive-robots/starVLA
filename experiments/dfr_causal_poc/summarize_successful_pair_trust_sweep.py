#!/usr/bin/env python3
"""Select a flow-residual trust region using closed-loop validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(root: Path, arm: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted((root / arm).glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["domain"], row["case_id"])
            if key in rows:
                raise RuntimeError(f"duplicate case for {arm}: {key}")
            if row.get("split") != "validation":
                raise RuntimeError(f"non-validation row in trust sweep: {row.get('split')!r}")
            rows[key] = row
    return rows


def _rate(rows: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "successes": successes,
        "episodes": len(rows),
        "rate": successes / len(rows) if rows else None,
    }


def _paired_change(base: list[dict], candidate: list[dict], seed: int) -> dict:
    base_values = np.asarray([row["success"] for row in base], dtype=float)
    candidate_values = np.asarray([row["success"] for row in candidate], dtype=float)
    differences = candidate_values - base_values
    tasks = np.asarray([f"{row['suite']}:{row['base_task']}" for row in base])
    unique_tasks = np.unique(tasks)
    task_differences = np.asarray([differences[tasks == task].mean() for task in unique_tasks])
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            task_differences[
                rng.integers(0, len(task_differences), len(task_differences))
            ].mean()
            for _ in range(10000)
        ]
    )
    return {
        "baseline": _rate(base),
        "candidate": _rate(candidate),
        "absolute_change": float(differences.mean()),
        "wins": int(np.sum(differences > 0)),
        "losses": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
        "task_cluster_bootstrap_95_interval": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-clean", type=int, required=True)
    parser.add_argument("--expected-perturbed", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    configurations = json.loads(args.config.read_text())
    names = [row["name"] for row in configurations]
    if not names or names[0] != "strength_0":
        raise RuntimeError("first sweep configuration must be strength_0 baseline")
    arms = {name: _load(args.root, name) for name in names}
    coverage = set(arms[names[0]])
    for name, rows in arms.items():
        if set(rows) != coverage:
            raise RuntimeError(f"coverage mismatch for {name}")
    clean_keys = sorted(key for key in coverage if key[0] == "clean")
    perturbed_keys = sorted(key for key in coverage if key[0] == "perturbed")
    if len(clean_keys) != args.expected_clean or len(perturbed_keys) != args.expected_perturbed:
        raise RuntimeError(
            f"coverage clean={len(clean_keys)}/{args.expected_clean}, "
            f"perturbed={len(perturbed_keys)}/{args.expected_perturbed}"
        )

    baseline = arms[names[0]]
    reports = {}
    for index, config in enumerate(configurations):
        name = config["name"]
        rows = arms[name]
        clean = _paired_change(
            [baseline[key] for key in clean_keys],
            [rows[key] for key in clean_keys],
            args.seed + index,
        )
        perturbed = _paired_change(
            [baseline[key] for key in perturbed_keys],
            [rows[key] for key in perturbed_keys],
            args.seed + 100 + index,
        )
        by_category = {}
        categories = sorted({baseline[key]["category"] for key in perturbed_keys})
        for category_index, category in enumerate(categories):
            keys = [key for key in perturbed_keys if baseline[key]["category"] == category]
            by_category[category] = _paired_change(
                [baseline[key] for key in keys],
                [rows[key] for key in keys],
                args.seed + 200 + 10 * index + category_index,
            )
        reports[name] = {
            "configuration": config,
            "clean": clean,
            "perturbed": perturbed,
            "by_category": by_category,
            "clean_noninferior": clean["absolute_change"] >= -0.05,
        }

    eligible = [name for name in names if reports[name]["clean_noninferior"]]
    # Predeclared rule: among clean-safe candidates, maximize validation
    # perturbed success. Ties prefer the earlier, less aggressive candidate.
    selected = max(
        eligible,
        key=lambda name: (reports[name]["perturbed"]["candidate"]["rate"], -names.index(name)),
    )
    selected_report = reports[selected]
    proceed = (
        selected != "strength_0"
        and selected_report["perturbed"]["absolute_change"] > 0.0
    )
    output = {
        "format": "starvla_successful_pair_trust_sweep_v1",
        "selection_split": "validation",
        "selection_rule": (
            "maximize perturbed success among candidates with clean change >= -5pp; "
            "ties prefer the earlier/less aggressive candidate"
        ),
        "reports": reports,
        "selected": selected,
        "proceed_to_fresh_confirmation": proceed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
