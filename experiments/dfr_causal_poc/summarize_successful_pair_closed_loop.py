#!/usr/bin/env python3
"""Summarize paired baseline/readout closed-loop outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(root: Path, arm: str) -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted((root / arm).glob("shard_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["domain"], row["case_id"])
            if key in result:
                raise RuntimeError(f"duplicate {arm} case {key}")
            result[key] = row
    return result


def _rate(rows: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in rows)
    return {"successes": successes, "episodes": len(rows), "rate": successes / len(rows) if rows else None}


def _paired_report(base: list[dict], corrected: list[dict], seed: int) -> dict:
    if [row["case_id"] for row in base] != [row["case_id"] for row in corrected]:
        raise RuntimeError("paired case order mismatch")
    base_values = np.asarray([row["success"] for row in base], dtype=float)
    corrected_values = np.asarray([row["success"] for row in corrected], dtype=float)
    differences = corrected_values - base_values
    tasks = np.asarray([f"{row['suite']}:{row['base_task']}" for row in base])
    unique_tasks = np.unique(tasks)
    task_differences = np.asarray([differences[tasks == task].mean() for task in unique_tasks])
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        task_differences[rng.integers(0, len(task_differences), len(task_differences))].mean()
        for _ in range(10000)
    ])
    return {
        "baseline": _rate(base),
        "paired_pi_late": _rate(corrected),
        "absolute_change": float(differences.mean()),
        "wins": int(np.sum(differences > 0)),
        "losses": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
        "task_cluster_bootstrap_95_interval": [float(x) for x in np.quantile(bootstrap, (0.025, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-perturbed", type=int, default=104)
    parser.add_argument("--expected-clean", type=int, default=26)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    arms = {arm: _load(args.root, arm) for arm in ("baseline", "paired_pi_late")}
    source_replay = _load(args.root, "source_replay")
    if set(arms["baseline"]) != set(arms["paired_pi_late"]):
        missing = set(arms["baseline"]) ^ set(arms["paired_pi_late"])
        raise RuntimeError(f"arm coverage mismatch: {sorted(missing)[:10]}")
    keys = sorted(arms["baseline"])
    by_domain = {}
    for domain in ("clean", "perturbed"):
        selected = [key for key in keys if key[0] == domain]
        expected = args.expected_clean if domain == "clean" else args.expected_perturbed
        if expected > 0 and len(selected) != expected:
            raise RuntimeError(f"expected {expected} {domain} cases, found {len(selected)}")
        base = [arms["baseline"][key] for key in selected]
        corrected = [arms["paired_pi_late"][key] for key in selected]
        report = _paired_report(base, corrected, args.seed)
        replay_keys = [key for key in selected if key in source_replay]
        if replay_keys:
            replay_rows = [source_replay[key] for key in replay_keys]
            report["source_action_replay"] = _rate(replay_rows)
            valid_keys = [key for key in replay_keys if source_replay[key]["success"]]
            if valid_keys:
                report["source_replay_valid_subset"] = _paired_report(
                    [arms["baseline"][key] for key in valid_keys],
                    [arms["paired_pi_late"][key] for key in valid_keys],
                    args.seed + 500,
                )
        if domain == "perturbed":
            report["by_category"] = {}
            for category in sorted({row["category"] for row in base}):
                indices = [i for i, row in enumerate(base) if row["category"] == category]
                report["by_category"][category] = _paired_report(
                    [base[i] for i in indices], [corrected[i] for i in indices], args.seed + len(indices)
                )
        report["by_suite"] = {}
        for suite in sorted({row["suite"] for row in base}):
            indices = [i for i, row in enumerate(base) if row["suite"] == suite]
            report["by_suite"][suite] = _paired_report(
                [base[i] for i in indices], [corrected[i] for i in indices], args.seed + 100 + len(indices)
            )
        by_domain[domain] = report

    pert = by_domain["perturbed"]
    clean = by_domain["clean"]
    category_changes = [row["absolute_change"] for row in pert["by_category"].values()]
    criteria = {
        "perturbed_absolute_gain_at_least_5pp": pert["absolute_change"] >= 0.05,
        "perturbed_task_bootstrap_lower_above_zero": pert["task_cluster_bootstrap_95_interval"][0] > 0.0,
        "clean_drop_no_more_than_5pp": clean["absolute_change"] >= -0.05,
        "nonnegative_in_at_least_three_categories": sum(value >= 0.0 for value in category_changes) >= 3,
    }
    output = {
        "format": "starvla_successful_pair_closed_loop_summary_v1",
        "design": {
            "initial_states": "first state of held-out clean trajectory successfully completed by source policy",
            "perturbations": "held-out LIBERO-Plus appearance variants",
            "flow_noise": "identical deterministic seed per paired case and action chunk",
            "unit": "one rollout per task/category for perturbed; one per task for clean",
        },
        "domains": by_domain,
        "predeclared_criteria": criteria,
        "passes_all": all(criteria.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
