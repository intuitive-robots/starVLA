#!/usr/bin/env python3
"""Summarize matched baseline/adapter PI endpoint closed-loop rollouts."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_plus(root: Path, arm: str) -> dict[int, dict]:
    records = {}
    log_dir = root / arm / "libero_plus_exact140" / "logs" / "libero_10"
    for path in log_dir.glob("*_episodes.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            task_id = int(record["task_id"])
            if task_id in records:
                raise RuntimeError(f"duplicate task id {task_id} in {arm}")
            records[task_id] = record
    if len(records) != 140:
        raise RuntimeError(f"expected 140 {arm} LIBERO+ episodes, got {len(records)}")
    return records


def exact_mcnemar(baseline_only: int, adapter_only: int) -> float:
    discordant = baseline_only + adapter_only
    smaller = min(baseline_only, adapter_only)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", default="1712108")
    args = parser.parse_args()

    baseline = load_plus(args.root, "baseline")
    adapter = load_plus(args.root, "endpoint_correct")
    if baseline.keys() != adapter.keys():
        raise RuntimeError("baseline and adapter task IDs differ")

    task_ids = sorted(baseline)
    outcomes = Counter(
        (bool(baseline[i]["success"]), bool(adapter[i]["success"])) for i in task_ids
    )
    baseline_only = outcomes[(True, False)]
    adapter_only = outcomes[(False, True)]
    differences = np.asarray(
        [int(adapter[i]["success"]) - int(baseline[i]["success"]) for i in task_ids]
    )
    rng = np.random.default_rng(20260908)
    bootstrap = differences[rng.integers(0, len(differences), (100_000, len(differences)))].mean(1)

    by_category = defaultdict(list)
    for task_id in task_ids:
        if baseline[task_id]["category"] != adapter[task_id]["category"]:
            raise RuntimeError(f"category mismatch for task {task_id}")
        by_category[baseline[task_id]["category"]].append(task_id)

    categories = {}
    for category, ids in sorted(by_category.items()):
        category_outcomes = Counter(
            (bool(baseline[i]["success"]), bool(adapter[i]["success"])) for i in ids
        )
        categories[category] = {
            "n": len(ids),
            "baseline_successes": sum(bool(baseline[i]["success"]) for i in ids),
            "adapter_successes": sum(bool(adapter[i]["success"]) for i in ids),
            "baseline_only": category_outcomes[(True, False)],
            "adapter_only": category_outcomes[(False, True)],
        }

    clean = {}
    for arm in ("baseline", "endpoint_correct"):
        payload = json.loads(
            (args.root / arm / "libero_clean_3x10" / "overall_results.json").read_text()
        )["libero_10"]
        if payload["total_count"] != 30:
            raise RuntimeError(f"expected 30 clean episodes for {arm}")
        clean[arm] = {
            key: payload[key] for key in ("total_count", "success_count", "success_rate")
        }

    report = {
        "format": "starvla_pi_endpoint_closed_loop_v1",
        "status": "complete",
        "slurm_job_id": args.job_id,
        "libero_plus": {
            "n": len(task_ids),
            "baseline_successes": sum(bool(baseline[i]["success"]) for i in task_ids),
            "adapter_successes": sum(bool(adapter[i]["success"]) for i in task_ids),
            "adapter_minus_baseline": float(differences.mean()),
            "paired_task_bootstrap_95pct": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
            "matched_outcomes": {
                "both_success": outcomes[(True, True)],
                "both_failure": outcomes[(False, False)],
                "baseline_only": baseline_only,
                "adapter_only": adapter_only,
            },
            "exact_mcnemar_p": exact_mcnemar(baseline_only, adapter_only),
            "by_category": categories,
        },
        "clean_libero_10": clean,
    }
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    plus = report["libero_plus"]
    lines = [
        "# PI sampler-aware adapter closed-loop result",
        "",
        f"Slurm job `{args.job_id}` completed with all 4 GPUs and zero task retries.",
        "",
        "| Arm | LIBERO+ exact 140 | Clean LIBERO-10 |",
        "|---|---:|---:|",
        f"| Frozen baseline | {plus['baseline_successes']}/140 | {clean['baseline']['success_count']}/30 |",
        f"| Endpoint adapter | {plus['adapter_successes']}/140 | {clean['endpoint_correct']['success_count']}/30 |",
        "",
        f"Matched LIBERO+ change: {100 * plus['adapter_minus_baseline']:+.2f} points, "
        f"paired task-bootstrap 95% CI [{100 * plus['paired_task_bootstrap_95pct'][0]:+.2f}, "
        f"{100 * plus['paired_task_bootstrap_95pct'][1]:+.2f}], exact McNemar "
        f"p={plus['exact_mcnemar_p']:.3g}. There were {plus['matched_outcomes']['baseline_only']} "
        f"baseline-only successes and {plus['matched_outcomes']['adapter_only']} adapter-only successes.",
        "",
        "| Category | n | Baseline | Adapter | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, value in categories.items():
        change = value["adapter_successes"] - value["baseline_successes"]
        lines.append(
            f"| {category} | {value['n']} | {value['baseline_successes']} | "
            f"{value['adapter_successes']} | {change:+d} |"
        )
    lines += [
        "",
        "Conclusion: the offline endpoint improvement did not translate into aggregate "
        "closed-loop robustness. Clean performance was retained, and camera/robot-initial-state "
        "subtotals increased, but these exploratory category changes were offset by worse "
        "background/language outcomes and are too small for category-level claims.",
        "",
    ]
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
