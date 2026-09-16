#!/usr/bin/env python3
"""Merge the sampler-aware projection-LoRA ablation reports."""

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("endpoint_correct", "endpoint_no_pair", "endpoint_shuffled", "local_flow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    reports = {}
    for arm in ARMS:
        report = json.loads((args.root / arm / "report.json").read_text())
        reports[arm] = report
        metric = report["validation"]["overall"]
        interval = report["validation"]["task_cluster_bootstrap"]["successful_endpoint_improvement_95pct"]
        rows.append({
            "arm": arm, "objective": report["objective"], "parameters": report["trainable_parameters"],
            "pair_endpoint_improvement": metric["pair_endpoint_improvement"],
            "successful_endpoint_improvement": metric["successful_endpoint_improvement"],
            "successful_endpoint_improvement_95pct": interval,
            "clean_successful_endpoint_improvement": metric["clean_successful_endpoint_improvement"],
            "clean_drift_over_successful_action_energy": metric["clean_drift_over_successful_action_energy"],
            "passes_individual_screen": report["screen"]["passes_individual_screen"],
        })
    mapped = {row["arm"]: row for row in rows}
    correct = mapped["endpoint_correct"]
    correct_beats_controls = (
        correct["successful_endpoint_improvement"] > mapped["endpoint_no_pair"]["successful_endpoint_improvement"]
        and correct["successful_endpoint_improvement"] > mapped["endpoint_shuffled"]["successful_endpoint_improvement"]
        and correct["successful_endpoint_improvement"] > mapped["local_flow"]["successful_endpoint_improvement"]
    )
    proceed = correct["passes_individual_screen"] and correct_beats_controls
    # Because every arm uses exactly the same validation pairs, quantify the
    # incremental benefit of correct pairing with a paired task bootstrap.
    record_maps = {
        arm: {record["pair_id"]: record for record in report["validation"]["records"]}
        for arm, report in reports.items()
    }
    pair_ids = set(record_maps["endpoint_correct"])
    if any(set(records) != pair_ids for records in record_maps.values()):
        raise RuntimeError("validation pair IDs differ across arms")
    clusters = {}
    for pair_id, record in record_maps["endpoint_correct"].items():
        clusters.setdefault((record["suite"], record["task_id"]), []).append(pair_id)
    cluster_keys = sorted(clusters)
    rng = np.random.default_rng(20262933)

    def successful_improvement(arm, selected_ids):
        records = record_maps[arm]
        base = np.mean([records[pair_id]["base_perturbed_successful_mse"] for pair_id in selected_ids])
        adapted = np.mean([records[pair_id]["adapted_perturbed_successful_mse"] for pair_id in selected_ids])
        return 1.0 - adapted / base

    control_names = ("endpoint_no_pair", "endpoint_shuffled", "local_flow")
    differences = {control: [] for control in control_names}
    for _ in range(5000):
        sampled = rng.choice(len(cluster_keys), len(cluster_keys), replace=True)
        selected_ids = [pair_id for index in sampled for pair_id in clusters[cluster_keys[int(index)]]]
        correct_value = successful_improvement("endpoint_correct", selected_ids)
        for control in control_names:
            differences[control].append(correct_value - successful_improvement(control, selected_ids))
    paired_control_differences = {
        control: {
            "point_difference": correct["successful_endpoint_improvement"] - mapped[control]["successful_endpoint_improvement"],
            "task_bootstrap_95pct": np.quantile(values, (0.025, 0.975)).tolist(),
        }
        for control, values in differences.items()
    }
    robustly_beats_all_controls = all(
        value["task_bootstrap_95pct"][0] > 0 for value in paired_control_differences.values()
    )
    summary = {
        "format": "starvla_pi_unrolled_endpoint_ablation_summary_v1", "status": "complete",
        "scope": "validation only; test split untouched; no closed-loop success claim",
        "rows": rows, "correct_pair_beats_all_controls": correct_beats_controls,
        "paired_correct_minus_control": paired_control_differences,
        "correct_pair_robustly_beats_all_controls": robustly_beats_all_controls,
        "proceed_to_sealed_test_and_rollout": proceed,
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# PI sampler-aware endpoint adaptation results", "",
        "Frozen alternating PI; rank-4 projection LoRA only. All arms use identical balanced pairs, deterministic noise, adapter size, and validation endpoints.", "",
        "| Arm | Objective | Params | Pair endpoint | Successful endpoint (95% task CI) | Clean successful endpoint | Clean drift / action energy | Individual screen |", "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lo, hi = row["successful_endpoint_improvement_95pct"]
        lines.append(
            f"| {row['arm']} | {row['objective']} | {row['parameters']:,} | {row['pair_endpoint_improvement']:+.2%} | "
            f"{row['successful_endpoint_improvement']:+.2%} [{lo:+.2%}, {hi:+.2%}] | "
            f"{row['clean_successful_endpoint_improvement']:+.2%} | {row['clean_drift_over_successful_action_energy']:.3%} | {row['passes_individual_screen']} |"
        )
    lines += ["", "Paired task-bootstrap differences for successful-endpoint improvement (correct minus control):", ""]
    for control, value in paired_control_differences.items():
        lo, hi = value["task_bootstrap_95pct"]
        lines.append(f"- `{control}`: {value['point_difference']:+.2%} [{lo:+.2%}, {hi:+.2%}]")
    lines += [
        "", f"Correct pairs beat all controls by point estimate: **{correct_beats_controls}**.",
        f"Correct pairs robustly beat all controls (all paired 95% lower bounds > 0): **{robustly_beats_all_controls}**.",
        f"Preregistered proceed decision: **{proceed}**.",
    ]
    (args.root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
