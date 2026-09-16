#!/usr/bin/env python3
"""Merge locus reports and apply the frozen offline gate for paired refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOCI = ("raw_preprojector", "projected_memory", "pi_input", "pi_early", "pi_middle", "pi_late")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shift-threshold", type=float, default=0.10)
    parser.add_argument("--advantage-threshold", type=float, default=0.05)
    parser.add_argument("--clean-floor", type=float, default=-0.02)
    parser.add_argument("--gap-reduction-threshold", type=float, default=0.10)
    args = parser.parse_args()
    reports = {}
    for locus in LOCI:
        matches = list((args.root / "probe_shards").glob(f"rank_*_{locus}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one report for {locus}, found {matches}")
        reports[locus] = json.loads(matches[0].read_text())
    checkpoint = reports[LOCI[0]]["checkpoint"]
    architecture = reports[LOCI[0]]["architecture"]
    results = []
    for locus in LOCI:
        report = reports[locus]
        if report["checkpoint"] != checkpoint:
            raise RuntimeError("locus reports use different checkpoints")
        clean_arm = report["arms"]["clean_only"]["test"]
        paired = report["arms"]["paired_balanced"]["test"]
        shifted = float(paired["perturbed"]["relative_improvement"])
        clean = float(paired["clean"]["relative_improvement"])
        clean_only_shifted = float(clean_arm["perturbed"]["relative_improvement"])
        advantage = shifted - clean_only_shifted
        interval = paired["perturbed"]["trajectory_bootstrap_95_interval"]
        gap_reduction = float(paired["counterfactual_gap"]["relative_reduction"])
        passed = (
            shifted >= args.shift_threshold
            and advantage >= args.advantage_threshold
            and clean >= args.clean_floor
            and interval[0] > 0
            and gap_reduction >= args.gap_reduction_threshold
        )
        results.append({
            "locus": locus,
            "clean_only_perturbed_improvement": clean_only_shifted,
            "paired_perturbed_improvement": shifted,
            "paired_clean_improvement": clean,
            "paired_advantage": advantage,
            "perturbed_bootstrap_95_interval": interval,
            "source_perturbed_mse": paired["perturbed"]["source_mse"],
            "source_clean_mse": paired["clean"]["source_mse"],
            "counterfactual_gap_reduction": gap_reduction,
            "by_category": {
                category: {
                    "perturbed_improvement": values["relative_improvement"],
                    "counterfactual_gap_reduction": paired["counterfactual_gap"]["by_category"][category]["relative_reduction"],
                }
                for category, values in paired["perturbed"]["by_category"].items()
            },
            "by_suite": {
                suite: {
                    "perturbed_improvement": values["relative_improvement"],
                    "counterfactual_gap_reduction": paired["counterfactual_gap"]["by_suite"][suite]["relative_reduction"],
                }
                for suite, values in paired["perturbed"]["by_suite"].items()
            },
            "pass": passed,
        })
    summary = {
        "format": "starvla_successful_pair_readout_summary_v1",
        "checkpoint": checkpoint,
        "architecture": architecture,
        "coverage": reports[LOCI[0]]["coverage"],
        "decision_rule": {
            "paired_perturbed_improvement_min": args.shift_threshold,
            "paired_advantage_min": args.advantage_threshold,
            "paired_clean_improvement_min": args.clean_floor,
            "perturbed_bootstrap_lower_must_exceed_zero": True,
            "counterfactual_gap_reduction_min": args.gap_reduction_threshold,
        },
        "any_pass": any(row["pass"] for row in results),
        "results": results,
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Successful-trajectory paired counterfactual readout result",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "| Locus | Clean-only perturbed | Paired perturbed | Paired clean | Paired advantage | CF gap reduction | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in results:
        percent = lambda value: f"{100 * value:+.2f}%"
        lines.append(
            f"| `{row['locus']}` | {percent(row['clean_only_perturbed_improvement'])} | "
            f"{percent(row['paired_perturbed_improvement'])} | {percent(row['paired_clean_improvement'])} | "
            f"{percent(row['paired_advantage'])} | {percent(row['counterfactual_gap_reduction'])} | "
            f"{'yes' if row['pass'] else 'no'} |"
        )
    lines.extend([
        "",
        "Action/flow error is an offline screening metric, not policy performance. "
        + ("At least one locus passes; paired closed-loop evaluation is warranted." if summary["any_pass"] else "No locus passes; do not claim or launch recovery rollout from this screen."),
        "",
    ])
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
