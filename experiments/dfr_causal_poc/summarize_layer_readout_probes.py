#!/usr/bin/env python3
"""Merge layer-readout probe shards and write JSON/Markdown summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOCI = ("raw_preprojector", "projected_memory", "pi_input", "pi_early", "pi_middle", "pi_late")


def _percent(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    probe_dir = args.root / "probe_shards"
    reports = {}
    for locus in LOCI:
        matches = list(probe_dir.glob(f"rank_*_{locus}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one report for {locus}, found {matches}")
        reports[locus] = json.loads(matches[0].read_text())

    summary = {
        "format": "starvla_layer_readout_sufficiency_summary_v1",
        "question": (
            "Does a frozen P-causal representation contain linearly accessible flow-residual "
            "information that a group-balanced readout can use on successful generalized "
            "LIBERO-Plus trajectories without damaging clean LIBERO?"
        ),
        "loci": {},
        "decision_rule": {
            "plus_improvement_min": 0.10,
            "balanced_advantage_over_clean_only_min": 0.05,
            "clean_improvement_min": -0.02,
            "plus_bootstrap_lower_bound_min": 0.0,
        },
        "caveats": [
            "This is a teacher-forced flow-residual screen, not closed-loop success.",
            "Raw/projected memory probes use masked mean pooling plus the source action token; failure there cannot prove information absence.",
            "The generalized dataset has successful expert actions but no retained LIBERO-Plus perturbation-category label, so it is one broad shifted group.",
            "The source model was trained on the clean dataset; trajectory-disjoint splits prevent probe leakage but cannot undo source pretraining exposure to clean trajectories.",
        ],
    }
    passing = []
    for locus, report in reports.items():
        clean_only = report["arms"]["clean_only"]["test"]
        balanced = report["arms"]["balanced"]["test"]
        plus_gain = balanced["plus"]["relative_residual_mse_improvement"]
        clean_gain = balanced["clean"]["relative_residual_mse_improvement"]
        advantage = plus_gain - clean_only["plus"]["relative_residual_mse_improvement"]
        lower = balanced["plus"]["trajectory_bootstrap_95_interval"][0]
        passed = plus_gain >= 0.10 and advantage >= 0.05 and clean_gain >= -0.02 and lower > 0.0
        summary["loci"][locus] = {
            "feature_dimension": report["feature_dimension"],
            "pi_depths": report["pi_depths"],
            "clean_only_test": clean_only,
            "balanced_test": balanced,
            "balanced_advantage_on_plus": advantage,
            "passes_screen": passed,
            "source_report": str(next(probe_dir.glob(f"rank_*_{locus}.json")).resolve()),
        }
        if passed:
            passing.append(locus)
    summary["passing_loci"] = passing
    summary["conclusion"] = (
        "At least one locus passes the offline sufficiency screen; implement the shallowest passing residual as a closed-loop intervention."
        if passing
        else "No locus passes the offline sufficiency screen; do not launch closed-loop residual-readout evaluation."
    )
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Layer-localized action-readout sufficiency result",
        "",
        summary["question"],
        "",
        "| Locus | Clean-only: Plus | Balanced: Plus | Balanced: Clean | Balanced advantage | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for locus in LOCI:
        item = summary["loci"][locus]
        clean_only_plus = item["clean_only_test"]["plus"]["relative_residual_mse_improvement"]
        balanced_plus = item["balanced_test"]["plus"]["relative_residual_mse_improvement"]
        balanced_clean = item["balanced_test"]["clean"]["relative_residual_mse_improvement"]
        lines.append(
            f"| `{locus}` | {_percent(clean_only_plus)} | {_percent(balanced_plus)} | "
            f"{_percent(balanced_clean)} | {_percent(item['balanced_advantage_on_plus'])} | "
            f"{'yes' if item['passes_screen'] else 'no'} |"
        )
    lines.extend([
        "",
        "Positive percentages mean that the residual probe reduces the frozen source policy's teacher-forced flow error.",
        "",
        f"**Decision:** {summary['conclusion']}",
        "",
        "This result is only a representation/readout diagnostic. Any passing readout must still improve repeated paired closed-loop rollouts while retaining clean success.",
        "",
    ])
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
