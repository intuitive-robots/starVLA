#!/usr/bin/env python3
"""Create the cross-architecture paired-counterfactual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=SUMMARY_JSON")
    args = parser.parse_args()
    models = []
    for specification in args.model:
        name, path = specification.split("=", 1)
        payload = json.loads(Path(path).read_text())
        best = max(payload["results"], key=lambda row: row["paired_perturbed_improvement"])
        models.append((name, payload, best))
    lines = [
        "# Paired successful-trajectory comparison",
        "",
        "Each row uses that policy's own clean-successful trajectories and actions. Percentages are offline flow-residual changes, not rollout success.",
        "",
        "| Model | Tasks (suite coverage) | Best locus | Perturbed | Clean | Paired advantage | CF gap reduction | Pass |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    percent = lambda value: f"{100 * value:+.2f}%"
    for name, payload, row in models:
        coverage = payload["coverage"]
        by_suite = ", ".join(
            f"{suite.removeprefix('libero_')}={count}"
            for suite, count in coverage["tasks_by_suite"].items()
        )
        lines.append(
            f"| {name} | {coverage['tasks']} ({by_suite}) | `{row['locus']}` | {percent(row['paired_perturbed_improvement'])} | "
            f"{percent(row['paired_clean_improvement'])} | {percent(row['paired_advantage'])} | "
            f"{percent(row['counterfactual_gap_reduction'])} | {'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(["", "## Best-locus perturbation breakdown", ""])
    categories = sorted({category for _, _, row in models for category in row["by_category"]})
    lines.append("| Model | " + " | ".join(categories) + " |")
    lines.append("|---|" + "---:|" * len(categories))
    for name, _, row in models:
        lines.append(
            f"| {name} | "
            + " | ".join(percent(row["by_category"][category]["perturbed_improvement"]) for category in categories)
            + " |"
        )
    lines.extend([
        "",
        "A passing offline locus only authorizes paired closed-loop evaluation on the same clean-successful initial-state seeds. It is not itself a performance result.",
        "",
    ])
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
