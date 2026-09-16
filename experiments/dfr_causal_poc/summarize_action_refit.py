#!/usr/bin/env python3
"""Combine offline refit screens and closed-loop results into one table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = (
    "baseline",
    "fresh_final_balanced",
    "fresh_decoder_balanced",
    "warm_decoder_balanced",
    "fresh_decoder_clean",
)
TARGET_CATEGORIES = ("Background Textures", "Light Conditions", "Sensor Noise")


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_plus_episodes(root: Path, arm: str, plus_exact: int) -> dict:
    log_dir = (
        root
        / arm
        / "closed_loop"
        / f"libero_plus_exact{plus_exact}"
        / "logs"
        / "libero_10"
    )
    episodes = {}
    for path in sorted(log_dir.glob("*_episodes.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record["task_id"], record["episode_idx"])
            value = {
                "success": bool(record["success"]),
                "category": record["category"],
            }
            if key in episodes and episodes[key] != value:
                raise ValueError(f"conflicting duplicate episode {key} in {log_dir}")
            episodes[key] = value
    return episodes


def _load_clean_episodes(root: Path, arm: str, clean_trials: int) -> dict:
    log_dir = (
        root
        / arm
        / "closed_loop"
        / f"libero_clean_{clean_trials}x10"
        / "logs"
        / "libero_10"
    )
    episodes = {}
    for path in sorted(log_dir.glob("*_to_*.json")):
        report = json.loads(path.read_text())
        for record in report.get("episodes", []):
            key = (record["task_id"], record["episode_idx"])
            value = bool(record["success"])
            if key in episodes and episodes[key] != value:
                raise ValueError(f"conflicting duplicate episode {key} in {log_dir}")
            episodes[key] = value
    return episodes


def _mcnemar(
    arm: dict, baseline: dict, category: str | tuple[str, ...] | None = None
) -> dict:
    common = sorted(set(arm) & set(baseline))
    if category is not None:
        categories = (category,) if isinstance(category, str) else category
        common = [key for key in common if arm[key]["category"] in categories]
    arm_only = sum(
        arm[key]["success"] and not baseline[key]["success"] for key in common
    )
    baseline_only = sum(
        baseline[key]["success"] and not arm[key]["success"] for key in common
    )
    discordant = arm_only + baseline_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(arm_only, baseline_only) + 1))
        exact_p = min(1.0, 2.0 * tail / (2**discordant))
    else:
        exact_p = 1.0
    return {
        "matched_episodes": len(common),
        "arm_only_successes": arm_only,
        "baseline_only_successes": baseline_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": exact_p,
    }


def _mcnemar_clean(arm: dict, baseline: dict) -> dict:
    common = sorted(set(arm) & set(baseline))
    wrapped_arm = {key: {"success": arm[key]} for key in common}
    wrapped_baseline = {key: {"success": baseline[key]} for key in common}
    return _mcnemar(wrapped_arm, wrapped_baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plus-exact", type=int, default=140)
    parser.add_argument("--clean-trials", type=int, default=3)
    args = parser.parse_args()

    rows = []
    plus_episodes = {}
    clean_episodes = {}
    for arm in ARMS:
        report = _read(args.root / arm / "report.json") if arm != "baseline" else None
        plus = _read(
            args.root
            / arm
            / "closed_loop"
            / f"libero_plus_exact{args.plus_exact}"
            / "libero_10"
            / "overall_results.json"
        )
        clean = _read(
            args.root
            / arm
            / "closed_loop"
            / f"libero_clean_{args.clean_trials}x10"
            / "overall_results.json"
        )
        row = {"arm": arm}
        if report is not None:
            row["offline_before"] = report["validation_flow_loss_before"]
            row["offline_after"] = report["validation_flow_loss_after"]
            row["offline_relative_change"] = report["relative_change"]
        if plus is not None:
            row["libero_plus"] = {
                "overall": plus["overall"],
                **{category: plus.get(category) for category in TARGET_CATEGORIES},
            }
            plus_episodes[arm] = _load_plus_episodes(args.root, arm, args.plus_exact)
        if clean is not None:
            row["libero_clean"] = clean
            clean_episodes[arm] = _load_clean_episodes(
                args.root, arm, args.clean_trials
            )
        rows.append(row)

    matched_comparisons = {}
    if "baseline" in plus_episodes:
        for arm, episodes in plus_episodes.items():
            if arm == "baseline":
                continue
            matched_comparisons[arm] = {
                "libero_plus_overall": _mcnemar(episodes, plus_episodes["baseline"]),
                "libero_plus_target_categories_combined": _mcnemar(
                    episodes,
                    plus_episodes["baseline"],
                    category=TARGET_CATEGORIES,
                ),
                "libero_plus_by_target_category": {
                    category: _mcnemar(
                        episodes, plus_episodes["baseline"], category=category
                    )
                    for category in TARGET_CATEGORIES
                },
            }
            if arm in clean_episodes and "baseline" in clean_episodes:
                matched_comparisons[arm]["libero_clean"] = _mcnemar_clean(
                    clean_episodes[arm], clean_episodes["baseline"]
                )

    summary = {
        "format": "starvla_dfr_action_refit_summary_v1",
        "plus_exact": args.plus_exact,
        "clean_trials_per_task": args.clean_trials,
        "target_categories": TARGET_CATEGORIES,
        "arms": rows,
        "matched_comparisons_to_baseline": matched_comparisons,
        "matched_test_caveat": (
            "Episodes/configurations are matched, but policy flow-noise streams were not "
            "explicitly coupled across independently served checkpoints. McNemar results "
            "control task difficulty, not all inference randomness."
        ),
        "decision_rule": (
            "Promising only if a balanced refit improves simulator-native target-category "
            "success against the matched baseline and retains clean success. Offline flow "
            "loss alone is not a positive result."
        ),
    }
    destination = args.root / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
