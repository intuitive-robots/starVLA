#!/usr/bin/env python3
"""Collect raw LIBERO(/plus) and RoboCasa evaluation JSONs without mixing protocols."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

TREES = [pathlib.Path("/e/project1/m3/blank4/code/starVLA"),
         pathlib.Path("/e/project1/m3/blank4/code/starVLA-upstream-merge")]
FIELDS = ["tree", "run", "arm", "seed", "benchmark", "result_dir", "suite_or_task", "tag",
          "n_episodes", "successes", "success_rate", "se", "checkpoint_step", "source_path",
          "is_full_protocol"]


def counts(node: object) -> tuple[int, int] | None:
    """Read count fields from plus ``overall`` or plain-LIBERO shape."""
    if not isinstance(node, dict):
        return None
    node = node.get("overall", node)
    if isinstance(node, dict) and {"success_count", "total_count"} <= node.keys():
        return int(node["success_count"]), int(node["total_count"])
    return None


def arm_of(run: str) -> str:
    name = run.lower()
    if "zbase" in name:
        return "encoder/shared-z-off-control"
    if "zsup" in name:
        return "shared-z/zsup"
    if "zonly" in name or "sharedz" in name:
        return "shared-z/zonly"
    if "mlm" in name:
        return "encoder/mlm"
    if (re.search(r"(?:^|_)w2?(?:_|$)", name) or "encoder" in name or "encdec" in name
            or "bidir" in name or "v5_pi_actiononly" in name or "robocasa365_pi_v5" in name):
        return "encoder"
    if "causal" in name:
        return "causal"
    if "cot" in name or "trace" in name:
        return "CoT-trace"
    if "gr00t" in name:
        return "GR00T/other-backbone"
    return "other"


def seed_of(run: str, payload: dict | None = None) -> str:
    # The payload seed is an evaluation seed, not a training replicate.
    found = re.search(r"(?:^|[_-])s(?:eed)?(\d+)(?:$|[_-])", run, re.I)
    return found.group(1) if found else ""


def training_seed(tree: pathlib.Path, run: str) -> str:
    config = tree / "playground/Checkpoints" / run / "config.full.yaml"
    if config.is_file():
        found = re.search(r"^seed:\s*['\"]?(\d+)", config.read_text(), re.M)
        if found:
            return found.group(1)
    return seed_of(run)


def rate_se(successes: int, episodes: int) -> tuple[float, float]:
    rate = successes / episodes if episodes else math.nan
    return rate, math.sqrt(rate * (1 - rate) / episodes) if episodes else math.nan


def collect_libero(tree: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    # rglob is intentional: suite-local JSONs are independent evidence artifacts.
    for path in sorted((tree / "playground/Checkpoints").glob("*/results/**/overall_results.json")):
        rel = path.relative_to(tree / "playground/Checkpoints")
        run, _, benchmark, *tail = rel.parts
        result_dir = "/".join((benchmark, *tail[:-1]))
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"! unreadable {path}: {exc}", file=sys.stderr)
            continue
        file_rows: list[dict] = []
        # A root file maps suite -> result. A suite-local file itself has an
        # ``overall`` result plus perturbation categories; those categories
        # must not become extra pseudo-suites in the master table.
        direct = counts(blob)
        units = [(tail[-2], blob)] if direct is not None and len(tail) > 1 else blob.items()
        for suite, node in units:
            found = counts(node)
            if found is None:
                continue
            successes, episodes = found
            rate, se = rate_se(successes, episodes)
            file_rows.append({"tree": tree.name, "run": run, "arm": arm_of(run),
                              "seed": training_seed(tree, run), "benchmark": benchmark,
                              "result_dir": result_dir, "suite_or_task": suite, "tag": "",
                              "n_episodes": episodes, "successes": successes,
                              "success_rate": f"{rate:.8f}", "se": f"{se:.8f}",
                              "checkpoint_step": "", "source_path": str(path),
                              "is_full_protocol": "False"})
        failures = list(path.parent.glob("failed_shards_*.txt"))
        full = (sum(int(r["n_episodes"]) for r in file_rows) >= 4000
                and {r["suite_or_task"] for r in file_rows} ==
                {"libero_10", "libero_goal", "libero_object", "libero_spatial"}
                and not any(p.read_text().strip() for p in failures))
        for row in file_rows:
            row["is_full_protocol"] = str(full)
        rows.extend(file_rows)
    return rows


def collect_robocasa(tree: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    root = tree / "playground/Checkpoints"
    for path in sorted(root.glob("*/checkpoints/*.eval/robocasa_*.json")):
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"! unreadable {path}: {exc}", file=sys.stderr)
            continue
        outcomes = blob.get("successes")
        if not isinstance(outcomes, list):
            continue
        if any(v not in (True, False, 0, 1) for v in outcomes):
            raise ValueError(f"Non-binary successes in {path}")
        successes, episodes = sum(int(v) for v in outcomes), len(outcomes)
        computed, se = rate_se(successes, episodes)
        reported = blob.get("success_rate")
        if reported is not None and not math.isclose(float(reported), computed, abs_tol=1e-12):
            print(f"! rate mismatch {path}: json={reported} computed={computed}", file=sys.stderr)
        rel = path.relative_to(root)
        run, _, checkpoint = rel.parts[:3]
        step = re.search(r"steps_(\d+)_pytorch_model\.eval", checkpoint)
        stem = path.stem.removeprefix("robocasa_")
        task = str(blob.get("env", stem)).removeprefix("robocasa/")
        # Include all filename protocol fields: env count, horizon, action steps,
        # resize mode and evaluation seed must never be pooled accidentally.
        protocol = stem.removeprefix(task).lstrip("_") or "legacy"
        tag = protocol.split("_native_", 1)[1] if "_native_" in protocol else protocol
        rows.append({"tree": tree.name, "run": run, "arm": arm_of(run),
                     "seed": training_seed(tree, run), "benchmark": "robocasa365",
                     "result_dir": f"{checkpoint}/{protocol}", "suite_or_task": task, "tag": tag,
                     "n_episodes": episodes, "successes": successes,
                     "success_rate": f"{computed:.8f}", "se": f"{se:.8f}",
                     "checkpoint_step": step.group(1) if step else "", "source_path": str(path),
                     "is_full_protocol": "False"})
    # Full here means the declared internal 17-task atomic sweep, NOT the
    # official 50-task RoboCasa365 leaderboard. Ablations are always partial.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["result_dir"])].append(row)
    for group in groups.values():
        tag = group[0]["tag"]
        manifest = pathlib.Path("/e/scratch/m3/blank4/rc365_smoke") / f"manifest_full_s{group[0]['seed']}.txt"
        if not re.fullmatch(r"s\d+_50k", tag) or not manifest.is_file():
            continue
        expected = {line.split()[1].removeprefix("robocasa/")
                    for line in manifest.read_text().splitlines() if line.strip()
                    and line.split()[0] == group[0]["run"]}
        full = (bool(expected) and {r["suite_or_task"] for r in group} == expected
                and all(int(r["n_episodes"]) == 48 for r in group))
        for row in group:
            row["is_full_protocol"] = str(full)
    return rows


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="substring filter")
    parser.add_argument("--run-filter", help="substring filter")
    parser.add_argument("--min-episodes", type=int, default=0, help="summary threshold")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results_collected"))
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    rows = [row for tree in TREES if tree.exists()
            for row in (collect_libero(tree) + collect_robocasa(tree))]
    if args.benchmark:
        rows = [r for r in rows if args.benchmark in r["benchmark"]]
    if args.run_filter:
        rows = [r for r in rows if args.run_filter in r["run"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: tuple(r[f] for f in ("tree", "benchmark", "run", "result_dir", "tag", "suite_or_task")))
    write_csv(args.out_dir / "results_master.csv", rows, FIELDS)
    compact = [{"tree": r["tree"], "benchmark": r["benchmark"], "run": r["run"],
                "unit": r["suite_or_task"], "successes": r["successes"], "episodes": r["n_episodes"],
                "success_rate": r["success_rate"], "source": r["source_path"]} for r in rows]
    write_csv(args.out_dir / "results_all.csv", compact,
              ["tree", "benchmark", "run", "unit", "successes", "episodes", "success_rate", "source"])
    pooled: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0, 0])
    keys = ("tree", "benchmark", "run", "arm", "seed", "result_dir", "tag", "checkpoint_step")
    for row in rows:
        aggregate = pooled[tuple(row[f] for f in keys)]
        aggregate[0] += int(row["successes"]); aggregate[1] += int(row["n_episodes"]); aggregate[2] += 1
    summary = []
    for key, (successes, episodes, units) in pooled.items():
        if episodes < args.min_episodes:
            continue
        rate, se = rate_se(successes, episodes)
        summary.append(dict(zip(keys, key), units=units, episodes=episodes, successes=successes,
                            success_rate=f"{rate:.8f}", se=f"{se:.8f}"))
    summary.sort(key=lambda r: (r["benchmark"], -float(r["success_rate"])))
    summary_fields = [*keys, "units", "episodes", "successes", "success_rate", "se"]
    write_csv(args.out_dir / "results_summary.csv", summary, summary_fields)
    print(f"{len(rows)} raw unit rows -> {args.out_dir/'results_master.csv'}")
    print(f"{len(summary)} protocol-specific pooled rows -> {args.out_dir/'results_summary.csv'}")
    for row in summary[:args.top]:
        print(f"{row['benchmark']:18s} {row['run'][:45]:45s} {float(row['success_rate']):.3f} ±{float(row['se']):.3f} n={row['episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
