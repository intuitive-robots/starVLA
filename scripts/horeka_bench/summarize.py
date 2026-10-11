"""Build the tables of benchmark_results_gh200.md from /e/scratch/m3/blank4/horeka_bench.
Usage: python3 summarize.py [bench_dir]  -> prints markdown for tables A and B."""

import csv
import glob
import json
import os
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/e/scratch/m3/blank4/horeka_bench"
NAMES = {"2b": "Qwen3.5-2B + GR00T DiT-B", "4b": "Qwen3.5-4B + GR00T DiT-B", "9b": "Qwen3.5-9B + GR00T DiT-B"}


def table_a():
    rows = []
    for p in glob.glob(f"{D}/results/*_g*_n*_b*_z*.json"):
        m = re.match(r"(\w+?)_g(\d+)_n(\d+)_b(\d+)_z\d", os.path.basename(p))
        rows.append((m.group(1), json.load(open(p))))
    # OOM / failed configs appear only as logs.
    done = {os.path.basename(p)[:-5] for p in glob.glob(f"{D}/results/*.json")}
    ooms = []
    for p in glob.glob(f"{D}/logs/*_r0.log"):
        tag = os.path.basename(p)[:-7]
        m = re.match(r"(\w+?)_g(\d+)_n(\d+)_b(\d+)_z\d$", tag)
        if m and tag not in done:
            oom = bool(re.search(r"CUDA out of memory|OutOfMemoryError", open(p, errors="ignore").read()))
            ooms.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), "OOM" if oom else "failed"))
    out = ["| model | #GPUs | #nodes | per-GPU batch | global batch | step time [s] | samples/s | speedup vs smallest config | ideal speedup | parallel efficiency | peak mem/GPU [GB] | data wait [s] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in ("2b", "4b", "9b"):
        mrows = [r for mm, r in rows if mm == model]
        # One weak-scaling series per per-GPU batch; baseline = its smallest GPU count.
        for bs in sorted({r["per_gpu_batch"] for r in mrows}):
            rs = sorted((r for r in mrows if r["per_gpu_batch"] == bs), key=lambda r: r["world_size"])
            base = rs[0]
            for r in rs:
                sp = r["samples_per_s"] / base["samples_per_s"]
                ideal = r["world_size"] / base["world_size"]
                out.append(f"| {NAMES[model]} | {r['world_size']} | {r['num_nodes']} | {r['per_gpu_batch']} | {r['global_batch']} | "
                           f"{r['step_time_median_s']:.3f} | {r['samples_per_s']:.1f} | {sp:.2f} | {ideal:.0f} | {sp / ideal:.0%} | "
                           f"{r['peak_mem_per_gpu_gb']:.1f} | {r['data_time_median_s']:.3f} |")
        for mm, g, n, b, why in sorted(ooms):
            if mm == model:
                out.append(f"| {NAMES[model]} | {g} | {n} | {b} | {b * g} | {why} | | | | | | |")
    return "\n".join(out)


def table_b():
    out = ["| #GPUs | #CPU cores (allocated / busy avg) | #parallel envs | #episodes | success | rollout wall [s] | episodes/hour | env steps/s | GPU util [%] |",
           "|---|---|---|---|---|---|---|---|---|"]
    samples = []
    for p in glob.glob(f"{D}/sim/sampler_*.csv"):
        samples += [list(map(float, r)) for r in csv.reader(open(p)) if r]
    for envs in (8, 16, 32, 64):
        d = f"{D}/sim/envs{envs}"
        if not os.path.exists(f"{d}/run.log"):
            continue
        log = open(f"{d}/run.log").read().splitlines()
        t0 = next((float(l.split()[0]) for l in log if "policy servers ready" in l), None)
        t1 = next((float(l.split()[0]) for l in log if "All workers finished" in l), None)
        eps = []
        for p in glob.glob(f"{d}/logs/*/*_to_*.json"):
            eps += json.load(open(p)).get("episodes", [])
        if t0 is None or t1 is None or not eps:
            out.append(f"| 2 | 144 | {envs} | {len(eps)} | | incomplete | | | |")
            continue
        wall = t1 - t0
        steps = sum(e.get("env_steps", 0) for e in eps)
        half = 0 if envs in (8, 32) else 1  # GPUs 0,1 / cores 0-143 run 8 and 32 envs
        win = [s for s in samples if t0 <= s[0] <= t1]
        busy = sum(s[1 + half] for s in win) / len(win) if win else float("nan")
        util = sum((s[3 + 2 * half] + s[4 + 2 * half]) / 2 for s in win) / len(win) if win else float("nan")
        succ = sum(e["success"] for e in eps) / len(eps)
        out.append(f"| 2 | 144 / {busy:.0f} | {envs} | {len(eps)} | {succ:.0%} | {wall:.0f} | {len(eps) / wall * 3600:.0f} | "
                   f"{steps / wall:.0f} | {util:.0f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## A\n\n" + table_a() + "\n\n## B\n\n" + table_b())
    v = next((json.load(open(p)) for p in glob.glob(f"{D}/results/*.json")), None)
    if v:
        print(f"\nversions: torch {v['torch']}, CUDA {v['cuda']}, NCCL {v['nccl']}, transformers {v['transformers']}, "
              f"DeepSpeed {v['deepspeed']}, GPU {v['gpu']}")
