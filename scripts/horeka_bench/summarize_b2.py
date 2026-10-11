"""Table B2 (LIBERO eval scaling at a fixed workload) from /e/scratch/m3/blank4/horeka_bench/sim_scale.
Workload time = first node's policy servers ready -> last node's final "All workers finished".
Start-up (job start -> servers ready on every node) is reported separately. Usage: python3 summarize_b2.py [dir]"""

import glob
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/e/scratch/m3/blank4/horeka_bench/sim_scale"
RUNS = [("g2a", 2, 0.5), ("g2b", 2, 0.5), ("g4", 4, 1), ("g8", 8, 2), ("g16", 16, 4)]


def stamps(log, needle):
    return [int(l.split()[0]) for l in open(log, errors="ignore") if needle in l and l.split()[0].isdigit()]


rows = []
for tag, ng, nodes in RUNS:
    d = f"{D}/{tag}"
    logs = sorted(glob.glob(f"{d}/node*.log"))
    if not logs or not os.path.exists(f"{d}/job_start"):
        continue
    job_start = int(open(f"{d}/job_start").read())
    ready = [min(stamps(l, "policy servers ready"), default=None) for l in logs]
    # "All workers finished" is printed once per suite; a node is done after its last suite.
    # Chained-suite runs (chain_suites=true) print it once, after all suites.
    end = []
    for l in logs:
        st = stamps(l, "All workers finished")
        need = 1 if any("chained suite" in x for x in open(l, errors="ignore")) else 4
        end.append(max(st) if len(st) >= need else None)
    eps = [e for p in glob.glob(f"{d}/logs/*/*_to_*.json") for e in json.load(open(p)).get("episodes", [])]
    if None in ready or None in end:
        rows.append((tag, ng, nodes, len(eps), None))
        continue
    wall = max(end) - min(ready)
    rows.append((tag, ng, nodes, len(eps), dict(
        wall=wall, startup=max(ready) - job_start, steps=sum(e.get("env_steps", 0) for e in eps),
        success=sum(e["success"] for e in eps) / max(len(eps), 1))))

base = [r[4]["wall"] for r in rows if r[1] == 2 and r[4]]
base = sum(base) / len(base) if base else None
print("| run | #GPUs | #nodes | #parallel envs | #episodes | success | wall time [min] | start-up [min] | episodes/hour | env steps/s | speedup | ideal speedup | efficiency |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag, ng, nodes, n, m in rows:
    if not m:
        print(f"| {tag} | {ng} | {nodes} | {16 * ng} | {n} | | incomplete | | | | | | |")
        continue
    sp = base / m["wall"] if base else float("nan")
    print(f"| {tag} | {ng} | {nodes} | {16 * ng} | {n} | {m['success']:.1%} | {m['wall'] / 60:.1f} | {m['startup'] / 60:.1f} | "
          f"{n / m['wall'] * 3600:.0f} | {m['steps'] / m['wall']:.0f} | {sp:.2f} | {ng / 2:.0f} | {sp / (ng / 2):.0%} |")
