"""Steady-state evaluation throughput for B2: episodes completed between the LO and HI quantiles of the
per-episode completion times (default 10% and 90%), i.e. without the simulator launch ramp at the start and the
slowest episodes at the end. Completion times come from each episode's "INFO - Success:" log line.
Usage: python3 steady_state_b2.py [run dir, default sim_scale_chain] [lo hi]"""

import glob
import re
import sys
from datetime import datetime

D = sys.argv[1] if len(sys.argv) > 1 else "/e/scratch/m3/blank4/horeka_bench/sim_scale_chain"
LO, HI = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.10, 0.90)
RUNS = [("g2a", 2, 0.5), ("g2b", 2, 0.5), ("g4", 4, 1), ("g8", 8, 2), ("g16", 16, 4)]
PAT = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - INFO - Success:")

rows = []
for tag, ng, nodes in RUNS:
    t = sorted(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
               for log in glob.glob(f"{D}/{tag}/node*.log") for line in open(log, errors="ignore")
               if (m := PAT.search(line)))
    if len(t) < 100:
        continue
    i, j = int(LO * len(t)), int(HI * len(t)) - 1
    rate = (j - i) / (t[j] - t[i]) * 3600          # episodes per hour in steady state
    rows.append((tag, ng, nodes, len(t), rate, (t[-1] - t[0]) / 60))

base = sum(r[4] for r in rows if r[1] == 2) / max(1, sum(1 for r in rows if r[1] == 2))
print(f"steady state = episodes completed between the {LO:.0%} and {HI:.0%} completion quantiles")
print("| run | #GPUs | #nodes | #parallel envs | #episodes | steady-state episodes/hour | episodes/hour per GPU | speedup | ideal speedup | efficiency |")
print("|---|---|---|---|---|---|---|---|---|---|")
for tag, ng, nodes, n, rate, span in rows:
    sp = rate / base
    print(f"| {tag} | {ng} | {nodes} | {16 * ng} | {n} | {rate:,.0f} | {rate / ng:,.0f} | {sp:.2f} | {ng / 2:.0f} | {sp / (ng / 2):.0%} |")
