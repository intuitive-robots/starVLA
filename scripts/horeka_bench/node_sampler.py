"""Every PERIOD s append: unix time, busy cores per CPU range, GPU utilisation per GPU (nvidia-smi).
Usage: python3 node_sampler.py out.csv 0-143 144-287 [period_s]"""

import subprocess
import sys
import time


def cpu_jiffies():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *v = line.split()
                v = list(map(int, v))
                idle = v[3] + v[4]
                out[int(name[3:])] = (sum(v) - idle, sum(v))
    return out


def main():
    out, ranges = sys.argv[1], []
    period = 2.0
    for a in sys.argv[2:]:
        if "-" in a:
            lo, hi = map(int, a.split("-"))
            ranges.append(range(lo, hi + 1))
        else:
            period = float(a)
    prev = cpu_jiffies()
    with open(out, "a") as f:
        while True:
            time.sleep(period)
            cur = cpu_jiffies()
            busy = []
            for r in ranges:
                b = sum(cur[c][0] - prev[c][0] for c in r if c in cur)
                t = sum(cur[c][1] - prev[c][1] for c in r if c in cur)
                busy.append(len([c for c in r if c in cur]) * b / t if t else 0.0)
            prev = cur
            util = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.split()
            f.write(",".join([f"{time.time():.1f}", *(f"{b:.1f}" for b in busy), *util]) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
