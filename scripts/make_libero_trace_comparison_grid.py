#!/usr/bin/env python3
import argparse
import pathlib
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ours, det = pathlib.Path(a.ours), pathlib.Path(a.det)
    pairs = [(Image.open(ours / f"task_{i:02d}_ep_00.png"), Image.open(det / f"task_{i:02d}_ep_00.png")) for i in range(10)]
    w, h = pairs[0][0].size
    grid = Image.new("RGB", (2 * w, 10 * h + 35), "black")
    draw = ImageDraw.Draw(grid)
    draw.text((w // 2 - 20, 8), "OURS", fill="white")
    draw.text((w + w // 2 - 35, 8), "DETECTOR", fill="white")
    for i, (left, right) in enumerate(pairs):
        grid.paste(left, (0, 35 + i * h))
        grid.paste(right, (w, 35 + i * h))
    pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
