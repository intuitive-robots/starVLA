#!/usr/bin/env python
"""Visualize the batch/horizon flattening failure seen in DROID open-loop eval.

The StarVLA PI head returns ``(batch, horizon, action_dim)``.  The canonical
Marigold evaluator concatenates complete chunks in batch order.  Flattening in
time-major order instead interleaves the same horizon position from different
episode anchors and produces an artificial high-frequency sawtooth.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = np.load(args.npz)
    ground_truth = payload["gt_physical"]
    prediction = payload["pred_physical"]
    chunk_lengths = payload["chunk_lengths"].astype(int)

    chunks = []
    cursor = 0
    for chunk_length in chunk_lengths:
        chunks.append(prediction[cursor : cursor + chunk_length])
        cursor += chunk_length
    if cursor != prediction.shape[0]:
        raise ValueError(
            f"chunk lengths sum to {cursor}, prediction has {prediction.shape[0]} rows"
        )

    # Deliberately WRONG: horizon-major traversal interleaves episode anchors.
    interleaved = np.stack(
        [
            chunk[horizon_idx]
            for horizon_idx in range(max(map(len, chunks)))
            for chunk in chunks
            if horizon_idx < len(chunk)
        ]
    )
    assert interleaved.shape == prediction.shape

    action_dim = prediction.shape[-1]
    ncols = min(4, action_dim)
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for dim, axis in enumerate(axes.ravel()):
        if dim >= action_dim:
            axis.axis("off")
            continue
        axis.plot(ground_truth[:, dim], label="ground truth", linewidth=1.8, color="C0")
        axis.plot(
            interleaved[:, dim],
            label="prediction interleaved (wrong)",
            linewidth=1.0,
            color="C1",
        )
        axis.set_title(f"action dim {dim}")
        axis.set_xlabel("plotted timestep")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Diagnostic: correct chunks flattened time-major (WRONG ordering)")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"prediction shape: {prediction.shape}")
    print(f"chunk lengths: {chunk_lengths.tolist()}")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
