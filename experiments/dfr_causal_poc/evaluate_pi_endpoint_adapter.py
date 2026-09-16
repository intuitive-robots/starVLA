#!/usr/bin/env python3
"""Evaluate the selected PI endpoint adapter on a disjoint manifest split."""

import argparse
import json
from pathlib import Path

import torch

from starVLA.model.framework.base_framework import baseframework
from fit_pi_flow_field_adapters import PIFlowAdapter, _balanced_rows, _load_action_bounds, _load_rows
from fit_pi_unrolled_endpoint_adapters import _evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-pairs", type=int, default=256)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    rows = _load_rows(args.manifest)
    selected = _balanced_rows(rows, "test", args.test_pairs, args.seed + 2)
    shard = selected[args.shard_id :: args.num_shards]
    model = baseframework.from_pretrained(str(args.checkpoint), is_inference=False).cuda().eval().requires_grad_(False)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    adapter = PIFlowAdapter(model, "projection_lora", int(payload["rank"])).cuda().eval()
    adapter.load_state_dict(payload["state_dict"], strict=True)
    result = _evaluate(model, adapter, shard, _load_action_bounds(args.stats), 4, args.seed + 2000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "starvla_pi_endpoint_test_shard_v1", "split": "test",
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "pairs": len(shard), "records": result["records"],
    }, indent=2) + "\n")
    adapter.close()


if __name__ == "__main__":
    main()
