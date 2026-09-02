#!/usr/bin/env python3
"""Benchmark the real StarVLA -> Marigold DROID input path.

Run under ``accelerate launch``.  The script keeps StarVLA's sample adapter,
normalization, image conversion, and augmentation, but omits the model so the
worker-batched and main-process-batched loader designs can be compared beyond
the initial prefetch window.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/DROID/train_files/train_droid_delta_eef.yaml",
    )
    parser.add_argument(
        "--data-mix",
        choices=("droid_lerobot_delta_eef", "droid_lerobot_resized_success_delta_eef"),
        default="droid_lerobot_delta_eef",
    )
    parser.add_argument("--mode", choices=("round_robin", "worker_batched"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--consumer-delay", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    # StarVLA/Marigold already assigns disjoint global worker identities.
    accelerator.dataloader_config.dispatch_batches = False

    cfg = OmegaConf.load(args.config)
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_mix = args.data_mix
    data_cfg.per_device_batch_size = args.batch_size
    data_cfg.num_workers = args.num_workers
    data_cfg.prefetch_factor = args.prefetch_factor
    data_cfg.persistent_workers = True
    data_cfg.round_robin_dataloader = args.mode == "round_robin"
    data_cfg.marigold_resume_batches_per_rank = 0

    run_dir = args.output.parent / f"{args.output.stem}_dataset"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    cfg.output_dir = str(run_dir)

    build_start = time.perf_counter()
    loader = build_dataloader(cfg, dataset_py="marigold_data_datasets")
    get_inner = getattr(loader, "accelerator_prepare_component", None)
    finish = getattr(loader, "after_accelerator_prepare", None)
    if callable(get_inner) and callable(finish):
        loader = finish(accelerator.prepare(get_inner()))
    else:
        loader = accelerator.prepare(loader)
    iterator_start = time.perf_counter()
    iterator = iter(loader)
    iterator_ready = time.perf_counter()

    global_max_waits: list[float] = []
    global_mean_waits: list[float] = []
    local_waits: list[float] = []
    for step in range(args.steps):
        fetch_start = time.perf_counter()
        batch = next(iterator)
        wait = time.perf_counter() - fetch_start
        local_waits.append(wait)

        wait_tensor = torch.tensor([wait], device=accelerator.device, dtype=torch.float64)
        gathered = accelerator.gather(wait_tensor).detach().cpu().numpy().tolist()
        global_max_waits.append(max(gathered))
        global_mean_waits.append(statistics.fmean(gathered))

        if accelerator.is_main_process:
            print(
                f"step={step + 1:02d} fetch_rank_max={max(gathered):.3f}s "
                f"fetch_rank_mean={statistics.fmean(gathered):.3f}s",
                flush=True,
            )
        del batch
        if args.consumer_delay > 0:
            time.sleep(args.consumer_delay)

    local_summary = {
        "rank": accelerator.process_index,
        "fetch_mean_s": statistics.fmean(local_waits),
        "fetch_max_s": max(local_waits),
    }
    gathered_summaries = [local_summary]
    if torch.distributed.is_initialized():
        gathered_summaries = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_summaries, local_summary)

    if accelerator.is_main_process:
        elapsed_fetch = sum(global_max_waits)
        result = {
            "data_mix": args.data_mix,
            "mode": args.mode,
            "world_size": accelerator.num_processes,
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * accelerator.num_processes,
            "num_workers_per_rank": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "consumer_delay_s": args.consumer_delay,
            "steps": args.steps,
            "build_s": iterator_start - build_start,
            "iterator_create_s": iterator_ready - iterator_start,
            "rank_max_fetch_s": global_max_waits,
            "rank_mean_fetch_s": global_mean_waits,
            "fetch_rank_max_mean_s": statistics.fmean(global_max_waits),
            "fetch_rank_max_p50_s": percentile(global_max_waits, 50),
            "fetch_rank_max_p90_s": percentile(global_max_waits, 90),
            "fetch_rank_max_s": max(global_max_waits),
            "fetch_only_global_samples_s": (
                args.batch_size * accelerator.num_processes * args.steps / elapsed_fetch
            ),
            "per_rank": gathered_summaries,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "LEROBOT_PARQUET_CACHE_SIZE",
                    "LEROBOT_VIDEO_DECODER_CACHE_SIZE",
                    "LEROBOT_PREFETCH_MP4",
                    "LEROBOT_SKIP_FILE_CHECK",
                )
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
