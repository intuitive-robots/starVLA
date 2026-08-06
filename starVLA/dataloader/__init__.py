import json
import hashlib
import os
import random
from functools import partial
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)


def _mix_seed(*values: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for value in values:
        h.update(str(int(value)).encode("utf-8"))
        h.update(b":")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def _seed_worker_with_global_scope(worker_id: int, base_seed: int = 0, torch_num_threads: int = 1) -> None:
    # Marigold-style dataloader workers get parallelism from worker count, not
    # from an intra-op thread pool per worker. This avoids CPU oversubscription
    # on large multi-node jobs with many video-decoding workers.
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    rank, world_size = _rank_world()
    seed64 = _mix_seed(
        base_seed,
        torch.initial_seed(),
        rank,
        world_size,
        worker_id,
    )
    seed32 = seed64 % (2**32)
    random.seed(seed64)
    np.random.seed(seed32)
    torch.manual_seed(seed64)


def _build_loader_kwargs(data_cfg) -> dict:
    """Construct DataLoader kwargs from config with safe defaults."""
    num_workers = int(getattr(data_cfg, "num_workers", 4))
    pin_memory = bool(getattr(data_cfg, "pin_memory", True))
    seed_workers = bool(getattr(data_cfg, "seed_workers_with_global_scope", True))
    worker_torch_num_threads = int(getattr(data_cfg, "worker_torch_num_threads", 1))

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(data_cfg, "persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(getattr(data_cfg, "prefetch_factor", 2))
        if seed_workers:
            loader_kwargs["worker_init_fn"] = partial(
                _seed_worker_with_global_scope,
                base_seed=int(getattr(data_cfg, "seed", 0)),
                torch_num_threads=worker_torch_num_threads,
            )
        multiprocessing_context = getattr(data_cfg, "multiprocessing_context", None)
        if multiprocessing_context not in (None, "", "none", "None"):
            loader_kwargs["multiprocessing_context"] = multiprocessing_context

    return loader_kwargs


def _estimate_in_flight_samples(batch_size: int, loader_kwargs: dict) -> int | None:
    """Estimate how many samples workers may prefetch before step 1."""
    num_workers = int(loader_kwargs.get("num_workers", 0))
    prefetch_factor = int(loader_kwargs.get("prefetch_factor", 0))
    if num_workers <= 0 or prefetch_factor <= 0:
        return None
    return batch_size * num_workers * prefetch_factor

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        loader_kwargs = _build_loader_kwargs(vla_dataset_cfg)

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        estimated_in_flight_samples = _estimate_in_flight_samples(
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            loader_kwargs=loader_kwargs,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Building VLA DataLoader: batch_size=%s, num_workers=%s, prefetch_factor=%s, "
                "persistent_workers=%s, pin_memory=%s, worker_init=%s, worker_torch_num_threads=%s, "
                "estimated_in_flight_samples=%s",
                cfg.datasets.vla_data.per_device_batch_size,
                loader_kwargs.get("num_workers", 0),
                loader_kwargs.get("prefetch_factor", "n/a"),
                loader_kwargs.get("persistent_workers", False),
                loader_kwargs.get("pin_memory", False),
                "global_scope_seed" if "worker_init_fn" in loader_kwargs else "default",
                getattr(vla_dataset_cfg, "worker_torch_num_threads", 1),
                estimated_in_flight_samples if estimated_in_flight_samples is not None else "n/a",
            )
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            **loader_kwargs,
            # shuffle=True
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Built VLA DataLoader with dataset_len=%s", len(vla_dataset))
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "marigold_lerobot_datasets":
        from starVLA.dataloader.marigold_datasets import get_marigold_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        loader_kwargs = _build_loader_kwargs(vla_dataset_cfg)

        vla_dataset = get_marigold_vla_dataset(data_cfg=vla_dataset_cfg)
        estimated_in_flight_samples = _estimate_in_flight_samples(
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            loader_kwargs=loader_kwargs,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Building Marigold VLA DataLoader: batch_size=%s, num_workers=%s, prefetch_factor=%s, "
                "persistent_workers=%s, pin_memory=%s, worker_init=%s, worker_torch_num_threads=%s, "
                "estimated_in_flight_samples=%s",
                cfg.datasets.vla_data.per_device_batch_size,
                loader_kwargs.get("num_workers", 0),
                loader_kwargs.get("prefetch_factor", "n/a"),
                loader_kwargs.get("persistent_workers", False),
                loader_kwargs.get("pin_memory", False),
                "global_scope_seed" if "worker_init_fn" in loader_kwargs else "default",
                getattr(vla_dataset_cfg, "worker_torch_num_threads", 1),
                estimated_in_flight_samples if estimated_in_flight_samples is not None else "n/a",
            )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Built Marigold VLA DataLoader with dataset_len=%s", len(vla_dataset))
        if dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "marigold_data_datasets":
        from starVLA.dataloader.marigold_data_datasets import get_marigold_data_vla_dataset, collate_fn

        vla_dataset_cfg = cfg.datasets.vla_data
        loader_kwargs = _build_loader_kwargs(vla_dataset_cfg)

        vla_dataset = get_marigold_data_vla_dataset(data_cfg=vla_dataset_cfg)
        estimated_in_flight_samples = _estimate_in_flight_samples(
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            loader_kwargs=loader_kwargs,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Building full marigold_data VLA DataLoader: batch_size=%s, num_workers=%s, "
                "prefetch_factor=%s, persistent_workers=%s, pin_memory=%s, worker_init=%s, "
                "worker_torch_num_threads=%s, estimated_in_flight_samples=%s",
                cfg.datasets.vla_data.per_device_batch_size,
                loader_kwargs.get("num_workers", 0),
                loader_kwargs.get("prefetch_factor", "n/a"),
                loader_kwargs.get("persistent_workers", False),
                loader_kwargs.get("pin_memory", False),
                "global_scope_seed" if "worker_init_fn" in loader_kwargs else "default",
                getattr(vla_dataset_cfg, "worker_torch_num_threads", 1),
                estimated_in_flight_samples if estimated_in_flight_samples is not None else "n/a",
            )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Built full marigold_data VLA DataLoader")
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader
