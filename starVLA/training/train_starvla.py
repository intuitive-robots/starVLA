# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def _read_proc_rss_bytes(pid: int) -> int | None:
    """Read resident set size from /proc/<pid>/status in bytes."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    return None


def _read_proc_ppid_and_name(pid: int) -> tuple[int | None, str | None]:
    """Read parent PID and process name from /proc/<pid>/status."""
    ppid = None
    name = None
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Name:"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                elif line.startswith("PPid:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ppid = int(parts[1])
                if ppid is not None and name is not None:
                    break
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None, None
    return ppid, name


class StepTimeAccumulator:
    """Aggregate per-step data/model wall time across a logging window.

    ``timing/data`` used to be the cost of the single step that happened to
    land on the logging boundary. That step almost always finds the prefetch
    queue full, so it reads ~0.001s even when the dataloader is stalling on
    most other steps — on LIBERO-plus it reported 0.001s/step against a real
    3.1s/step wall clock. Reporting mean/max/p90 over the whole window instead
    makes a starved input pipeline visible in the logs.
    """

    def __init__(self):
        self._data: list[float] = []
        self._model: list[float] = []

    def add(self, data_time: float, model_time: float) -> None:
        self._data.append(data_time)
        self._model.append(model_time)

    def summary(self) -> dict:
        """Return aggregated timings for the window and start a new one."""
        if not self._data:
            return {}

        data = np.asarray(self._data, dtype=np.float64)
        model = np.asarray(self._model, dtype=np.float64)
        self._data, self._model = [], []

        total = float(data.sum() + model.sum())
        return {
            # Mean, not instantaneous — keeps the existing key meaningful.
            "timing/data": float(data.mean()),
            "timing/data_max": float(data.max()),
            "timing/data_p90": float(np.percentile(data, 90)),
            "timing/model": float(model.mean()),
            "timing/model_max": float(model.max()),
            # Share of wall time spent waiting on the input pipeline. This is
            # the number to watch: >0.2 means the GPUs are input-starved.
            "timing/data_fraction": float(data.sum() / total) if total > 0 else 0.0,
            "timing/step": float((data + model).mean()),
            "timing/window_steps": int(data.size),
        }


class MemoryMonitor:
    """Lightweight host-memory monitor for rank processes and DataLoader workers."""

    def __init__(self, cfg, accelerator: Accelerator):
        trainer_cfg = getattr(cfg, "trainer", None)
        memory_cfg = getattr(trainer_cfg, "memory_debug", None) if trainer_cfg is not None else None

        env_enabled = os.environ.get("STARVLA_MEMORY_DEBUG", "").strip().lower()
        env_enabled = env_enabled in {"1", "true", "yes", "on"}
        cfg_enabled = False
        if memory_cfg is not None:
            cfg_enabled = bool(memory_cfg.get("enabled", False) if hasattr(memory_cfg, "get") else getattr(memory_cfg, "enabled", False))

        self.enabled = bool(env_enabled or cfg_enabled)
        self.accelerator = accelerator
        self.interval = 250
        self.log_worker_details = False

        if memory_cfg is not None:
            if hasattr(memory_cfg, "get"):
                self.interval = int(memory_cfg.get("interval", self.interval))
                self.log_worker_details = bool(memory_cfg.get("log_worker_details", self.log_worker_details))
            else:
                self.interval = int(getattr(memory_cfg, "interval", self.interval))
                self.log_worker_details = bool(
                    getattr(memory_cfg, "log_worker_details", self.log_worker_details)
                )

        env_interval = os.environ.get("STARVLA_MEMORY_DEBUG_INTERVAL", "").strip()
        if env_interval:
            self.interval = int(env_interval)

    def _get_worker_pids(self, data_iter) -> list[int]:
        workers = getattr(data_iter, "_workers", None)
        pids = []
        if workers is not None:
            for worker in workers:
                pid = getattr(worker, "pid", None)
                if pid is not None:
                    pids.append(int(pid))
        if pids:
            return pids

        # Accelerate / DataLoader wrappers can hide the underlying iterator
        # worker handles. Fall back to scanning direct child processes and pick
        # PyTorch dataloader workers by process name.
        main_pid = os.getpid()
        proc_root = Path("/proc")
        fallback_pids = []
        try:
            for entry in proc_root.iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                ppid, name = _read_proc_ppid_and_name(pid)
                if ppid != main_pid or name is None:
                    continue
                if name in {"pt_data_worker", "torch_shm_manager"}:
                    fallback_pids.append(pid)
        except OSError:
            return []
        return sorted(fallback_pids)

    def collect(self, completed_steps: int, data_iter) -> dict:
        if not self.enabled or completed_steps <= 0 or completed_steps % self.interval != 0:
            return {}

        local_rank = self.accelerator.process_index
        main_pid = os.getpid()
        worker_pids = self._get_worker_pids(data_iter)
        worker_rss = []
        for pid in worker_pids:
            rss = _read_proc_rss_bytes(pid)
            if rss is not None:
                worker_rss.append({"pid": pid, "rss_bytes": rss})

        local_snapshot = {
            "rank": local_rank,
            "main_pid": main_pid,
            "main_rss_bytes": _read_proc_rss_bytes(main_pid) or 0,
            "worker_count": len(worker_pids),
            "worker_rss": worker_rss,
        }

        gathered = [local_snapshot]
        if dist.is_initialized():
            gathered = [None for _ in range(self.accelerator.num_processes)]
            dist.all_gather_object(gathered, local_snapshot)

        if not self.accelerator.is_main_process:
            return {}

        total_main_rss = sum(int(item["main_rss_bytes"]) for item in gathered if item is not None)
        total_worker_rss = sum(
            int(worker["rss_bytes"])
            for item in gathered if item is not None
            for worker in item["worker_rss"]
        )
        worker_rss_values = [
            int(worker["rss_bytes"])
            for item in gathered if item is not None
            for worker in item["worker_rss"]
        ]

        metrics = {
            "memory/main_rss_gib": total_main_rss / (1024 ** 3),
            "memory/dataloader_workers_rss_gib": total_worker_rss / (1024 ** 3),
            "memory/combined_rss_gib": (total_main_rss + total_worker_rss) / (1024 ** 3),
            "memory/worker_count": float(sum(int(item["worker_count"]) for item in gathered if item is not None)),
        }
        if worker_rss_values:
            metrics["memory/max_worker_rss_gib"] = max(worker_rss_values) / (1024 ** 3)
            metrics["memory/avg_worker_rss_gib"] = (
                sum(worker_rss_values) / len(worker_rss_values)
            ) / (1024 ** 3)

        if self.log_worker_details:
            rank_summaries = []
            for item in gathered:
                if item is None:
                    continue
                rank_worker_total = sum(int(worker["rss_bytes"]) for worker in item["worker_rss"])
                rank_summaries.append(
                    f"rank={item['rank']} main={item['main_rss_bytes'] / (1024 ** 3):.2f}GiB "
                    f"workers={item['worker_count']} worker_rss={rank_worker_total / (1024 ** 3):.2f}GiB"
                )
            logger.info("Memory snapshot @ step %s :: %s", completed_steps, " | ".join(rank_summaries))
        else:
            logger.info(
                "Memory snapshot @ step %s :: main=%.2fGiB workers=%.2fGiB combined=%.2fGiB workers=%s",
                completed_steps,
                metrics["memory/main_rss_gib"],
                metrics["memory/dataloader_workers_rss_gib"],
                metrics["memory/combined_rss_gib"],
                int(metrics["memory/worker_count"]),
            )

        return metrics


def parse_bool_flag(value):
    """Accept common string forms for CLI booleans."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def create_accelerator(use_deepspeed: bool, gradient_accumulation_steps: int = 1) -> Accelerator:
    """Create the accelerator after CLI parsing so launch flags can control backend choice."""
    accelerate_use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    if not use_deepspeed and accelerate_use_deepspeed:
        raise ValueError(
            "--use_deepspeed false was requested, but Accelerate was launched with a DeepSpeed config. "
            "Please launch with a non-DeepSpeed Accelerate config instead of "
            "`starVLA/config/deepseeds/deepspeed_zero2.yaml`."
        )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if use_deepspeed:
        accelerator = Accelerator(
            deepspeed_plugin=DeepSpeedPlugin(),
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    accelerator.print(accelerator.state)
    return accelerator


def log_eval_backend_banner(cfg, accelerator: Accelerator) -> None:
    """Print a visible startup banner for attention backend and eval behavior."""
    if not accelerator.is_main_process:
        return

    attn_impl = str(getattr(cfg.framework.qwenvl, "attn_implementation", "unknown")).strip()
    open_loop_disabled_by_config = not bool(getattr(cfg.trainer, "open_loop_eval", True))
    open_loop_disabled_by_backend = "flash_attention" in attn_impl.lower()
    open_loop_disabled = open_loop_disabled_by_config or open_loop_disabled_by_backend

    banner_lines = [
        "",
        "=" * 88,
        "STARTUP CONFIG",
        f"Attention backend: {attn_impl}",
        (
            "Open-loop eval: DISABLED by trainer.open_loop_eval or attention backend"
            if open_loop_disabled
            else "Open-loop eval: ENABLED"
        ),
        "=" * 88,
        "",
    ]
    logger.warning("\n".join(banner_lines))


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def _latest_model_checkpoint(checkpoint_dir: str | Path) -> tuple[str | None, int]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0

    checkpoints = []
    for path in checkpoint_dir.iterdir():
        match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", path.name)
        if match and path.is_file():
            checkpoints.append((path, int(match.group(1))))
    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda item: item[1])
    latest_path, completed_steps = checkpoints[-1]
    return str(latest_path), completed_steps


def _dataloader_state_path(checkpoint_path: str | Path) -> Path:
    name = Path(checkpoint_path).name
    match = re.match(r"(steps_\d+)_(?:pytorch_model\.pt|model\.safetensors)$", name)
    stem = match.group(1) if match else Path(checkpoint_path).stem
    return Path(checkpoint_path).with_name(f"{stem}_dataloader_state.json")


def _prune_old_checkpoints(checkpoint_dir: str | Path, save_total_limit: int | None, logger_fn=logger.info) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints_by_step: dict[int, list[Path]] = {}
    for path in checkpoint_dir.iterdir():
        match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", path.name)
        if match and path.is_file():
            checkpoints_by_step.setdefault(int(match.group(1)), []).append(path)

    steps = sorted(checkpoints_by_step)
    old_steps = steps[: max(0, len(steps) - save_total_limit)]
    for step in old_steps:
        for model_path in checkpoints_by_step[step]:
            state_path = _dataloader_state_path(model_path)
            for path in (model_path, state_path):
                if path.exists():
                    path.unlink()
                    logger_fn("Deleted old checkpoint file: %s", path)


def _estimated_resume_batches(cfg, completed_steps: int) -> int:
    grad_accum = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    eval_interval = int(getattr(cfg.trainer, "eval_interval", 0) or 0)
    eval_batches = completed_steps // eval_interval if eval_interval > 0 else 0
    return completed_steps * grad_accum + eval_batches


def configure_dataloader_resume(cfg, output_dir: Path) -> None:
    """Populate VLA data config with deterministic resume controls before workers fork."""
    data_cfg = cfg.datasets.vla_data
    if "seed" not in data_cfg or getattr(data_cfg, "seed", None) is None:
        data_cfg.seed = int(getattr(cfg, "seed", 0))

    data_cfg.marigold_resume_batches_per_rank = 0
    data_cfg.marigold_resume_steps = 0

    if getattr(data_cfg, "dataset_py", "") != "marigold_data_datasets":
        return
    if not bool(getattr(cfg.trainer, "is_resume", False)):
        return

    checkpoint_path, completed_steps = _latest_model_checkpoint(output_dir / "checkpoints")
    if checkpoint_path is None:
        logger.warning("[marigold_data] Resume requested, but no model checkpoint found for dataloader resume.")
        return

    state_path = _dataloader_state_path(checkpoint_path)
    resume_batches = None
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            resume_batches = int(state.get("dataloader_batches_consumed", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("[marigold_data] Could not read dataloader state %s: %s", state_path, exc)

    if resume_batches is None:
        resume_batches = _estimated_resume_batches(cfg, completed_steps)
        logger.warning(
            "[marigold_data] No dataloader sidecar found at %s; estimating consumed batches as %s "
            "from completed_steps=%s.",
            state_path,
            resume_batches,
            completed_steps,
        )

    data_cfg.marigold_resume_batches_per_rank = max(0, int(resume_batches))
    data_cfg.marigold_resume_steps = int(completed_steps)
    logger.warning(
        "[marigold_data] Dataloader resume configured from %s: completed_steps=%s, "
        "resume_batches_per_rank=%s",
        checkpoint_path,
        completed_steps,
        data_cfg.marigold_resume_batches_per_rank,
    )


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_eval_dataset = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.memory_monitor = MemoryMonitor(cfg, accelerator)
        self.step_timers = StepTimeAccumulator()
        self._last_timing_flush_step = -1
        self._stopped_for_preemption = False

        self.completed_steps = 0
        self.dataloader_batches_consumed = int(
            getattr(self.config.datasets.vla_data, "marigold_resume_batches_per_rank", 0) or 0
        )
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        if getattr(self.config.datasets.vla_data, "dataset_py", "") == "lerobot_datasets":
            try:
                self.vla_eval_dataset = get_vla_dataset(data_cfg=self.config.datasets.vla_data, mode="eval")
            except ValueError as exc:
                if self.accelerator.is_main_process:
                    logger.warning(f"Skipping dedicated open-loop eval dataset construction: {exc}")
                self.vla_eval_dataset = None

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
            for metric_name in (
                "train/cot_loss",
                "train_cot_loss",
                "cot_loss",
                "train/cot_coverage",
                "cot_coverage",
                "train/cot_keep_rate",
                "cot_keep_rate",
                "eval/cot_loss",
                "eval_cot_loss",
                "eval/cot_coverage",
                "eval_cot_coverage",
                "eval/cot_keep_rate",
                "eval_cot_keep_rate",
            ):
                wandb.define_metric(metric_name)

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                model_path = Path(checkpoint_path + "_model.safetensors")
                partial_model_path = model_path.with_name(
                    f"{model_path.name}.partial.{os.getpid()}"
                )
                save_file(state_dict, partial_model_path)
            elif save_format == "pt":
                model_path = Path(checkpoint_path + "_pytorch_model.pt")
                partial_model_path = model_path.with_name(
                    f"{model_path.name}.partial.{os.getpid()}"
                )
                torch.save(state_dict, partial_model_path)
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            # A killed writer must never leave a filename that the automatic
            # resume scanner mistakes for a complete checkpoint.
            os.replace(partial_model_path, model_path)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            dataloader_state = {
                "steps": self.completed_steps,
                "dataloader_batches_consumed": self.dataloader_batches_consumed,
                "per_device_batch_size": int(self.config.datasets.vla_data.per_device_batch_size),
                "gradient_accumulation_steps": int(getattr(self.config.trainer, "gradient_accumulation_steps", 1)),
                "eval_interval": int(getattr(self.config.trainer, "eval_interval", 0) or 0),
                "num_workers": int(getattr(self.config.datasets.vla_data, "num_workers", 0)),
                "world_size": int(self.accelerator.num_processes),
            }
            state_path = _dataloader_state_path(model_path)
            partial_state_path = state_path.with_name(
                f"{state_path.name}.partial.{os.getpid()}"
            )
            with partial_state_path.open("w", encoding="utf-8") as f:
                json.dump(dataloader_state, f, indent=2)
                f.write("\n")
            os.replace(partial_state_path, state_path)
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            _prune_old_checkpoints(
                self.checkpoint_dir,
                getattr(self.config.trainer, "save_total_limit", None),
                logger_fn=logger.info,
            )

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            # Infinite/iterable loaders (including Marigold's
            # MainProcessBatchLoader) deliberately have no meaningful length.
            # Logging must not turn that into a training failure.
            try:
                dataloader_length = len(self.vla_train_dataloader)
            except (TypeError, NotImplementedError):
                dataloader_length = None
            if dataloader_length:
                metrics["epoch"] = round(self.completed_steps / dataloader_length, 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        t_start = time.perf_counter()
        self.vla_iter = iter(self.vla_train_dataloader)
        t_end = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info(f"DataLoader iterator created in {t_end - t_start:.3f}s")

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        self.dataloader_batches_consumed += 1
        return batch_vla

    def _log_first_batch_startup(self):
        """Time the first batch fetch so startup stalls are visible in logs."""
        if getattr(self, "_first_batch_startup_logged", False):
            return

        if self.accelerator.is_main_process:
            logger.info("Fetching first training batch from DataLoader...")

        t_start = time.perf_counter()
        batch_vla = self._get_next_batch()
        t_end = time.perf_counter()

        if self.accelerator.is_main_process:
            batch_size = len(batch_vla) if hasattr(batch_vla, "__len__") else "unknown"
            logger.info(
                "First training batch fetched in %.3fs (batch_size=%s)",
                t_end - t_start,
                batch_size,
            )

        self._pending_first_batch = batch_vla
        self._first_batch_startup_logged = True

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        self._log_first_batch_startup()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            if hasattr(self, "_pending_first_batch"):
                batch_vla = self._pending_first_batch
                del self._pending_first_batch
            else:
                batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            self._maybe_stage_decoder()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            sync = getattr(self, "_ds_sync_gradients", None)
            if sync is None:
                sync = self.accelerator.sync_gradients
            if sync:
                progress_bar.update(1)
                self.completed_steps += 1

            # Slurm's early-warning signal is translated by the batch launcher
            # into a shared-filesystem flag. Check it only after a synchronized
            # optimizer step, then agree across ranks before entering checkpoint
            # collectives. Saving directly inside a Unix signal handler would be
            # unsafe and could leave a partial distributed checkpoint.
            if sync and self._preemption_requested():
                if self.accelerator.is_main_process:
                    logger.warning(
                        "Pre-time-limit checkpoint requested at completed step %s.",
                        self.completed_steps,
                    )
                self._save_checkpoint()
                self._stopped_for_preemption = True
                break

            if self.accelerator.is_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            self.step_timers.add(t_end_data - t_start_data, t_end_model - t_start_model)
            # Flush once per logging window. Under gradient accumulation several
            # consecutive micro-steps share the same completed_steps, so guard on
            # the last flushed step rather than the modulo alone.
            if (
                self.completed_steps % self.config.trainer.logging_frequency == 0
                and self.completed_steps != self._last_timing_flush_step
            ):
                self._last_timing_flush_step = self.completed_steps
                step_metrics.update(self.step_timers.summary())
            step_metrics.update(self.memory_monitor.collect(self.completed_steps, self.vla_iter))
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _preemption_requested(self) -> bool:
        flag_path = os.environ.get("STARVLA_PREEMPT_FLAG")
        if not flag_path:
            return False
        poll_steps = max(1, int(os.environ.get("STARVLA_PREEMPT_POLL_STEPS", "10")))
        if self.completed_steps % poll_steps != 0:
            return False
        local_request = Path(flag_path).exists()
        request = torch.tensor(
            int(local_request),
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.all_reduce(request, op=dist.ReduceOp.MAX)
        return bool(request.item())

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        step_metrics = step_metrics or {}
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots
            if self._should_skip_open_loop_eval():
                logger.info("Skipping open-loop eval because it is disabled by config or attention backend.")
            else:
                open_loop_metrics = self._eval_open_loop_trajectories()
                step_metrics.update(open_loop_metrics)

        # Eval CoT loss: run forward (not predict_action) on the same batch to get
        # the language CE loss over CoT tokens. Logged separately from train/cot_loss.
        with torch.no_grad():
            eval_fwd = self.accelerator.unwrap_model(self.model).forward(examples)
            eval_cot_loss = eval_fwd.get("cot_loss", None)
            if eval_cot_loss is not None and self.accelerator.is_main_process:
                step_metrics["eval/cot_loss"] = eval_cot_loss.item()
                step_metrics["eval_cot_loss"] = eval_cot_loss.item()
            eval_cot_coverage = eval_fwd.get("cot_coverage", None)
            if eval_cot_coverage is not None and self.accelerator.is_main_process:
                step_metrics["eval/cot_coverage"] = float(eval_cot_coverage)
                step_metrics["eval_cot_coverage"] = float(eval_cot_coverage)
            eval_cot_keep_rate = eval_fwd.get("cot_keep_rate", None)
            if eval_cot_keep_rate is not None and self.accelerator.is_main_process:
                step_metrics["eval/cot_keep_rate"] = float(eval_cot_keep_rate)
                step_metrics["eval_cot_keep_rate"] = float(eval_cot_keep_rate)
            if self.accelerator.is_main_process:
                for key in (
                    "choice_loss", "score_loss", "choice_min_error",
                    "choice_score_mae", "choice_diversity",
                ):
                    value = eval_fwd.get(key)
                    if value is not None:
                        step_metrics[f"eval/{key}"] = float(value)
                winner_histogram = eval_fwd.get("choice_winner_histogram")
                if winner_histogram is not None:
                    for idx, value in enumerate(winner_histogram):
                        step_metrics[f"eval/choice_winner_{idx}"] = float(value)

        del examples
        dist.barrier()
        return step_metrics

    def _eval_open_loop_trajectories(self) -> dict:
        """Evaluate stitched open-loop chunk prediction on one holdout trajectory per subdataset."""

        action_horizon = int(getattr(self.config.framework.action_model, "action_horizon", 1))
        eval_root = Path(self.config.output_dir) / "open_loop_eval" / f"step_{self.completed_steps}"
        eval_root.mkdir(parents=True, exist_ok=True)

        if self.vla_eval_dataset is None:
            logger.warning("No dedicated eval dataset available for open-loop eval.")
            return {}

        trajectories = self._select_open_loop_trajectories(self.vla_eval_dataset)
        if not trajectories:
            logger.warning("No trajectories available for open-loop eval.")
            return {}

        model = self.accelerator.unwrap_model(self.model)
        all_squared_errors = []
        metrics = {}

        for dataset_name, dataset, trajectory_id in trajectories:
            gt_chunks = []
            pred_chunks = []
            trajectory_length = int(dataset.trajectory_lengths[dataset.get_trajectory_index(trajectory_id)])
            trajectory_step_to_index = {
                base_index: sample_index
                for sample_index, (traj_id, base_index) in enumerate(dataset.all_steps)
                if traj_id == trajectory_id
            }

            for step in range(0, trajectory_length, action_horizon):
                if step not in trajectory_step_to_index:
                    raise KeyError(
                        f"Trajectory step not found in dataset.all_steps: dataset={dataset_name}, "
                        f"trajectory_id={trajectory_id}, step={step}"
                    )
                sample = dataset[trajectory_step_to_index[step]]
                try:
                    pred_actions = model.predict_action(examples=[sample], use_ddim=True, num_ddim_steps=20)[
                        "normalized_actions"
                    ][0]
                except Exception:
                    logger.error(
                        "Open-loop eval failure at dataset=%s trajectory_id=%s step=%s",
                        dataset_name,
                        trajectory_id,
                        step,
                    )
                    raise
                gt_actions = np.asarray(sample["action"], dtype=np.float32)

                valid_horizon = min(action_horizon, trajectory_length - step)
                gt_chunks.append(gt_actions[:valid_horizon])
                pred_chunks.append(np.asarray(pred_actions, dtype=np.float32)[:valid_horizon])

            if not gt_chunks:
                continue

            gt_traj = np.concatenate(gt_chunks, axis=0)
            pred_traj = np.concatenate(pred_chunks, axis=0)
            squared_error = (pred_traj - gt_traj) ** 2
            all_squared_errors.append(squared_error)

            per_dim_mse = squared_error.mean(axis=0)
            traj_prefix = f"open_loop/{dataset_name}/traj_{trajectory_id}"
            metrics[f"{traj_prefix}/mse_mean"] = float(per_dim_mse.mean())
            for dim_idx, dim_mse in enumerate(per_dim_mse):
                metrics[f"{traj_prefix}/mse_dim_{dim_idx}"] = float(dim_mse)

            plot_path = eval_root / f"{dataset_name}_traj_{trajectory_id}.png"
            self._plot_open_loop_trajectory(
                gt_traj=gt_traj,
                pred_traj=pred_traj,
                save_path=plot_path,
                title=f"{dataset_name} trajectory {trajectory_id} @ step {self.completed_steps}",
            )
            metrics[f"{traj_prefix}/plot"] = wandb.Image(str(plot_path))

        if not all_squared_errors:
            return metrics

        stacked_squared_error = np.concatenate(all_squared_errors, axis=0)
        per_dim_mse = stacked_squared_error.mean(axis=0)
        metrics["open_loop/mse_mean"] = float(per_dim_mse.mean())
        for dim_idx, dim_mse in enumerate(per_dim_mse):
            metrics[f"open_loop/mse_dim_{dim_idx}"] = float(dim_mse)

        # Log plots and aggregate metrics immediately so eval visuals are not gated by logging_frequency.
        wandb.log(metrics, step=self.completed_steps)

        return metrics

    def _should_skip_open_loop_eval(self) -> bool:
        """Skip stitched open-loop eval when explicitly disabled or unsupported."""
        if not bool(getattr(self.config.trainer, "open_loop_eval", True)):
            return True
        attn_impl = str(getattr(self.config.framework.qwenvl, "attn_implementation", "")).strip().lower()
        return "flash_attention" in attn_impl

    def _select_open_loop_trajectories(self, mixture_dataset):
        """Pick one deterministic holdout trajectory from each subdataset."""
        selected = []
        for dataset in getattr(mixture_dataset, "datasets", []):
            trajectory_ids = list(getattr(dataset, "trajectory_ids", []))
            dataset_name = getattr(dataset, "dataset_name", getattr(dataset, "_dataset_name", "dataset"))
            if not trajectory_ids:
                continue
            # Use the final trajectory in each subdataset as a simple deterministic holdout.
            trajectory_id = trajectory_ids[-1]
            selected.append((dataset_name, dataset, trajectory_id))
        return selected

    def _plot_open_loop_trajectory(self, gt_traj: np.ndarray, pred_traj: np.ndarray, save_path: Path, title: str) -> None:
        """Save a per-dimension line plot for one stitched open-loop trajectory."""
        import math

        import matplotlib.pyplot as plt

        action_dim = gt_traj.shape[-1]
        ncols = min(4, action_dim)
        nrows = math.ceil(action_dim / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
        axes = axes.flatten()

        for dim_idx in range(action_dim):
            ax = axes[dim_idx]
            ax.plot(gt_traj[:, dim_idx], label="gt", linewidth=2)
            ax.plot(pred_traj[:, dim_idx], label="pred", linewidth=1.5)
            ax.set_title(f"action_dim_{dim_idx}")
            ax.set_xlabel("timestep")
            ax.grid(True, alpha=0.3)

        for ax in axes[action_dim:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _maybe_stage_decoder(self):
        """Two-stage schedule for the auxiliary-reasoning decoder (experiment arm E).

        Stage 1 (steps < `cot.decoder_unfreeze_step`): the pretrained 2B decoder core is
        frozen. The encoder, vision tower, and DiT train. Merged attention is implemented
        inside those decoder blocks, so its weights are frozen too, but its fixed operations
        still pass the CoT-loss gradient back to the encoder. The decoder already knows
        language; freezing it forces the ENCODER to supply the physical content rather than
        letting the decoder absorb the task.
        Stage 2: the decoder trains too, at the reduced LR its own param group specifies
        (`trainer.learning_rate.<decoder path>`), so it acts as a supervision head rather
        than the place new embodied knowledge is stored.

        NOTE `lm.layers` are the DECODER blocks; `lm.encoder_layers` are the encoder. In
        `full_duplicate` the encoder is a separate deepcopy, so freezing/unfreezing
        `lm.layers` does not touch the encoder -- and unfreezing the wrong one trains
        nothing at all.
        """
        cot_cfg = getattr(getattr(self.config.datasets, "vla_data", None), "cot", None)
        step_at = None
        if cot_cfg is not None:
            step_at = (cot_cfg.get("decoder_unfreeze_step") if hasattr(cot_cfg, "get")
                       else getattr(cot_cfg, "decoder_unfreeze_step", None))
        if not step_at:
            return
        want_frozen = self.completed_steps < int(step_at)
        if getattr(self, "_decoder_frozen", None) is want_frozen:
            return
        blocks = self._decoder_blocks()
        if blocks is None:
            if self.accelerator.is_main_process:
                logger.warning("decoder_unfreeze_step set but no decoder stack found; "
                               "the staged schedule is a no-op")
            self._decoder_frozen = want_frozen
            return
        modules = [blocks]
        stage_private_io = bool(
            cot_cfg.get("stage_private_decoder_io", False)
            if hasattr(cot_cfg, "get")
            else getattr(cot_cfg, "stage_private_decoder_io", False)
        )
        if stage_private_io:
            model = self.accelerator.unwrap_model(self.model)
            itf = getattr(model, "qwen_vl_interface", None)
            lm = itf._text_model() if itf is not None and hasattr(itf, "_text_model") else None
            for name in ("decoder_embed_tokens", "decoder_norm"):
                module = getattr(lm, name, None) if lm is not None else None
                if module is not None:
                    modules.append(module)
            outer = getattr(itf, "model", None) if itf is not None else None
            lm_head = getattr(outer, "lm_head", None)
            if lm_head is not None:
                modules.append(lm_head)

        n = 0
        seen = set()
        for prm in (prm for module in modules for prm in module.parameters()):
            if id(prm) in seen:
                continue
            seen.add(id(prm))
            prm.requires_grad = not want_frozen
            n += prm.numel()
        self._decoder_frozen = want_frozen
        if self.accelerator.is_main_process:
            scope = "core + private embedding/norm/head" if stage_private_io else "core"
            logger.info(f"[stage] step {self.completed_steps}: decoder {scope} "
                        f"{'FROZEN' if want_frozen else 'UNFROZEN'} ({n/1e6:.0f}M params)")

    def _decoder_blocks(self):
        """The causal decoder stack of the enc-dec backbone, or None."""
        model = self.accelerator.unwrap_model(self.model)
        itf = getattr(model, "qwen_vl_interface", None)
        if itf is None or not hasattr(itf, "_text_model"):
            return None
        lm = itf._text_model()
        if hasattr(lm, "encoder_layers"):
            return getattr(lm, "layers", None)
        # In parameter-matched layer-split mode the two stacks share one ModuleList,
        # partitioned at n_encoder_layers. Freeze/stage only the causal suffix.
        if hasattr(lm, "n_encoder_layers"):
            return lm.layers[int(lm.n_encoder_layers):]
        return None

    def _log_encoder_grad_split(
        self, action_loss, cot_loss, metrics, choice_loss=None, score_loss=None,
        cot_scale=1.0, choice_scale=1.0, score_scale=1.0,
    ):
        """Report how hard each loss pulls on the shared encoder.

        The CE term scales with the number of supervised tokens, so it can dominate the
        flow loss for reasons that have nothing to do with the hypothesis. Logging the two
        norms separately is what tells us whether `loss_scale.cot` needs rescaling; the
        target is the same order of magnitude. Only run at the logging interval -- it costs
        two extra backward passes through the encoder.
        """
        blocks = None
        # After ``accelerator.prepare`` self.model is a DeepSpeed/DDP wrapper.  Looking
        # directly on that wrapper silently returned None, so all completed distributed
        # runs skipped this diagnostic.  Always inspect the underlying framework model.
        unwrapped = self.accelerator.unwrap_model(self.model)
        itf = getattr(unwrapped, "qwen_vl_interface", None)
        if itf is not None and hasattr(itf, "_text_model"):
            lm = itf._text_model()
            blocks = getattr(lm, "encoder_layers", None)
            if blocks is None and hasattr(lm, "n_encoder_layers"):
                blocks = lm.layers[:int(lm.n_encoder_layers)]
        if blocks is None:
            return
        # A full-encoder autograd probe materializes gradients for ~1.4B encoder
        # parameters and needs another ~4.6 GiB at the production micro-batch,
        # even though the ordinary backward fits.  Use one fixed, representative
        # matrix from the final encoder block instead.  Ratios across objectives
        # remain directly comparable, and the diagnostic no longer perturbs the
        # memory envelope of the actual training step.
        last_block = blocks[-1]
        named_probe = [
            (name, prm) for name, prm in last_block.named_parameters()
            if prm.requires_grad and name.endswith("self_attn.q_proj.weight")
        ]
        if not named_probe:
            named_probe = [
                (name, prm) for name, prm in last_block.named_parameters()
                if prm.requires_grad
            ][:1]
        params = [prm for _, prm in named_probe]
        if not params:
            return
        metrics["grad/enc_probe_numel"] = float(sum(prm.numel() for prm in params))

        def norm(loss):
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return float(sum((x.detach() ** 2).sum() for x in g if x is not None) ** 0.5)

        try:
            metrics["grad/enc_from_flow"] = norm(action_loss)
            if cot_loss is not None:
                metrics["grad/enc_from_reason"] = norm(cot_loss)
                if metrics["grad/enc_from_flow"] > 0:
                    metrics["grad/reason_over_flow"] = (
                        metrics["grad/enc_from_reason"] / metrics["grad/enc_from_flow"])
                    metrics["grad/weighted_reason_over_flow"] = (
                        float(cot_scale) * metrics["grad/reason_over_flow"])
            if choice_loss is not None:
                metrics["grad/enc_from_choice"] = norm(choice_loss)
                if metrics["grad/enc_from_flow"] > 0:
                    metrics["grad/choice_over_flow"] = (
                        metrics["grad/enc_from_choice"] / metrics["grad/enc_from_flow"])
                    metrics["grad/weighted_choice_over_flow"] = (
                        float(choice_scale) * metrics["grad/choice_over_flow"])
            if score_loss is not None:
                metrics["grad/enc_from_score"] = norm(score_loss)
                if metrics["grad/enc_from_flow"] > 0:
                    metrics["grad/score_over_flow"] = (
                        metrics["grad/enc_from_score"] / metrics["grad/enc_from_flow"])
                    metrics["grad/weighted_score_over_flow"] = (
                        float(score_scale) * metrics["grad/score_over_flow"])
        except RuntimeError as exc:
            # Graph already freed (e.g. under DeepSpeed); diagnostics must never
            # take down a training run.
            logger.warning(f"encoder grad split unavailable: {exc}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        grad_accum = self.accelerator.gradient_accumulation_steps
        use_deepspeed = getattr(self.config, "use_deepspeed", False)

        if grad_accum > 1 and use_deepspeed:
            # DeepSpeed ZeRO2 manages gradient accumulation internally via its
            # engine: it skips the reduce-scatter on non-sync backward calls
            # automatically. Accelerate's accumulate() uses DDP's no_sync()
            # which is incompatible with ZeRO2 gradient partitioning.
            self._accum_count = getattr(self, "_accum_count", 0) + 1
            self._ds_sync_gradients = (self._accum_count % grad_accum == 0)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                cot_loss = output_dict.get("cot_loss", None)
                structured_aux_loss = output_dict.get("structured_aux_loss", None)
                choice_loss = output_dict.get("choice_loss", None)
                score_loss = output_dict.get("score_loss", None)

                loss_scale = getattr(self.config.trainer, "loss_scale", None)
                cot_scale = getattr(loss_scale, "cot", 0.1)
                structured_aux_scale = getattr(loss_scale, "structured_aux", 1.0)
                choice_scale = getattr(loss_scale, "choice", 1.0)
                score_scale = getattr(loss_scale, "score", 1.0)
                total_loss = action_loss
                if cot_loss is not None:
                    total_loss = total_loss + cot_scale * cot_loss
                if structured_aux_loss is not None:
                    total_loss = total_loss + structured_aux_scale * structured_aux_loss
                if choice_loss is not None:
                    total_loss = total_loss + choice_scale * choice_loss
                if score_loss is not None:
                    total_loss = total_loss + score_scale * score_loss

            # Probe the final micro-step only: every micro-step builds an equivalent
            # graph, while probing all of them needlessly repeats the extra autograd
            # work. This must happen before backward while the graph is still alive.
            grad_split = {}
            if (
                self._ds_sync_gradients
                and self.config.trainer.logging_frequency
                and self.completed_steps % self.config.trainer.logging_frequency == 0
            ):
                self._log_encoder_grad_split(
                    action_loss,
                    cot_loss,
                    grad_split,
                    choice_loss=choice_loss,
                    score_loss=score_loss,
                    cot_scale=cot_scale,
                    choice_scale=choice_scale,
                    score_scale=score_scale,
                )

            self.accelerator.backward(total_loss / grad_accum)

            if self._ds_sync_gradients:
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            metrics = {"action_dit_loss": action_loss.item()}
            metrics.update(grad_split)
            if cot_loss is not None:
                metrics["train/cot_loss"] = cot_loss.item()
                metrics["train_cot_loss"] = cot_loss.item()
                metrics["cot_loss"] = cot_loss.item()
            if structured_aux_loss is not None:
                metrics["train/structured_aux_loss"] = structured_aux_loss.item()
                metrics["structured_aux_loss"] = structured_aux_loss.item()
                for key, value in output_dict.items():
                    if key.startswith("structured_aux/"):
                        metrics[f"train/{key}"] = float(value)
            if choice_loss is not None:
                metrics["train/choice_loss"] = choice_loss.item()
                metrics["choice_loss"] = choice_loss.item()
            if score_loss is not None:
                metrics["train/score_loss"] = score_loss.item()
                metrics["score_loss"] = score_loss.item()
            for key in ("choice_min_error", "choice_score_mae", "choice_diversity"):
                value = output_dict.get(key)
                if value is not None:
                    metrics[f"train/{key}"] = float(value)
            winner_histogram = output_dict.get("choice_winner_histogram")
            if winner_histogram is not None:
                for idx, value in enumerate(winner_histogram):
                    metrics[f"train/choice_winner_{idx}"] = float(value)
            cot_coverage = output_dict.get("cot_coverage", None)
            if cot_coverage is not None:
                metrics["train/cot_coverage"] = float(cot_coverage)
                metrics["cot_coverage"] = float(cot_coverage)
            cot_keep_rate = output_dict.get("cot_keep_rate", None)
            if cot_keep_rate is not None:
                metrics["train/cot_keep_rate"] = float(cot_keep_rate)
                metrics["cot_keep_rate"] = float(cot_keep_rate)
            return metrics

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                cot_loss = output_dict.get("cot_loss", None)
                structured_aux_loss = output_dict.get("structured_aux_loss", None)
                choice_loss = output_dict.get("choice_loss", None)
                score_loss = output_dict.get("score_loss", None)

                cot_scale = getattr(getattr(self.config.trainer, "loss_scale", None), "cot", 0.1)
                structured_aux_scale = getattr(
                    getattr(self.config.trainer, "loss_scale", None), "structured_aux", 1.0
                )
                choice_scale = getattr(
                    getattr(self.config.trainer, "loss_scale", None), "choice", 1.0
                )
                score_scale = getattr(
                    getattr(self.config.trainer, "loss_scale", None), "score", 1.0
                )
                total_loss = action_loss
                if cot_loss is not None:
                    total_loss = total_loss + cot_scale * cot_loss
                if structured_aux_loss is not None:
                    total_loss = total_loss + structured_aux_scale * structured_aux_loss
                if choice_loss is not None:
                    total_loss = total_loss + choice_scale * choice_loss
                if score_loss is not None:
                    total_loss = total_loss + score_scale * score_loss

            # Must run BEFORE backward(): it needs the graph both losses still hold.
            grad_split = {}
            if (self.config.trainer.logging_frequency
                    and self.completed_steps % self.config.trainer.logging_frequency == 0):
                self._log_encoder_grad_split(
                    action_loss, cot_loss, grad_split,
                    choice_loss=choice_loss, score_loss=score_loss,
                    cot_scale=cot_scale, choice_scale=choice_scale,
                    score_scale=score_scale,
                )

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        metrics = {"action_dit_loss": action_loss.item()}
        metrics.update(grad_split)
        if cot_loss is not None:
            metrics["train/cot_loss"] = cot_loss.item()
            metrics["train_cot_loss"] = cot_loss.item()
            metrics["cot_loss"] = cot_loss.item()
        if structured_aux_loss is not None:
            metrics["train/structured_aux_loss"] = structured_aux_loss.item()
            metrics["structured_aux_loss"] = structured_aux_loss.item()
            for key, value in output_dict.items():
                if key.startswith("structured_aux/"):
                    metrics[f"train/{key}"] = float(value)
        if choice_loss is not None:
            metrics["train/choice_loss"] = choice_loss.item()
            metrics["choice_loss"] = choice_loss.item()
        if score_loss is not None:
            metrics["train/score_loss"] = score_loss.item()
            metrics["score_loss"] = score_loss.item()
        for key in ("choice_min_error", "choice_score_mae", "choice_diversity"):
            value = output_dict.get(key)
            if value is not None:
                metrics[f"train/{key}"] = float(value)
        winner_histogram = output_dict.get("choice_winner_histogram")
        if winner_histogram is not None:
            for idx, value in enumerate(winner_histogram):
                metrics[f"train/choice_winner_{idx}"] = float(value)
        cot_coverage = output_dict.get("cot_coverage", None)
        if cot_coverage is not None:
            metrics["train/cot_coverage"] = float(cot_coverage)
            metrics["cot_coverage"] = float(cot_coverage)
        cot_keep_rate = output_dict.get("cot_keep_rate", None)
        if cot_keep_rate is not None:
            metrics["train/cot_keep_rate"] = float(cot_keep_rate)
            metrics["cot_keep_rate"] = float(cot_keep_rate)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self._stopped_for_preemption:
            if self.accelerator.is_main_process:
                logger.info(
                    "Training stopped cleanly after the pre-time-limit checkpoint at step %s.",
                    self.completed_steps,
                )
                wandb.finish()
            self.accelerator.wait_for_everyone()
            return

        save_final_checkpoint = bool(getattr(self.config.trainer, "save_final_checkpoint", True))
        if self.accelerator.is_main_process and save_final_checkpoint:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        elif self.accelerator.is_main_process:
            logger.info("Training complete. Final checkpoint disabled by config.")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg, accelerator: Accelerator) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    log_eval_backend_banner(cfg, accelerator)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    configure_dataloader_resume(cfg, output_dir)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--use_deepspeed",
        type=parse_bool_flag,
        default=True,
        help="Whether to initialize Accelerate with DeepSpeedPlugin. Default: true.",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml
    cfg.use_deepspeed = args.use_deepspeed

    grad_accum = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    accelerator = create_accelerator(use_deepspeed=args.use_deepspeed, gradient_accumulation_steps=grad_accum)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg, accelerator)
