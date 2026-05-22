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


def create_accelerator(use_deepspeed: bool) -> Accelerator:
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
        accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(), kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    accelerator.print(accelerator.state)
    return accelerator


def log_eval_backend_banner(cfg, accelerator: Accelerator) -> None:
    """Print a visible startup banner for attention backend and eval behavior."""
    if not accelerator.is_main_process:
        return

    attn_impl = str(getattr(cfg.framework.qwenvl, "attn_implementation", "unknown")).strip()
    open_loop_disabled = "flash_attention" in attn_impl.lower()

    banner_lines = [
        "",
        "=" * 88,
        "STARTUP CONFIG",
        f"Attention backend: {attn_impl}",
        (
            "Open-loop eval: DISABLED because Flash Attention backend is configured"
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

        self.completed_steps = 0
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
                "eval/cot_loss",
                "eval_cot_loss",
                "eval/cot_coverage",
                "eval_cot_coverage",
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

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

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
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
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
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            if hasattr(self, "_pending_first_batch"):
                batch_vla = self._pending_first_batch
                del self._pending_first_batch
            else:
                batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

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
                logger.info("Skipping open-loop eval because Flash Attention backend is configured.")
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
        """Skip stitched open-loop eval for Flash Attention based configs."""
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

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                cot_loss = output_dict.get("cot_loss", None)

                cot_scale = getattr(getattr(self.config.trainer, "loss_scale", None), "cot", 0.1)
                total_loss = action_loss
                if cot_loss is not None:
                    total_loss = total_loss + cot_scale * cot_loss

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
        if cot_loss is not None:
            metrics["train/cot_loss"] = cot_loss.item()
            metrics["train_cot_loss"] = cot_loss.item()
            metrics["cot_loss"] = cot_loss.item()
        cot_coverage = output_dict.get("cot_coverage", None)
        if cot_coverage is not None:
            metrics["train/cot_coverage"] = float(cot_coverage)
            metrics["cot_coverage"] = float(cot_coverage)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
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

    accelerator = create_accelerator(use_deepspeed=args.use_deepspeed)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg, accelerator)
