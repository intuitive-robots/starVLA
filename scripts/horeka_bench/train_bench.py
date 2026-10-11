"""Throughput wrapper around starVLA/training/train_starvla.py for the HoreKa proposal benchmark.

Runs the unmodified trainer (same CLI, model, data and DeepSpeed setup) and adds, per optimizer step:
CUDA-synchronised model time, step-to-step wall time (includes data loading), and the peak
torch.cuda.max_memory_allocated over all ranks. Rank 0 writes a JSON summary to $BENCH_OUT:
median over the steps after $BENCH_WARMUP (default 10).

Launch exactly like train_starvla.py, e.g.
  accelerate launch ... scripts/horeka_bench/train_bench.py --config_yaml X --trainer.max_train_steps 40
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import starVLA.training.train_starvla as T

_model_times: list[float] = []
_wall_times: list[float] = []
_last_start = [None]

_orig_train_step = T.VLATrainer._train_step
_orig_finalize = T.VLATrainer._finalize_training


def _timed_train_step(self, batch_vla, batch_vlm=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if _last_start[0] is not None:
        _wall_times.append(t0 - _last_start[0])
    _last_start[0] = t0
    out = _orig_train_step(self, batch_vla, batch_vlm)
    torch.cuda.synchronize()
    _model_times.append(time.perf_counter() - t0)
    return out


def _finalize(self):
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
    if dist.is_initialized():
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if (not dist.is_initialized() or dist.get_rank() == 0) and os.environ.get("BENCH_OUT"):
        model = _model_times[warmup:]
        # _wall_times[i] spans step i -> i+1 (data + model of step i); last step has no successor.
        wall = _wall_times[warmup:]
        per_gpu_bs = int(self.config.datasets.vla_data.per_device_batch_size)
        med_wall = statistics.median(wall) if wall else float("nan")
        result = {
            "world_size": world,
            "num_nodes": int(os.environ.get("NUM_MACHINES", "1")),
            "per_gpu_batch": per_gpu_bs,
            "global_batch": per_gpu_bs * world,
            "measured_steps": len(model),
            "warmup_steps": warmup,
            "step_time_median_s": med_wall,
            "model_time_median_s": statistics.median(model) if model else float("nan"),
            "data_time_median_s": statistics.median(w - m for w, m in zip(wall, model)) if wall else float("nan"),
            "samples_per_s": per_gpu_bs * world / med_wall,
            "peak_mem_per_gpu_gb": round(peak.item(), 2),
            "model_times": model,
            "wall_times": wall,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(),
        }
        import transformers, deepspeed  # noqa: E401

        result.update(transformers=transformers.__version__, deepspeed=deepspeed.__version__)
        with open(os.environ["BENCH_OUT"], "w") as f:
            json.dump(result, f, indent=1)
        print(f"BENCH_RESULT {json.dumps({k: v for k, v in result.items() if not k.endswith('_times')})}")
    return _orig_finalize(self)


T.VLATrainer._train_step = _timed_train_step
T.VLATrainer._finalize_training = _finalize


if __name__ == "__main__":
    # Same CLI handling as train_starvla.py's __main__ block.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--use_deepspeed", type=T.parse_bool_flag, default=True)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(T.normalize_dotlist_args(clipargs)))
    cfg = T.apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    cfg.use_deepspeed = args.use_deepspeed

    grad_accum = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    accelerator = T.create_accelerator(use_deepspeed=args.use_deepspeed, gradient_accumulation_steps=grad_accum)
    T.main(cfg, accelerator)
