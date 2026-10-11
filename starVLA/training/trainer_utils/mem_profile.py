"""Per-phase GPU memory profile for the first training steps (STARVLA_MEM_PROFILE=1, rank 0 only).

Hooks the VLM interface and the action head. Each line reports the peak since the previous line, so the line
"before VLM" carries the peak of the previous backward + optimizer step. `non_torch` is GPU memory in use that
PyTorch's allocator does not hold (contexts, NCCL, other processes).
"""

import os

import torch

_GIB = 2 ** 30


def install_mem_profile(model, max_steps: int = 8) -> None:
    if os.environ.get("STARVLA_MEM_PROFILE") != "1" or int(os.environ.get("RANK", "0")) != 0:
        return
    state = {"step": 0}

    def shapes(args, kwargs) -> str:
        items = list(kwargs.items()) + [(f"arg{i}", v) for i, v in enumerate(args)]
        return " ".join(f"{k}{tuple(v.shape)}" for k, v in items if torch.is_tensor(v))[:500]

    def snap(tag: str, extra: str = "") -> None:
        if state["step"] >= max_steps:
            return
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved() / _GIB
        used = (total - free) / _GIB
        print(f"[memprof] step {state['step']} {tag}: peak {torch.cuda.max_memory_allocated() / _GIB:.1f} "
              f"alloc {torch.cuda.memory_allocated() / _GIB:.1f} reserved {reserved:.1f} gpu_used {used:.1f} "
              f"non_torch {used - reserved:.1f} grad={torch.is_grad_enabled()} {extra}", flush=True)
        torch.cuda.reset_peak_memory_stats()

    def head_post(module, args, kwargs, output):
        snap("after head forward")
        state["step"] += 1

    model.qwen_vl_interface.register_forward_pre_hook(lambda m, a, k: snap("before VLM", shapes(a, k)), with_kwargs=True)
    model.qwen_vl_interface.register_forward_hook(lambda m, a, k, o: snap("after VLM forward"), with_kwargs=True)
    model.action_model.register_forward_pre_hook(lambda m, a, k: snap("before head", shapes(a, k)), with_kwargs=True)
    model.action_model.register_forward_hook(head_post, with_kwargs=True)
    print("[memprof] installed", flush=True)
