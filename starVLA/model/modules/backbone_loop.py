"""Looped backbone blocks (latent recurrent refinement), default off.

``framework.backbone_loop`` re-runs backbone blocks ``start..end`` (inclusive) up to ``max_iters`` times on
their own output before the remaining blocks run once (LOOPING_2026-09-30.md, design L1). Iteration i >= 2:

    x_i = h_{i-1} + inject(h_pre)                     # h_pre = the input of block ``start`` (re-injection)
    h_i = h_{i-1} + gate[i] * (F(x_i) - h_{i-1})      # F = blocks start..end, gate = damping

``inject`` (Linear, no bias) and ``gate`` start at zero, so a freshly enabled loop computes exactly the
unlooped network: training from a pretrained checkpoint begins at its behaviour and learns how much each
extra pass contributes. Training samples the number of iterations K uniformly from 1..max_iters per step
(the same K on every rank: drawn from the step counter) and backpropagates only through the last ``bptt``
iterations. Inference uses ``eval_iters`` (default max_iters) or ``STARVLA_LOOP_ITERS``.

The EMA target pass of fusion_registers must stay a single pass: wrap it in ``loop_suspended()``. Hooks
check that the firing module is one of THIS model's blocks, so a deep copy of the backbone (the EMA copy)
never loops through the original blocks.

Config::
    backbone_loop:
      enabled: false
      start: 16
      end: 19
      max_iters: 3
      bptt: 2
      eval_iters: null     # null -> max_iters
"""
import contextlib
import os
import random
from typing import List

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_SUSPEND = [0]


@contextlib.contextmanager
def loop_suspended():
    """Run the backbone once (no extra iterations) inside this block, e.g. for EMA target passes."""
    _SUSPEND[0] += 1
    try:
        yield
    finally:
        _SUSPEND[0] -= 1


class BackboneLoopMixin:
    def _configure_backbone_loop(self, config) -> None:
        cfg = dict(config.framework.get("backbone_loop", {}) or {})
        self.backbone_loop_enabled = bool(cfg.get("enabled", False))
        if not self.backbone_loop_enabled:
            return
        self.loop_start = int(cfg.get("start", 16))
        self.loop_end = int(cfg.get("end", 19))
        self.loop_max_iters = int(cfg.get("max_iters", 3))
        self.loop_bptt = max(1, int(cfg.get("bptt", 2)))
        # fixed_iters: always max_iters passes in training (use with per-pass targets, which give every pass
        # its own meaning; random K exists for the case where only the final output is supervised)
        self.loop_fixed_iters = bool(cfg.get("fixed_iters", False))
        self._loop_pass_out = None
        ev = cfg.get("eval_iters", None)
        self.loop_eval_iters = self.loop_max_iters if ev in (None, "null", "") else int(ev)
        if self.loop_max_iters < 1 or self.loop_start > self.loop_end:
            raise ValueError("backbone_loop: need max_iters >= 1 and start <= end")
        self._loop_step = 0
        self._loop_k = 1
        self._loop_ctx = None
        self._loop_inside = False
        self._loop_logged = False

    def _build_backbone_loop(self, hidden_size: int) -> None:
        if not getattr(self, "backbone_loop_enabled", False):
            return
        blocks: List[nn.Module] = self._aux_blocks()
        if not 0 <= self.loop_start <= self.loop_end < len(blocks):
            raise ValueError(f"backbone_loop {self.loop_start}-{self.loop_end} outside the {len(blocks)} blocks")
        self.loop_inject = nn.Linear(int(hidden_size), int(hidden_size), bias=False)
        nn.init.zeros_(self.loop_inject.weight)
        self.loop_gate = nn.Parameter(torch.zeros(max(1, self.loop_max_iters - 1)))
        first, last = blocks[self.loop_start], blocks[self.loop_end]
        loop_blocks = blocks[self.loop_start:self.loop_end + 1]

        def pre_hook(module, args, kwargs):
            if module is not first or self._loop_inside or _SUSPEND[0]:
                return None
            hs = kwargs.get("hidden_states") if kwargs.get("hidden_states") is not None else args[0]
            rest = args[1:] if kwargs.get("hidden_states") is None else args
            kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            self._loop_ctx = (hs, rest, kw)
            return None

        def post_hook(module, args, kwargs, out):
            if module is not last or self._loop_inside or _SUSPEND[0] or self._loop_ctx is None:
                return None
            h_pre, rest, kw = self._loop_ctx
            self._loop_ctx = None
            h = out[0] if isinstance(out, tuple) else out
            k = self._loop_k
            keep_passes = self.training and getattr(self, "_loop_keep_passes", False)
            self._loop_pass_out = [h] if keep_passes else None
            if k <= 1:
                if self.training:  # keep every rank's loop parameters in the graph (zero contribution)
                    h = h + 0.0 * (self.loop_gate.sum() + self.loop_inject.weight.sum()).to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return None
            inj = self.loop_inject(h_pre.to(self.loop_inject.weight.dtype)).to(h.dtype)
            self._loop_inside = True
            try:
                for i in range(1, k):
                    with_grad = (not self.training) or (i >= k - self.loop_bptt) or not torch.is_grad_enabled()
                    ctx = contextlib.nullcontext() if with_grad else torch.no_grad()
                    with ctx:
                        x = h + inj
                        for blk in loop_blocks:
                            o = blk(x, *rest, **kw)
                            x = o[0] if isinstance(o, tuple) else o
                        g = self.loop_gate[i - 1].to(h.dtype)
                        h_new = h + g * (x - h)
                    h = h_new if with_grad else h_new.detach()
                    if keep_passes:
                        self._loop_pass_out.append(h)
            finally:
                self._loop_inside = False
            if not self._loop_logged:
                self._loop_logged = True
                print(f"__BACKBONE_LOOP__ blocks {self.loop_start}-{self.loop_end} K={k} "
                      f"training={self.training} gate={[round(float(v), 4) for v in self.loop_gate]}", flush=True)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        first.register_forward_pre_hook(pre_hook, with_kwargs=True)
        last.register_forward_hook(post_hook, with_kwargs=True)
        logger.info("backbone loop ON: blocks %d-%d, max_iters %d, bptt %d, eval_iters %d",
                    self.loop_start, self.loop_end, self.loop_max_iters, self.loop_bptt, self.loop_eval_iters)

    def _loop_set_iters(self) -> None:
        """Choose K for the next backbone forward (call once per forward / predict_action)."""
        if not getattr(self, "backbone_loop_enabled", False):
            return
        if self.training:
            if self.loop_fixed_iters:
                self._loop_k = self.loop_max_iters
            else:
                rng = random.Random(1000003 * int(self._loop_step) + 17)
                self._loop_k = rng.randint(1, self.loop_max_iters)
            self._loop_step += 1
        else:
            env = os.environ.get("STARVLA_LOOP_ITERS", "").strip()
            self._loop_k = int(env) if env else int(self.loop_eval_iters)

    def _loop_metrics(self) -> dict:
        if not getattr(self, "backbone_loop_enabled", False):
            return {}
        m = {"structured_aux/loop_k": torch.tensor(float(self._loop_k))}
        for i, v in enumerate(self.loop_gate.detach()):
            m[f"structured_aux/loop_gate_{i + 2}"] = v.float()
        return m
