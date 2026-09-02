# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Collects concurrent predict_action requests from multiple websocket
connections into batched policy.predict_action() calls.

Why: the framework-level ``predict_action(examples: list[dict])`` already
batches correctly -- it runs one fused forward/generate call over the whole
list (see e.g. ``QwenOFT_CoT.predict_action`` / ``_predict_action_fast``,
which loop ``for i in range(batch_size)`` for exactly this reason). The
model was never the bottleneck; serving one request at a time was. Measured
directly on this model/hardware (0.8B Qwen3.5, fixed-length trace-format
CoT, GH200): batching to 32 concurrent requests gives ~30x throughput, 128
gives ~112x, for only a ~15% wall-clock increase per batch -- because decode
at batch=1 is memory-bandwidth-bound, leaving most of the GPU idle.

Design (adapted from the vla-evaluation-harness's ``predict.py`` pattern): a
single background dispatch loop pulls up to ``max_batch_size`` pending
requests, waiting up to ``max_wait_time`` for more to arrive once at least
one is queued, then runs ONE batched ``predict_action()`` call in a worker
thread (via ``asyncio.to_thread``, so the event loop keeps accepting new
connections/requests while the GPU call is in flight) and fans results back
out to each caller.

Requests are grouped by their non-``examples`` kwargs (``unnorm_key``,
``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...) before batching --
different concurrent clients could in principle request different settings,
and blindly merging them would silently apply one request's settings to
another's examples.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _PendingRequest:
    examples: List[dict]
    shared_kwargs: Dict[str, Any]
    kwargs_key: tuple
    future: "asyncio.Future"
    n_examples: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_examples = len(self.examples)


def _kwargs_signature(kwargs: Dict[str, Any]) -> tuple:
    """Hashable signature for grouping requests that must batch together.

    Values here (unnorm_key, do_sample, use_ddim, num_ddim_steps, ...) are
    always simple primitives in practice (str/bool/int/None) -- raises
    loudly if that assumption is ever violated rather than silently grouping
    incompatible requests together.
    """
    try:
        return tuple(sorted(kwargs.items()))
    except TypeError as e:
        raise TypeError(
            f"BatchDispatcher: predict_action kwargs must be hashable primitives to "
            f"safely group concurrent requests, got {kwargs!r}"
        ) from e


class BatchDispatcher:
    def __init__(self, policy, max_batch_size: int = 32, max_wait_time: float = 1.0):
        self._policy = policy
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._queue: "asyncio.Queue[_PendingRequest]" = asyncio.Queue()
        self._loop_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Idempotent; lazily starts the background dispatch loop on first use
        (avoids requiring callers to have a running event loop at construction time).
        """
        if self._loop_task is None:
            self._loop_task = asyncio.ensure_future(self._dispatch_loop())

    async def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Enqueue one caller's request; returns that caller's own predict_action() result."""
        self.start()
        examples = payload.get("examples")
        if examples is None:
            raise ValueError("predict_action payload must include 'examples'")
        if not isinstance(examples, list):
            examples = [examples]
        shared_kwargs = {k: v for k, v in payload.items() if k != "examples"}
        req = _PendingRequest(
            examples=examples,
            shared_kwargs=shared_kwargs,
            kwargs_key=_kwargs_signature(shared_kwargs),
            future=asyncio.get_event_loop().create_future(),
        )
        await self._queue.put(req)
        return await req.future

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return

            # Collect more requests sharing the SAME kwargs signature as `first`,
            # waiting up to max_wait_time (from `first`'s arrival) for the batch to
            # fill. Requests with a different signature are set aside and put back
            # for the next dispatch cycle rather than dropped.
            batch = [first]
            deferred: List[_PendingRequest] = []
            deadline = time.monotonic() + self.max_wait_time

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                (batch if item.kwargs_key == first.kwargs_key else deferred).append(item)

            # Drain anything else already queued (non-blocking) without extending the wait.
            while len(batch) < self.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                (batch if item.kwargs_key == first.kwargs_key else deferred).append(item)

            for item in deferred:
                self._queue.put_nowait(item)

            await self._run_batch(batch)

    async def _run_batch(self, batch: List[_PendingRequest]) -> None:
        combined_examples: List[dict] = []
        for req in batch:
            combined_examples.extend(req.examples)
        shared_kwargs = batch[0].shared_kwargs

        # Realized batch size vs. the configured ceiling: how well max_wait_time is
        # actually working at your real request-arrival rate. If this sits well
        # below max_batch_size under real load, max_wait_time is too short for how
        # spread out your workers' requests actually are -- raise it.
        batch_n = len(combined_examples)
        fill = batch_n / self.max_batch_size

        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(self._policy.predict_action, examples=combined_examples, **shared_kwargs)
        except Exception as e:
            logging.exception("BatchDispatcher: batched predict_action failed (batch_size=%d)", batch_n)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
            return
        elapsed = time.monotonic() - t0
        logging.info(
            f"[BatchDispatcher] batch_size={batch_n}/{self.max_batch_size} ({fill:.0%} full), "
            f"predict_action={elapsed:.3f}s"
        )

        actions = result["actions"]  # [sum(n_examples), T, D]
        # Any other per-sample field must be sliced alongside actions -- a bare
        # {**result} would hand every client the whole batch's values.
        per_sample_keys = [
            k for k, v in result.items()
            if k != "actions" and isinstance(v, (list, tuple)) and len(v) == actions.shape[0]
        ]
        offset = 0
        for req in batch:
            n = req.n_examples
            per_req_result = {**result, "actions": actions[offset : offset + n]}
            for k in per_sample_keys:
                per_req_result[k] = list(result[k][offset : offset + n])
            offset += n
            if not req.future.done():
                req.future.set_result(per_req_result)
