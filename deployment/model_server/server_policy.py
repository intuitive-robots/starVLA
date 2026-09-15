# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.seed_utils import set_seed_everywhere
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    seed = getattr(args, "seed", None)
    if not isinstance(seed, int):
        seed = None
    if seed is not None:
        set_seed_everywhere(seed)

    config_overrides = getattr(args, "config_override", [])
    if config_overrides:
        override_keys = [item.split("=", 1)[0] for item in config_overrides]
        logging.info("Applying config override keys: %s", override_keys)
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        is_inference=True,
        cot_max_new_tokens=args.cot_max_new_tokens,
        config_overrides=config_overrides,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt_path`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    metadata = dict(wrapper.metadata)
    metadata["seed"] = seed
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
    )
    logging.info("server running ... metadata=%s", metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed Python, NumPy, and PyTorch before loading the policy")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--cot_max_new_tokens",
        type=int,
        default=None,
        help="Optional inference-only override for framework.cot.max_new_tokens. "
        "Leaves the checkpoint configuration unchanged.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="CEILING on how many concurrent requests are batched into one predict_action() "
        "call. 1 (default) = unchanged behavior, one request at a time. Measured on this "
        "model: ~30x throughput at 32, ~112x at 128, for a ~15%% latency cost per batch. "
        "Batches fill on their own from requests that queue up during the previous batch's "
        "inference, so this is a ceiling to leave room above the real client count, NOT a "
        "target the dispatcher waits to reach (it does not wait at all unless you also set "
        "--max_wait_time).",
    )
    parser.add_argument(
        "--max_wait_time",
        type=float,
        default=0.0,
        help="How long the batch dispatcher may block waiting for a batch to reach its target "
        "size. 0 (default) = never block: dispatch whatever has queued up and let the next "
        "batch collect during this one's inference. A positive value is a real stall whenever "
        "the target is not reached -- each client blocks on its own response, so no more than "
        "'connected clients' requests can ever be in flight, and the dispatcher caps its "
        "target at that count (a 1.0s default against an unreachable --max_batch_size 32 used "
        "to add a full second in front of every 0.19s inference). Only raise it if clients "
        "genuinely burst and you measured that the larger batches pay for the wait.",
    )
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf dotlist override applied before model construction.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    # Cluster environments may define DEBUG=0/false globally.  Treat only
    # explicit truthy values as enabling debugpy; bool("0") is True and made
    # every policy-server replica contend for the fixed debug port 10095.
    debug_enabled = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    if debug_enabled:
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
