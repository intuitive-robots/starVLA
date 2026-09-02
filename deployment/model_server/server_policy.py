# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        is_inference=True,
        cot_max_new_tokens=args.cot_max_new_tokens,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
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
        help="Batch concurrent requests from multiple connections into one predict_action() "
        "call. 1 (default) = unchanged behavior, one request at a time. Measured on this "
        "model: ~30x throughput at 32, ~112x at 128, for a ~15%% latency cost per batch -- "
        "set to (at least) your expected number of concurrent env workers per server.",
    )
    parser.add_argument(
        "--max_wait_time",
        type=float,
        default=1.0,
        help="CEILING (not a fixed delay) on how long the batch dispatcher waits for more "
        "requests before running an under-full batch -- it exits the moment the batch fills "
        "or the queue goes idle, so raising this doesn't add latency on the happy path. Only "
        "relevant when --max_batch_size > 1. Watch the '[BatchDispatcher] batch_size=X/Y "
        "(Z% full)' log line and raise this if fill rate is consistently low under real load.",
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
