# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from .batch_dispatcher import BatchDispatcher


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_batch_size: int = 1,
        max_wait_time: float = 1.0,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # Opt-in: max_batch_size=1 (default) preserves the exact previous
        # behavior (call predict_action inline per connection). max_batch_size>1
        # routes requests through a BatchDispatcher that collects concurrent
        # requests from multiple connections into ONE predict_action(examples=[...])
        # call -- see batch_dispatcher.py for why this is a large win for this
        # model (measured ~30x throughput at batch=32, ~112x at batch=128, for a
        # ~15% per-batch latency cost) rather than one-request-at-a-time serving.
        self._dispatcher = BatchDispatcher(policy, max_batch_size, max_wait_time) if max_batch_size > 1 else None
        if self._dispatcher is not None:
            logging.info(
                f"[WebsocketPolicyServer] batching ON: max_batch_size={max_batch_size}, "
                f"max_wait_time={max_wait_time}s"
            )

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # The default 20s ping_interval/ping_timeout assumes a responsive event
            # loop, but `_handler` runs `policy.predict_action` SYNCHRONOUSLY inline
            # (GPU inference blocks the loop -- it can't answer keepalive pings on
            # ANY connection while a request is in flight). Under GPU contention
            # (e.g. multiple policy servers sharing one GPU) a single inference call
            # can legitimately take 30-50s+, which blows past a 20s ping_timeout and
            # gets otherwise-healthy connections killed with "keepalive ping
            # timeout". 300s gives real headroom without masking an actually-dead
            # server for long.
            ping_interval=300,
            ping_timeout=300,
        ) as server:
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                ret = await self._route_message(msg)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    async def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                if self._dispatcher is not None:
                    output_dict = await self._dispatcher.submit(payload)
                else:
                    # max_batch_size=1: unchanged from before -- call inline, blocking
                    # the event loop for the duration of this one request.
                    output_dict = await asyncio.to_thread(self._policy.predict_action, **payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
