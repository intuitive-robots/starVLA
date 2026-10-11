#!/usr/bin/env python3
"""Time breakdown of a RoboTwin evaluation: runs the image's script/eval_policy.py main() unchanged, with timers
wrapped around the task env's setup_demo / play_once / get_obs / take_action / close_env and the policy client's
predict_action. Same CLI as eval_policy.py (eval.sh's arguments). Prints [PROFILE] totals at exit."""
import atexit
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "script"))
sys.path.insert(0, os.getcwd())
import eval_policy as ep  # noqa: E402

T = collections.defaultdict(float)
N = collections.defaultdict(int)


def timed(name, fn):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] += time.perf_counter() - t0
            N[name] += 1
    return wrapper


_orig_class_decorator = ep.class_decorator


def class_decorator(task_name):
    env = _orig_class_decorator(task_name)
    for name in ("setup_demo", "play_once", "get_obs", "take_action", "close_env", "check_success"):
        setattr(env, name, timed(name, getattr(env, name)))
    return env


ep.class_decorator = class_decorator

from deployment.model_server.tools import websocket_policy_client as wpc  # noqa: E402
wpc.WebsocketClientPolicy.predict_action = timed("predict_action", wpc.WebsocketClientPolicy.predict_action)
T0 = time.perf_counter()


@atexit.register
def report():
    wall = time.perf_counter() - T0
    parts = " ".join(f"{k}={T[k]:.1f}s/{N[k]}" for k in sorted(T))
    print(f"\n[PROFILE] skip={os.environ.get('ROBOTWIN_SKIP_SUBSTEP_RENDER', '0')} wall={wall:.1f}s {parts}", flush=True)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    ep.main(ep.parse_args_and_config())
