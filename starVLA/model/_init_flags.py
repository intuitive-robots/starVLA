# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Process-wide switch for skipping redundant weight initialisation at inference.

``baseframework.from_pretrained`` loads the fine-tuned checkpoint with
``strict=True``. Strict means the checkpoint's key set *equals* the model's key
set, so every parameter and persistent buffer in the model is overwritten by the
checkpoint. Weights read while *constructing* the model are therefore discarded
without exception -- not usually, not mostly: by construction.

At inference that construction-time reading is pure startup latency. The largest
item is the enc-dec pretrain checkpoint (``qwenvl.encdec_ckpt``): 7.2 GB and 934
``load_state_dict`` copies, paid once per policy server, i.e. once per GPU. On a
4-GPU eval node that is ~29 GB of reads that change nothing.

Empirically confirmed on every enc-dec checkpoint we hold (gr00t_enc_v2,
piv4_enc_v2, lit_stage2): all 934 pretrain tensors are present in the fine-tuned
checkpoint, 0 uncovered.

TRAINING IS UNTOUCHED. There the initialisation *is* the starting point and the
flag is never set -- it is armed only by ``from_pretrained(is_inference=True)``,
and only for the duration of model construction.

Escape hatch: ``STARVLA_FULL_INIT=1`` forces the full initialisation back on, for
bisecting a suspected load-path difference.
"""

import os

_SKIP_REDUNDANT_INIT = False


def set_skip_redundant_init(value: bool) -> None:
    """Arm/disarm the switch. Called only by ``baseframework.from_pretrained``."""
    global _SKIP_REDUNDANT_INIT
    _SKIP_REDUNDANT_INIT = bool(value)


def skip_redundant_init() -> bool:
    """True when construction-time weight loading may be skipped."""
    if os.environ.get("STARVLA_FULL_INIT", "").strip() == "1":
        return False
    return _SKIP_REDUNDANT_INIT
