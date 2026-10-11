# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Closed-loop image-token transplant (insight analysis; opt-in, off by default).

Enabled only when ``STARVLA_TRANSPLANT_IMAGE_FROM`` is set. For every policy query the
server then runs two backbone passes on the same observation:

1. a *donor* pass, identical except that the instruction is the donor instruction, which
   captures the hidden states of the image tokens (both cameras, the CAM0/CAM1 groups of
   ``attn_probe.TokenLabeler``, exactly as ``analysis/cam_patch.py`` labels them) at the
   output of backbone layer ``STARVLA_TRANSPLANT_LAYER`` (default 26; same 0-based layer
   list and convention as cam_patch.py);
2. the *main* pass with the policy's own instruction, in which those image-token hidden
   states are replaced by the donor's at that layer. Everything else (instruction tokens,
   state tokens, markers) stays the policy's own. Its actions are returned.

Both passes use the same flow-matching noise: each example gets ``flow_seed`` derived from the
client's stable ``_state_corrupt_key`` (``suite|task|episode|step``) and the global torch RNG is
reseeded before each pass (for heads that ignore ``flow_seed``), so the transplant is the only
difference between the main pass and an untransplanted pass with the same seed.

Donor spec (``STARVLA_TRANSPLANT_IMAGE_FROM``):
  ``permute_goal``  donor = the next LIBERO-Goal task of the policy's own instruction
                    (the ``STARVLA_PERMUTE_GOAL_INSTRUCTION`` derangement);
  ``env``           donor = the client's (environment's) original instruction;
  ``self``          donor = the policy's own instruction (must be a no-op: check);
  ``none``          no donor pass, no patch, but the same seeding (seeded clean control);
  ``{json}``        explicit mapping {own instruction: donor instruction} (normalised keys).

Retention (env A, own A, donor B=next(A)): ``STARVLA_TRANSPLANT_IMAGE_FROM=permute_goal``.
Override (env B, own A=prev(B), donor B):  ``STARVLA_PERMUTE_GOAL_INSTRUCTION=prev`` plus
``STARVLA_TRANSPLANT_IMAGE_FROM=env``. Both conditions use the same (A, next(A)) pairs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


def norm_instruction(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").strip().rstrip(".").split())


def stable_seed(key: str) -> int:
    digest = hashlib.blake2b(f"transplant|{key}".encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


class ImageTransplant:
    def __init__(self, framework, spec: str, layer: int, goal_next: Dict[str, str], probe=None):
        from starVLA.model.modules import attn_probe as ap

        self.framework = framework
        self.spec_raw = spec.strip()
        self.mapping: Optional[Dict[str, str]] = None
        if self.spec_raw.startswith("{"):
            self.mapping = {norm_instruction(k): str(v) for k, v in json.loads(self.spec_raw).items()}
            self.mode = "json"
        else:
            self.mode = self.spec_raw.lower()
            if self.mode not in {"permute_goal", "env", "self", "none"}:
                raise ValueError(f"STARVLA_TRANSPLANT_IMAGE_FROM={spec!r}: expected permute_goal|env|self|none|<json>")
        self.goal_next = goal_next
        self.layer = int(layer)
        # Token groups only (native attention kernels) unless a knockout/parity probe already exists.
        self.probe = probe if probe is not None else ap.AttnProbe(framework, knockout=None, patch_attention=False)
        iface = framework.qwen_vl_interface
        if hasattr(iface, "_encoder_blocks") and hasattr(iface, "_text_model"):
            layers = list(iface._encoder_blocks(iface._text_model()))
        else:
            layers = list(iface.model.model.language_model.layers)
        assert [l.self_attn for l in layers] == list(self.probe.modules)
        if not 0 <= self.layer < len(layers):
            raise ValueError(f"STARVLA_TRANSPLANT_LAYER={self.layer} outside [0, {len(layers)})")
        self.num_layers = len(layers)
        self.ids = (ap.CAM0, ap.CAM1)
        self.ctl = {"cap": None, "patch": None, "stats": None}
        layers[self.layer].register_forward_hook(self._hook)
        self.calls = 0
        self.unmapped = 0
        logging.warning(
            "IMAGE TRANSPLANT ACTIVE spec=%r layer=%d/%d (0-based; last=%d) groups=CAM0,CAM1 probe_patches_attention=%s",
            self.spec_raw, self.layer, self.num_layers, self.num_layers - 1, self.probe.patch_attention,
        )

    # -- forward hook on the transplant layer
    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        g = self.probe.ctx["groups"]
        t = torch.tensor(self.ids, device=h.device)
        if self.ctl["cap"] is not None:
            self.ctl["cap"]["h"] = h.detach().clone()
            self.ctl["cap"]["g"] = g.clone()
            return out
        if self.ctl["patch"] is not None:
            src, gsrc = self.ctl["patch"]["h"], self.ctl["patch"]["g"]
            sel, ssel = torch.isin(g, t), torch.isin(gsrc, t)
            if h.shape[0] != src.shape[0] or not torch.equal(sel.sum(1), ssel.sum(1)):
                raise RuntimeError(
                    f"image transplant: image-token counts differ per row {sel.sum(1).tolist()} vs {ssel.sum(1).tolist()}"
                )
            new = h.clone()
            donor_vals = src[ssel].to(h.dtype)
            own_vals = h[sel]
            new[sel] = donor_vals
            diff = (donor_vals.float() - own_vals.float()).norm().item()
            ref = own_vals.float().norm().item()
            self.ctl["stats"] = {"n_img_tokens": sel.sum(1).tolist(), "rel_delta": diff / max(ref, 1e-12)}
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return out

    def donor_lang(self, own: str, env: str) -> str:
        key = norm_instruction(own)
        if self.mode == "self":
            return own
        if self.mode == "env":
            return env
        if self.mode == "permute_goal":
            if key in self.goal_next:
                return self.goal_next[key]
        elif self.mode == "json":
            if key in self.mapping:
                return self.mapping[key]
        self.unmapped += 1
        return own

    def _seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def predict(self, predict_fn: Callable, raw_examples: List[dict], prepared: List[dict], **kwargs) -> dict:
        own = []
        seeds = []
        for raw, ex in zip(raw_examples, prepared):
            ex = dict(ex)
            key = str(raw.get("_state_corrupt_key", ""))
            if "flow_seed" not in ex:
                if not key:
                    raise ValueError("image transplant needs the client's _state_corrupt_key to seed the flow noise")
                ex["flow_seed"] = stable_seed(key)
            seeds.append(int(ex["flow_seed"]))
            own.append(ex)
        batch_seed = stable_seed("|".join(str(s) for s in seeds))
        donor_langs = [self.donor_lang(str(ex.get("lang", "")), str(raw.get("lang", ""))) for raw, ex in zip(raw_examples, own)]
        stats = None
        if self.mode != "none":
            donor = [dict(ex, lang=dl) for ex, dl in zip(own, donor_langs)]
            self.ctl["cap"] = {}
            try:
                self._seed(batch_seed)
                predict_fn(examples=donor, **kwargs)
                cap = self.ctl["cap"]
            finally:
                self.ctl["cap"] = None
            if "h" not in cap:
                raise RuntimeError("image transplant: donor pass did not reach the transplant layer")
            self.ctl["patch"] = cap
        try:
            self._seed(batch_seed)
            out = predict_fn(examples=own, **kwargs)
            stats = self.ctl["stats"]
        finally:
            self.ctl["patch"] = None
            self.ctl["stats"] = None
        if self.calls < 3 or self.calls % 500 == 0:
            logging.warning(
                "__TRANSPLANT__ call=%d mode=%s own=%r donor=%r env=%r seeds=%s stats=%s unmapped=%d",
                self.calls, self.mode, str(own[0].get("lang", "")), donor_langs[0],
                str(raw_examples[0].get("lang", "")), seeds[:4], stats, self.unmapped,
            )
        self.calls += 1
        return out


def attach_transplant_from_env(framework, goal_next: Dict[str, str], probe=None) -> Optional[ImageTransplant]:
    spec = os.environ.get("STARVLA_TRANSPLANT_IMAGE_FROM", "").strip()
    if not spec:
        return None
    layer = int(os.environ.get("STARVLA_TRANSPLANT_LAYER", "26"))
    return ImageTransplant(framework, spec, layer, goal_next, probe=probe)
