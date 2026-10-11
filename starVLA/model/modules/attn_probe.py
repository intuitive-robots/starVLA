"""Attention probe for Qwen3-VL policy backbones: measurement and knockout.

One dense re-implementation of every attention topology a starVLA Qwen3-VL policy
can run with:

    causal      the stock decoder stack (``attention_topology: causal``)
    bidir       the converted/copied encoder stack (``enc_dec: true``) and the
                single-stack ``full_prefix_bidir`` topology
    modblock    ``modality_block_causal``: bidirectional within the camera,
                language and state blocks, causal between blocks

Installing the probe replaces each text self-attention ``forward`` with a version
that computes q/k/v exactly as ``Qwen3VLTextAttention`` does, builds the
topology's boolean mask densely, optionally removes (query group -> key group)
pairs, and runs either SDPA (fast path) or an fp32 softmax whose probabilities are
reduced to per-token-group statistics (record path).

Removing pairs *before* the softmax is exactly "zero the attention and
re-normalise": every surviving key keeps its relative weight. A query always keeps
its own key, so no row can become empty.

Uniform-aim interventions (``uni_<preset>``, e.g. ``uni_instr2img``): the (query group ->
key group) pairs are not removed. After the fp32 softmax, for every head and every query of
the source group, the total probability M on the (visible) keys of the target group is
spread evenly over those keys (each gets M / n). Every other key keeps its weight, so each
row still sums to 1 and the channel's mass is preserved while its aim is removed. These
layers run the eager fp32 path (``eager`` alone runs that path with no change: its
numerical control).

Token groups are assigned per prompt from the token ids alone (image tokens and
their camera index from the vision delimiters; state tokens from the splice mask
or, for text-mode state, from the ``[STATE] ... [ACTION]`` span; the instruction
from its character span in the decoded text region). Nothing here depends on the
framework beyond ``qwen_vl_interface``.

Environment hooks (read by ``deployment/model_server/policy_wrapper.py``):
    STARVLA_ATTN_KNOCKOUT   preset name(s) from ``KNOCKOUT_PRESETS`` joined by
                            '+', or explicit ``qset>kset`` pairs joined by ','.
                            Opt-in extensions (``parse_spec``; plain specs are unchanged):
                            ``uni_<preset>`` uniform aim at preserved mass (see below),
                            a ``_L<a>-<b>`` suffix restricts the whole spec to layers
                            a..b inclusive, ``eager`` forces the fp32 eager path (no-op).
    STARVLA_ATTN_PROBE=1    install the dense probe with no knockout (parity runs)
    STARVLA_HEAD_KEYS       comma list of GROUP_SETS names the action head may read
                            (e.g. ``image``); native backbone attention unless combined
    STARVLA_OBJ_BIAS        E4/E5 aim bias (default off; ``parse_obj_bias``): instruction
                            queries -> a per-request image-key region (task object, background or
                            a same-area random region from the client's ``_obj_tok``), weight
                            e^b, mass preserved per camera; see ``aim_bias``. Installed by
                            policy_wrapper.py; nothing here changes when it is unset.
"""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from types import MethodType
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PAD, OTHER, CAM0, CAM1, INSTR, STATE, MARKER = -1, 0, 1, 2, 3, 4, 5
GROUP_NAMES = ("other", "cam0", "cam1", "instr", "state", "marker")
NUM_GROUPS = len(GROUP_NAMES)

GROUP_SETS: Dict[str, Tuple[int, ...]] = {
    "image": (CAM0, CAM1),
    "cam0": (CAM0,),
    "cam1": (CAM1,),
    "instr": (INSTR,),
    "state": (STATE,),
    "text": (INSTR, STATE),
    "marker": (MARKER,),
    "other": (OTHER,),
}

KNOCKOUT_PRESETS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "img2instr": (("image", "instr"),),
    "img2state": (("image", "state"),),
    "img2text": (("image", "instr"), ("image", "state")),
    "instr2img": (("instr", "image"),),
    "state2img": (("state", "image"),),
    "text2img": (("instr", "image"), ("state", "image")),
    # Relay-closed variants: every non-image group, so the instruction cannot reach the image
    # tokens (or vice versa) through template/marker/state tokens in later layers.
    "img2nonimg": (("image", "instr"), ("image", "state"), ("image", "other"), ("image", "marker")),
    "nonimg2img": (("instr", "image"), ("state", "image"), ("other", "image"), ("marker", "image")),
}

STATE_MARKER = "[STATE]"
ACTION_MARKER = "[ACTION]"
_STATE_SUFFIX_RE = re.compile(r"\s*\[STATE\].*?\[ACTION\]\s*$", re.S)


def parse_knockout(spec: Optional[str]) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """``"img2instr+state2img"`` or ``"image>instr,state>image"`` -> group-id pairs."""
    if spec is None or not str(spec).strip():
        return ()
    pairs: List[Tuple[str, str]] = []
    for part in str(spec).replace("+", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part in KNOCKOUT_PRESETS:
            pairs.extend(KNOCKOUT_PRESETS[part])
        elif ">" in part:
            q, k = (s.strip() for s in part.split(">", 1))
            pairs.append((q, k))
        else:
            raise ValueError(
                f"unknown knockout {part!r}; presets {sorted(KNOCKOUT_PRESETS)} or 'qset>kset' "
                f"with sets {sorted(GROUP_SETS)}"
            )
    out = []
    for q, k in pairs:
        if q not in GROUP_SETS or k not in GROUP_SETS:
            raise ValueError(f"knockout pair {q}>{k}: sets must be in {sorted(GROUP_SETS)}")
        out.append((GROUP_SETS[q], GROUP_SETS[k]))
    return tuple(out)


_LAYER_SUFFIX_RE = re.compile(r"^(.*)_L(\d+)-(\d+)$")


def parse_spec(spec: Optional[str]):
    """Split a knockout spec into (hard pairs, uniform-aim pairs, layer set or None, eager).

    Plain specs (no ``uni_`` / ``eager`` part and no ``_L<a>-<b>`` suffix) give exactly
    ``(parse_knockout(spec), (), None, False)``. A layer suffix may appear on any part and
    must be the same on all parts that carry one (one layer range per probe).
    """
    if spec is None or not str(spec).strip():
        return (), (), None, False
    parts = [p.strip() for p in str(spec).replace("+", ",").split(",") if p.strip()]
    if not any(p.startswith("uni_") or p == "eager" or _LAYER_SUFFIX_RE.match(p) for p in parts):
        return parse_knockout(spec), (), None, False
    hard, uni, ranges, eager = [], [], set(), False
    for p in parts:
        m = _LAYER_SUFFIX_RE.match(p)
        if m:
            p = m.group(1)
            lo, hi = int(m.group(2)), int(m.group(3))
            if hi < lo:
                raise ValueError(f"layer range L{lo}-{hi} is empty")
            ranges.add((lo, hi))
        if p == "eager":
            eager = True
        elif p.startswith("uni_"):
            uni.append(p[len("uni_"):])
        else:
            hard.append(p)
    if len(ranges) > 1:
        raise ValueError(f"one layer range per spec, got {sorted(ranges)}")
    layers = None
    if ranges:
        lo, hi = ranges.pop()
        layers = set(range(lo, hi + 1))
    return (parse_knockout(",".join(hard)) if hard else (),
            parse_knockout(",".join(uni)) if uni else (), layers, eager)


def uniform_aim(probs: torch.Tensor, allowed: torch.Tensor, pairs) -> torch.Tensor:
    """Spread each source query's mass on the target keys evenly over them.

    probs [B, H, L, L] (post-softmax), allowed [B, 1, L, L], pairs: [(qmask [B, L], kmask [B, L])].
    """
    for qm, km in pairs:
        sel = allowed & qm[:, None, :, None] & km[:, None, None, :]  # [B, 1, L, L]
        n = sel.sum(-1, keepdim=True).to(probs.dtype)
        mass = (probs * sel).sum(-1, keepdim=True)
        probs = torch.where(sel, mass / n.clamp(min=1), probs)
    return probs


def aim_bias(probs: torch.Tensor, allowed: torch.Tensor, kb: dict) -> torch.Tensor:
    """Re-aim instruction-query attention inside each camera toward a key region, mass preserved.

    For every instruction query and camera c, the probabilities on c's image keys are multiplied by
    e^b on the region keys and renormalised to the camera's original total, so only the aim within
    the camera changes (the instruction's total mass on each camera, and on every other key, is
    untouched). b = 0 is the identity. probs [B, H, L, L]; kb from ``AttnProbe._prepare``.
    """
    qm = kb["q"]  # [B, L]
    w = math.exp(float(kb["b"]))
    for cam_m, reg_m in kb["cams"]:  # [B, L] bool each
        sel = allowed & qm[:, None, :, None] & cam_m[:, None, None, :]  # [B, 1, L, L]
        mass = (probs * sel).sum(-1, keepdim=True)
        pw = torch.where(sel & reg_m[:, None, None, :], probs * w, probs * sel)
        tot = pw.sum(-1, keepdim=True)
        probs = torch.where(sel, pw * (mass / tot.clamp(min=1e-30)), probs)
    return probs


def parse_obj_bias(spec: Optional[str]) -> Optional[dict]:
    """``"b=2,region=object,layers=0-12,tau=0.25,seed=0"`` -> config dict (None if unset).

    region: ``object`` (task-object tokens), ``background`` (the camera's other tokens) or
    ``random`` (same-area random region, token_regions.random_region, seeded per episode).
    layers: ``a-b`` inclusive, default all probed layers.
    """
    if spec is None or not str(spec).strip():
        return None
    cfg = {"b": None, "region": "object", "layers": None, "tau": 0.25, "seed": 0}
    for part in str(spec).split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k == "b":
            cfg["b"] = float(v)
        elif k == "region":
            if v not in ("object", "background", "random"):
                raise ValueError(f"STARVLA_OBJ_BIAS region must be object|background|random, got {v!r}")
            cfg["region"] = v
        elif k == "layers":
            lo, hi = (int(x) for x in v.split("-"))
            if hi < lo:
                raise ValueError(f"STARVLA_OBJ_BIAS layers {v} is empty")
            cfg["layers"] = set(range(lo, hi + 1))
        elif k == "tau":
            cfg["tau"] = float(v)
        elif k == "seed":
            cfg["seed"] = int(v)
        else:
            raise ValueError(f"unknown STARVLA_OBJ_BIAS field {k!r}")
    if cfg["b"] is None:
        raise ValueError("STARVLA_OBJ_BIAS needs b=<float>")
    return cfg


def strip_state_suffix(text: str) -> str:
    """Instruction as the user wrote it, without the appended state block."""
    return _STATE_SUFFIX_RE.sub("", str(text)).strip()


# --------------------------------------------------------------------------- labels
class TokenLabeler:
    """Assign every prompt token to one of ``GROUP_NAMES`` (``PAD`` for padding)."""

    def __init__(self, tokenizer, image_token_id: int, vision_start_id: int, vision_end_id: int):
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.vision_start_id = int(vision_start_id)
        self.vision_end_id = int(vision_end_id)
        self._pieces: "OrderedDict[int, str]" = OrderedDict()
        self.fallbacks = 0

    def piece(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._pieces.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            self._pieces[token_id] = cached
            if len(self._pieces) > 200_000:
                self._pieces.popitem(last=False)
        return cached

    def _label_pre_image_text(
        self,
        ids: Sequence[int],
        valid: Sequence[bool],
        groups: List[int],
        first_vision_start: int,
        instruction: str,
        has_state_mask: bool,
    ) -> bool:
        """Label instruction / markers / text-mode state in the text before the first image.

        Returns whether the instruction was found there. Only exact (or whitespace-squeezed)
        matches are labelled; there is deliberately no fallback in this region, so chat
        boilerplate ("<|im_start|>user\\n") never becomes INSTR.
        """
        region = [i for i in range(first_vision_start) if valid[i]]
        starts, ends, chunks, pos = [], [], [], 0
        for i in region:
            p = self.piece(ids[i])
            starts.append(pos)
            pos += len(p)
            ends.append(pos)
            chunks.append(p)
        text = "".join(chunks)

        def mark(lo: int, hi: int, group: int) -> None:
            for j, i in enumerate(region):
                if starts[j] < hi and ends[j] > lo and groups[i] == OTHER:
                    groups[i] = group

        state_at = text.find(STATE_MARKER)
        instr_end = state_at if state_at >= 0 else len(text)
        if state_at >= 0:
            mark(state_at, state_at + len(STATE_MARKER), MARKER)
            action_at = text.find(ACTION_MARKER, state_at)
            if action_at >= 0:
                mark(action_at, action_at + len(ACTION_MARKER), MARKER)
                if not has_state_mask:
                    mark(state_at + len(STATE_MARKER), action_at, STATE)
        if not instruction:
            return False
        lo = text.rfind(instruction, 0, instr_end + 1)
        if lo < 0:
            instruction = " ".join(instruction.split())
            lo = text.rfind(instruction, 0, instr_end + 1)
        if lo < 0:
            return False
        mark(lo, lo + len(instruction), INSTR)
        return True

    def label_row(
        self,
        ids: Sequence[int],
        valid: Sequence[bool],
        state_mask: Optional[Sequence[bool]] = None,
        instruction: Optional[str] = None,
    ) -> List[int]:
        n = len(ids)
        groups = [OTHER if valid[i] else PAD for i in range(n)]
        cam = -1
        last_vision_end = -1
        first_vision_start = -1
        for i in range(n):
            if not valid[i]:
                continue
            t = int(ids[i])
            if t == self.vision_start_id:
                cam += 1
                if first_vision_start < 0:
                    first_vision_start = i
            elif t == self.vision_end_id:
                last_vision_end = i
            elif t == self.image_token_id:
                groups[i] = CAM0 if cam <= 0 else CAM1
        has_state_mask = state_mask is not None and any(bool(x) for x in state_mask)
        if has_state_mask:
            for i in range(n):
                if valid[i] and state_mask[i]:
                    groups[i] = STATE

        # Text region: everything after the last image, as one decoded string whose
        # character offsets are built from the same per-token pieces.
        region = [i for i in range(last_vision_end + 1, n) if valid[i]]
        starts, ends, chunks, pos = [], [], [], 0
        for i in region:
            p = self.piece(ids[i])
            starts.append(pos)
            pos += len(p)
            ends.append(pos)
            chunks.append(p)
        text = "".join(chunks)

        def mark(lo: int, hi: int, group: int, only_other: bool = True) -> int:
            hits = 0
            for j, i in enumerate(region):
                if starts[j] < hi and ends[j] > lo and (not only_other or groups[i] == OTHER):
                    groups[i] = group
                    hits += 1
            return hits

        # Text placed BEFORE the first image (framework.qwenvl.prompt_order text_first /
        # sandwich). With the default image_first layout this region is only
        # "<|im_start|>user\n", which never contains the instruction, so nothing changes.
        pre_found = False
        if instruction and first_vision_start > 0:
            pre_found = self._label_pre_image_text(
                ids, valid, groups, first_vision_start, strip_state_suffix(instruction),
                has_state_mask,
            )

        state_at = text.find(STATE_MARKER)
        instr_end = state_at if state_at >= 0 else len(text)
        if state_at >= 0:
            mark(state_at, state_at + len(STATE_MARKER), MARKER)
            action_at = text.find(ACTION_MARKER, state_at)
            if action_at >= 0:
                mark(action_at, action_at + len(ACTION_MARKER), MARKER)
                if not has_state_mask:
                    # Text-mode state: the numbers between the markers are the state.
                    mark(state_at + len(STATE_MARKER), action_at, STATE)
        if instruction:
            instruction = strip_state_suffix(instruction)
            lo = text.rfind(instruction, 0, instr_end + 1) if instruction else -1
            if lo < 0:
                # Whitespace normalisation by the chat template: match on a squeezed copy.
                squeezed = " ".join(instruction.split())
                lo = text.rfind(squeezed, 0, instr_end + 1)
                instruction = squeezed
            if lo >= 0:
                mark(lo, lo + len(instruction), INSTR)
            elif pre_found:
                # text_first: the instruction lives only before the images.
                return groups
            else:
                self.fallbacks += 1
                instruction = None
        if not instruction:
            # Fallback: the user text between the last image and the state block (or
            # the end of the user turn), minus special tokens.
            user_end = text.find("<|im_end|>")
            hi = instr_end if state_at >= 0 else (user_end if user_end >= 0 else len(text))
            for j, i in enumerate(region):
                if starts[j] < hi and groups[i] == OTHER and not self.piece(ids[i]).startswith("<|"):
                    groups[i] = INSTR

        return groups


def group_counts(groups: torch.Tensor) -> Dict[str, int]:
    return {name: int((groups == g).sum()) for g, name in enumerate(GROUP_NAMES)}


# --------------------------------------------------------------------------- masks
def build_allowed(
    topology: str,
    valid: torch.Tensor,
    groups: torch.Tensor,
    block_ids: Optional[torch.Tensor] = None,
    knockout: Sequence[Tuple[Sequence[int], Sequence[int]]] = (),
) -> torch.Tensor:
    """Dense ``[B, 1, L, L]`` boolean mask (True = key visible to query)."""
    valid = valid.bool()
    B, L = valid.shape
    ar = torch.arange(L, device=valid.device)
    if topology == "causal":
        allowed = (ar[None, :, None] >= ar[None, None, :]) & valid[:, None, :]
    elif topology == "bidir":
        allowed = valid[:, None, :].expand(B, L, L).clone()
    elif topology == "modblock":
        if block_ids is None:
            raise ValueError("modblock topology needs block_ids")
        bq, bk = block_ids[:, :, None], block_ids[:, None, :]
        allowed = (bk <= bq) & (bk >= 0) & (bq >= 0)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    for qset, kset in knockout:
        qm = torch.isin(groups, torch.as_tensor(list(qset), device=groups.device))
        km = torch.isin(groups, torch.as_tensor(list(kset), device=groups.device))
        allowed &= ~(qm[:, :, None] & km[:, None, :])
    allowed |= torch.eye(L, dtype=torch.bool, device=valid.device)[None]
    return allowed[:, None]


# --------------------------------------------------------------------------- forward
def _apply_rotary(q, k, cos, sin):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    return apply_rotary_pos_emb(q, k, cos, sin)


def _probe_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Drop-in for ``Qwen3VLTextAttention.forward`` (and the packed FA3 variants)."""
    probe: AttnProbe = self._starvla_probe
    ctx = probe.ctx
    if ctx is None or ctx["allowed"].shape[-1] != hidden_states.shape[1]:
        raise RuntimeError(
            "attention probe context missing or stale: the probe supports full-prompt "
            "forwards through qwen_vl_interface only"
        )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = _apply_rotary(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)
        if k.shape[2] != q.shape[2]:
            raise RuntimeError("attention probe does not support incremental decoding")
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    layer = self._starvla_probe_layer
    kl = getattr(probe, "knockout_layers", None)
    allowed = ctx["allowed"] if (kl is None or layer in kl) else ctx["allowed_base"]
    uni = ctx.get("uniform") if (kl is None or layer in kl) else None
    kb = ctx.get("kbias")
    if kb is not None and not (kb["layers"] is None or layer in kb["layers"]):
        kb = None
    eager = bool(uni) or (getattr(probe, "force_eager", False) and (kl is None or layer in kl)) or kb is not None
    if eager and not (probe.record and (probe.record_layers is None or layer in probe.record_layers)):
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        probs = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        if uni:
            probs = uniform_aim(probs, allowed, uni)
        if kb is not None:
            probs = aim_bias(probs, allowed, kb)
        out = torch.matmul(probs.to(v.dtype), v)
    elif probe.record and (probe.record_layers is None or layer in probe.record_layers):
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        probs = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        if uni:
            probs = uniform_aim(probs, allowed, uni)
        if kb is not None:
            probs = aim_bias(probs, allowed, kb)
        probe._record(layer, probs)
        out = torch.matmul(probs.to(v.dtype), v)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0, scale=self.scaling)
    out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


# --------------------------------------------------------------------------- probe
class AttnProbe:
    """Install on a starVLA framework; see the module docstring.

    ``records`` (record mode) holds one dict per batch row of the most recent VLM
    forward: ``groups`` (list[int]) and per recorded layer ``mass`` ([G, G] mean
    attention mass from query group to key group, heads averaged), ``nq`` (queries per
    group), ``ent_instr``/``ent_state`` (normalised entropy of image-query attention
    restricted to instruction/state keys), and ``instr_profile`` (image-query mass per
    instruction token).
    """

    def __init__(
        self,
        framework,
        knockout: Optional[str] = None,
        record: bool = False,
        record_layers: Optional[Iterable[int]] = None,
        topology: Optional[str] = None,
        patch_attention: bool = True,
    ):
        self.framework = framework
        self.patch_attention = bool(patch_attention)  # False: token groups only, native attention
        self.iface = framework.qwen_vl_interface
        self.knockout_spec = knockout or ""
        self.knockout, self.uniform, _layers, self.force_eager = parse_spec(knockout)
        if _layers is not None:
            self.knockout_layers = _layers  # only set by an explicit _L<a>-<b> suffix
        self.record = bool(record)
        self.record_layers = None if record_layers is None else set(int(x) for x in record_layers)
        auto_topology, modules = self._discover()
        self.topology = topology or auto_topology
        self.modules = modules
        self.ctx: Optional[dict] = None
        self.records: List[dict] = []
        self.last_instructions: Optional[List[str]] = None
        self.forwards = 0
        # E4/E5 aim bias (STARVLA_OBJ_BIAS; None = off): {"b", "layers", "rows": [{CAM0: idx, CAM1: idx}]}
        # with within-camera row-major token indices per batch row, set before each forward.
        self.key_bias: Optional[dict] = None
        cfg = self.iface.model.config
        self.labeler = TokenLabeler(
            self.iface.processor.tokenizer,
            getattr(cfg, "image_token_id", 151655),
            cfg.vision_start_token_id,
            cfg.vision_end_token_id,
        )
        self._saved = []
        self._install()

    # ---- discovery / install
    def _discover(self):
        iface = self.iface
        if hasattr(iface, "_encoder_blocks") and hasattr(iface, "_text_model"):
            if not getattr(iface, "skip_decoder", True):
                logger.warning("attention probe patches encoder blocks only; the decoder is untouched")
            blocks = list(iface._encoder_blocks(iface._text_model()))
            return "bidir", [b.self_attn for b in blocks]
        topo = str(getattr(iface, "attention_topology", "causal"))
        mapping = {"causal": "causal", "modality_block_causal": "modblock", "full_prefix_bidir": "bidir"}
        if topo not in mapping:
            raise NotImplementedError(f"attention probe: unsupported topology {topo!r}")
        layers = iface.model.model.language_model.layers
        return mapping[topo], [layer.self_attn for layer in layers]

    def _install(self):
        for idx, mod in enumerate(self.modules if self.patch_attention else ()):
            self._saved.append((mod, mod.__dict__.get("forward")))
            mod._starvla_probe = self
            mod._starvla_probe_layer = idx
            mod.forward = MethodType(_probe_attention_forward, mod)
        iface = self.iface
        self._orig_build = iface.__dict__.get("build_qwenvl_inputs")
        self._orig_forward = iface.__dict__.get("forward")
        build = iface.build_qwenvl_inputs
        fwd = iface.forward

        def build_qwenvl_inputs(*args, **kwargs):
            instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
            self.last_instructions = list(instructions) if instructions is not None else None
            return build(*args, **kwargs)

        def forward(**kwargs):
            self._prepare(kwargs)
            return fwd(**kwargs)

        iface.build_qwenvl_inputs = build_qwenvl_inputs
        iface.forward = forward

    def remove(self):
        for mod, saved in self._saved:
            if saved is None:
                mod.__dict__.pop("forward", None)
            else:
                mod.forward = saved
            for attr in ("_starvla_probe", "_starvla_probe_layer"):
                mod.__dict__.pop(attr, None)
        for name, saved in (("build_qwenvl_inputs", self._orig_build), ("forward", self._orig_forward)):
            if saved is None:
                self.iface.__dict__.pop(name, None)
            else:
                setattr(self.iface, name, saved)
        self._saved = []

    # ---- per-forward context
    def _prepare(self, kwargs: dict):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("attention probe needs input_ids")
        B, L = input_ids.shape
        valid = kwargs.get("attention_mask")
        valid = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid.bool()
        state_mask = kwargs.get("_state_token_mask")
        if state_mask is None:
            state_mask = getattr(self.framework, "_last_state_token_mask", None)
        if state_mask is not None and tuple(state_mask.shape) != (B, L):
            state_mask = None
        instr = self.last_instructions if self.last_instructions and len(self.last_instructions) == B else [None] * B
        ids_cpu = input_ids.detach().cpu().tolist()
        valid_cpu = valid.detach().cpu().tolist()
        sm_cpu = state_mask.detach().cpu().tolist() if state_mask is not None else [None] * B
        groups = torch.tensor(
            [self.labeler.label_row(ids_cpu[b], valid_cpu[b], sm_cpu[b], instr[b]) for b in range(B)],
            dtype=torch.long,
            device=input_ids.device,
        )
        block_ids = None
        if self.topology == "modblock":
            from starVLA.model.modules.vlm.QWen3 import build_modality_block_ids

            cfg = self.iface.model.config
            block_ids = build_modality_block_ids(
                input_ids, valid.long(), int(cfg.vision_start_token_id), int(cfg.vision_end_token_id), state_mask
            )
        allowed = build_allowed(self.topology, valid, groups, block_ids, self.knockout)
        # Optional layer restriction (knockout_layers): the other layers use the unmodified mask.
        allowed_base = (build_allowed(self.topology, valid, groups, block_ids, ())
                        if getattr(self, "knockout_layers", None) is not None else allowed)
        onehot = F.one_hot(groups.clamp(min=0), NUM_GROUPS).float() * (groups >= 0)[..., None].float()
        self.ctx = {"allowed": allowed, "allowed_base": allowed_base, "groups": groups, "onehot": onehot,
                    "valid": valid}
        if getattr(self, "uniform", ()):
            self.ctx["uniform"] = [
                (torch.isin(groups, torch.as_tensor(list(qs), device=groups.device)),
                 torch.isin(groups, torch.as_tensor(list(ks), device=groups.device)))
                for qs, ks in self.uniform
            ]
            if self.forwards == 0:
                logger.warning("__ATTN_PROBE__ uniform-aim pairs=%s layers=%s", self.uniform,
                               sorted(self.knockout_layers) if getattr(self, "knockout_layers", None) else "all")
        if self.key_bias is not None:
            self.ctx["kbias"] = self._key_bias_ctx(groups)
        self.records = [
            {"groups": groups[b].tolist(), "input_ids": ids_cpu[b], "layers": {}} for b in range(B)
        ] if self.record else []
        if self.forwards == 0:
            logger.warning(
                "__ATTN_PROBE__ topology=%s modules=%d knockout=%r record=%s row0_groups=%s "
                "knocked_pairs_row0=%d instr_fallbacks=%d",
                self.topology, len(self.modules), self.knockout_spec, self.record,
                group_counts(groups[0]),
                int((~allowed[0, 0] & build_allowed(self.topology, valid[:1], groups[:1], None if block_ids is None else block_ids[:1])[0, 0]).sum()),
                self.labeler.fallbacks,
            )
        self.forwards += 1

    def _key_bias_ctx(self, groups: torch.Tensor) -> dict:
        kb = self.key_bias
        rows = kb["rows"]
        B = groups.shape[0]
        if len(rows) != B:
            raise RuntimeError(f"STARVLA_OBJ_BIAS: {len(rows)} region rows for a batch of {B}")
        cams = []
        for gid in (CAM0, CAM1):
            cam_m = groups == gid
            reg_m = torch.zeros_like(cam_m)
            for b in range(B):
                idx = rows[b].get(gid)
                if idx is None or not len(idx):
                    continue  # object not visible in this camera: no bias there
                pos = torch.nonzero(cam_m[b]).squeeze(-1)
                if len(pos) != 64:
                    raise RuntimeError(f"STARVLA_OBJ_BIAS: camera group {gid} has {len(pos)} tokens, expected 8x8")
                reg_m[b, pos[torch.as_tensor(list(idx), device=pos.device, dtype=torch.long)]] = True
            cams.append((cam_m, reg_m))
        if self.forwards == 0 or not getattr(self, "_kbias_logged", False):
            self._kbias_logged = True
            logger.warning("__ATTN_PROBE__ OBJ BIAS b=%s layers=%s region_tokens_row0=%s", kb["b"],
                           sorted(kb["layers"]) if kb.get("layers") else "all",
                           [int(r[0].sum()) for _, r in [(c, m) for c, m in cams]])
        return {"q": groups == INSTR, "cams": cams, "b": float(kb["b"]), "layers": kb.get("layers")}

    # ---- recording
    @torch.no_grad()
    def _record(self, layer: int, probs: torch.Tensor):
        groups = self.ctx["groups"]
        onehot = self.ctx["onehot"]  # [B, L, G]
        mass = torch.einsum("bhqk,bkg->bqg", probs, onehot) / probs.shape[1]  # heads averaged
        for b, rec in enumerate(self.records):
            g = groups[b]
            m = torch.zeros(NUM_GROUPS, NUM_GROUPS, device=probs.device)
            nq = torch.zeros(NUM_GROUPS, device=probs.device)
            for qg in range(NUM_GROUPS):
                sel = g == qg
                n = int(sel.sum())
                nq[qg] = n
                if n:
                    m[qg] = mass[b, sel].mean(0)
            entry = {"mass": m.cpu().numpy(), "nq": nq.cpu().numpy()}
            img_q = (g == CAM0) | (g == CAM1)
            for name, grp in (("instr", INSTR), ("state", STATE)):
                keys = g == grp
                nk = int(keys.sum())
                if not img_q.any() or nk == 0:
                    entry[f"ent_{name}"] = float("nan")
                    continue
                p = probs[b][:, img_q][:, :, keys]  # [H, Nq, Nk]
                s = p.sum(-1, keepdim=True)
                ok = s.squeeze(-1) > 1e-6
                if not ok.any() or nk < 2:
                    entry[f"ent_{name}"] = float("nan")
                else:
                    pn = p / s.clamp(min=1e-12)
                    ent = -(pn * pn.clamp(min=1e-12).log()).sum(-1) / math.log(nk)
                    entry[f"ent_{name}"] = float(ent[ok].mean())
                if name == "instr":
                    entry["instr_profile"] = p.mean(dim=(0, 1)).cpu().numpy()
            # Instruction queries -> image keys: how concentrated is the grounding? (1 = uniform)
            iq, ik = g == INSTR, img_q
            nk = int(ik.sum())
            if iq.any() and nk > 1:
                p = probs[b][:, iq][:, :, ik]
                s_ = p.sum(-1, keepdim=True)
                ok = s_.squeeze(-1) > 1e-6
                pn = p / s_.clamp(min=1e-12)
                ent = -(pn * pn.clamp(min=1e-12).log()).sum(-1) / math.log(nk)
                entry["ent_instr2img"] = float(ent[ok].mean()) if ok.any() else float("nan")
            else:
                entry["ent_instr2img"] = float("nan")
            # Optional spatial read-outs for attention maps (off by default):
            # key_profiles {name: query group ids} -> mean attention over heads and those
            # queries for every key position; keep_full {layers} -> head-averaged [L, L].
            for pname, qgroups in (getattr(self, "key_profiles", None) or {}).items():
                sel = torch.isin(g, torch.as_tensor(list(qgroups), device=g.device))
                if sel.any():
                    entry[f"kprof_{pname}"] = probs[b][:, sel].mean(dim=(0, 1)).cpu().numpy()
            if layer in (getattr(self, "keep_full", None) or ()):
                entry["full"] = probs[b].mean(0).cpu().numpy()
            rec["layers"][layer] = entry


def restrict_head_keys(framework, probe: "AttnProbe", key_sets: Sequence[str] = ("image",)) -> None:
    """Let the action head read only the listed token groups (``GROUP_SETS`` names).

    Wraps QwenGR00T's ``_dit_valid_mask`` so the DiT cross-attention mask is ANDed with the
    group mask of the current prompt. Under a causal backbone the image tokens precede all
    text, so an image-only head is instruction- and state-blind by construction; under a
    bidirectional encoder it reads whatever the image tokens absorbed from the text.
    """
    if not hasattr(framework, "_dit_valid_mask"):
        raise NotImplementedError("head key restriction implemented for QwenGR00T (_dit_valid_mask) only")
    ids = sorted({g for name in key_sets for g in GROUP_SETS[name]})
    orig = framework._dit_valid_mask

    def _dit_valid_mask(qwen_inputs, last_hidden):
        valid = orig(qwen_inputs, last_hidden)
        groups = probe.ctx["groups"] if probe.ctx is not None else None
        if groups is None or groups.shape != valid.shape:
            raise RuntimeError("head key restriction needs prompt-aligned token groups")
        keep = torch.isin(groups, torch.as_tensor(ids, device=groups.device))
        if not bool(keep.any(dim=1).all()):
            raise RuntimeError("head key restriction left a row with no keys")
        return valid.bool() & keep

    framework._dit_valid_mask = _dit_valid_mask
    framework._starvla_head_keys = tuple(key_sets)


def attach_from_env(framework, env: Optional[dict] = None) -> Optional[AttnProbe]:
    """Server hook: install a knockout / parity probe when the env asks for one."""
    import os

    env = os.environ if env is None else env
    spec = str(env.get("STARVLA_ATTN_KNOCKOUT", "")).strip()
    parity = str(env.get("STARVLA_ATTN_PROBE", "")).strip() == "1"
    head_keys = [k.strip() for k in str(env.get("STARVLA_HEAD_KEYS", "")).split(",") if k.strip()]
    if not spec and not parity and not head_keys:
        return None
    # Head-key restriction alone keeps the native backbone attention (groups only).
    probe = AttnProbe(framework, knockout=spec or None, patch_attention=bool(spec or parity))
    if head_keys:
        restrict_head_keys(framework, probe, head_keys)
        logger.warning("HEAD KEY RESTRICTION ACTIVE keys=%s", head_keys)
    logger.warning(
        "ATTN PROBE ACTIVE topology=%s modules=%d knockout=%r",
        probe.topology, len(probe.modules), spec,
    )
    return probe
