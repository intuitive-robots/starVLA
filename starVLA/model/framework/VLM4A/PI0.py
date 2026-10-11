from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from starVLA.model.modules.vlm.openpi_transformers.gemma import modeling_gemma

from deployment.model_server.tools.image_tools import resize_with_pad, to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.OpenPI_ActionHead import (
    GemmaDims,
    OpenPI0ActionHead,
    OpenPIActionHeadBase,
    make_att_2d_masks,
    make_att_4d_masks,
)
from starVLA.model.modules.pi_recipe import PIRecipeMixin
from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIPaliGemma
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass
class Observation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor


@dataclass
class PI0DefaultConfig:
    name: str = "PI0"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    action_dim: int = 32
    action_horizon: int = 50
    max_state_dim: int = 32
    discrete_state_input: bool = False
    max_token_len: int = 200
    num_inference_steps: int = 10

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))

    tokenizer: dict = field(default_factory=lambda: {"model_path": "", "pad_state_value": -2.0})
    paligemma: dict = field(default_factory=lambda: {"model_name_or_path": ""})
    paligemma_config: dict = field(
        default_factory=lambda: {
            "width": 2048,
            "depth": 18,
            "mlp_dim": 16384,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )
    action_expert_config: dict = field(
        default_factory=lambda: {
            "width": 1024,
            "depth": 18,
            "mlp_dim": 4096,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )

    # ── StarVLA additions (2026-10-05). Defaults reproduce the upstream port exactly. ──
    # train_forward: "joint" = upstream (prefix and action expert in one masked sequence);
    # "split" = prefix stream once, action expert attending to its per-layer K/V (mathematically the same: the
    # prefix never attends to the suffix). Needed for repeated_diffusion_steps > 1 and the robustness recipe.
    train_forward: str = "joint"
    # GR00T-style repeated noise samples per observation in the flow loss (split forward only).
    repeated_diffusion_steps: int = 1
    # Flow loss over the first k action dims (0 = all action_dim dims, upstream). The padded dims carry no data.
    action_loss_dims: int = 0
    # predict_action returns only the first k dims (0 = all).
    output_action_dim: int = 0
    # "upstream": state padded to max_state_dim with pad_state_value and digitised unclipped (upstream port);
    # "starvla": the unpadded state, clipped to [-1, 1] and binned exactly as StateTokenMixin._state_bins.
    discrete_state_mode: str = "upstream"


def _as_dims(config) -> GemmaDims:
    return GemmaDims(
        width=int(config.width),
        depth=int(config.depth),
        mlp_dim=int(config.mlp_dim),
        num_heads=int(config.num_heads),
        num_kv_heads=int(config.num_kv_heads),
        head_dim=int(config.head_dim),
    )


_SENTENCEPIECE_FALLBACK = "/e/scratch/m3/blank4/pylibs_openpi"


class LazyPaliGemmaTokenizer:
    def __init__(self, max_len: int, model_path: str = "", extra_pythonpath: str = ""):
        self.max_len = int(max_len)
        self.model_path = str(model_path or "")
        self.extra_pythonpath = str(extra_pythonpath or "")
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.model_path:
            raise FileNotFoundError(
                "PI0/PI05 tokenizer.model_path is empty. Set framework.tokenizer.model_path "
                "to a local paligemma_tokenizer.model path for raw-text examples."
            )
        try:
            import sentencepiece
        except ImportError:
            # the starVLA image has no sentencepiece; a pip --target install lives on scratch
            for path in [p for p in (self.extra_pythonpath or _SENTENCEPIECE_FALLBACK).split(":") if p]:
                if path not in sys.path:
                    sys.path.append(path)
            import sentencepiece

        with open(self.model_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self._tokenizer

    def piece_id(self, piece: str) -> int:
        idx = int(self._load().piece_to_id(piece))
        if idx == int(self._load().unk_id()):
            raise ValueError(f"PaliGemma tokenizer has no piece {piece!r}")
        return idx

    def tokenize(self, prompt: str, state: np.ndarray | torch.Tensor | None = None):
        tokenizer = self._load()
        cleaned_text = str(prompt).strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))
            prompt_text = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = tokenizer.encode(prompt_text, add_bos=True)
        else:
            tokens = tokenizer.encode(cleaned_text, add_bos=True) + tokenizer.encode("\n")

        token_len = len(tokens)
        if token_len < self.max_len:
            padding = [0] * (self.max_len - token_len)
            mask = [True] * token_len + [False] * len(padding)
            tokens = tokens + padding
        else:
            if token_len > self.max_len:
                logger.warning("PI tokenizer length %s exceeds max_token_len=%s; truncating", token_len, self.max_len)
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)

    def tokenize_with_spans(self, prompt: str, bins: np.ndarray) -> dict:
        """Same ids as tokenize(prompt, state) for the pi0.5 state prompt, built piece by piece so the token
        spans of the instruction and of every state value are known:
            "Task:" | " <instruction>" | ", State:" | " <bin_0>" ... " <bin_n-1>" | ";\\nAction: "
        Checked against the one-shot encoding; on a mismatch the row gets no spans (it is then never masked)."""
        sp = self._load()
        clean = str(prompt).strip().replace("_", " ").replace("\n", " ")
        full = f"Task: {clean}, State: " + " ".join(str(int(b)) for b in bins) + ";\nAction: "
        ref = sp.encode(full, add_bos=True)
        ids = list(sp.encode("Task:", add_bos=True))
        instr = list(sp.encode(" " + clean))
        instr_span = (len(ids), len(ids) + len(instr))
        ids += instr + list(sp.encode(", State:"))
        dim_spans = []
        for b in bins:
            piece = list(sp.encode(" " + str(int(b))))
            # "▁" is its own token before the digits; the value's span is its digits
            start = len(ids) + (1 if piece and sp.id_to_piece(piece[0]) == "▁" else 0)
            ids += piece
            dim_spans.append((start, len(ids)))
        state_span = (dim_spans[0][0], dim_spans[-1][1]) if dim_spans else None
        ids += list(sp.encode(";\nAction: "))
        if ids != ref:
            return {"ids": list(ref), "instr_span": None, "dim_spans": None, "state_span": None}
        return {"ids": ids, "instr_span": instr_span, "dim_spans": dim_spans, "state_span": state_span}


_KO_PAIRS = {  # (query group, key group) removed from the prefix attention, as in attn_probe.py
    "text2img": [("text", "image")], "instr2img": [("text", "image")],
    "img2text": [("image", "text")], "img2instr": [("image", "text")],
}
_KO_LOGGED: set = set()


def _knockout_spec() -> list[tuple[str, str]]:
    """STARVLA_ATTN_KNOCKOUT for PI0/PI05: presets joined by '+' (text2img, img2text, ...)."""
    spec = os.environ.get("STARVLA_ATTN_KNOCKOUT", "").strip()
    pairs = []
    for name in [x for x in spec.split("+") if x]:
        if name not in _KO_PAIRS:
            raise ValueError(f"PI0 knockout {name!r} not supported; choose from {sorted(_KO_PAIRS)}")
        pairs += _KO_PAIRS[name]
    if spec and spec not in _KO_LOGGED:
        _KO_LOGGED.add(spec)
        logger.warning(f"ATTN PROBE ACTIVE topology=pi0-prefix knockout={spec!r} pairs={pairs}")
    return pairs


def _head_keys() -> list[str]:
    """STARVLA_HEAD_KEYS for PI0/PI05: prefix groups the action expert may read (e.g. image)."""
    keys = [k.strip() for k in os.environ.get("STARVLA_HEAD_KEYS", "").split(",") if k.strip()]
    if keys and ("head", tuple(keys)) not in _KO_LOGGED:
        _KO_LOGGED.add(("head", tuple(keys)))
        logger.warning(f"HEAD KEY RESTRICTION ACTIVE keys={keys} (pi0 prefix groups: image, text)")
    return keys


def _apply_prefix_knockout(att_2d: torch.Tensor, groups: dict, pairs) -> torch.Tensor:
    """Remove (query group -> key group) edges from a [B, L, L] prefix mask; surviving keys renormalise."""
    if not pairs:
        return att_2d
    att_2d = att_2d.clone()
    for q, k in pairs:
        qm, km = groups[q], groups[k]
        att_2d &= ~(qm[:, :, None] & km[:, None, :])
    return att_2d


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None):
    return x + y if gate is None else x + y * gate


def _norm(layernorm, hidden_states: torch.Tensor, cond: torch.Tensor | None):
    try:
        out = layernorm(hidden_states, cond=cond)
    except TypeError as exc:
        if cond is not None:
            dense = getattr(layernorm, "dense", None)
            if dense is None:
                raise RuntimeError(
                    "PI05 requires Gemma adaRMS support, but this layernorm has no dense module."
                ) from exc
            dtype = hidden_states.dtype
            eps = getattr(layernorm, "variance_epsilon", getattr(layernorm, "eps", 1e-6))
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            out = hidden_states * torch.rsqrt(variance + eps)
            scale, shift, gate = dense(cond.to(dense.weight.dtype))[:, None, :].chunk(3, dim=-1)
            return (out * (1 + scale) + shift).to(dtype), gate.to(dtype)
        out = layernorm(hidden_states)
    if isinstance(out, tuple):
        return out
    return out, None


def _state_bins_starvla(state: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """StateTokenMixin._state_bins: clip to [-1, 1], uniform bins over [-1, 1]."""
    edges = np.linspace(-1, 1, num_bins + 1)[:-1]
    bins = np.digitize(np.clip(state, -1.0, 1.0), bins=edges) - 1
    return np.clip(bins, 0, num_bins - 1).astype(np.int64)


@FRAMEWORK_REGISTRY.register("PI0")
@FRAMEWORK_REGISTRY.register("Pi0")
class PI0(PIRecipeMixin, baseframework):
    default_config_cls = PI0DefaultConfig
    action_head_cls: type[OpenPIActionHeadBase] = OpenPI0ActionHead
    model_name = "PI0"
    discrete_state_input = False
    # policy_wrapper checks this before honouring STARVLA_ZERO_IMAGE_INDICES (gray image, tokens kept)
    supports_zero_image_indices = True

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self._init_runtime_config(config)
        self._build_tokenizer()
        self._build_model_modules()
        self._configure_pi_recipe(self.config)
        self._build_pi_recipe_modules()
        self.to_precision(self.config.framework.precision)
        torch.set_float32_matmul_precision("highest")

    def _init_runtime_config(self, config):
        self.config = merge_framework_config(self.default_config_cls, config)
        self.config.framework.name = self.model_name
        self.discrete_state_input = bool(self.config.framework.discrete_state_input)

        self.action_dim = int(self.config.framework.action_dim)
        self.action_horizon = int(self.config.framework.action_horizon)
        self.max_state_dim = int(self.config.framework.max_state_dim)
        self.num_inference_steps = int(self.config.framework.num_inference_steps)
        self.image_resolution = tuple(int(x) for x in self.config.framework.image_resolution)
        self.image_keys = tuple(self.config.framework.image_keys)
        self.state_pad_value = float(self.config.framework.tokenizer.pad_state_value)

        fw = self.config.framework
        self.train_forward = str(fw.train_forward).lower()
        self.repeated_diffusion_steps = int(fw.repeated_diffusion_steps)
        self.action_loss_dims = int(fw.action_loss_dims)
        self.output_action_dim = int(fw.output_action_dim)
        self.discrete_state_mode = str(fw.discrete_state_mode).lower()
        if self.train_forward not in ("joint", "split"):
            raise ValueError("framework.train_forward must be joint or split")
        if self.discrete_state_mode not in ("upstream", "starvla"):
            raise ValueError("framework.discrete_state_mode must be upstream or starvla")
        if self.repeated_diffusion_steps < 1 or (self.repeated_diffusion_steps > 1 and self.train_forward != "split"):
            raise ValueError("repeated_diffusion_steps > 1 needs framework.train_forward: split")
        if not 0 <= self.action_loss_dims <= self.action_dim or not 0 <= self.output_action_dim <= self.action_dim:
            raise ValueError("action_loss_dims / output_action_dim must be in [0, action_dim]")
        # framework.elide_masked_image_slots (default off, 2026-10-11): trailing image slots whose mask is False for
        # every row (openpi's padded slots, e.g. pi0.5-DROID's right_wrist_0_rgb) are left out of the prefix. Exact:
        # padded tokens are never attended and do not advance the position ids (cumsum of the pad mask), so the
        # valid tokens see the same sequence; it only saves the SigLIP pass, the 256 prefix tokens and their memory.
        self.elide_masked_image_slots = bool(fw.get("elide_masked_image_slots", False))
        # policy_wrapper._expects_state reads state_token_mode: the state rides in the prompt as text
        self.state_token_mode = fw.get("state_token_mode", None)
        if self.state_token_mode not in (None, "text"):
            raise ValueError("PI0/PI05: framework.state_token_mode can only be 'text' (state digits in the prompt)")
        # The policy server sizes action chunks from framework.action_model.action_horizon.
        action_model_cfg = fw.get("action_model", None)
        if action_model_cfg is not None and "action_horizon" in action_model_cfg:
            if int(action_model_cfg.action_horizon) != self.action_horizon:
                raise ValueError("framework.action_model.action_horizon must equal framework.action_horizon")

    def _build_tokenizer(self):
        tok_cfg = self.config.framework.tokenizer
        self.tokenizer = LazyPaliGemmaTokenizer(
            max_len=int(self.config.framework.max_token_len),
            model_path=str(tok_cfg.model_path),
            extra_pythonpath=str(tok_cfg.get("extra_pythonpath", "") or "") if hasattr(tok_cfg, "get") else "",
        )

    def _build_model_modules(self):
        vlm_config = _as_dims(self.config.framework.paligemma_config)
        expert_config = _as_dims(self.config.framework.action_expert_config)
        self.vlm = OpenPIPaliGemma(
            vlm_config,
            use_adarms=False,
            model_name_or_path=str(self.config.framework.paligemma.model_name_or_path),
        )
        self.action_head = self.action_head_cls(
            expert_config,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_state_dim=self.max_state_dim,
        )

    def to_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported PI precision: {precision}")
        self.to(dtype=torch.bfloat16)
        keep_fp32 = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            if any(selector in name for selector in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def _pad_array_2d(self, arr: np.ndarray, target_time: int, target_dim: int, pad_value: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > target_time or arr.shape[1] > target_dim:
            raise ValueError(f"Input shape {arr.shape} exceeds target {(target_time, target_dim)}")
        out = np.full((target_time, target_dim), pad_value, dtype=np.float32)
        out[: arr.shape[0], : arr.shape[1]] = arr
        return out

    def _pil_to_tensor(self, image) -> torch.Tensor:
        image_np = np.asarray(image.convert("RGB")).copy()
        image_np = resize_with_pad(image_np[None], self.image_resolution[0], self.image_resolution[1])[0]
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0 * 2.0 - 1.0

    def _images_to_tensor(self, raw_images) -> tuple[torch.Tensor, list[bool]]:
        raw_images = to_pil_preserve(raw_images)
        if isinstance(raw_images, tuple):
            raw_images = list(raw_images)
        elif not isinstance(raw_images, list):
            raw_images = [raw_images]
        out, present = [], []
        for idx, _key in enumerate(self.image_keys):
            has_image = idx < len(raw_images)
            if has_image:
                image = raw_images[idx]
            else:
                image = raw_images[0].copy()
                image.paste(0, (0, 0, image.size[0], image.size[1]))
            out.append(self._pil_to_tensor(image))
            present.append(has_image)
        return torch.stack(out), present

    def _prepare_examples(self, examples: list[dict], include_actions: bool):
        images, masks, states, actions, prompts, raw_states = [], [], [], [], [], []
        for example in examples:
            sample_images, has = self._images_to_tensor(example["image"])
            provided_mask = example.get("image_mask")
            provided_mask = list(provided_mask) if provided_mask is not None else None
            sample_masks = [
                bool(provided_mask[idx]) if provided_mask is not None and idx < len(provided_mask) else has[idx]
                for idx in range(len(self.image_keys))
            ]
            images.append(sample_images)
            masks.append(sample_masks)
            prompts.append(str(example.get("lang", example.get("language", ""))))

            state_value = example.get("state")
            if state_value is None:
                fill = self.state_pad_value if self.config.framework.discrete_state_input else 0.0
                state_np = np.full((1, self.max_state_dim), fill, dtype=np.float32)
                raw_states.append(None)
            else:
                state_np = np.asarray(state_value, dtype=np.float32)
                if state_np.ndim == 2 and state_np.shape[0] > 1:
                    state_np = state_np[-1:]
                raw_states.append(state_np.reshape(-1).copy())
                state_np = self._pad_array_2d(
                    state_np,
                    1 if state_np.ndim == 1 else state_np.shape[0],
                    self.max_state_dim,
                    self.state_pad_value if self.config.framework.discrete_state_input else 0.0,
                )
            states.append(torch.from_numpy(state_np))
            if include_actions:
                action_np = self._pad_array_2d(np.asarray(example["action"], dtype=np.float32), self.action_horizon, self.action_dim, 0.0)
                actions.append(torch.from_numpy(action_np))

        batch = {
            "image": torch.stack(images),
            "image_mask": torch.as_tensor(np.asarray(masks), dtype=torch.bool),
            "state": torch.stack(states),
            "state_raw": raw_states,
            "lang": prompts,
        }
        if include_actions:
            batch["action"] = torch.stack(actions)
        return batch

    # ── prompt ───────────────────────────────────────────────────────────────
    def _row_prompt(self, prompt: str, raw_state) -> dict:
        """Token ids of one row (and, in starvla mode, the instruction / state spans)."""
        if self.discrete_state_input and self.discrete_state_mode == "starvla":
            if raw_state is None:
                raise ValueError("discrete_state_mode starvla needs a state on every example")
            bins = _state_bins_starvla(np.asarray(raw_state, dtype=np.float64).reshape(-1))
            return self.tokenizer.tokenize_with_spans(prompt, bins)
        return None

    def _build_text(self, batch, draws=None) -> dict:
        """Online and true token batches, padded to the longest row (capped at max_token_len).
        draws: the recipe's state / instruction masking decisions (None = nothing masked)."""
        B = len(batch["lang"])
        rows = [self._row_prompt(batch["lang"][i], batch["state_raw"][i]) for i in range(B)]
        if rows[0] is None:  # upstream tokenisation, no spans
            toks, masks = [], []
            for idx, prompt in enumerate(batch["lang"]):
                token_state = batch["state"][idx].squeeze(0) if self.discrete_state_input else None
                t, m = self.tokenizer.tokenize(prompt, token_state)
                toks.append(t)
                masks.append(m)
            tokens = torch.as_tensor(np.asarray(toks))
            tmask = torch.as_tensor(np.asarray(masks))
            return {"tokens_online": tokens, "mask_online": tmask, "tokens_true": tokens, "mask_true": tmask,
                    "state_fill": None, "instr_fill": None, "state_pos": None, "dim_spans": None,
                    "instr_mask_true": None, "state_masked": np.zeros(B, dtype=bool)}
        n_dims = len(rows[0]["dim_spans"] or batch["state_raw"][0].reshape(-1))
        state_keep = draws["state_keep"] if draws is not None else np.ones(B, dtype=bool)
        instr_draw = draws["instr_draw"] if draws is not None else np.zeros(B, dtype=bool)
        online, true, state_pos, instr_spans, state_masked = [], [], [], [], np.zeros(B, dtype=bool)
        for i, row in enumerate(rows):
            ids = row["ids"]
            true.append(ids)
            if instr_draw[i]:
                self._pi_counts["instr_rows"] += 1
                if row["instr_span"] is None:
                    self._pi_counts["instr_fallback"] += 1
            instr_spans.append(row["instr_span"] if instr_draw[i] else None)
            if not state_keep[i] and row["state_span"] is not None:
                a, b = row["state_span"]
                online.append(ids[:a] + [0] * n_dims + ids[b:])  # placeholder ids, embeddings overwritten
                state_pos.append(list(range(a, a + n_dims)))
                state_masked[i] = True
            else:
                online.append(list(ids))
                state_pos.append(None)

        def pad(seqs):
            L = min(max(len(s) for s in seqs), int(self.config.framework.max_token_len))
            if max(len(s) for s in seqs) > L:
                logger.warning("PI prompt longer than max_token_len=%d; truncating", L)
            t = torch.zeros(B, L, dtype=torch.long)
            m = torch.zeros(B, L, dtype=torch.bool)
            for r, s in enumerate(seqs):
                s = s[:L]
                t[r, : len(s)] = torch.as_tensor(s, dtype=torch.long)
                m[r, : len(s)] = True
            return t, m

        tok_on, m_on = pad(online)
        tok_tr, m_tr = pad(true)
        state_fill = torch.zeros_like(m_on)
        instr_fill = torch.zeros_like(m_on)
        instr_true = torch.zeros_like(m_tr)
        for i in range(B):
            if state_pos[i] is not None:
                state_fill[i, state_pos[i]] = True
            if instr_spans[i] is not None:
                a, b = instr_spans[i]
                instr_fill[i, a:b] = True
                instr_true[i, a:b] = True
        return {"tokens_online": tok_on, "mask_online": m_on, "tokens_true": tok_tr, "mask_true": m_tr,
                "state_fill": state_fill if bool(state_fill.any()) else None,
                "instr_fill": instr_fill if bool(instr_fill.any()) else None,
                "state_pos": state_pos, "dim_spans": [r["dim_spans"] for r in rows],
                "instr_mask_true": instr_true, "state_masked": state_masked}

    def _build_observation_from_batch(self, batch):
        device = self.action_head.action_in_proj.weight.device
        images = {key: batch["image"][:, idx].to(device=device, dtype=torch.float32) for idx, key in enumerate(self.image_keys)}
        masks = {key: batch["image_mask"][:, idx].to(device=device, dtype=torch.bool) for idx, key in enumerate(self.image_keys)}
        text = self._build_text(batch)
        return Observation(
            images=images,
            image_masks=masks,
            tokenized_prompt=text["tokens_online"].to(device),
            tokenized_prompt_mask=text["mask_online"].to(device),
            state=batch["state"].to(device=device, dtype=torch.float32),
        )

    def _n_active_slots(self, img_masks) -> int:
        """Number of leading image slots to embed: all of them, or (elide_masked_image_slots) all but the trailing
        slots whose mask is False in every row. img_masks: list per slot of [B] bool, or a [B, C] tensor."""
        masks = list(img_masks) if isinstance(img_masks, (list, tuple)) else [img_masks[:, c] for c in range(img_masks.shape[1])]
        n = len(masks)
        if self.elide_masked_image_slots and not getattr(self, "pi_fc", False):
            while n > 1 and not bool(masks[n - 1].any()):
                n -= 1
        return n

    def _preprocess_observation(self, observation: Observation):
        keys = self.image_keys[: self._n_active_slots([observation.image_masks[k] for k in self.image_keys])]
        return (
            [observation.images[key] for key in keys],
            [observation.image_masks[key] for key in keys],
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    # ── prefix ───────────────────────────────────────────────────────────────
    def _embed_prefix_tensors(self, vlm, images, img_masks, lang_tokens, lang_masks, *, fills=None,
                              copies: bool = False, set_groups: bool = False):
        """images: list (per camera) of [B, 3, H, W]; img_masks: list of [B]. fills: [(mask [B, Lt], embedding
        module)] replacing text embeddings (mask tokens). copies: append the future copies of the cameras after the
        text, each with its learned marker added (block-0 input). Returns (embs [B, Lp, D], pad_masks [B, Lp])."""
        embs, pad_masks = [], []
        img_embs = []
        for image, image_mask in zip(images, img_masks, strict=True):
            image_emb = vlm.embed_image(image)
            bsize, num_img_embs = image_emb.shape[:2]
            img_embs.append(image_emb)
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(bsize, num_img_embs))
        lang_emb = vlm.embed_language_tokens(lang_tokens)
        scale = math.sqrt(lang_emb.shape[-1])
        lang_emb = lang_emb * scale
        for fill_mask, module in (fills or []):
            if fill_mask is None:
                continue
            vec = (module.weight[0] * scale).to(lang_emb.dtype)
            lang_emb = torch.where(fill_mask.to(lang_emb.device)[..., None], vec, lang_emb)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        n_img = sum(e.shape[1] for e in img_embs)
        if copies:
            for j, image_emb in enumerate(img_embs):
                marker = self.fcopy_marker[j, : image_emb.shape[1]]
                embs.append(image_emb + marker.to(image_emb.dtype))  # the same (possibly grayed) camera
                pad_masks.append(torch.ones(image_emb.shape[:2], dtype=torch.bool, device=image_emb.device))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        if set_groups:
            is_img = torch.zeros_like(pad_masks)
            is_img[:, :n_img] = True
            if copies:
                is_img[:, n_img + lang_emb.shape[1]:] = True
            self._prefix_groups = {"image": is_img & pad_masks, "text": (~is_img) & pad_masks}
        return embs, pad_masks

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks = self._embed_prefix_tensors(
            self.vlm, images, img_masks, lang_tokens, lang_masks,
            copies=bool(getattr(self, "pi_fc", False)), set_groups=True)
        att_masks = torch.zeros(pad_masks.shape, dtype=torch.bool, device=pad_masks.device)
        return embs, pad_masks, att_masks

    @staticmethod
    def _attend(q, k, v, mask, scaling):
        """q [B, H, Lq, d], k/v [B, Hkv, Lk, d], mask bool broadcastable to [B, H, Lq, Lk] (True = attend)."""
        if k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = modeling_gemma.repeat_kv(k, rep)
            v = modeling_gemma.repeat_kv(v, rep)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    @staticmethod
    def _layer_qkv(layer, h, cond, head_dim):
        x, gate = _norm(layer.input_layernorm, h, cond)
        shape = (*x.shape[:-1], -1, head_dim)
        q = layer.self_attn.q_proj(x).view(shape).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(shape).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(shape).transpose(1, 2)
        return q, k, v, gate

    @staticmethod
    def _layer_out(layer, h, att, gate, cond):
        if att.dtype != layer.self_attn.o_proj.weight.dtype:
            att = att.to(layer.self_attn.o_proj.weight.dtype)
        out = _gated_residual(h, layer.self_attn.o_proj(att), gate)
        x, gate2 = _norm(layer.post_attention_layernorm, out, cond)
        if x.dtype != layer.mlp.up_proj.weight.dtype:
            x = x.to(layer.mlp.up_proj.weight.dtype)
        return _gated_residual(out, layer.mlp(x), gate2)

    def _run_prefix(self, vlm, embs, pad_masks, *, capture=(), stop_after=None, keep_kv=False):
        """The prefix stream alone (it never attends to the action expert). Returns (hidden, {layer: output},
        [(k, v)] per layer after RoPE when keep_kv)."""
        lm = vlm.language_model
        if lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embs = embs.to(torch.bfloat16)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        head_dim = lm.layers[0].self_attn.head_dim
        dummy = torch.zeros(embs.shape[0], embs.shape[1], head_dim, device=embs.device, dtype=embs.dtype)
        cos, sin = lm.rotary_emb(dummy, position_ids)
        key_mask = pad_masks[:, None, None, :]
        h, caps, kvs = embs, {}, []
        for idx, layer in enumerate(lm.layers):
            q, k, v, gate = self._layer_qkv(layer, h, None, head_dim)
            q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            att = self._attend(q, k, v, key_mask, layer.self_attn.scaling)
            h = self._layer_out(layer, h, att, gate, None)
            if keep_kv:
                kvs.append((k, v))
            if idx in capture:
                caps[idx] = h
            if stop_after is not None and idx >= stop_after:
                break
        return h, caps, kvs

    def _run_suffix(self, kvs, prefix_pad, suffix_embs, suffix_pad, suffix_att, cond, repeats: int):
        """Action expert over B*R suffix rows (row r*B + b repeats observation b) attending to the prefix K/V of b
        and to its own block of suffix tokens (block-diagonal across the R repeats)."""
        expert = self.action_head.model
        vlm_layers = self.vlm.language_model.layers
        BR, S = suffix_embs.shape[:2]
        B, Lp = prefix_pad.shape
        R = int(repeats)
        if BR != B * R:
            raise RuntimeError(f"suffix rows {BR} != {B} x {R}")
        if expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)
        n_valid = prefix_pad.sum(dim=-1).repeat(R)
        position_ids = n_valid[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        head_dim = expert.layers[0].self_attn.head_dim
        dummy = torch.zeros(BR, S, head_dim, device=suffix_embs.device, dtype=suffix_embs.dtype)
        cos, sin = self.vlm.language_model.rotary_emb(dummy, position_ids)
        s2s = make_att_2d_masks(suffix_pad, suffix_att)                      # [BR, S, S]
        s2s = s2s.view(R, B, S, S).permute(1, 0, 2, 3)                       # [B, R, S, S]
        block = torch.zeros(B, R * S, R * S, dtype=torch.bool, device=s2s.device)
        for r in range(R):
            block[:, r * S:(r + 1) * S, r * S:(r + 1) * S] = s2s[:, r]
        mask = torch.cat([prefix_pad[:, None, :].expand(B, R * S, Lp), block], dim=2)[:, None]  # [B, 1, RS, Lp+RS]

        def to_blocks(x):  # [BR, h, S, d] -> [B, h, R*S, d]
            hh, d = x.shape[1], x.shape[3]
            return x.view(R, B, hh, S, d).permute(1, 2, 0, 3, 4).reshape(B, hh, R * S, d)

        s = suffix_embs
        for idx, layer in enumerate(expert.layers):
            q, k, v, gate = self._layer_qkv(layer, s, cond, head_dim)
            q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            pk, pv = kvs[idx]
            K = torch.cat([pk, to_blocks(k)], dim=2)
            V = torch.cat([pv, to_blocks(v)], dim=2)
            att = self._attend(to_blocks(q), K, V, mask, vlm_layers[idx].self_attn.scaling)  # [B, RS, h*d]
            att = att.view(B, R, S, -1).permute(1, 0, 2, 3).reshape(BR, S, -1)
            s = self._layer_out(layer, s, att, gate, cond)
        out, _ = _norm(expert.norm, s, cond)
        return out

    # ── upstream joint forward ───────────────────────────────────────────────
    def forward_with_action_expert(
        self,
        *,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: torch.Tensor | None,
    ):
        streams = [prefix_embs, suffix_embs]
        conds = [None, adarms_cond]
        models = [self.vlm.language_model, self.action_head.model]
        for layer_idx in range(self.vlm.config.text_config.num_hidden_layers):
            streams = self.forward_shared_gemma_layer(layer_idx, models, streams, conds, attention_mask, position_ids)
        suffix_out, _ = _norm(models[1].norm, streams[1], conds[1])
        return suffix_out

    def forward_shared_gemma_layer(self, layer_idx, models, streams, conds, attention_mask, position_ids):
        query_states, key_states, value_states, gates = [], [], [], []
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            hidden_states, gate = _norm(layer.input_layernorm, hidden_states, conds[idx])
            gates.append(gate)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        dummy = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.vlm.language_model.rotary_emb(dummy, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        att_output, _ = modeling_gemma.eager_attention_forward(
            self.vlm.language_model.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.vlm.language_model.layers[layer_idx].self_attn.scaling,
        )
        head_dim = self.vlm.language_model.layers[layer_idx].self_attn.head_dim
        num_heads = self.vlm.language_model.config.num_attention_heads
        att_output = att_output.reshape(query_states.shape[0], -1, num_heads * head_dim)

        next_streams = []
        start = 0
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            end = start + hidden_states.shape[1]
            layer_att = att_output[:, start:end]
            if layer_att.dtype != layer.self_attn.o_proj.weight.dtype:
                layer_att = layer_att.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(layer_att)
            out = _gated_residual(hidden_states, out, gates[idx])
            residual = out.clone()
            out, gate = _norm(layer.post_attention_layernorm, out, conds[idx])
            if out.dtype != layer.mlp.up_proj.weight.dtype:
                out = out.to(layer.mlp.up_proj.weight.dtype)
            out = layer.mlp(out)
            next_streams.append(_gated_residual(residual, out, gate))
            start = end
        return next_streams

    def _flow_loss(self, u_t, v_t):
        k = self.action_loss_dims or u_t.shape[-1]
        return F.mse_loss(u_t[..., :k], v_t[..., :k], reduction="none").mean()

    def forward_from_observation(self, observation: Observation, actions: torch.Tensor, noise=None, time=None, return_debug=False):
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        x_t, u_t, noise, time = self.action_head.prepare_flow_targets(actions, noise=noise, time=time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.action_head.embed_action_suffix(state.squeeze(1), x_t, time)
        if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention_mask = make_att_4d_masks(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suffix_out = self.forward_with_action_expert(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )
        v_t = self.action_head.decode_action_velocity(suffix_out)
        out = {"v_t": v_t, "action_loss": self._flow_loss(u_t, v_t)}
        if return_debug:
            out.update({"u_t": u_t, "x_t": x_t, "time": time, "noise": noise})
        return out

    # ── split forward (StarVLA): prefix once, R repeated noise samples, recipe ──
    def _future_batch(self, examples, rows):
        imgs = []
        for i in rows:
            ex = examples[i]
            multi = ex.get("future_images_multi")
            fut = multi[self.pi_fc_offset] if multi is not None and self.pi_fc_offset in multi else ex.get("future_image")
            imgs.append(self._images_to_tensor(fut)[0])
        return torch.stack(imgs)

    def _fc_rows(self, examples):
        rows = []
        for i, ex in enumerate(examples):
            multi = ex.get("future_images_multi")
            if multi is not None and self.pi_fc_offset in multi:
                fut = multi[self.pi_fc_offset]
            elif ex.get("future_image") is not None and not multi:
                fut = ex["future_image"]
            else:
                fut = None
            n_img = len(ex["image"]) if isinstance(ex["image"], (list, tuple)) else 1
            if fut is not None and len(fut) == n_img:
                rows.append(i)
        return rows

    def forward_split(self, examples, batch, actions, noise=None, time=None, return_debug=False):
        device = self.action_head.action_in_proj.weight.device
        B = len(examples)
        if getattr(self, "pi_fusion", False) and self._pi_aux_active() and "_pi_ema_state" not in self.__dict__:
            # create the EMA prefix copy now, before this step's activations exist, instead of lazily inside the
            # recipe loss on a full GPU (creation does not update; the first update moves it toward the same weights)
            self._pi_ema_update()
        draws = self._pi_draw(B) if self.pi_recipe_on else None
        text = self._build_text(batch, draws)
        images = batch["image"].to(device=device, dtype=torch.float32)        # [B, C, 3, H, W], augmented, full
        img_masks = batch["image_mask"].to(device)
        dropped = draws["dropped"] if draws is not None else [[] for _ in range(B)]
        online_images = images
        if any(dropped):
            online_images = images.clone()
            for r, cams in enumerate(dropped):
                for c in cams:
                    online_images[r, c] = 0.0   # uniform mid-gray, tokens kept and attended
        n_cams = self._n_active_slots(img_masks)
        fills = [(text["state_fill"], getattr(self, "state_mask_embed", None)),
                 (text["instr_fill"], getattr(self, "instr_mask_embed", None))]
        prefix_embs, prefix_pad = self._embed_prefix_tensors(
            self.vlm, [online_images[:, c] for c in range(n_cams)], [img_masks[:, c] for c in range(n_cams)],
            text["tokens_online"].to(device), text["mask_online"].to(device), fills=fills,
            copies=bool(self.pi_fc), set_groups=True)
        aux = self._pi_aux_active()
        capture = set()
        if aux:
            capture = {getattr(self, "pi_fusion_layer", None), getattr(self, "pi_fc_layer", None)} - {None}
        _, caps, kvs = self._run_prefix(self.vlm, prefix_embs, prefix_pad, capture=capture, keep_kv=True)
        R = self.repeated_diffusion_steps
        actions_rep = actions.repeat(R, 1, 1)
        if noise is not None:
            noise = noise.repeat(R, 1, 1) if noise.shape[0] == B else noise
        if time is not None:
            time = time.repeat(R) if time.shape[0] == B else time
        x_t, u_t, noise, time = self.action_head.prepare_flow_targets(actions_rep, noise=noise, time=time)
        state = batch["state"].to(device=device, dtype=torch.float32).squeeze(1).repeat(R, 1)
        suffix_embs, suffix_pad, suffix_att, cond = self.action_head.embed_action_suffix(state, x_t, time)
        suffix_out = self._run_suffix(kvs, prefix_pad, suffix_embs, suffix_pad, suffix_att, cond, R)
        v_t = self.action_head.decode_action_velocity(suffix_out)
        out = {"v_t": v_t, "action_loss": self._flow_loss(u_t, v_t)}
        if return_debug:
            out.update({"u_t": u_t, "x_t": x_t, "time": time, "noise": noise})
        if self.pi_recipe_on:
            n_dropped = [len(d) for d in dropped]
            out["vision_keep_rate"] = float(draws["vision_keep"].mean())
            out["vision_drop_rate"] = self._pi_current_vision_rate()
            out["state_keep_rate"] = float(1.0 - text["state_masked"].mean())
        if aux:
            n_img = self._num_image_tokens()
            text_off = n_cams * n_img
            ctx = {
                "device": device, "batch": B, "dropped": dropped, "n_img_tokens": n_img,
                "online_capture": caps, "state_masked": text["state_masked"],
                "copy_offset": text_off + text["tokens_online"].shape[1],
                "target_inputs": {"images": [images[:, c] for c in range(n_cams)],
                                  "img_masks": [img_masks[:, c] for c in range(n_cams)],
                                  "tokens": text["tokens_true"].to(device), "token_masks": text["mask_true"].to(device)},
            }
            if text["state_fill"] is not None:
                ctx["state_pos_online"] = [None if p is None else [text_off + x for x in p] for p in text["state_pos"]]
                ctx["state_dim_spans_target"] = [None if s is None else [(text_off + a, text_off + b) for a, b in s]
                                                 for s in text["dim_spans"]]
            if text["instr_fill"] is not None:
                im_on = torch.zeros(prefix_pad.shape, dtype=torch.bool)
                im_on[:, text_off:text_off + text["instr_fill"].shape[1]] = text["instr_fill"]
                im_tr = torch.zeros(B, text_off + text["tokens_true"].shape[1], dtype=torch.bool)
                im_tr[:, text_off:] = text["instr_mask_true"]
                ctx["instr_mask_online"], ctx["instr_mask_target"] = im_on, im_tr
            else:
                ctx["instr_mask_online"] = None
            if self.pi_fc:
                rows = self._fc_rows(examples)
                ctx["fc_rows"] = rows
                if rows:
                    fut = self._future_batch(examples, rows).to(device=device, dtype=torch.float32)
                    rows_t = torch.as_tensor(rows, device=device)
                    ctx["future_inputs"] = {"images": [fut[:, c] for c in range(n_cams)],
                                            "img_masks": [img_masks[rows_t, c] for c in range(n_cams)],
                                            "tokens": text["tokens_true"].to(device)[rows_t],
                                            "token_masks": text["mask_true"].to(device)[rows_t]}
            aux_loss, metrics = self._pi_recipe_losses(ctx)
            metrics["structured_aux/cam_drop_rows"] = torch.tensor(float(np.mean([n > 0 for n in n_dropped])), device=device)
            for c in range(n_cams):
                metrics[f"structured_aux/cam{c}_drop_rate"] = torch.tensor(
                    float(np.mean([c in d for d in dropped])), device=device)
            if self.pi_instr_rate > 0:
                metrics["structured_aux/instr_rows_masked"] = torch.tensor(
                    float(text["instr_fill"].any(1).float().mean()) if text["instr_fill"] is not None else 0.0, device=device)
                if self._pi_counts["instr_rows"]:
                    metrics["structured_aux/instr_mask_fallback"] = torch.tensor(
                        self._pi_counts["instr_fallback"] / self._pi_counts["instr_rows"], device=device)
            out["structured_aux_loss"] = aux_loss
            out.update(metrics)
            self._pi_log_once("aux", f"__PI_RECIPE__ prefix_len={prefix_embs.shape[1]} text_len={text['tokens_online'].shape[1]} "
                                     f"captures={sorted(caps)} dropped={dropped} state_masked={text['state_masked'].tolist()} "
                                     f"instr_masked={0 if text['instr_fill'] is None else int(text['instr_fill'].any(1).sum())} "
                                     f"fc_rows={len(ctx.get('fc_rows', []))}")
        return out

    def forward(self, examples: list[dict] = None, **kwargs):
        '''
        Training forward for PI0 flow-matching action prediction.
        '''
        batch = self._prepare_examples(examples, include_actions=True)
        actions = batch["action"].to(device=self.action_head.action_in_proj.weight.device, dtype=torch.float32)
        if self.train_forward == "split" or self.pi_recipe_on:
            return self.forward_split(
                examples, batch, actions, noise=kwargs.get("noise"), time=kwargs.get("time"),
                return_debug=bool(kwargs.get("return_debug", False)))
        observation = self._build_observation_from_batch(batch)
        return self.forward_from_observation(
            observation,
            actions,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            return_debug=bool(kwargs.get("return_debug", False)),
        )

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict = None, **kwargs):
        '''
        Prediction forward for PI0 flow-matching action prediction
        '''
        if not isinstance(examples, list):
            examples = [examples]
        batch = self._prepare_examples(examples, include_actions=False)
        zero_positions = examples[0].get("_zero_image_indices")
        if zero_positions:
            if any(ex.get("_zero_image_indices") != zero_positions for ex in examples):
                raise ValueError("_zero_image_indices must be identical for every example")
            for pos in zero_positions:
                if not 0 <= int(pos) < len(self.image_keys):
                    raise ValueError(f"cannot zero image position {pos} of {len(self.image_keys)}")
                batch["image"][:, int(pos)] = 0.0  # gray, as camera dropout in training; copies follow
        observation = self._build_observation_from_batch(batch)
        noise = kwargs.get("noise")
        seeds = [ex.get("flow_seed") for ex in examples]
        if noise is None and all(s is not None for s in seeds):
            device = self.action_head.action_in_proj.weight.device
            noise = torch.stack([
                torch.randn((self.action_horizon, self.action_dim), generator=torch.Generator().manual_seed(int(s)))
                for s in seeds]).to(device)
        actions = self.sample_actions(
            observation,
            noise=noise,
            num_steps=int(kwargs.get("num_steps", kwargs.get("num_inference_steps", self.num_inference_steps))),
        )
        actions = actions.float().cpu().numpy()
        if self.output_action_dim:
            actions = actions[..., : self.output_action_dim]
        return {"normalized_actions": actions}

    def sample_actions(self, observation: Observation, noise=None, num_steps: int = 10):
        device = self.action_head.action_in_proj.weight.device
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = _apply_prefix_knockout(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks), self._prefix_groups, _knockout_spec()
        )
        prefix_att_2d_masks_4d = make_att_4d_masks(prefix_att_2d)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.vlm.language_model.config._attn_implementation = "eager"
        if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
        prefix_output = self.vlm.forward_prefix(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        bsize = state.shape[0]
        shape = (bsize, self.action_horizon, self.action_dim)
        x_t = self.action_head.sample_action_noise(shape, device) if noise is None else torch.as_tensor(noise, device=device, dtype=torch.float32)
        if x_t.ndim == 2:
            x_t = x_t[None]
        dt = torch.tensor(-1.0 / max(num_steps, 1), dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state.squeeze(1), prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, _adarms_cond = self.action_head.embed_action_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        head_pad = prefix_pad_masks
        keys = _head_keys()
        if keys:
            head_pad = prefix_pad_masks & torch.stack([self._prefix_groups[k] for k in keys]).any(0)
        prefix_pad_2d_masks = head_pad[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention_mask = make_att_4d_masks(torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2))
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        suffix_embs = suffix_embs.to(torch.bfloat16) if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16 else suffix_embs
        return self.action_head(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=_adarms_cond,
        )

    def _is_expected_openpi_missing_key(self, key: str) -> bool:
        if not getattr(self.action_head, "use_adarms", False):
            return False
        prefix = "action_head.gemma_expert.model."
        if key == prefix + "norm.weight":
            return True
        if not key.startswith(prefix + "layers."):
            return False
        return key.endswith(".input_layernorm.weight") or key.endswith(".post_attention_layernorm.weight")

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        if path.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt", "ckpt.pt"):
                candidate = path / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No supported PI checkpoint file found under {path}")
        return path

    @classmethod
    def _load_checkpoint_state_dict(cls, checkpoint_path: str | Path):
        checkpoint_path = cls._resolve_checkpoint_path(checkpoint_path)
        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict):
                for key in ("state_dict", "model_state_dict", "module", "model"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
        return state_dict

    def _normalize_state_dict(self, state_dict: dict):
        current = self.state_dict()
        normalized = {}
        for key, value in state_dict.items():
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in ("model.", "module.", "_orig_mod."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            if clean_key.startswith("paligemma_with_expert.paligemma.model."):
                clean_key = clean_key.replace(
                    "paligemma_with_expert.paligemma.model.",
                    "vlm.model.model.",
                    1,
                )
            elif clean_key.startswith("paligemma_with_expert.paligemma."):
                clean_key = clean_key.replace("paligemma_with_expert.paligemma.", "vlm.model.", 1)
            elif clean_key.startswith("paligemma_with_expert.gemma_expert."):
                clean_key = clean_key.replace("paligemma_with_expert.gemma_expert.", "action_head.gemma_expert.", 1)
            elif clean_key.startswith("action_in_proj."):
                clean_key = clean_key.replace("action_in_proj.", "action_head.action_in_proj.", 1)
            elif clean_key.startswith("action_out_proj."):
                clean_key = clean_key.replace("action_out_proj.", "action_head.action_out_proj.", 1)
            elif clean_key.startswith("state_proj."):
                clean_key = clean_key.replace("state_proj.", "action_head.state_proj.", 1)
            elif clean_key.startswith("action_time_mlp_in."):
                clean_key = clean_key.replace("action_time_mlp_in.", "action_head.action_time_mlp_in.", 1)
            elif clean_key.startswith("action_time_mlp_out."):
                clean_key = clean_key.replace("action_time_mlp_out.", "action_head.action_time_mlp_out.", 1)
            elif clean_key.startswith("time_mlp_in."):
                clean_key = clean_key.replace("time_mlp_in.", "action_head.time_mlp_in.", 1)
            elif clean_key.startswith("time_mlp_out."):
                clean_key = clean_key.replace("time_mlp_out.", "action_head.time_mlp_out.", 1)
            normalized[clean_key] = value
        embed_key = "vlm.model.model.language_model.embed_tokens.weight"
        lm_head_key = "vlm.model.lm_head.weight"
        if embed_key not in normalized and lm_head_key in normalized and embed_key in current:
            if normalized[lm_head_key].shape == current[embed_key].shape:
                normalized[embed_key] = normalized[lm_head_key]
        return normalized

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path, reload_modules: str | None = None):
        state_dict = self._normalize_state_dict(self._load_checkpoint_state_dict(checkpoint_path))
        if reload_modules:
            for path in [p.strip() for p in reload_modules.split(",") if p.strip()]:
                module = self
                for part in path.split("."):
                    module = getattr(module, part)
                prefix = path + "."
                sub_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                module.load_state_dict(sub_state, strict=False)
            return self

        current = self.state_dict()
        filtered, shape_mismatch = {}, []
        for key, value in state_dict.items():
            if key in current and getattr(value, "shape", None) == current[key].shape:
                filtered[key] = value
            elif key in current:
                shape_mismatch.append((key, tuple(value.shape), tuple(current[key].shape)))
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        expected_missing = [key for key in missing if self._is_expected_openpi_missing_key(key)]
        real_missing = [key for key in missing if key not in expected_missing]
        logger.info(
            "%s checkpoint load: checkpoint=%s loaded=%s model=%s missing=%s expected_missing=%s unexpected=%s shape_mismatch=%s",
            type(self).__name__,
            len(state_dict),
            len(filtered),
            len(current),
            len(real_missing),
            len(expected_missing),
            len(unexpected),
            len(shape_mismatch),
        )
        if real_missing:
            logger.info("%s first missing keys: %s", type(self).__name__, real_missing[:10])
        if shape_mismatch:
            logger.info("%s first shape mismatches: %s", type(self).__name__, shape_mismatch[:10])
        return self
