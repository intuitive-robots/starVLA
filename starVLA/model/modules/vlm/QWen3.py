# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from types import MethodType
from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


# ``framework.qwenvl.prompt_order``: where the user text goes relative to the images.
#   image_first (default, unset): [images] [instruction (+ [STATE] ... [ACTION])], the
#       historical layout, byte-identical to before this flag existed.
#   text_first: [instruction (+ state)] [images]. Under a causal mask image tokens can
#       read the instruction and state, but instruction tokens can no longer read images.
#   sandwich: [instruction copy, no state] [images] [instruction (+ state)]. The trailing
#       block is exactly the image_first text, so the instruction->image route and the
#       state position are unchanged, and image tokens can additionally read the copy.
PROMPT_ORDERS = ("image_first", "text_first", "sandwich")
_STATE_SUFFIX_MARKER = " [STATE] "


def _resolve_prompt_order(value) -> str:
    order = "image_first" if value is None else str(value).strip().lower()
    if order not in PROMPT_ORDERS:
        raise ValueError(f"framework.qwenvl.prompt_order must be one of {PROMPT_ORDERS}, got {value!r}")
    return order


def _instruction_copy_without_state(prompt: str) -> str:
    """Sandwich lead text: the prompt up to the appended state block (StateTokenMixin
    writes ``f"{instruction} [STATE] {slots} [ACTION]"``). A row whose state was dropped
    has no block, so the copy is the whole prompt."""
    lead = prompt.split(_STATE_SUFFIX_MARKER, 1)[0]
    if "<|fim_pad|>" in lead or "[STATE]" in lead:
        raise RuntimeError(f"sandwich lead text still carries state placeholders: {lead!r}")
    return lead


def arrange_prompt_content(image_items: list, prompt: str, prompt_order: str) -> list:
    """User-turn content list for one sample (see PROMPT_ORDERS)."""
    text_item = {"type": "text", "text": prompt}
    if prompt_order == "image_first":
        return list(image_items) + [text_item]
    if prompt_order == "text_first":
        return [text_item] + list(image_items)
    if prompt_order == "sandwich":
        lead = {"type": "text", "text": _instruction_copy_without_state(prompt)}
        return [lead] + list(image_items) + [text_item]
    raise ValueError(f"unknown prompt_order {prompt_order!r}")


def build_modality_block_ids(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    vision_start_token_id: int,
    vision_end_token_id: int,
    state_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assign valid prompt tokens to camera, language, and state blocks.

    Every block is internally bidirectional. Information between blocks follows prompt
    order: a block may attend to itself and earlier blocks, never later blocks. ``-1`` is
    padding. Chat boilerplate before the first image belongs to camera 1, separators
    between images belong to the following camera, and the suffix from the first state
    token onward belongs to the state block.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, length], got {tuple(input_ids.shape)}")
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask and input_ids must have the same shape")
        valid = attention_mask.to(dtype=torch.bool)
    if state_token_mask is not None and state_token_mask.shape != input_ids.shape:
        raise ValueError("state_token_mask and input_ids must have the same shape")

    block_ids = torch.full_like(input_ids, -1, dtype=torch.long)
    for row in range(input_ids.shape[0]):
        valid_pos = valid[row].nonzero(as_tuple=True)[0]
        if valid_pos.numel() == 0:
            continue
        starts = ((input_ids[row] == vision_start_token_id) & valid[row]).nonzero(as_tuple=True)[0]
        ends = ((input_ids[row] == vision_end_token_id) & valid[row]).nonzero(as_tuple=True)[0]
        if starts.numel() != ends.numel():
            raise RuntimeError(
                f"row {row}: unmatched vision delimiters ({starts.numel()} starts, "
                f"{ends.numel()} ends)"
            )

        cursor = int(valid_pos[0])
        block = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            if start < cursor or end < start:
                raise RuntimeError(f"row {row}: malformed or overlapping vision span")
            block_ids[row, cursor : end + 1] = block
            cursor = end + 1
            block += 1

        valid_end = int(valid_pos[-1]) + 1
        if starts.numel() == 0:
            block = 0
            cursor = int(valid_pos[0])

        state_start = None
        if state_token_mask is not None:
            state_pos = (state_token_mask[row].to(dtype=torch.bool) & valid[row]).nonzero(as_tuple=True)[0]
            if state_pos.numel() > 0:
                state_start = int(state_pos[0])

        language_end = state_start if state_start is not None else valid_end
        if cursor < language_end:
            block_ids[row, cursor:language_end] = block
            block += 1
        if state_start is not None:
            block_ids[row, state_start:valid_end] = block

        if bool((block_ids[row, valid_pos] < 0).any()):
            raise RuntimeError(f"row {row}: one or more valid tokens were not assigned a block")
    return block_ids


def build_modality_block_pack_plan(block_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precompute the invariant gather/scatter indices for one prompt batch."""
    groups = []
    max_block = int(block_ids.max().item())
    for source_row in range(block_ids.shape[0]):
        for block in range(max_block + 1):
            current_pos = (block_ids[source_row] == block).nonzero(as_tuple=True)[0]
            if current_pos.numel() == 0:
                continue
            previous_pos = (
                (block_ids[source_row] >= 0) & (block_ids[source_row] < block)
            ).nonzero(as_tuple=True)[0]
            groups.append((source_row, current_pos, previous_pos))
    if not groups:
        raise RuntimeError("modality block plan contains no query groups")

    max_current = max(current_pos.numel() for _, current_pos, _ in groups)
    max_previous = max(previous_pos.numel() for _, _, previous_pos in groups)
    key_length = max_previous + max_current
    device = block_ids.device
    source_rows = torch.tensor(
        [source_row for source_row, _, _ in groups], dtype=torch.long, device=device
    )
    query_indices = torch.zeros((len(groups), max_current), dtype=torch.long, device=device)
    key_indices = torch.zeros((len(groups), key_length), dtype=torch.long, device=device)
    query_mask = torch.zeros((len(groups), max_current), dtype=torch.bool, device=device)
    key_mask = torch.zeros((len(groups), key_length), dtype=torch.bool, device=device)
    for packed_row, (_, current_pos, previous_pos) in enumerate(groups):
        n_current = current_pos.numel()
        n_previous = previous_pos.numel()
        query_offset = max_current - n_current
        key_current_offset = max_previous + query_offset
        query_indices[packed_row, query_offset:] = current_pos
        query_mask[packed_row, query_offset:] = True
        if n_previous:
            key_indices[packed_row, :n_previous] = previous_pos
            key_mask[packed_row, :n_previous] = True
        key_indices[packed_row, key_current_offset:] = current_pos
        key_mask[packed_row, key_current_offset:] = True
    return {
        "source_rows": source_rows,
        "query_indices": query_indices,
        "key_indices": key_indices,
        "query_mask": query_mask,
        "key_mask": key_mask,
    }


def build_full_prefix_pack_plan(attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Right-align every valid prompt for one non-causal FA3 call.

    QwenGR00T has no teacher-forced answer in the plain policy recipe: cameras,
    instruction, and tokenized state are all observation-prefix tokens.  This is
    therefore the policy-time half of a PaLiGemma-style PrefixLM mask.
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, length], got {tuple(attention_mask.shape)}"
        )
    valid = attention_mask.to(dtype=torch.bool)
    lengths = valid.sum(dim=1)
    if bool((lengths == 0).any()):
        bad = (lengths == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"full-prefix rows contain no valid tokens: {bad}")
    width = int(lengths.max().item())
    indices = torch.zeros((valid.shape[0], width), dtype=torch.long, device=valid.device)
    keep = torch.zeros((valid.shape[0], width), dtype=torch.bool, device=valid.device)
    for row in range(valid.shape[0]):
        pos = valid[row].nonzero(as_tuple=True)[0]
        indices[row, width - pos.numel() :] = pos
        keep[row, width - pos.numel() :] = True
    return {"indices": indices, "keep": keep, "valid": valid}


def _modality_block_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    **kwargs,
):
    """FA3 attention: bidirectional within blocks, causal between blocks.

    FA3 does not consume arbitrary dense masks. For block ``g`` this packs keys from
    blocks ``<= g`` and queries from block ``g``, then calls FA3 non-causally. All
    (sample, block) rows are packed into one enlarged batch, so this costs one kernel call
    per layer rather than one call per sample or semantic block.
    """
    if past_key_values is not None:
        raise RuntimeError(
            "modality_block_causal is a full-prompt policy topology and does not support KV-cache generation"
        )
    block_ids = getattr(self, "_starvla_modality_block_ids", None)
    if block_ids is None or block_ids.shape != hidden_states.shape[:2]:
        raise RuntimeError("modality block ids were not installed for this attention forward")
    pack_plan = getattr(self, "_starvla_modality_pack_plan", None)
    if pack_plan is None:
        # Unit/direct callers may not go through the wrapper. Production computes this
        # once per input batch and shares it across all layers and checkpoint replays.
        pack_plan = build_modality_block_pack_plan(block_ids)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.config._attn_implementation != "flash_attention_3":
        raise RuntimeError("modality_block_causal requires flash_attention_3")
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    batch, query_heads, _, head_dim = query_states.shape
    key_value_heads = key_states.shape[1]
    output = query_states.new_zeros((batch, input_shape[1], query_heads, head_dim))
    # The wrapper builds these indices once. Vectorized gathers avoid reconstructing the
    # same groups in every layer and again during gradient-checkpoint backward replays.
    source_rows = pack_plan["source_rows"]
    query_indices = pack_plan["query_indices"]
    key_indices = pack_plan["key_indices"]
    query_mask = pack_plan["query_mask"]
    kv_mask = pack_plan["key_mask"]
    q_pack = query_states[source_rows[:, None], :, query_indices].permute(0, 2, 1, 3)
    k_pack = key_states[source_rows[:, None], :, key_indices].permute(0, 2, 1, 3)
    v_pack = value_states[source_rows[:, None], :, key_indices].permute(0, 2, 1, 3)

    block_output, _ = attention_interface(
        self,
        q_pack,
        k_pack,
        v_pack,
        kv_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        is_causal=False,
        **kwargs,
    )
    destination = source_rows[:, None] * input_shape[1] + query_indices
    output = output.view(batch * input_shape[1], query_heads, head_dim).index_copy(
        0, destination[query_mask], block_output[query_mask]
    ).view(batch, input_shape[1], query_heads, head_dim)

    output = output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(output), None


def _full_prefix_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    **kwargs,
):
    """FA3 full-attention forward over the complete valid observation prefix."""
    if past_key_values is not None:
        raise RuntimeError(
            "full_prefix_bidir is a policy representation topology and does not support KV-cache generation"
        )
    plan = getattr(self, "_starvla_full_prefix_plan", None)
    if plan is None:
        raise RuntimeError("full-prefix pack plan was not installed before attention")
    if self.config._attn_implementation != "flash_attention_3":
        raise RuntimeError("full_prefix_bidir requires flash_attention_3")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    batch, query_heads, seq_len, head_dim = query_states.shape
    rows = torch.arange(batch, device=hidden_states.device)[:, None]
    indices = plan["indices"]
    keep = plan["keep"]
    q_pack = query_states[rows, :, indices].permute(0, 2, 1, 3)
    k_pack = key_states[rows, :, indices].permute(0, 2, 1, 3)
    v_pack = value_states[rows, :, indices].permute(0, 2, 1, 3)
    call_kwargs = dict(kwargs)
    call_kwargs.pop("is_causal", None)
    packed_output, _ = attention_interface(
        self,
        q_pack,
        k_pack,
        v_pack,
        keep,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        is_causal=False,
        **call_kwargs,
    )
    output = query_states.new_zeros((batch, seq_len, query_heads, head_dim))
    destination = rows * seq_len + indices
    output = output.view(batch * seq_len, query_heads, head_dim).index_copy(
        0, destination[keep], packed_output[keep]
    ).view(batch, seq_len, query_heads, head_dim)
    output = output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(output), None


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        self.attention_topology = str(qwenvl_config.get("attention_topology", "causal"))
        if self.attention_topology not in {"causal", "modality_block_causal", "full_prefix_bidir"}:
            raise ValueError(
                "framework.qwenvl.attention_topology must be 'causal', "
                "'modality_block_causal', or 'full_prefix_bidir'; "
                f"got {self.attention_topology!r}"
            )
        if self.attention_topology in {"modality_block_causal", "full_prefix_bidir"} and attn_implementation != "flash_attention_3":
            raise ValueError(f"{self.attention_topology} requires attn_implementation: flash_attention_3")
        # Position of the user text relative to the camera images (rebuttal R4 control).
        # Read from the (checkpoint) config, so eval reproduces the training layout.
        self.prompt_order = _resolve_prompt_order(qwenvl_config.get("prompt_order", None))
        if self.prompt_order != "image_first" and self.attention_topology == "modality_block_causal":
            # build_modality_block_ids assumes [images][language][state]: text placed
            # before the first image would be folded into the camera-1 block.
            raise ValueError(
                "framework.qwenvl.prompt_order != image_first is not supported with "
                "attention_topology=modality_block_causal"
            )
        if self.prompt_order != "image_first":
            print(f"__PROMPT_ORDER__ {self.prompt_order}", flush=True)

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        if self.attention_topology == "modality_block_causal":
            for layer in self.model.model.language_model.layers:
                layer.self_attn.forward = MethodType(
                    _modality_block_attention_forward, layer.self_attn
                )
            print(
                "__ATTENTION_TOPOLOGY__ modality_block_causal "
                "(bidirectional within camera/language/state blocks; causal across blocks; FA3)",
                flush=True,
            )
        elif self.attention_topology == "full_prefix_bidir":
            for layer in self.model.model.language_model.layers:
                layer.self_attn.forward = MethodType(
                    _full_prefix_attention_forward, layer.self_attn
                )
            print(
                "__ATTENTION_TOPOLOGY__ full_prefix_bidir "
                "(single causal Qwen stack; all valid observation tokens mutually visible; FA3)",
                flush=True,
            )

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        # Metadata used by the encoder-decoder adapter to split prompt from answer.
        # The causal Qwen model has no use for it and must not receive an unknown kwarg.
        kwargs.pop("_cot_encoder_prefix_length", None)

        state_token_mask = kwargs.pop("_state_token_mask", None)
        attentions = []
        if self.attention_topology == "modality_block_causal":
            if kwargs.get("input_ids") is None:
                raise RuntimeError("modality_block_causal requires input_ids")
            # This topology is used as a policy encoder, never for autoregressive token
            # generation. Disabling the cache also prevents HF from materializing an
            # empty DynamicCache and presenting it to the patched attention layers.
            kwargs["use_cache"] = False
            block_ids = build_modality_block_ids(
                kwargs["input_ids"],
                kwargs.get("attention_mask"),
                int(self.model.config.vision_start_token_id),
                int(self.model.config.vision_end_token_id),
                state_token_mask,
            )
            pack_plan = build_modality_block_pack_plan(block_ids)
            attentions = [layer.self_attn for layer in self.model.model.language_model.layers]
            for attention in attentions:
                attention._starvla_modality_block_ids = block_ids
                attention._starvla_modality_pack_plan = pack_plan
        elif self.attention_topology == "full_prefix_bidir":
            if kwargs.get("attention_mask") is None:
                if kwargs.get("input_ids") is None:
                    raise RuntimeError("full_prefix_bidir requires attention_mask or input_ids")
                valid_mask = torch.ones_like(kwargs["input_ids"], dtype=torch.long)
            else:
                valid_mask = kwargs["attention_mask"]
            kwargs["use_cache"] = False
            pack_plan = build_full_prefix_pack_plan(valid_mask)
            attentions = [layer.self_attn for layer in self.model.model.language_model.layers]
            for attention in attentions:
                attention._starvla_full_prefix_plan = pack_plan

        # Keep block_ids installed through backward: gradient checkpointing replays each
        # layer after this wrapper's forward has returned. The next batch overwrites them
        # before its own forward, after the preceding backward is complete.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        # bf16, not fp16: weights are cast to bf16 elsewhere (matches forward()'s
        # autocast). An fp16 autocast here forces every op to cast bf16 weights to
        # fp16 and back each decode step -- pure overhead, no accuracy upside.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        cot_conversations=None,
        cot_modes=None,
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions/cot_conversations).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        Args:
            images:            List[List[PIL.Image]]  — one inner list per sample
            instructions:      List[str]
            solutions:         List[str] | None  — fast-tokenizer action token sequences
            cot_conversations: List[list|None] | None  — per-sample ShareGPT conversations,
                               each [{from: human, value: ...}, {from: gpt, value: ...}] or None
                               (unannotated → no CoT loss for that sample). The human value may
                               contain {instruction}, which is filled in here.
            cot_modes:         List[str|None] | None — explicit ERVLA-style ``cot`` /
                               ``no_cot`` condition. The marker is appended to the user prompt.

        NOTE: this parameter used to be absent, so every framework that passed
        ``cot_conversations=`` (QwenGR00T, QwenGR00T_CoTrain) had it silently swallowed by
        **kwargs — no labels were built, ``qwenvl_outputs.loss`` stayed None, and no
        ``cot_loss`` key ever reached the trainer. Ported from QWen3_5.py, which already had
        this path; see RESULTS_TIER3.md §3 for the runs that were affected.
        """

        # Create user-only messages (prompt side).
        # For CoT samples the human turn comes from the conversation, not the CoT_prompt config.
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        if cot_modes is not None and len(cot_modes) != len(instructions):
            raise ValueError("cot_modes length must match batch size")
        prompt_order = getattr(self, "prompt_order", "image_first")
        # future_copies: the last _n_tail_images of each row go after the text (default 0: unchanged)
        n_tail = int(getattr(self, "_n_tail_images", 0) or 0)
        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            image_items = [{"type": "image", "image": img} for img in imgs]
            tail_items = []
            if n_tail:
                image_items, tail_items = image_items[:-n_tail], image_items[-n_tail:]

            conv = cot_conversations[i] if cot_conversations is not None else None
            if conv is not None:
                prompt = conv[0]["value"].replace("{instruction}", instruction)
            elif "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            mode = cot_modes[i] if cot_modes is not None else None
            if mode is not None:
                if mode not in {"cot", "no_cot"}:
                    raise ValueError(f"unsupported cot mode: {mode!r}")
                prompt = f"{prompt}\n/{mode}"

            content = arrange_prompt_content(image_items, prompt, prompt_order) + tail_items
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # ── CoT conversation path ─────────────────────────────────────────────────
        # Loss is computed only on gpt (assistant) tokens; human/system tokens are masked.
        if cot_conversations is not None:
            assert len(cot_conversations) == len(messages), "cot_conversations length must match batch size"

            messages_with_cot = []
            for msg, conv in zip(messages, cot_conversations):
                gpt_text = conv[1]["value"] if conv is not None else ""
                messages_with_cot.append(
                    msg + [{"role": "assistant", "content": [{"type": "text", "text": gpt_text}]}]
                )

            # Encode prompt-only to measure per-sample prompt lengths. With
            # add_generation_prompt=True the encoding ends exactly where the assistant starts.
            prompt_only_inputs = self.processor.apply_chat_template(
                messages,  # user-only messages
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            batch_inputs = self.processor.apply_chat_template(
                messages_with_cot,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Build labels: mask everything up to the assistant turn, supervise only the CoT
            # tokens. With LEFT padding the valid tokens are right-aligned, so the assistant
            # span is the last (full_valid - prompt_valid) positions.
            #
            # For the enc-dec backbone this is also what defines the architecture split:
            # _EncDecOuterMixin sets encoder_prefix_length = (labels != -100).argmax(dim=1),
            # so the bidirectional encoder sees exactly the prompt and the causal decoder
            # sees exactly the CoT answer.
            max_full_len = batch_inputs["input_ids"].shape[1]
            labels = batch_inputs["input_ids"].clone()
            encoder_prefix_length = []
            for i in range(labels.size(0)):
                prompt_valid = int(prompt_only_inputs["attention_mask"][i].sum())
                full_valid = int(batch_inputs["attention_mask"][i].sum())
                asst_len = full_valid - prompt_valid
                asst_start = max_full_len - max(asst_len, 0)
                encoder_prefix_length.append(asst_start)
                labels[i, :asst_start] = IGNORE_INDEX
                # An unmatched frame deliberately has no auxiliary target.  Appending an
                # empty assistant turn still adds chat-template terminators, so mask the
                # complete row instead of accidentally training on those special tokens.
                if cot_conversations[i] is None:
                    labels[i, :] = IGNORE_INDEX
                # Mask any residual pad tokens that ended up in the label.
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
            # Keep the prompt boundary even for all-ignored rows.  Deriving it with
            # `(labels != -100).argmax()` would return zero and remove that sample's image
            # and instruction from the bidirectional encoder.  The ordinary causal adapter
            # drops this private metadata in forward(); QWen3_EncDec consumes it.
            batch_inputs["_cot_encoder_prefix_length"] = torch.tensor(
                encoder_prefix_length, device=labels.device, dtype=torch.long)
            return batch_inputs.to(self.model.device)

        # Preparation for inference (no CoT)

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
