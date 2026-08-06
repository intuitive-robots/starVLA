# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""
QwenOFT_CoT — QwenOFT with step-based Chain-of-Thought supervision.

Same architecture as :class:`QwenOFT` (action special tokens + MLP head, parallel
continuous decoding). The addition is CoT supervision from a step-based mapping,
as in :class:`QwenGR00T_CoT`.

Key difference from the GR00T variant
-------------------------------------
GR00T's DiT reads the *whole* hidden-state sequence, so CoT tokens influence the
action head wherever they sit. QwenOFT instead gathers hidden states at specific
``<action>`` token positions, and attention is causal — so if the action tokens
stayed in the user turn (where plain QwenOFT puts them) they could never attend to
a CoT answer emitted in the assistant turn afterwards, and CoT would degrade to a
plain auxiliary VLM loss.

This framework therefore emits the action-token block at the END of the assistant
turn, after the reasoning:

    user:      <images> "Your task is X. Generate the 2D trajectory ..."
    assistant: "<|trace|>{...}<|/trace|>" + " Please predict the next N robot actions: <action>🔍..🔍<action>."
                └── supervised by CoT cross-entropy ──┘ └── masked out of the LM labels ──┘

Set ``framework.cot.action_tokens_after_cot: false`` to fall back to the plain
QwenOFT layout (action tokens in the user turn); CoT is then an auxiliary VLM loss
only and does not condition the action head.
"""

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from starVLA.dataloader.cot_resolver import assert_cot_prompt_consistent, build_cot_resolver
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)


class _ForceSuffixLogitsProcessor(LogitsProcessor):
    """Redirects greedy CoT decoding into a fixed action-token suffix in-place.

    Each row generates its own CoT freely (unconstrained argmax) until the FIRST
    step where that row's own argmax would pick EOS -- i.e. the model itself
    decided its reasoning is done -- or a step budget is hit. At that exact step,
    instead of letting the turn actually close, this forces the fixed
    ``suffix_ids`` token-by-token (the same ``<action>...<action>.`` block
    ``predict_action``'s two-pass path builds via a second full forward), then
    forces a real EOS once the suffix is exhausted.

    This lets one ``generate()`` call produce [CoT][action-block] end-to-end, so
    the action-token hidden states can be read directly from
    ``generate(..., output_hidden_states=True)`` -- no second full-sequence
    re-encode needed. Per-row state, so batches of independently-lengthed CoTs
    are handled correctly; the whole batch still stops as soon as every row has
    emitted its real, final EOS (same semantics HF's own early-stopping gives).
    """

    def __init__(self, suffix_ids: List[int], eos_token_id: int, cot_budget: int):
        self.suffix_ids = suffix_ids
        self.eos_token_id = eos_token_id
        self.cot_budget = max(1, cot_budget)
        self._mode: List[str] = []  # per-row: "cot" | "suffix" | "done"
        self._suffix_pos: List[int] = []
        self._cot_steps: List[int] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        if not self._mode:
            self._mode = ["cot"] * batch_size
            self._suffix_pos = [0] * batch_size
            self._cot_steps = [0] * batch_size

        neg_inf = torch.finfo(scores.dtype).min
        for i in range(batch_size):
            if self._mode[i] == "cot":
                self._cot_steps[i] += 1
                wants_to_stop = int(torch.argmax(scores[i]).item()) == self.eos_token_id
                if wants_to_stop or self._cot_steps[i] >= self.cot_budget:
                    self._mode[i] = "suffix"
                    self._suffix_pos[i] = 0
            if self._mode[i] == "suffix":
                pos = self._suffix_pos[i]
                forced_id = self.suffix_ids[pos] if pos < len(self.suffix_ids) else self.eos_token_id
                scores[i] = neg_inf
                scores[i, forced_id] = 0.0
                self._suffix_pos[i] = pos + 1
                if self._suffix_pos[i] > len(self.suffix_ids):
                    self._mode[i] = "done"
        return scores


@FRAMEWORK_REGISTRY.register("QwenOFT_CoT")
class QwenOFT_CoT(Qwenvl_OFT):
    """QwenOFT + step-based CoT supervision."""

    def __init__(self, config: Optional[dict] = None, is_inference: bool = False, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.is_inference = is_inference

        # Step-based CoT resolver — NullCoTResolver when cot.source is absent/none.
        self.cot_resolver = build_cot_resolver(self.config, is_inference=self.is_inference)
        assert_cot_prompt_consistent(self.cot_resolver, self.config)

        cot_cfg = getattr(self.config.framework, "cot", None)
        self.action_tokens_after_cot = bool(
            getattr(cot_cfg, "action_tokens_after_cot", True) if cot_cfg is not None else True
        )
        self.generate_at_inference = bool(
            getattr(cot_cfg, "generate_at_inference", False) if cot_cfg is not None else False
        )
        # STARVLA_FAST_ACTION_DECODE=1 overrides an existing checkpoint's saved
        # config.yaml without having to edit it (mirrors STARVLA_COMPILE_GENERATE
        # in QWen3_5.py).
        self.fast_action_decode = bool(
            getattr(cot_cfg, "fast_action_decode", False) if cot_cfg is not None else False
        ) or (os.environ.get("STARVLA_FAST_ACTION_DECODE", "") == "1")
        logger.info(
            f"[QwenOFT_CoT] action_tokens_after_cot={self.action_tokens_after_cot}, "
            f"generate_at_inference={self.generate_at_inference}, fast_action_decode={self.fast_action_decode}"
        )
        # Inference-time call counter, used to throttle the per-call timing logs below.
        self._infer_call_count = 0
        # Fixed action-token suffix ids, tokenized once (constant given a fixed chunk_len).
        # Used by the fast single-pass decode path (see _predict_action_fast).
        self._suffix_ids: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _action_prompt_suffix(self) -> str:
        """The ``<action>`` token block appended to the prompt (identical to QwenOFT)."""
        action_tokens = self.action_token * self.chunk_len
        return f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."

    def _resolve_cot(self, examples: List[dict]) -> Tuple[Optional[list], float]:
        """Resolve per-sample CoT conversations; returns (conversations_or_None, coverage)."""
        cot_conversations = [
            self.cot_resolver.resolve(ex.get("trajectory_name", ""), ex.get("frame_index", 0)) for ex in examples
        ]
        has_cot = any(c is not None for c in cot_conversations)
        coverage = sum(c is not None for c in cot_conversations) / max(len(cot_conversations), 1)
        return (cot_conversations if has_cot else None), coverage

    def _build_inputs(self, batch_images, instructions, cot_conversations):
        """Build QwenVL inputs, placing the action tokens per ``action_tokens_after_cot``."""
        suffix = self._action_prompt_suffix()
        if cot_conversations is not None and self.action_tokens_after_cot:
            # Action tokens go at the end of the assistant turn, behind the reasoning.
            return self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                cot_conversations=cot_conversations,
                assistant_suffix=[suffix] * len(instructions),
            )
        # Plain QwenOFT layout: action tokens live in the user turn.
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=[i + suffix for i in instructions],
            cot_conversations=cot_conversations,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        cot_conversations, cot_coverage = self._resolve_cot(examples)
        qwen_inputs = self._build_inputs(batch_images, instructions, cot_conversations)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # CoT CE loss from the VLM language head (labels only attached on the CoT path).
        cot_loss = qwenvl_outputs.loss if (cot_conversations is not None and qwenvl_outputs.loss is not None) else None

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            action_loss = self.l1_loss(pred_actions, actions_target)

        result = {"action_loss": action_loss, "cot_coverage": cot_coverage}
        if cot_loss is not None:
            result["cot_loss"] = cot_loss
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        """Predict a normalized action chunk.

        Implicit CoT (``generate_at_inference: false``, default):
            One pass. The assistant turn holds the action-token block with an EMPTY CoT
            answer; reasoning is implicit in the backbone from training. Cheaper, but
            asymmetric with training, where the action tokens follow a real CoT answer.

        Explicit CoT (``generate_at_inference: true``):
            Pass 1 — greedily generate the CoT text from the user prompt alone.
            Pass 2 — re-encode [prompt + generated CoT + action tokens] so the action
                     queries attend over the reasoning, exactly as in training.
            Costs an extra autoregressive decode of up to ``cot.max_new_tokens`` per step.

            Set ``framework.cot.fast_action_decode: true`` to instead do this in a
            SINGLE ``generate()`` call (see ``_predict_action_fast``): the fixed
            action-token suffix is forced onto the end of the same decode that
            produced the CoT, so there's no second full-sequence re-encode. Falls
            back to the two-pass path automatically if anything about the fast
            path errors.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        if self.action_tokens_after_cot and self.generate_at_inference and self.fast_action_decode:
            try:
                return self._predict_action_fast(batch_images, instructions)
            except Exception:
                logger.error(
                    "[QwenOFT_CoT] fast_action_decode path failed; falling back to the two-pass decode for this call.",
                    exc_info=True,
                )

        t_start = time.time()
        if self.action_tokens_after_cot:
            if self.generate_at_inference:
                # Pass 1: generate the reasoning from the user prompt alone.
                generated_cot = self._generate_cot(batch_images, instructions)
                cot_text = generated_cot
            else:
                # Implicit CoT: empty answer, action tokens still in the assistant turn
                # so their positions match training as closely as possible.
                cot_text = [""] * len(instructions)

            # Pass 2: re-encode [prompt + CoT + action tokens].
            human = self._cot_human_prompt()
            cot_conversations = [
                [{"from": "human", "value": human}, {"from": "gpt", "value": txt}] for txt in cot_text
            ]
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                cot_conversations=cot_conversations,
                assistant_suffix=[self._action_prompt_suffix()] * len(instructions),
            )
            # Inference needs hidden states only, not the CE loss.
            qwen_inputs.pop("labels", None)
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=[i + self._action_prompt_suffix() for i in instructions],
            )

        t_after_cot = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        # Occasional timing log: which pass is actually eating the latency
        # (CoT generation vs. the single forward + action head).
        self._infer_call_count += 1
        if self._infer_call_count % 5 == 1:
            t_end = time.time()
            logger.info(
                f"[QwenOFT_CoT] predict_action call #{self._infer_call_count}: "
                f"cot_gen={t_after_cot - t_start:.3f}s, forward+head={t_end - t_after_cot:.3f}s, "
                f"total={t_end - t_start:.3f}s, batch_size={len(instructions)}"
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    @torch.inference_mode()
    def _predict_action_fast(self, batch_images: list, instructions: list) -> dict:
        """Single-``generate()``-call explicit-CoT inference (no second re-encode).

        Replaces the two-pass ``_generate_cot`` + full-forward with one greedy
        decode: :class:`_ForceSuffixLogitsProcessor` lets each row generate CoT
        freely, then transparently redirects it into the fixed action-token
        suffix the moment that row would otherwise emit EOS, then closes with a
        real EOS. Action-token hidden states come straight out of
        ``generate(..., output_hidden_states=True)`` -- assembled into the same
        ``[B, L, H]`` shape ``_gather_action_token_embeddings`` already expects,
        so it's reused unmodified (its per-row action-token-count check is also
        our correctness guard: if the forced suffix wasn't actually inserted for
        some row, it raises instead of silently returning garbage actions).

        NOTE: opt-in via ``framework.cot.fast_action_decode: true``. This bypasses
        the chat-template re-tokenization of [CoT text + suffix] that the two-pass
        path does, forcing the pre-tokenized suffix ids directly onto the model's
        own generated tokens instead -- verify predictions match the two-pass path
        on a few examples before trusting it for a full eval run.
        """
        max_new = self._cot_cfg_get("max_new_tokens", 128)
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise RuntimeError("fast_action_decode requires tokenizer.eos_token_id to be set.")

        if self._suffix_ids is None:
            self._suffix_ids = tokenizer(self._action_prompt_suffix(), add_special_tokens=False)["input_ids"]

        prompt_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        cot_budget = max_new - len(self._suffix_ids) - 1  # leave room for suffix + final EOS
        processor = _ForceSuffixLogitsProcessor(self._suffix_ids, eos_token_id, cot_budget)

        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_out = self.qwen_vl_interface.generate(
                **prompt_inputs,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                logits_processor=LogitsProcessorList([processor]),
            )
        elapsed = time.time() - t0

        input_ids = gen_out.sequences  # [B, prompt_len + T]
        # `hidden_states`: tuple of T entries (one per generated token); entry 0 covers
        # the whole prompt (the prefill), entries 1..T-1 cover one new token each. That's
        # prompt_len + (T-1) positions total -- one short of `input_ids` because nothing
        # is generated *from* the final token, so its hidden state was never computed.
        # Pad one dummy row to line up with input_ids; harmless because the suffix
        # template always has closing tokens ("<action>." + EOS) after the last real
        # action token, so an action-token position is never that final row.
        last_hidden = torch.cat([step_hs[-1] for step_hs in gen_out.hidden_states], dim=1)
        if last_hidden.shape[1] < input_ids.shape[1]:
            last_hidden = torch.cat([last_hidden, last_hidden[:, -1:, :]], dim=1)
        if last_hidden.shape[1] != input_ids.shape[1]:
            raise RuntimeError(
                f"fast_action_decode: hidden_states length {last_hidden.shape[1]} != "
                f"sequences length {input_ids.shape[1]}; refusing to guess action-token positions."
            )

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            pred_actions = self.action_model.predict_action(action_queries)

        self._infer_call_count += 1
        if self._infer_call_count % 5 == 1:
            logger.info(
                f"[QwenOFT_CoT] fast predict_action call #{self._infer_call_count}: "
                f"{elapsed:.3f}s single-pass (cot+action, no re-encode), "
                f"cot_lens={processor._cot_steps}, gen_len={input_ids.shape[1] - prompt_len}, "
                f"batch_size={len(instructions)}"
            )

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _cot_human_prompt(self) -> str:
        """The configured CoT prompt (``{instruction}`` is substituted downstream)."""
        vla = self.config.datasets.vla_data
        if hasattr(vla, "get"):
            return vla.get("CoT_prompt", "{instruction}")
        return getattr(vla, "CoT_prompt", "{instruction}")

    def _cot_cfg_get(self, key: str, default):
        cot_cfg = getattr(self.config.framework, "cot", None)
        if cot_cfg is None:
            return default
        if hasattr(cot_cfg, "get"):
            return cot_cfg.get(key, default)
        return getattr(cot_cfg, key, default)

    @torch.inference_mode()
    def _generate_cot(self, batch_images: list, instructions: list) -> List[str]:
        """Pass 1 of explicit CoT inference: greedily generate one CoT string per sample.

        Mirrors :meth:`QwenGR00T_CoT._generate_cot`. The prompt deliberately excludes the
        action-token suffix so the model produces reasoning only; the suffix is appended in
        pass 2, behind the generated text.
        """
        max_new = self._cot_cfg_get("max_new_tokens", 128)

        # cot_conversations=None → build_qwenvl_inputs uses CoT_prompt and
        # add_generation_prompt=True, i.e. the sequence ends where the assistant starts.
        prompt_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_ids = self.qwen_vl_interface.generate(
                **prompt_inputs,
                max_new_tokens=max_new,
                do_sample=False,
            )
        elapsed = time.time() - t0

        new_ids = gen_ids[:, prompt_len:]
        decoded = self.qwen_vl_interface.processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        # HF `generate` on a batch only stops once EVERY sequence has emitted EOS (or hit
        # max_new_tokens) -- one straggler sample forces the whole batch through all
        # `max_new_tokens` steps. Log actual vs. max tokens/sec so a slow run is visible.
        if self._infer_call_count % 5 == 0:
            gen_lens = (new_ids != self.qwen_vl_interface.processor.tokenizer.pad_token_id).sum(dim=1).tolist()
            logger.info(
                f"[QwenOFT_CoT] _generate_cot: {elapsed:.3f}s for max_new_tokens={max_new} "
                f"({elapsed / max_new * 1000:.1f} ms/token-slot), prompt_len={prompt_len}, "
                f"actual_gen_lens={gen_lens}, sample0='{decoded[0][:100]}'"
            )
        return decoded
