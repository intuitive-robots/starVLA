# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import os
from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        # Opt-in fast decode: static KV-cache + torch.compile(mode="reduce-overhead")
        # on the per-step forward used by generate(). Off by default -- first call
        # pays a real compilation warmup (tens of seconds), and only the decode-step
        # call inside generate() is ever compiled (see `generate()` below); the plain
        # `forward()` used for training and the fused-forward inference paths is
        # untouched. Enable via `framework.qwenvl.compile_generate: true` in the
        # model config, or the STARVLA_COMPILE_GENERATE=1 env var for an existing
        # checkpoint without editing its saved config.yaml.
        self._compile_generate = bool(qwenvl_config.get("compile_generate", False)) or (
            os.environ.get("STARVLA_COMPILE_GENERATE", "") == "1"
        )
        self._compiled_forward = None
        if self._compile_generate and not self.model._supports_default_dynamic_cache():
            # Qwen3_5ForConditionalGeneration sets `_is_stateful = True`, which makes
            # `_supports_default_dynamic_cache()` return False -- HF's own generate()
            # then silently IGNORES `cache_implementation="static"` and keeps using
            # whatever legacy/growing cache the model manages internally. That means
            # the decode-step shape still changes every step, so the "compiled"
            # forward would recompile on every single token instead of reusing one
            # graph -- strictly slower than plain eager, not just unhelpful. Refuse
            # to enable it rather than silently regressing throughput.
            # (Confirmed still broken as of transformers 5.3.0; Qwen2.5-VL and
            # Qwen3-VL do NOT have this problem, only the 3.5 line.)
            logger.warning(
                "[QWen3_5] compile_generate requested but this model class "
                "(Qwen3_5ForConditionalGeneration) does not support the standard "
                "Cache API in the installed transformers version, so "
                "cache_implementation='static' would be silently ignored and the "
                "compiled forward would recompile every decode step -- net SLOWER "
                "than eager. Disabling compile_generate; leaving fast_action_decode "
                "and the bf16 generate() fix (both unaffected by this) in place."
            )
            self._compile_generate = False
        if self._compile_generate:
            # Dynamo's default guard cache (8) is too small once prompts vary in
            # length across calls (e.g. different instructions/episodes) -- each new
            # prompt length is a new shape and needs its own compiled graph. Past
            # the limit, dynamo silently falls back to eager instead of erroring, but
            # that quietly loses the speedup, so raise the ceiling.
            import torch._dynamo as _dynamo

            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            logger.info(
                "[QWen3_5] compile_generate=True: the decode-step forward will be "
                "torch.compile'd (mode='reduce-overhead') + static KV-cache on first "
                "use. Expect a one-time warmup of ~tens of seconds per distinct "
                "prompt length seen."
            )

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

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
        # bf16, not fp16: weights are cast to bf16 in __init__ (matches forward()'s
        # autocast). An fp16 autocast here forces every op to cast bf16 weights to
        # fp16 and back each step -- pure overhead on top of an already per-step-heavy
        # decode loop, with no accuracy upside.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self._compile_generate:
                generation_output = self._generate_compiled(**kwargs)
            else:
                generation_output = self.model.generate(
                    **kwargs,
                )
        return generation_output

    def _generate_compiled(self, **kwargs):
        """Static-cache + compiled-forward generate().

        A dynamic KV-cache grows every step, so its shape changes every step, so
        torch.compile has to recompile every step -- useless. A static cache
        pre-allocates a fixed-size buffer instead, so once the cache is warm the
        decode-step forward always sees the SAME input/cache shape (`[B, 1]` +
        fixed-length cache) and one compiled graph (with CUDA graph capture, under
        `mode="reduce-overhead"`) covers every remaining step.

        `self.model.forward` is only swapped for a compiled version for the
        duration of this call, then restored -- training's `forward()` (variable
        image counts / sequence lengths, `output_hidden_states=True`) and the
        fused-forward inference path never see the compiled callable, so they
        can't be destabilized by this.
        """
        if self._compiled_forward is None:
            self._compiled_forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=False)

        kwargs.setdefault("cache_implementation", "static")
        original_forward = self.model.forward
        self.model.forward = self._compiled_forward
        try:
            return self.model.generate(**kwargs)
        except Exception:
            # Disable permanently (not just for this call): a compile/static-cache
            # failure is almost always shape- or op-related and will recur on every
            # subsequent call, so retrying it forever would just waste a compile
            # attempt's worth of time on every single request.
            self._compile_generate = False
            logger.error(
                "[QWen3_5] compiled generate() failed; disabling compile_generate "
                "and falling back to the uncompiled path for the rest of this run.",
                exc_info=True,
            )
            self.model.forward = original_forward
            return self.model.generate(
                **{k: v for k, v in kwargs.items() if k != "cache_implementation"},
            )
        finally:
            self.model.forward = original_forward

    def build_qwenvl_inputs(
        self, images, instructions, solutions=None, cot_conversations=None, assistant_suffix=None, **kwargs
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions/cot_conversations).
        Follow Official Qwen3.5-VL Instruct format: https://huggingface.co/Qwen/Qwen3.5-VL-4B-Instruct

        Args:
            images:            List[List[PIL.Image]]  — one inner list per sample
            instructions:      List[str]
            solutions:         List[str] | None  — fast-tokenizer action token sequences (existing path)
            cot_conversations: List[list|None] | None  — per-sample ShareGPT conversations.
                               Each entry is [{from: human, value: ...}, {from: gpt, value: ...}]
                               or None (holdout / unannotated → no CoT loss for that sample).
                               The human value may contain {instruction} which is filled in here.
            assistant_suffix:  List[str] | None — appended to each sample's ASSISTANT turn,
                               after the CoT answer. Used by QwenOFT_CoT to place its
                               <action> token block behind the reasoning so the action
                               queries can attend to it (attention is causal). These suffix
                               tokens are excluded from the CoT cross-entropy labels: only
                               the CoT answer itself is supervised.
                               Requires cot_conversations (ignored otherwise).
        """

        # Create user-only messages (prompt side).
        # For CoT samples, the human turn comes from the conversation (not CoT_prompt config).
        # For non-CoT samples, fall back to CoT_prompt config or bare instruction.
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for i, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            conv = cot_conversations[i] if cot_conversations is not None else None
            if conv is not None:
                prompt = conv[0]["value"].replace("{instruction}", instruction)
            elif "CoT_prompt" in self.config.datasets.vla_data:
                prompt = self.config.datasets.vla_data.get("CoT_prompt", "").replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[i]}]})
            messages.append(msg)

        # ── CoT conversation path ─────────────────────────────────────────────────
        # Loss is computed only on gpt (assistant) tokens; human/system tokens are masked.
        if cot_conversations is not None:
            assert len(cot_conversations) == len(messages), "cot_conversations length must match batch size"

            if assistant_suffix is not None:
                assert len(assistant_suffix) == len(messages), "assistant_suffix length must match batch size"

            # Append assistant (gpt) turn. None entries → empty string → no loss contribution.
            # ``messages_with_cot`` may additionally carry the action-token suffix; we keep a
            # separate ``messages_cot_only`` (answer without suffix) to locate the end of the
            # supervised span below.
            messages_with_cot = []
            messages_cot_only = []
            for i, (msg, conv) in enumerate(zip(messages, cot_conversations)):
                gpt_text = conv[1]["value"] if conv is not None else ""
                suffix = assistant_suffix[i] if assistant_suffix is not None else ""

                msg_cot = list(msg)
                msg_cot.append({"role": "assistant", "content": [{"type": "text", "text": gpt_text + suffix}]})
                messages_with_cot.append(msg_cot)

                msg_only = list(msg)
                msg_only.append({"role": "assistant", "content": [{"type": "text", "text": gpt_text}]})
                messages_cot_only.append(msg_only)

            # Encode prompt-only (no assistant) to measure per-sample prompt lengths.
            # With add_generation_prompt=True the encoding ends exactly where the assistant starts.
            prompt_only_inputs = self.processor.apply_chat_template(
                messages,  # user-only messages
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Encode full sequence (user + assistant CoT).
            batch_inputs = self.processor.apply_chat_template(
                messages_with_cot,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )

            # Build labels: mask everything up to (and including) the prompt, supervise only
            # the assistant CoT tokens. With left padding the valid tokens are right-aligned.
            # When an action-token suffix was appended, encode the answer-only variant so we
            # can stop supervision at the end of the CoT answer.
            cot_only_inputs = None
            empty_asst_inputs = None
            if assistant_suffix is not None:
                cot_only_inputs = self.processor.apply_chat_template(
                    messages_cot_only,
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                # Same messages with an EMPTY assistant turn. The difference from the
                # prompt-only length is exactly the assistant scaffolding (<|im_end|> etc.),
                # which we must subtract or the supervised span would run past the CoT
                # answer and into the action-token suffix.
                messages_empty_asst = []
                for msg in messages:
                    m = list(msg)
                    m.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})
                    messages_empty_asst.append(m)
                empty_asst_inputs = self.processor.apply_chat_template(
                    messages_empty_asst,
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )

            max_full_len = batch_inputs["input_ids"].shape[1]
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                prompt_valid = int(prompt_only_inputs["attention_mask"][i].sum())
                full_valid = int(batch_inputs["attention_mask"][i].sum())
                asst_len = full_valid - prompt_valid
                # With left padding: last `full_valid` positions are real tokens;
                # last `asst_len` of those are the assistant response.
                asst_start = max_full_len - max(asst_len, 0)
                labels[i, :asst_start] = IGNORE_INDEX
                if cot_only_inputs is not None:
                    # Supervise only [asst_start, asst_start + cot_len): the CoT answer.
                    # Everything after it is the action-token block, which must not be
                    # trained as a language target. ``scaffold`` removes the assistant
                    # wrapper tokens (<|im_end|> ...) counted by the answer-only encoding;
                    # note this also drops EOS supervision, which is intentional here since
                    # the sequence does not end after the answer.
                    cot_valid = int(cot_only_inputs["attention_mask"][i].sum())
                    scaffold = int(empty_asst_inputs["attention_mask"][i].sum()) - prompt_valid
                    cot_len = max(cot_valid - prompt_valid - max(scaffold, 0), 0)
                    labels[i, asst_start + cot_len :] = IGNORE_INDEX
                # Mask any residual pad tokens that ended up in the label.
                if self.processor.tokenizer.pad_token_id is not None:
                    labels[i][labels[i] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
            return batch_inputs.to(self.model.device)

        # Preparation for inference (no CoT / no solutions)
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
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
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

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
