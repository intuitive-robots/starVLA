"""Proprioceptive state as dedicated vocabulary tokens.

Shared by every framework that routes state through the VLM rather than injecting it at
the action head. ``Qwen_PI_v3`` had an inline text-only version of this; ``Qwen_GR00T``
had no state pathway at all, so `include_state: true` there only reached the DiT head --
which is exactly the late conditioning ThinkProprio (arXiv:2602.06575, Tab. 7) measures as
indistinguishable from omitting state entirely.

Two encodings, selected by ``framework.state_token_mode``:

``text`` (default, back-compatible)
    Bins are rendered as decimal text, "95 133 203", and tokenized by the BPE tokenizer.
    One state dimension becomes 1-3 tokens depending on the magnitude of its bin index, so
    the sequence length varies per sample, bin 95 and bin 9 share a sub-token embedding,
    and every bin inherits whatever the pretrained model already believes about that
    numeral.

``reserved_ids``
    Bin ``b`` maps to token id ``vocab_size - 1 - b`` (the scheme in ThinkProprio).
    Qwen3-VL-2B declares ``text_config.vocab_size = 151936`` while the highest real or
    special token id is 151668, so ids 151669..151935 are 267 embedding rows that no token
    maps to. 256 bins land inside that region with 11 rows to spare: no resize, no
    collision with any real token, exactly one dedicated embedding per bin, and a fixed
    ``state_dim`` tokens per sample.

    No string tokenizes to a chosen id, so the prompt instead carries ``state_dim`` copies
    of a placeholder guaranteed to be one token, and the ids are spliced into ``input_ids``
    after the processor runs. The token count is unchanged by the splice, so
    ``attention_mask`` and the mrope position derivation are untouched.
"""

from typing import List, Optional

import numpy as np
import torch

STATE_PLACEHOLDER = "<|fim_pad|>"


class StateTokenMixin:
    """Quantised proprioception routed through the VLM's token pathway."""

    STATE_PLACEHOLDER = STATE_PLACEHOLDER

    def _configure_state_tokens(self, config) -> None:
        cfg = config.framework
        state_routing = str(cfg.get("state_routing", "legacy"))
        if state_routing not in ("legacy", "token_only"):
            raise ValueError(
                "framework.state_routing must be 'legacy' or 'token_only', got "
                f"{state_routing!r}"
            )
        if state_routing == "token_only":
            include_state = bool(config.datasets.vla_data.get("include_state", False))
            state_dim = int(cfg.action_model.get("state_dim", 0) or 0)
            if not include_state:
                raise ValueError("state_routing=token_only requires datasets.vla_data.include_state=true")
            if str(cfg.get("state_token_mode", "text")) != "reserved_ids":
                raise ValueError("state_routing=token_only requires state_token_mode=reserved_ids")
            if state_dim != 0:
                raise ValueError(
                    "state_routing=token_only requires action_model.state_dim=0; "
                    f"got {state_dim}, which would create a continuous action-head bypass"
                )
        # A scalar, or a ramp {start, end, start_step, end_step} over optimizer steps (same shape as
        # modality_dropout.vision_rate; 2026-10-05, annealed state masking). state_dropout_rate stays a float
        # (the larger endpoint) for the validity checks elsewhere; the per-step rate comes from the schedule.
        rate_cfg = cfg.get("state_dropout_rate", 0.0)
        if hasattr(rate_cfg, "get"):
            self._state_rate_schedule = dict(rate_cfg)
            self.state_dropout_rate = max(float(rate_cfg.get("start", 0.0)),
                                          float(rate_cfg.get("end", rate_cfg.get("start", 0.0))))
        else:
            self._state_rate_schedule = None
            self.state_dropout_rate = float(rate_cfg)
        # state_follows_instruction (10-03, default off): mask the state on exactly the rows whose instruction
        # is masked (one draw at instruction_dropout_rate, reused by _mask_instruction_tokens) and on no other
        # row; state_dropout_rate is then not used. Arm A (instruction masked, state always present) learned the
        # task from the state and collapsed on LIBERO-goal; this keeps the state whenever the instruction is there.
        self.state_follows_instruction = bool(cfg.get("state_follows_instruction", False))
        self._instr_draw_pending = None
        if not 0.0 <= self.state_dropout_rate <= 1.0:
            raise ValueError(
                "framework.state_dropout_rate must be between 0 and 1, "
                f"got {self.state_dropout_rate}"
            )
        self.state_token_mode = str(cfg.get("state_token_mode", "text"))
        if self.state_token_mode not in ("text", "reserved_ids"):
            raise ValueError(
                "framework.state_token_mode must be 'text' or 'reserved_ids', "
                f"got {self.state_token_mode!r}"
            )
        self.state_num_bins = int(cfg.get("state_num_bins", 256))
        self.state_token_group_sizes = [
            int(size) for size in cfg.get("state_token_group_sizes", [])
        ]
        if any(size <= 0 for size in self.state_token_group_sizes):
            raise ValueError(
                "framework.state_token_group_sizes must contain positive integers, got "
                f"{self.state_token_group_sizes}"
            )
        self.state_token_group_names = [
            str(name) for name in cfg.get("state_token_group_names", [])
        ]
        if self.state_token_group_names and (
            len(self.state_token_group_names) != len(self.state_token_group_sizes)
        ):
            raise ValueError(
                "framework.state_token_group_names must have one entry per group: "
                f"names={self.state_token_group_names}, sizes={self.state_token_group_sizes}"
            )
        self.state_token_permutation = [
            int(index) for index in cfg.get("state_token_permutation", [])
        ]
        # BACKWARD COMPATIBILITY, and it is load-bearing. Checkpoints trained before the
        # rare_real scheme encode state at the dead reserved ids (vocab_size-1-b) and have
        # no state_bin_embed module. Defaulting to rare_real made `from_pretrained` build a
        # model those checkpoints cannot fill ("Missing key(s): state_bin_embed.weight") --
        # and, had the key loaded, it would have looked up entirely DIFFERENT embedding rows
        # than training used, producing plausible but meaningless numbers. So an unset
        # source means "old checkpoint": reserved rows, no trainable bins. New configs opt
        # in explicitly with state_token_source: rare_real.
        self.state_token_source = str(cfg.get("state_token_source", "reserved"))
        if self.state_token_source not in ("rare_real", "reserved", "fast_codec"):
            raise ValueError(
                "framework.state_token_source must be 'rare_real', 'reserved' or 'fast_codec', got "
                f"{self.state_token_source!r}"
            )
        if self.state_num_bins < 2:
            raise ValueError(
                f"framework.state_num_bins must be >= 2, got {self.state_num_bins}"
            )
        self._pending_state_bins: Optional[List[Optional[np.ndarray]]] = None
        self._state_placeholder_id: Optional[int] = None
        self._state_splice_logged = False
        self._last_state_token_mask: Optional[torch.Tensor] = None
        if self.state_token_mode != "reserved_ids":
            return

        tokenizer = self._resolve_tokenizer()
        ids = tokenizer.convert_tokens_to_ids([self.STATE_PLACEHOLDER])
        unk = getattr(tokenizer, "unk_token_id", None)
        if not ids or ids[0] is None or (unk is not None and ids[0] == unk):
            raise RuntimeError(
                f"state placeholder {self.STATE_PLACEHOLDER!r} is not a single token in "
                "this tokenizer; reserved_ids state encoding cannot be used"
            )
        self._state_placeholder_id = int(ids[0])

        num_group_tokens = len(self.state_token_group_sizes)
        num_state_ids = self.state_num_bins + num_group_tokens
        if self.state_token_source == "fast_codec":
            # The MolmoAct2-FAST co-trained VLMs (train_downstream final_action_*) added
            # <|state_000|>..<|state_255|>: 256 uniform bins of the q01..q99 range over
            # [-1, 1], the same grid as _state_bins. Reuse those learned rows directly.
            if self.state_num_bins != 256 or num_group_tokens:
                raise ValueError("fast_codec state tokens need state_num_bins 256 and no group tokens")
            names = [f"<|state_{b:03d}|>" for b in range(256)]
            ids = tokenizer.convert_tokens_to_ids(names)
            if any(i is None or (unk is not None and i == unk) for i in ids):
                raise RuntimeError("fast_codec: tokenizer lacks <|state_k|> tokens; load the "
                                   "processor from the FAST co-trained checkpoint (base_vlm)")
            self._state_bin_ids = [int(i) for i in ids]
            self._state_group_ids = []
        elif self.state_token_source == "rare_real":
            # Bin b -> the b-th Latin-free real token row (trained, distinct, proper norm).
            # Optional group markers (for example left/right arms) use the following rows,
            # so they are also fixed, tokenizer-native IDs and cannot collide with bins.
            state_ids = list(self._rare_token_ids(num_state_ids))
            self._state_bin_ids = state_ids[: self.state_num_bins]
            self._state_group_ids = state_ids[self.state_num_bins :]
        else:
            vocab_size = self._resolve_vocab_size()
            highest_real = max(int(i) for i in tokenizer.get_vocab().values())
            lowest_state_id = vocab_size - num_state_ids
            if lowest_state_id <= highest_real:
                raise ValueError(
                    f"reserved needs {num_state_ids} free embedding rows above the "
                    f"highest real token id {highest_real}, but vocab_size={vocab_size} "
                    f"leaves only {vocab_size - 1 - highest_real}."
                )
            self._state_bin_ids = [vocab_size - 1 - b for b in range(self.state_num_bins)]
            self._state_group_ids = [
                vocab_size - 1 - self.state_num_bins - group
                for group in range(num_group_tokens)
            ]
        self._state_token_base = None  # superseded by the explicit table
        # Trainable bins only make sense with rare_real, and enabling them by default would
        # break every pre-existing checkpoint (see the note on state_token_source above).
        if bool(cfg.get("trainable_bin_embedding", self.state_token_source == "rare_real")):
            self._install_trainable_bin_embedding(
                self._state_bin_ids + self._state_group_ids,
                name="state_bin_embed",
            )
        # state_dropout_mask (default off): a dropped state keeps its slots, filled with ONE learned
        # mask id, instead of being omitted. Same prompt layout as a present state, so the slots can
        # carry a reconstruction target (fusion_registers.state). Inference never drops the state.
        self.state_dropout_mask = bool(cfg.get("state_dropout_mask", False))
        self._state_true_bins = None
        if self.state_dropout_mask:
            if self.state_token_source != "rare_real":
                raise ValueError("state_dropout_mask needs state_token_source rare_real")
            n_state = len(self._state_bin_ids) + len(self._state_group_ids)
            # well past the ids fusion registers take after the state ids (n_state .. n_state + 8)
            self._state_mask_id = int(list(self._rare_token_ids(n_state + 65))[-1])
            if self._state_mask_id in set(self._state_bin_ids + self._state_group_ids):
                raise RuntimeError("state mask id collides with a state id")
            self._install_trainable_bin_embedding([self._state_mask_id], name="state_mask_embed")

    # ------------------------------------------------------------------
    # Where the per-bin embedding rows come from.
    #
    # "reserved" (the original scheme) claims the dead rows above the highest real token id.
    # Measured on Qwen3-VL-2B those 257 rows hold only **7 distinct vectors**, pairwise
    # distance 0.0008 against a norm of 0.36 -- they map to no token, so pretraining never
    # touched them. Every bin therefore looks identical to the model. With the embedding
    # table frozen (e.g. freeze_modules: qwen_vl_interface) that is fatal: bin 0 and bin 255
    # are the same input, and a run using them measures nothing. That is exactly what
    # happened to jobs 1898597/1899196.
    #
    # "rare_real" instead repurposes REAL token rows whose decoded text contains no Latin
    # character at all (Cyrillic, CJK, CJK punctuation, ...). 45,162 such tokens exist. They
    # cannot be produced by an English instruction, by the quantised state digits, or by the
    # [STATE]/[TRACE] markers, so repurposing them costs nothing we use -- and unlike the
    # dead rows they are properly trained and diverse:
    #
    #     dead reserved rows : norm 0.36, 7/257 distinct, pairwise distance 0.0008
    #     rare real rows     : norm 1.48, 256/256 distinct, pairwise distance 2.05
    #
    # Being distinct at initialisation is what lets a FROZEN embedding table still carry the
    # signal, which is what makes an image-free stage with a frozen backbone viable at all.
    # Token IDs are tokenizer-specific. Qwen3-VL and Qwen3.5 have different vocabulary
    # sizes and assign different IDs even to the shared placeholder, so a single class-wide
    # list can silently reuse Qwen3-VL rows when a process later constructs Qwen3.5.
    _RARE_ID_CACHE = {}

    def _install_trainable_bin_embedding(self, bin_ids, name="bin_embed"):
        """Make ONLY the repurposed bin rows trainable, leaving the backbone frozen.

        A repurposed token starts at an embedding that means something else -- `rare_real`
        rows are real Cyrillic/CJK tokens. Freezing them hardcodes a wrong prior: the row has
        to move from meaning "о" to meaning "bin 137". But the two obvious ways to allow that
        are both bad here:

          * unfreeze the whole table -> 311.2M parameters (151936 x 2048), ~3.7 GiB of
            optimizer state under ZeRO-2 for the ~600 rows we care about, and the LANGUAGE
            rows drift too. During an image-free stage that drift is fitted to a vision-free
            objective and then carried into the sighted stage.
          * leave it frozen (freeze_modules: qwen_vl_interface) -> the rows are distinct, so
            they carry signal, but they cannot specialise.

        So instead: a small nn.Embedding over just the bin rows, initialised FROM those rows
        so training starts exactly where the frozen lookup would, owned by the framework
        rather than by qwen_vl_interface so `freeze_modules: qwen_vl_interface` does not
        reach it. A forward hook on the backbone's embedding replaces the vectors at bin
        positions with this module's output. ~1.2M trainable parameters instead of 311.2M.
        """
        import torch.nn as nn

        table = None
        vocab = self._resolve_vocab_size()
        for mod in self.qwen_vl_interface.modules():
            if isinstance(mod, nn.Embedding) and mod.num_embeddings == vocab:
                table = mod
                break
        if table is None:
            raise RuntimeError(
                f"could not locate the input embedding (nn.Embedding with "
                f"num_embeddings={vocab}) inside qwen_vl_interface"
            )

        ids = list(bin_ids)
        hidden = table.embedding_dim
        emb = nn.Embedding(len(ids), hidden)
        with torch.no_grad():
            emb.weight.copy_(table.weight[torch.as_tensor(ids, dtype=torch.long)])
        emb.weight.requires_grad_(True)
        setattr(self, name, emb)

        lut = torch.full((vocab,), -1, dtype=torch.long)
        lut[torch.as_tensor(ids, dtype=torch.long)] = torch.arange(len(ids))
        self.register_buffer(f"_{name}_lut", lut, persistent=False)

        def _hook(module, inputs, output):
            token_ids = inputs[0]
            local = getattr(self, f"_{name}_lut").to(token_ids.device)[token_ids]
            hit = local >= 0
            if not bool(hit.any()):
                return output
            out = output.clone()
            out[hit] = getattr(self, name)(local[hit]).to(out.dtype)
            return out

        table.register_forward_hook(_hook)
        print(
            f"__TRAINABLE_BINS__ {name} rows={len(ids)} params={len(ids)*hidden} "
            f"(vs {vocab*hidden} for the full table)",
            flush=True,
        )

    def _rare_token_ids(self, count: int):
        """Deterministic Latin-free real-token ids, ascending, for bin embeddings."""
        tok = self._resolve_tokenizer()
        vocab = tok.get_vocab()
        vocab_ids = list(vocab.values())
        cache_key = (
            tok.__class__.__module__,
            tok.__class__.__qualname__,
            str(getattr(tok, "name_or_path", "")),
            len(vocab),
            min(vocab_ids),
            max(vocab_ids),
            int(tok.convert_tokens_to_ids(self.STATE_PLACEHOLDER)),
        )
        cache = type(self)._RARE_ID_CACHE
        # Tolerate an old test/caller explicitly resetting the historical scalar cache.
        if not isinstance(cache, dict):
            cache = {}
            type(self)._RARE_ID_CACHE = cache
        if cache_key not in cache:
            inv = {v: k for k, v in vocab.items()}
            ids = []
            for i in sorted(inv):
                if i < 1000:
                    continue
                try:
                    text = tok.decode([i])
                except Exception:
                    continue
                # Reject anything containing Basic Latin / Latin-1 / Latin Extended-A,B.
                if not text or any(ord(ch) < 0x250 for ch in text):
                    continue
                ids.append(i)
            cache[cache_key] = ids
        ids = cache[cache_key]
        if len(ids) < count:
            raise ValueError(
                f"need {count} Latin-free token rows for bin embeddings, tokenizer has "
                f"{len(ids)}"
            )
        return ids[:count]

    def _resolve_vocab_size(self) -> int:
        """Qwen3-VL keeps vocab_size under text_config, not at the top level."""
        cfg = self.qwen_vl_interface.model.config
        for holder in (cfg, getattr(cfg, "text_config", None)):
            size = getattr(holder, "vocab_size", None) if holder is not None else None
            if size:
                return int(size)
        raise RuntimeError(
            "could not resolve vocab_size from the VLM config (looked at the config and "
            "its text_config)"
        )

    def _resolve_tokenizer(self):
        for attr in ("tokenizer", "processor"):
            obj = getattr(self.qwen_vl_interface, attr, None)
            if obj is None:
                continue
            return getattr(obj, "tokenizer", obj)
        raise RuntimeError("could not resolve a tokenizer from the VLM interface")

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise into uniform bins and render as a space-separated decimal string.

        The pi-0.5 convention: bins span [-1, 1] uniformly.
        Example with 256 bins: [-0.5, 0.1, 0.8] -> "64 140 230"
        """
        return " ".join(map(str, self._state_bins(state)))

    def _state_bins(self, state: np.ndarray) -> np.ndarray:
        """Quantise a normalized state vector into uniform bins over [-1, 1].

        The dataloader's ``q99`` normalization already maps state into [-1, 1]
        (dataloader/gr00t_lerobot/transform/state_action.py). Values outside that range
        are clipped explicitly rather than folded into the edge bins by digitize's
        out-of-range behaviour.
        """
        edges = np.linspace(-1, 1, self.state_num_bins + 1)[:-1]
        bins = np.digitize(np.clip(state, -1.0, 1.0), bins=edges) - 1
        return np.clip(bins, 0, self.state_num_bins - 1).astype(np.int64)

    def _order_and_validate_state(self, state: np.ndarray) -> np.ndarray:
        """Return the configured token order and validate the arm-group contract."""
        vec = np.asarray(state, dtype=np.float64).reshape(-1)
        if vec.size < 2:
            raise RuntimeError(
                f"proprioceptive state flattened to {vec.size} element(s); expected the "
                "full state vector. Check the caller's shape contract -- a (state_dim,) "
                "array indexed as [0] yields a scalar."
            )
        if self.state_token_permutation:
            expected = list(range(vec.size))
            if sorted(self.state_token_permutation) != expected:
                raise RuntimeError(
                    "framework.state_token_permutation must be a permutation of every "
                    f"state dimension 0..{vec.size - 1}, got {self.state_token_permutation}"
                )
            vec = vec[self.state_token_permutation]
        if self.state_token_group_sizes and sum(self.state_token_group_sizes) != vec.size:
            raise RuntimeError(
                "framework.state_token_group_sizes must sum to the state width: "
                f"sizes={self.state_token_group_sizes}, state_width={vec.size}"
            )
        return vec

    def _state_ids_from_bins(self, bins: np.ndarray) -> list[int]:
        """Interleave fixed group-marker IDs with their scalar bin IDs."""
        if not self.state_token_group_sizes:
            return [self._state_bin_ids[int(value)] for value in bins]
        ids: list[int] = []
        cursor = 0
        for marker_id, size in zip(self._state_group_ids, self.state_token_group_sizes):
            ids.append(int(marker_id))
            ids.extend(
                self._state_bin_ids[int(value)] for value in bins[cursor : cursor + size]
            )
            cursor += size
        return ids

    def _add_state_conditioning(
        self,
        instructions: List[str],
        states: Optional[List[np.ndarray]],
    ) -> tuple:
        """Append discretised state, with per-example training-only dropout.

        Dropped state is omitted rather than replaced by zeros: zero is a valid
        normalized robot state and should not also mean "state unavailable".

        The per-row keep vector is returned as well, because pre-encoder vision dropout
        has to see it: a row that lost both its state and its images has no non-visual
        channel left to be forced through, and would just be a corrupted sample.
        """
        if states is None:
            self._pending_state_bins = None
            return instructions, 0.0, np.ones(len(instructions), dtype=bool)
        if len(instructions) != len(states):
            raise ValueError(
                f"instruction/state batch mismatch: {len(instructions)} vs {len(states)}"
            )

        if self.training and getattr(self, "state_follows_instruction", False):
            draw = np.random.random(len(states)) < float(getattr(self, "instruction_dropout_rate", 0.0))
            self._instr_draw_pending = draw
            keep_state = ~draw
        elif self.training and self.state_dropout_rate > 0.0:
            rate = self.state_dropout_rate
            if getattr(self, "_state_rate_schedule", None) is not None:
                from starVLA.model.modules.modality_dropout import _resolve_scheduled_rate
                rate = _resolve_scheduled_rate(self._state_rate_schedule, int(getattr(self, "_modality_step", 0)),
                                               "state_dropout_rate")
            self._last_state_dropout_rate = float(rate)
            keep_state = np.random.random(len(states)) >= rate
        else:
            keep_state = np.ones(len(states), dtype=bool)

        conditioned: List[str] = []
        pending: List[Optional[np.ndarray]] = []
        true_bins: List[Optional[np.ndarray]] = []
        mask_slots = bool(getattr(self, "state_dropout_mask", False)) and self.state_token_mode == "reserved_ids"
        for instruction, state, keep in zip(instructions, states, keep_state):
            if not keep and not mask_slots:
                conditioned.append(instruction)
                pending.append(None)
                true_bins.append(None)
                continue
            # Shape contract differs between paths and the mismatch was SILENT:
            #   train  -> dataloader gives (1, state_dim); the old `np.asarray(state)[0]`
            #            correctly yielded (state_dim,)
            #   eval   -> eval_libero_model.py:899 passes (state_dim,); `[0]` then yielded a
            #            SCALAR, so the prompt carried ONE state token instead of eight.
            # Nothing raised, because the splice only checks placeholder-count against
            # bin-count and both were consistently 1. Every state-conditioned eval before
            # this fix was off-distribution (gr00t_enc scored 1.94%, causal fell below its
            # own stateless baseline). Flatten instead, and assert the width.
            vec = self._order_and_validate_state(state)
            if getattr(self, "_state_width_seen", None) is None:
                self._state_width_seen = vec.size
            elif vec.size != self._state_width_seen:
                raise RuntimeError(
                    f"state width changed from {self._state_width_seen} to {vec.size} "
                    "within a run; train and inference must agree"
                )
            if self.state_token_mode == "reserved_ids":
                row_bins = self._state_bins(vec)
                # One placeholder per state dimension; ids are spliced in after the
                # processor runs. Grouped bimanual state additionally gets one fixed
                # marker token per group (left/right), so the token count is still fixed.
                n_slots = len(row_bins) + len(self.state_token_group_sizes)
                slots = " ".join([self.STATE_PLACEHOLDER] * n_slots)
                conditioned.append(f"{instruction} [STATE] {slots} [ACTION]")
                pending.append(row_bins if keep else "__STATE_MASK__")
                true_bins.append(row_bins)
            else:
                conditioned.append(
                    f"{instruction} [STATE] {self.state2str_transform(vec)} [ACTION]"
                )
                pending.append(None)
                true_bins.append(None)
        self._pending_state_bins = (
            pending if self.state_token_mode == "reserved_ids" else None
        )
        # true bins of every row (masked rows included), for full-information target passes
        self._state_true_bins = true_bins if (mask_slots and self.state_token_mode == "reserved_ids") else None
        return (
            conditioned,
            float(np.mean(keep_state)) if len(keep_state) else 0.0,
            keep_state,
        )

    def _splice_reserved(self, qwen_inputs: dict, per_row_ids, placeholder_id, tag: str,
                         log_attr: str = "_state_splice_logged") -> None:
        """Overwrite `placeholder_id` occurrences with explicit reserved ids, in place.

        Shared by the state path and the privileged-trace path. ``per_row_ids`` is one
        list of ids per batch row, or None for a row that carries no placeholders.
        Token count is unchanged, so attention_mask and mrope are untouched.
        """
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError(f"{tag}: reserved-id encoding requires input_ids")
        for row, ids_row in enumerate(per_row_ids):
            if ids_row is None:
                continue
            pos = (input_ids[row] == placeholder_id).nonzero(as_tuple=True)[0]
            if pos.numel() != len(ids_row):
                raise RuntimeError(
                    f"{tag} row {row}: found {pos.numel()} placeholders but have "
                    f"{len(ids_row)} ids -- the placeholder was split, dropped or "
                    "collided with prompt text"
                )
            ids = torch.as_tensor(ids_row, device=input_ids.device, dtype=input_ids.dtype)
            input_ids[row, pos] = ids
            if not getattr(self, log_attr, False):
                setattr(self, log_attr, True)
                print(
                    f"__{tag}__ n_tokens={len(ids_row)} "
                    f"id_range=[{int(ids.min())},{int(ids.max())}] "
                    f"placeholder={placeholder_id}",
                    flush=True,
                )

    def _splice_state_tokens(self, qwen_inputs: dict) -> None:
        """Replace placeholder tokens with the reserved per-bin ids, in place.

        Must run before the VLM forward: the placeholder carries no proprioceptive
        meaning, so a missed splice silently trains on a constant state.
        """
        bins = getattr(self, "_pending_state_bins", None)
        self._pending_state_bins = None
        self._last_state_token_mask = None
        if self.state_token_mode != "reserved_ids" or bins is None:
            return
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("reserved_ids state encoding requires input_ids")
        state_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, row_bins in enumerate(bins):
            if row_bins is None:  # state dropped for this row
                continue
            pos = (input_ids[row] == self._state_placeholder_id).nonzero(as_tuple=True)[0]
            if isinstance(row_bins, str) and row_bins == "__STATE_MASK__":  # state_dropout_mask
                input_ids[row, pos] = int(self._state_mask_id)
                state_token_mask[row, pos] = True
                continue
            ids_row = self._state_ids_from_bins(row_bins)
            if pos.numel() != len(ids_row):
                raise RuntimeError(
                    f"row {row}: found {pos.numel()} state placeholders but have "
                    f"{len(ids_row)} state tokens -- the placeholder was split, dropped or "
                    "collided with prompt text"
                )
            ids = torch.as_tensor(
                ids_row,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            state_token_mask[row, pos] = True
            input_ids[row, pos] = ids
            if not self._state_splice_logged:
                self._state_splice_logged = True
                # Printed once per process. A silently inactive state path is
                # indistinguishable from a working one in the loss curve, so the smoke
                # gate asserts on this line rather than on the exit status.
                print(
                    f"__STATE_SPLICE__ src={self.state_token_source} bins={self.state_num_bins} "
                    f"n_values={len(row_bins)} n_tokens={len(ids_row)} "
                    f"groups={self.state_token_group_names or self.state_token_group_sizes} "
                    f"group_ids={self._state_group_ids} "
                    f"id_range=[{int(ids.min())},{int(ids.max())}] "
                    f"placeholder={self._state_placeholder_id}",
                    flush=True,
                )
        # Retain this on the framework for action-facing interfaces that need exact state
        # span identity.  Do not put the private kwarg into ordinary HF calls: only the
        # modality-block attention wrapper consumes it.
        self._last_state_token_mask = state_token_mask
        qwenvl_cfg = self.config.framework.get("qwenvl", {})
        if str(qwenvl_cfg.get("attention_topology", "causal")) == "modality_block_causal":
            qwen_inputs["_state_token_mask"] = state_token_mask
