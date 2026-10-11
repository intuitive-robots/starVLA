"""M3 modality masking (Cheng et al. 2026, arXiv 2608.22419), ported to the QwenGR00T policy (default off).

M3 trains under "controlled partial observability": per training row it draws
  * u_v ~ Bernoulli(1 - wrist_rate): all non-anchor cameras are masked together (M3 masks both wrists jointly
    and always keeps the egocentric view; LIBERO has one external + one wrist camera, so the external view is
    the anchor and the wrist is masked);
  * u_l ~ Bernoulli(1 - lang_rate): the whole instruction is masked;
the two draws are independent, proprioception is never masked, and masked tokens are removed from attention
(M3 Eq. 5-6: a masked token neither sees nor is seen). Here a masked span gets attention_mask = 0: the encoder
does not attend to it and the action head, which reads the encoder's attention mask, ignores it. Rotary
positions are computed from the unmasked sequence and passed explicitly, so removing a span does not shift
the positions of later tokens (M3's additive mask leaves positions unchanged). Inference never masks.

Action queries (num_queries > 0): N learned query tokens are appended after the instruction (dedicated trainable
vocabulary rows), the action head reads only the image tokens and the queries (as the VLA-Adapter head in M3 reads
the visual latents and the query latents, never the text), and in training each query is removed with
query_rate (at least one kept) and the survivors are scaled by 1 / (1 - query_rate) at the block-0 input
(M3 Eq. 7). num_queries 0 keeps the GR00T head reading every encoder token.

Config::
    m3_mask:
      enabled: false
      wrist_rate: 0.5        # p_v (M3 does not report its value; our camera rate)
      lang_rate: 0.1         # p_l (M3 does not report its value; our instruction rate)
      anchor_views: [0]      # camera indices that are never masked (0 = external / egocentric)
      num_queries: 0         # learned action-query tokens in the encoder (0 = off)
      query_rate: 0.1        # p_q, M3's best ratio
      placeholder: "<|fim_suffix|>"
"""
from typing import List

import numpy as np
import torch

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class M3MaskMixin:
    def _configure_m3_mask(self, config) -> None:
        cfg = dict(config.framework.get("m3_mask", {}) or {})
        self.m3_enabled = bool(cfg.get("enabled", False))
        if not self.m3_enabled:
            return
        self.m3_wrist_rate = float(cfg.get("wrist_rate", 0.5))
        self.m3_lang_rate = float(cfg.get("lang_rate", 0.1))
        self.m3_anchor_views = [int(v) for v in (cfg.get("anchor_views") or [0])]
        for name, rate in (("wrist_rate", self.m3_wrist_rate), ("lang_rate", self.m3_lang_rate)):
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"framework.m3_mask.{name} must be in [0, 1)")
        fw = config.framework
        clash = {
            "modality_dropout": bool((fw.get("modality_dropout", {}) or {}).get("enabled", False)),
            "instruction_dropout_rate": float(fw.get("instruction_dropout_rate", 0.0) or 0.0) > 0.0,
            "state_dropout_rate": float(fw.get("state_dropout_rate", 0.0) or 0.0) > 0.0
            if not hasattr(fw.get("state_dropout_rate", 0.0), "get") else True,
            "future_copies": bool((fw.get("future_copies", {}) or {}).get("enabled", False)),
            "position_prompt": bool((fw.get("position_prompt", {}) or {}).get("enabled", False)),
        }
        on = [k for k, v in clash.items() if v]
        if on:
            raise ValueError(f"m3_mask replaces our masking recipe; disable {on}")
        self._m3_logged = False
        self._m3_stats = {"rows": 0, "wrist": 0, "lang": 0, "lang_fallback": 0, "queries_dropped": 0}
        self.m3_num_queries = int(cfg.get("num_queries", 0) or 0)
        self.m3_query_rate = float(cfg.get("query_rate", 0.1))
        self._m3_query_mask = None
        self._m3_pending_scale = None
        if self.m3_num_queries > 0:
            if not 0.0 <= self.m3_query_rate < 1.0:
                raise ValueError("framework.m3_mask.query_rate must be in [0, 1)")
            if getattr(self, "state_token_source", None) != "rare_real":
                raise ValueError("m3_mask.num_queries needs state_token_source rare_real (for the query ids)")
            if bool((fw.get("fusion_registers", {}) or {}).get("enabled", False)):
                raise ValueError("m3_mask.num_queries and fusion_registers both splice slot tokens; disable one")
            tok = self._resolve_tokenizer()
            self.m3_placeholder = str(cfg.get("placeholder", "<|fim_suffix|>"))
            ph = tok.encode(self.m3_placeholder, add_special_tokens=False)
            if len(ph) != 1:
                raise ValueError(f"m3_mask.placeholder {self.m3_placeholder!r} is not a single token")
            self._m3_placeholder_id = int(ph[0])
            n_state = len(self._state_bin_ids) + len(self._state_group_ids)
            # past the state ids, fusion register ids (+0..8), state mask (+64) and instruction mask (+65)
            ids = [int(i) for i in list(self._rare_token_ids(n_state + 100 + self.m3_num_queries))[n_state + 100:]]
            if len(set(ids)) != self.m3_num_queries:
                raise RuntimeError("m3 query ids are not distinct")
            self._m3_query_ids = ids
            self._install_trainable_bin_embedding(ids, name="m3_query_embed")
            blocks = self._aux_blocks()

            def pre_hook(module, args, kwargs):
                pend = self._m3_pending_scale
                if pend is None:
                    return None
                self._m3_pending_scale = None  # one forward only
                mask, scale = pend
                if "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                    hs = kwargs["hidden_states"].clone()
                    hs[mask] = hs[mask] * scale
                    kwargs["hidden_states"] = hs
                    return args, kwargs
                hs = args[0].clone()
                hs[mask] = hs[mask] * scale
                return (hs,) + tuple(args[1:]), kwargs

            blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        logger.info("M3 masking ON: wrist_rate=%.2f lang_rate=%.2f anchor_views=%s num_queries=%d query_rate=%.2f",
                    self.m3_wrist_rate, self.m3_lang_rate, self.m3_anchor_views, self.m3_num_queries,
                    self.m3_query_rate)

    # ── action queries ───────────────────────────────────────────────────────
    def _m3_add_query_slots(self, instructions: List[str]) -> List[str]:
        if not getattr(self, "m3_enabled", False) or getattr(self, "m3_num_queries", 0) <= 0:
            return instructions
        slots = " ".join([self.m3_placeholder] * self.m3_num_queries)
        return [f"{ins} {slots}" for ins in instructions]

    def _m3_splice_queries(self, qwen_inputs: dict) -> None:
        """Training and inference: swap the placeholders for the query ids; remember their positions."""
        self._m3_query_mask = None
        if not getattr(self, "m3_enabled", False) or getattr(self, "m3_num_queries", 0) <= 0:
            return
        input_ids = qwen_inputs["input_ids"]
        ids = torch.as_tensor(self._m3_query_ids, device=input_ids.device, dtype=input_ids.dtype)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            pos = (input_ids[row] == self._m3_placeholder_id).nonzero(as_tuple=True)[0]
            if pos.numel() != ids.numel():
                raise RuntimeError(f"row {row}: {pos.numel()} m3 query placeholders, expected {ids.numel()}")
            input_ids[row, pos] = ids
            mask[row, pos] = True
        self._m3_query_mask = mask

    def _m3_restrict_head(self, qwen_inputs: dict) -> None:
        """Training and inference, after the encoder: the action head reads only image tokens and queries."""
        if not getattr(self, "m3_enabled", False) or getattr(self, "m3_num_queries", 0) <= 0:
            return
        itf = self.qwen_vl_interface
        valid = getattr(itf, "_last_encoder_attention_mask", None)
        if valid is None:
            valid = qwen_inputs["attention_mask"]
        valid = valid.bool()
        ids = qwen_inputs["input_ids"][:, : valid.shape[1]]
        image_tok = ids == int(self.qwen_vl_interface.model.config.image_token_id)
        keep = valid & (image_tok | self._m3_query_mask[:, : valid.shape[1]].to(valid.device))
        itf._last_encoder_attention_mask = keep

    def _m3_image_spans(self, row_ids: List[int], vs_id: int, ve_id: int) -> List[tuple]:
        spans, start = [], None
        for i, t in enumerate(row_ids):
            if t == vs_id:
                start = i
            elif t == ve_id and start is not None:
                spans.append((start, i + 1))
                start = None
        return spans

    def _m3_apply(self, qwen_inputs: dict, instructions: List[str]) -> None:
        """Training only: remove the drawn wrist cameras and instructions from attention, in place."""
        if not getattr(self, "m3_enabled", False) or not self.training:
            return
        ids = qwen_inputs["input_ids"]
        attn = qwen_inputs["attention_mask"]
        B = ids.shape[0]
        drop_v = np.random.random(B) < self.m3_wrist_rate
        drop_l = np.random.random(B) < self.m3_lang_rate
        drop_q = None
        nq = getattr(self, "m3_num_queries", 0)
        if nq > 0 and self.m3_query_rate > 0.0:
            drop_q = np.random.random((B, nq)) < self.m3_query_rate
            full_rows = drop_q.all(axis=1)
            for row in np.nonzero(full_rows)[0].tolist():   # at least one active query per row
                drop_q[row, np.random.randint(nq)] = False
        self._m3_stats["rows"] += B
        if nq > 0 and self.m3_query_rate > 0.0:
            # survivors scaled by 1 / (1 - p_q) at the block-0 input (M3 Eq. 7)
            qmask = self._m3_query_mask.clone()
            for row in range(B):
                pos = qmask[row].nonzero(as_tuple=True)[0]
                qmask[row, pos[torch.as_tensor(drop_q[row], device=pos.device)]] = False
            self._m3_pending_scale = (qmask, 1.0 / (1.0 - self.m3_query_rate))
        if not (drop_v.any() or drop_l.any() or (drop_q is not None and drop_q.any())):
            self._m3_last = {"wrist_rate": 0.0, "lang_rate": 0.0}
            return
        # positions from the full sequence, before any span is removed
        if "position_ids" not in qwen_inputs:
            vlm = self.qwen_vl_interface.model
            inner = getattr(vlm, "model", vlm)
            pos, _ = inner.get_rope_index(ids, image_grid_thw=qwen_inputs.get("image_grid_thw"), video_grid_thw=None,
                                          attention_mask=attn, mm_token_type_ids=qwen_inputs.get("mm_token_type_ids"))
            qwen_inputs["position_ids"] = pos
        cfg = self.qwen_vl_interface.model.config
        vs_id, ve_id = int(cfg.vision_start_token_id), int(cfg.vision_end_token_id)
        tok = self._resolve_tokenizer()
        new_attn = attn.clone()
        for row in range(B):
            row_ids = ids[row].tolist()
            if drop_v[row]:
                spans = self._m3_image_spans(row_ids, vs_id, ve_id)
                for k, (a, b) in enumerate(spans):
                    if k not in self.m3_anchor_views:
                        new_attn[row, a:b] = 0
                self._m3_stats["wrist"] += 1
            if drop_l[row]:
                text = str(instructions[row]).strip()
                start, sub = -1, []
                for variant in (" " + text, text):
                    sub = tok.encode(variant, add_special_tokens=False)
                    start = self._find_subseq(row_ids, sub)
                    if start >= 0:
                        break
                if start < 0:
                    self._m3_stats["lang_fallback"] += 1
                else:
                    new_attn[row, start:start + len(sub)] = 0
                    self._m3_stats["lang"] += 1
        if drop_q is not None and drop_q.any():
            for row in range(B):
                pos = self._m3_query_mask[row].nonzero(as_tuple=True)[0]
                gone = pos[torch.as_tensor(drop_q[row], device=pos.device)]
                new_attn[row, gone] = 0
                self._m3_stats["queries_dropped"] += int(gone.numel())
        qwen_inputs["attention_mask"] = new_attn
        self._m3_last = {"wrist_rate": float(drop_v.mean()), "lang_rate": float(drop_l.mean())}
        if not self._m3_logged:
            self._m3_logged = True
            print(f"__M3_MASK__ rows={B} wrist={int(drop_v.sum())} lang={int(drop_l.sum())} "
                  f"removed_tokens={int((attn.bool() & ~new_attn.bool()).sum())}", flush=True)
