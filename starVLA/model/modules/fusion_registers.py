"""Missing-camera reconstruction against an EMA copy of the backbone (default off).

Two modes (``framework.fusion_registers.mode``), both trained with camera dropout
(``framework.modality_dropout`` camera / pixels) and both reconstructing the camera's block-L
representation from an EMA copy of the backbone run on the FULL images (FLARE-style target):

``registers`` (variant C): K=4 learned register tokens per camera, appended to the prompt after
    the state tokens (end of the sequence: a causal backbone's registers can still read every
    image, the instruction and the state). Present in training AND at inference; the action head
    reads them like any token. Register k of camera c predicts quadrant k (2x2 split of the
    camera's token grid) of the EMA block-L tokens of camera c. Loss weight 1 when camera c was
    dropped on the online pass, ``present_weight`` otherwise (a visible camera is close to a copy).

``masked_query`` (variant A): no extra tokens. For a dropped camera, a learned per-position query
    vector (zero-initialised, so training starts at the plain gray-pixel dropout) is ADDED to the
    camera's token states at the input of backbone block 0. Those tokens then predict, token by
    token, the EMA block-L tokens of the same camera with the camera visible. At eval the same
    queries are added to cameras removed with STARVLA_ZERO_IMAGE_INDICES; with all cameras
    present nothing changes.

Loss: per-channel normalised MSE against the batch-centred target, zero-initialised predictor
(loss starts at 1). EMA: a copy of ``qwen_vl_interface`` made on the first training step and kept
out of the module tree (self.__dict__: not trained, not saved, rebuilt from the current weights on
resume); frozen parameters share storage; trainable ones keep an fp32 shadow lerped every step.
The block-L capture hook and the embedding / block-0 hooks are copied with the module.

Config (``framework.fusion_registers``)::

    enabled: false
    mode: registers          # or masked_query
    views: [external, wrist]
    layer: 16
    weight: 1.0              # inner weight; the ramp lives in trainer.loss_scale.structured_aux
    present_weight: 0.1      # registers only
    ema: 0.995
    mlp_hidden: 2048
    tokens_per_camera: 64    # masked_query: size of the per-position query table
    placeholder: "<|fim_middle|>"   # registers only
"""

import copy
import os
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from starVLA.model.modules.future_latent import _VIEW_INDEX, _StopAfterFutureLatentLayer
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class FusionRegisterMixin:
    # ── config / modules ─────────────────────────────────────────────────────
    def _configure_fusion_registers(self, config) -> None:
        cfg = dict(config.framework.get("fusion_registers", {}) or {})
        self.fusion_registers_enabled = bool(cfg.get("enabled", False))
        self._fusion_pending_query = None
        self._last_fusion_mask = None
        self._last_mq_mask = None
        if not self.fusion_registers_enabled:
            return
        self.fusion_mode = str(cfg.get("mode", "registers")).lower()
        if self.fusion_mode not in ("registers", "masked_query"):
            raise ValueError("framework.fusion_registers.mode must be 'registers' or 'masked_query'")
        views = cfg.get("views", ["external", "wrist"])
        self.fusion_views = [str(v).lower() for v in ([views] if isinstance(views, str) else views)]
        if not self.fusion_views or any(v not in _VIEW_INDEX for v in self.fusion_views):
            raise ValueError(f"framework.fusion_registers.views must be a subset of {sorted(_VIEW_INDEX)}")
        self.fusion_k = 4
        self.fusion_layer = int(cfg.get("layer", 16))
        self.fusion_weight = float(cfg.get("weight", 1.0))
        self.fusion_present_weight = float(cfg.get("present_weight", 0.1))
        self.fusion_ema = float(cfg.get("ema", 0.995))
        self.fusion_mlp_hidden = int(cfg.get("mlp_hidden", 2048))
        self.fusion_tokens_per_camera = int(cfg.get("tokens_per_camera", 64))
        self.fusion_placeholder = str(cfg.get("placeholder", "<|fim_middle|>"))
        # masked_query only: False = no learned query table (dropout + reconstruction loss alone); the
        # dropped camera's gray-pixel tokens are the slot. Probes (09-30) found the queries inert at test time.
        self.fusion_use_queries = bool(cfg.get("use_queries", True))
        # state_queries / instruction_queries (10-02, default off): per-slot learned queries also on the masked
        # state slots (one query per state token) and masked instruction tokens (one per position within the
        # instruction, capped at instruction_query_tokens). Without them every masked slot carries the same
        # shared mask embedding (state_mask_embed / instr_mask_embed). Training only: nothing is masked at test.
        self.fusion_state_queries = bool(cfg.get("state_queries", False))
        self.fusion_instr_queries = bool(cfg.get("instruction_queries", False))
        self.fusion_state_query_tokens = int(cfg.get("state_query_tokens", 16))
        self.fusion_instr_query_tokens = int(cfg.get("instruction_query_tokens", 64))
        if (self.fusion_state_queries or self.fusion_instr_queries) and not (
                self.fusion_mode == "masked_query" and self.fusion_use_queries):
            raise ValueError("fusion_registers.state_queries / instruction_queries need mode masked_query and use_queries")
        # target_center (10-02, default "none"): masked_query only. Subtract a running per-(camera, position,
        # channel) mean of the EMA target instead of the batch mean, so an input-blind "usual patch" earns
        # nothing. "others": the mean over all OTHER tasks (leave-one-task-out: whatever the instruction,
        # state or other camera can tell is still rewarded); "task": the current task's own mean (only
        # within-task, frame-specific information is rewarded). Task = the raw instruction string. Means are
        # EMA (center_momentum) over the EMA target of every row, synced across ranks, kept in float32 outside
        # the module (a buffer would be cast to bf16) and re-warmed after a resume.
        self.fusion_center = str(cfg.get("target_center", "none") or "none").lower()
        if self.fusion_center not in ("none", "others", "task"):
            raise ValueError("fusion_registers.target_center must be none, others or task")
        if self.fusion_center != "none" and self.fusion_mode != "masked_query":
            raise ValueError("fusion_registers.target_center needs mode masked_query")
        self.fusion_center_momentum = float(cfg.get("center_momentum", 0.99))
        self.__dict__["_fusion_center_state"] = {}
        # state (E7): rows whose state was dropped with framework.state_dropout_mask carry mask tokens in
        # the state slots; those tokens predict the EMA copy's block-L tokens of the TRUE state.
        self.fusion_state = bool(cfg.get("state", False))
        # future (E9b): the dropped camera's tokens at block future_layer also predict that camera's CHANGE
        # over future_offset frames, EMA(t+k) - EMA(t) at future_layer (free-slot future, EMA targets).
        # instruction (E7): rows whose instruction was masked (framework.instruction_dropout_rate) predict the
        # EMA copy's block-L tokens of the TRUE instruction at the same positions.
        self.fusion_instruction = bool(cfg.get("instruction", False))
        if self.fusion_instruction and float(getattr(self, "instruction_dropout_rate", 0.0)) <= 0.0:
            raise ValueError("fusion_registers.instruction needs framework.instruction_dropout_rate > 0")
        fl = cfg.get("future_layer", None)
        self.fusion_future_layer = None if fl in (None, "null", "") else int(fl)
        # loop_future_offsets (L2): pass j of framework.backbone_loop -> the dropped camera's tokens at the loop's
        # last block predict EMA(t + offsets[j]) - EMA(t) at that block (a latent rollout through the action
        # chunk). Needs backbone_loop with fixed_iters and max_iters == len(offsets), and the dataloader's
        # future_offsets == these offsets. Uses the future-block capture at the loop end.
        lfo = cfg.get("loop_future_offsets", None)
        self.fusion_loop_offsets = [int(k) for k in (lfo or [])]
        # loop_target_offset (L3): the SAME future target at every pass (refinement toward one target, the
        # Ouro / LoopVLA form), one shared head per camera; works with random K (every executed pass supervised).
        lto = cfg.get("loop_target_offset", None)
        self.fusion_loop_same = lto not in (None, "null", "")
        if self.fusion_loop_same:
            if self.fusion_loop_offsets:
                raise ValueError("set loop_future_offsets or loop_target_offset, not both")
            bl0 = dict(config.framework.get("backbone_loop", {}) or {})
            self.fusion_loop_offsets = [int(lto)] * int(bl0.get("max_iters", 3))
        if self.fusion_loop_offsets:
            bl = dict(config.framework.get("backbone_loop", {}) or {})
            if not bool(bl.get("enabled", False)):
                raise ValueError("fusion_registers loop targets need framework.backbone_loop.enabled")
            if not self.fusion_loop_same and not bool(bl.get("fixed_iters", False)):
                raise ValueError("loop_future_offsets (progressive targets) needs backbone_loop.fixed_iters: true")
            if int(bl.get("max_iters", 3)) != len(self.fusion_loop_offsets):
                raise ValueError("backbone_loop.max_iters must equal len(fusion_registers.loop_future_offsets)")
            if self.fusion_future_layer is not None:
                raise ValueError("loop_future_offsets and future_layer are separate designs; set one")
            vla_fo = sorted({int(k) for k in (config.datasets.vla_data.get("future_offsets") or [])})
            if vla_fo != sorted(set(self.fusion_loop_offsets)):
                raise ValueError("datasets.vla_data.future_offsets must equal fusion_registers.loop_future_offsets")
            self.fusion_future_layer = int(bl.get("end", 19))  # EMA captures at the loop's last block
            self._fusion_loop_mode = True
        else:
            self._fusion_loop_mode = False
        self.fusion_future_weight = float(cfg.get("future_weight", 1.0))
        if self.fusion_future_layer is not None and not self._fusion_loop_mode:
            if self.fusion_mode != "masked_query":
                raise ValueError("fusion_registers.future_layer needs mode masked_query")
            if self.fusion_future_layer <= self.fusion_layer:
                raise ValueError("fusion_registers.future_layer must be deeper than fusion_registers.layer")
            vla = config.datasets.vla_data
            if int(vla.get("shared_z_future_offset", 0) or 0) <= 0 or not bool(vla.get("shared_z_future_keep_tail", False)):
                raise ValueError("fusion_registers.future_layer needs datasets.vla_data.shared_z_future_offset > 0 "
                                 "and shared_z_future_keep_tail: true (the dataloader's future_image)")
        if self.fusion_state and not (getattr(self, "state_dropout_mask", False)
                                      and (float(getattr(self, "state_dropout_rate", 0.0)) > 0.0
                                           or getattr(self, "state_follows_instruction", False))):
            raise ValueError("fusion_registers.state needs framework.state_dropout_mask: true and state_dropout_rate > 0 "
                             "(or state_follows_instruction)")
        for other in ("future_latent", "task_pred", "shared_z", "structured_aux"):
            if bool(config.framework.get(other, {}).get("enabled", False)):
                raise ValueError(f"fusion_registers and {other} share the structured_aux loss slot")
        md = config.framework.get("modality_dropout", {}) or {}
        if not (bool(md.get("enabled", False)) and str(md.get("mode", "")) in ("camera", "independent")
                and str(md.get("removal", "")) == "pixels"):
            raise ValueError("fusion_registers needs framework.modality_dropout {enabled, mode: camera | independent, "
                             "removal: pixels}")
        if self.fusion_mode == "registers":
            if getattr(self, "state_token_source", None) != "rare_real":
                raise ValueError("fusion registers take their ids after the rare_real state ids "
                                 "(framework.state_token_source: rare_real)")
            tok = self._resolve_tokenizer()
            ph = tok.encode(self.fusion_placeholder, add_special_tokens=False)
            if len(ph) != 1 or ph[0] == getattr(self, "_state_placeholder_id", None):
                raise ValueError(f"fusion placeholder {self.fusion_placeholder!r} must be one token "
                                 "distinct from the state placeholder")
            self._fusion_placeholder_id = int(ph[0])
            n_state = len(self._state_bin_ids) + len(self._state_group_ids)
            n_reg = self.fusion_k * len(self.fusion_views)
            self._fusion_reg_ids = [int(i) for i in list(self._rare_token_ids(n_state + n_reg))[n_state:]]
            if set(self._fusion_reg_ids) & set(self._state_bin_ids + self._state_group_ids):
                raise RuntimeError("fusion register ids collide with state ids")
            self._install_trainable_bin_embedding(self._fusion_reg_ids, name="fusion_reg_embed")
        self._fusion_logged = False
        logger.info("fusion reconstruction ON: mode=%s views=%s layer=%d ema=%.4f present_weight=%.2f",
                    self.fusion_mode, self.fusion_views, self.fusion_layer, self.fusion_ema,
                    self.fusion_present_weight)

    def _build_fusion_register_modules(self, hidden_size: int) -> None:
        if not getattr(self, "fusion_registers_enabled", False):
            return
        h, m = int(hidden_size), self.fusion_mlp_hidden
        self.fusion_predictor = nn.ModuleDict({
            v: nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
            for v in self.fusion_views
        })
        if getattr(self, "fusion_instruction", False):
            self.fusion_predictor["instr"] = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
        if getattr(self, "fusion_state", False):
            self.fusion_predictor["state"] = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
        if getattr(self, "fusion_future_layer", None) is not None:
            keys = ([f"{v}_{j}" for v in self.fusion_views for j in range(len(self.fusion_loop_offsets))]
                    if self._fusion_loop_mode and not self.fusion_loop_same else list(self.fusion_views))
            self.fusion_future_predictor = nn.ModuleDict({
                kk: nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
                for kk in keys
            })
            if self._fusion_loop_mode:
                self._loop_keep_passes = True  # backbone_loop keeps every pass's output at its last block
            for mlp in self.fusion_future_predictor.values():
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
            self._fu_capture = None

            def fu_hook(module, inputs, output):
                cap = self._fu_capture
                if cap is None or getattr(self, "_loop_inside", False):
                    return None
                cap["hidden"] = output[0] if isinstance(output, tuple) else output
                if cap.get("stop", False):
                    raise _StopAfterFutureLatentLayer()
                return None

            self._aux_blocks()[self.fusion_future_layer].register_forward_hook(fu_hook)
        for mlp in self.fusion_predictor.values():  # predict the batch mean at start: loss 1
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        if self.fusion_mode == "masked_query" and self.fusion_use_queries:
            self.fusion_query = nn.Parameter(torch.zeros(len(self.fusion_views), self.fusion_tokens_per_camera, h))
            if self.fusion_state_queries:
                self.fusion_state_query = nn.Parameter(torch.zeros(self.fusion_state_query_tokens, h))
            if self.fusion_instr_queries:
                self.fusion_instr_query = nn.Parameter(torch.zeros(self.fusion_instr_query_tokens, h))
            blocks = self._aux_blocks()

            def pre_hook(module, args, kwargs):
                pend = self._fusion_pending_query
                if pend is None:
                    return None
                self._fusion_pending_query = None  # one forward only
                mask, add = pend
                if "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                    hs = kwargs["hidden_states"]
                    hs = hs.clone()
                    hs[mask] = hs[mask] + add.to(hs.dtype)
                    kwargs["hidden_states"] = hs
                    return args, kwargs
                hs = args[0].clone()
                hs[mask] = hs[mask] + add.to(hs.dtype)
                return (hs,) + tuple(args[1:]), kwargs

            blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        self._aux_register_hook(self.fusion_layer)

    # ── prompt (registers) ───────────────────────────────────────────────────
    def _add_fusion_register_slots(self, instructions: List[str]) -> List[str]:
        if not getattr(self, "fusion_registers_enabled", False) or self.fusion_mode != "registers":
            return instructions
        slots = " ".join([self.fusion_placeholder] * len(self._fusion_reg_ids))
        return [f"{ins} {slots}" for ins in instructions]

    def _splice_fusion_registers(self, qwen_inputs: dict) -> None:
        if not getattr(self, "fusion_registers_enabled", False) or self.fusion_mode != "registers":
            return
        input_ids = qwen_inputs["input_ids"]
        ids = torch.as_tensor(self._fusion_reg_ids, device=input_ids.device, dtype=input_ids.dtype)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            pos = (input_ids[row] == self._fusion_placeholder_id).nonzero(as_tuple=True)[0]
            if pos.numel() != ids.numel():
                raise RuntimeError(f"row {row}: {pos.numel()} fusion placeholders, expected {ids.numel()}")
            input_ids[row, pos] = ids
            mask[row, pos] = True
        self._last_fusion_mask = mask
        if not self._fusion_logged:
            self._fusion_logged = True
            print(f"__FUSION_REGISTERS__ n={ids.numel()} ids={self._fusion_reg_ids} "
                  f"first_pos={int(mask[0].nonzero()[0])} seq_len={input_ids.shape[1]}", flush=True)

    # ── masked queries (A) ───────────────────────────────────────────────────
    def _arm_fusion_queries(self, qwen_inputs: dict, dropped: Optional[List[List[int]]]) -> None:
        """Mark the dropped cameras' tokens (the loss reads _last_mq_mask) and, unless disabled, arm the
        block-0 pre-hook that adds the learned queries to them (and, with state_queries / instruction_queries,
        to the masked state and instruction slots)."""
        self._last_mq_mask = None
        if not getattr(self, "fusion_registers_enabled", False) or self.fusion_mode != "masked_query":
            return
        input_ids = qwen_inputs["input_ids"]
        full = torch.zeros_like(input_ids, dtype=torch.bool)
        masks = None
        if dropped and any(dropped):
            masks = self._aux_camera_masks(input_ids, self.fusion_views)
            for row in range(input_ids.shape[0]):
                for vi, view in enumerate(self.fusion_views):
                    if _VIEW_INDEX[view] in (dropped[row] or []):
                        pos = masks[view][row].nonzero(as_tuple=True)[0]
                        if pos.numel() > self.fusion_tokens_per_camera:
                            raise RuntimeError(f"camera {view}: {pos.numel()} tokens > tokens_per_camera")
                        full[row, pos] = True
        if bool(full.any()):
            self._last_mq_mask = full
        smask = imask = None
        if self.training and getattr(self, "fusion_state_queries", False) and getattr(self, "_state_mask_id", None) is not None:
            smask = input_ids == int(self._state_mask_id)
        if self.training and getattr(self, "fusion_instr_queries", False) and getattr(self, "_instr_mask", None) is not None:
            imask = self._instr_mask.to(input_ids.device)
        n_s = 0 if smask is None else int(smask.sum())
        n_i = 0 if imask is None else int(imask.sum())
        if not bool(full.any()) and not n_s and not n_i:
            return
        # use_queries: false (E5): the gray-pixel tokens are the slot; nothing is added, the loss still runs.
        if not getattr(self, "fusion_use_queries", True):
            if not self._fusion_logged:
                self._fusion_logged = True
                print(f"__FUSION_QUERIES__ disabled (use_queries false) rows={int(full.any(1).sum())} "
                      f"tokens={int(full.sum())}", flush=True)
            return
        # Eval-time switch (E1): STARVLA_FUSION_QUERIES=0 skips the queries at inference only.
        if not self.training and os.environ.get("STARVLA_FUSION_QUERIES", "1").strip() == "0":
            if not getattr(self, "_fusion_queries_off_logged", False):
                self._fusion_queries_off_logged = True
                logger.warning("STARVLA_FUSION_QUERIES=0: masked queries disabled at inference")
            return
        q = self.fusion_query
        add_full = torch.zeros(*input_ids.shape, q.shape[-1], device=input_ids.device, dtype=q.dtype)
        for row in range(input_ids.shape[0]):
            if masks is not None:
                for vi, view in enumerate(self.fusion_views):
                    if full[row][masks[view][row]].any():
                        pos = masks[view][row].nonzero(as_tuple=True)[0]
                        add_full[row, pos] = q[vi, :pos.numel()]
            if n_s and bool(smask[row].any()):
                pos = smask[row].nonzero(as_tuple=True)[0]
                if pos.numel() > self.fusion_state_query_tokens:
                    raise RuntimeError(f"state: {pos.numel()} masked tokens > state_query_tokens")
                add_full[row, pos] = self.fusion_state_query[:pos.numel()]
            if n_i and bool(imask[row].any()):
                pos = imask[row].nonzero(as_tuple=True)[0]
                idx = torch.arange(pos.numel(), device=pos.device).clamp_max(self.fusion_instr_query_tokens - 1)
                add_full[row, pos] = self.fusion_instr_query[idx]
        tot = full.clone()
        if n_s:
            tot |= smask
        if n_i:
            tot |= imask
        self._fusion_pending_query = (tot, add_full[tot])
        self._fusion_query_counts = (n_s, n_i)
        if not self._fusion_logged:
            self._fusion_logged = True
            print(f"__FUSION_QUERIES__ armed rows={int(tot.any(1).sum())} camera_tokens={int(full.sum())} "
                  f"state_tokens={n_s} instruction_tokens={n_i}", flush=True)

    def _fusion_future_arm(self, stop: bool = False) -> None:
        if getattr(self, "fusion_registers_enabled", False) and getattr(self, "fusion_future_layer", None) is not None:
            self._fu_capture = {"hidden": None, "stop": bool(stop)}

    def _fusion_future_take(self):
        cap, self._fu_capture = getattr(self, "_fu_capture", None), None
        return None if cap is None else cap.get("hidden")

    # ── EMA copy of the backbone ─────────────────────────────────────────────
    @torch.no_grad()
    def _fusion_ema_update(self):
        st = self.__dict__.get("_fusion_ema_state")
        if st is None:
            # Non-leaf tensors held as plain attributes (the converted encoder keeps some) cannot
            # be deep-copied; the copy gets a detached clone of each.
            memo, n_nonleaf = {}, 0

            def _scan(v):
                nonlocal n_nonleaf
                if torch.is_tensor(v):
                    if v.grad_fn is not None and id(v) not in memo:
                        memo[id(v)] = v.detach().clone()
                        n_nonleaf += 1
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        _scan(x)
                elif isinstance(v, dict):
                    for x in v.values():
                        _scan(x)

            for mod in self.qwen_vl_interface.modules():
                for k, v in vars(mod).items():
                    if k not in ("_parameters", "_buffers", "_modules"):
                        _scan(v)
                for v in list(mod._buffers.values()):
                    _scan(v)
            ema = copy.deepcopy(self.qwen_vl_interface, memo)
            if n_nonleaf:
                logger.info("fusion: EMA copy detached %d non-leaf attribute tensors", n_nonleaf)
            ema.eval()
            pairs = []
            for (n, p), (n2, pe) in zip(self.qwen_vl_interface.named_parameters(), ema.named_parameters()):
                if n != n2:
                    raise RuntimeError("EMA copy parameter order differs")
                pe.requires_grad_(False)
                if p.requires_grad:
                    pairs.append((p, pe))
                else:
                    pe.data = p.data  # frozen: share storage
            shadow = [p.detach().float().clone() for p, _ in pairs]
            st = {"ema": ema, "pairs": pairs, "shadow": shadow}
            self.__dict__["_fusion_ema_state"] = st
            logger.info("fusion: EMA backbone created (%d trainable tensors, %.1fM params in fp32 shadow)",
                        len(pairs), sum(s.numel() for s in shadow) / 1e6)
            return st["ema"]
        w = 1.0 - self.fusion_ema
        for (p, pe), sh in zip(st["pairs"], st["shadow"]):  # per tensor: no model-size fp32 transient
            sh.lerp_(p.detach().float(), w)
            pe.data.copy_(sh)
        return st["ema"]

    def _fusion_center_prior(self, view: str, tok: torch.Tensor, instructions) -> torch.Tensor:
        """Per-row prior for target_center (float32, detached): computed from the running per-task means
        BEFORE this step's update (a row never predicts itself), then the means of every task in the global
        batch are updated with this step's EMA target (all rows, all ranks)."""
        st = self.__dict__["_fusion_center_state"].setdefault(view, {"mean": {}, "count": {}})
        tok = tok.detach().float()
        keys = [str(k) for k in instructions]
        means, counts = st["mean"], st["count"]
        fallback = tok.mean(dim=(0, 1), keepdim=True)[0]  # nothing known yet: the old batch centring
        if means and next(iter(means.values())).shape != tok.shape[1:]:
            raise RuntimeError(f"target_center: camera {view} token grid changed {tuple(tok.shape[1:])}")
        prior = []
        if self.fusion_center == "others" and means:
            names = list(means)
            w = torch.tensor([counts[k] for k in names], device=tok.device)
            stack = torch.stack([means[k] for k in names])
            total, wsum = (w[:, None, None] * stack).sum(0), float(w.sum())
        for k in keys:
            if self.fusion_center == "task":
                prior.append(means[k] if k in means else fallback.expand_as(tok[0]))
            elif means:
                own_w = float(counts.get(k, 0.0))
                if wsum - own_w <= 0:
                    prior.append(fallback.expand_as(tok[0]))
                else:
                    own = means[k] * own_w if k in means else 0.0
                    prior.append((total - own) / (wsum - own_w))
            else:
                prior.append(fallback.expand_as(tok[0]))
        prior = torch.stack(prior)
        local = sorted(set(keys))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            names = sorted({k for g in gathered for k in g})
        else:
            names = local
        idx = {k: i for i, k in enumerate(names)}
        rows = torch.tensor([idx[k] for k in keys], device=tok.device)
        sums = torch.zeros(len(names), *tok.shape[1:], device=tok.device).index_add_(0, rows, tok)
        cnt = torch.zeros(len(names), device=tok.device).index_add_(0, rows, torch.ones(len(keys), device=tok.device))
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums)
            dist.all_reduce(cnt)
        m = self.fusion_center_momentum
        for k, i in idx.items():
            bm = sums[i] / cnt[i]
            means[k] = bm.clone() if k not in means else m * means[k] + (1.0 - m) * bm
            counts[k] = counts.get(k, 0.0) + float(cnt[i])
        return prior.detach()

    def _fusion_target_pass(self, batch_images, instructions, state_bins, update: bool = True,
                            capture_main: bool = True):
        """EMA pass on full images. With future_layer set the pass runs on to that block and leaves its
        output in self._fu_target; update=False reuses this step's EMA weights (second pass)."""
        ema = self._fusion_ema_update() if update else self.__dict__["_fusion_ema_state"]["ema"]
        if update:
            self._ema_done_this_step = True  # future_copies reuses this step's EMA weights
        pp_lat = update and capture_main and hasattr(self, "_pp_latent_duplicate")
        images = self._pp_latent_duplicate(batch_images) if pp_lat else batch_images  # position_prompt latent_target
        target_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        if state_bins is not None:
            self._pending_state_bins = list(state_bins)
            self._splice_state_tokens(target_inputs)
        if hasattr(self, "_fc_splice_state_slots"):
            self._fc_splice_state_slots(target_inputs)  # future_copies.state: same query ids as online
        saved = self._last_fusion_mask
        self._splice_fusion_registers(target_inputs)
        self._last_fusion_mask = saved
        if pp_lat:
            self._pp_arm_target(target_inputs)  # no-op unless the online pass armed a latent target
        self._fusion_pending_query = None  # full images: no queries on the target pass
        fut = getattr(self, "fusion_future_layer", None) is not None
        if capture_main:
            self._future_latent_arm(stop=not fut)
        if fut:
            self._fusion_future_arm(stop=True)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    ema(**target_inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
                except _StopAfterFutureLatentLayer:
                    pass
            target_hidden = self._future_latent_take() if capture_main else None
            self._fu_target = self._fusion_future_take() if fut else None
            if pp_lat and target_hidden is not None:
                self._pp_set_target_hidden(target_hidden)
        finally:
            self._fl_capture = None
            if pp_lat:
                self._pp_pending_target = None  # never leak an armed addition into a later EMA pass
            if fut:
                self._fu_capture = None
        if capture_main and target_hidden is None:
            raise RuntimeError("fusion: the EMA target pass never reached block L")
        if fut and self._fu_target is None:
            raise RuntimeError("fusion: the EMA target pass never reached the future block")
        return target_hidden, target_inputs

    # ── loss ─────────────────────────────────────────────────────────────────
    def _fusion_align_mask(self, mask, online_ids, target_inputs):
        """Map an online-sequence mask onto the EMA target sequence. Equal shapes: unchanged. With
        future_copies the online rows carry the copy images after the text, so the two sequences differ;
        drop the padding and the copy-image spans (vision_start..vision_end of image index >= n) from the
        online row and match position by position with the target row's non-padding tokens."""
        tgt_ids = target_inputs["input_ids"]
        mask = mask.to(tgt_ids.device)
        if mask.shape == tgt_ids.shape:
            return mask
        # image copies after the text: future copies or position-prompt duplicates (never both)
        if getattr(self, "fcopy_enabled", False):
            n_fc = int(getattr(self, "_fc_n", 0) or 0)
        elif getattr(self, "pp_input", False):
            n_fc = int(getattr(self, "_pp_n", 0) or 0)
        else:
            n_fc = 0
        if n_fc == 0:
            raise RuntimeError(f"fusion: online {tuple(mask.shape)} and target {tuple(tgt_ids.shape)} differ")
        cfg = self.qwen_vl_interface.model.config
        on = online_ids.to(tgt_ids.device)
        vs, ve = (on == int(cfg.vision_start_token_id)), (on == int(cfg.vision_end_token_id))
        cam = torch.cumsum(vs.long(), dim=1) - 1
        inside = (torch.cumsum(vs.long(), dim=1) - torch.cumsum(ve.long(), dim=1) + ve.long()) > 0
        pad_id = int(self.qwen_vl_interface.processor.tokenizer.pad_token_id)
        on_keep = ~(inside & (cam >= n_fc)) & (on != pad_id)
        tgt_keep = target_inputs["attention_mask"].bool()
        # masked state / instruction slots hold the mask id online and the true ids in the target
        free = torch.zeros_like(on, dtype=torch.bool)
        for name in ("_state_mask_id", "_instr_mask_id"):
            if getattr(self, name, None) is not None:
                free |= on == int(getattr(self, name))
        out = torch.zeros_like(tgt_ids, dtype=torch.bool)
        for r in range(mask.shape[0]):
            ko, kt = on_keep[r].nonzero(as_tuple=True)[0], tgt_keep[r].nonzero(as_tuple=True)[0]
            if ko.numel() != kt.numel():
                raise RuntimeError(f"fusion: row {r} keeps {ko.numel()} online tokens vs {kt.numel()} target tokens")
            m_r = mask[r, ko]
            ok_r = m_r | free[r, ko]
            if not torch.equal(on[r, ko][~ok_r], tgt_ids[r, kt][~ok_r]):  # outside the masks the ids must agree
                bad = (on[r, ko] != tgt_ids[r, kt]) & ~ok_r
                i = bad.nonzero(as_tuple=True)[0][:12].tolist()
                raise RuntimeError(f"fusion: row {r} online and target tokens do not line up at kept idx {i}: "
                                   f"online {on[r, ko][i].tolist()} target {tgt_ids[r, kt][i].tolist()} "
                                   f"(kept {ko.numel()}, masked {int(m_r.sum())})")
            out[r, kt] = m_r
        if int(out.sum()) != int(mask.sum()):
            raise RuntimeError("fusion: masked positions fall in the future-copy span")
        return out

    def _fusion_register_loss(self, online_hidden, online_input_ids, batch_images, instructions,
                              state_bins, dropped_images, examples=None):
        self._aux_check_online(online_hidden)
        online_future = self._fusion_future_take() if getattr(self, "fusion_future_layer", None) is not None else None
        device = online_hidden.device
        batch = online_hidden.shape[0]
        zero = sum(p.sum() for p in self.fusion_predictor.parameters()) * 0.0
        if getattr(self, "fusion_future_layer", None) is not None:
            zero = zero + sum(p.sum() for p in self.fusion_future_predictor.parameters()) * 0.0
        if self.fusion_mode == "masked_query":
            if getattr(self, "fusion_use_queries", True):
                zero = zero + self.fusion_query.sum() * 0.0
                for name in ("fusion_state_query", "fusion_instr_query"):
                    if hasattr(self, name):
                        zero = zero + getattr(self, name).sum() * 0.0
            state_masked = (getattr(self, "fusion_state", False)
                            and bool((online_input_ids == self._state_mask_id).any()))
            instr_masked = getattr(self, "fusion_instruction", False) and getattr(self, "_instr_mask", None) is not None
            # nothing dropped in this batch. With target_center every rank must still reach the synced mean
            # update (all_gather / all_reduce), so the early return is skipped (no loss term is added below).
            # teacher_align mode masked (T1): dropped cameras are supervised by the teacher, not by the EMA copy
            cam_masked = self._last_mq_mask is not None and not getattr(self, "ta_replace_camera_ema", False)
            if (not cam_masked and not state_masked and not instr_masked
                    and getattr(self, "fusion_center", "none") == "none"):
                n_drop = float(sum(1 for d in (dropped_images or []) if d)) if not cam_masked else 0.0
                return zero, {"structured_aux/fusion_rows_dropped": torch.tensor(n_drop / batch, device=device)}
        target_hidden, target_inputs = self._fusion_target_pass(batch_images, instructions, state_bins)
        cam_masks = self._aux_camera_masks(target_inputs["input_ids"], self.fusion_views)
        on_masks = self._aux_camera_masks(online_input_ids, self.fusion_views)
        metrics, losses, weights = {}, [], []
        if hasattr(self, "fusion_state_query") or hasattr(self, "fusion_instr_query"):
            n_s, n_i = getattr(self, "_fusion_query_counts", (0, 0))
            self._fusion_query_counts = (0, 0)
            metrics["structured_aux/fusion_query_state_tokens"] = torch.tensor(float(n_s), device=device)
            metrics["structured_aux/fusion_query_instr_tokens"] = torch.tensor(float(n_i), device=device)
            for name, key in (("fusion_state_query", "state"), ("fusion_instr_query", "instr")):
                if hasattr(self, name):
                    metrics[f"structured_aux/fusion_query_norm_{key}"] = getattr(self, name).detach().float().norm()
        if self.fusion_mode == "registers":
            reg_mask = self._last_fusion_mask
            n_reg = len(self._fusion_reg_ids)
            reg_h = online_hidden.float()[reg_mask].reshape(batch, n_reg, -1)
        for vi, view in enumerate(self.fusion_views):
            cam = _VIEW_INDEX[view]
            m = cam_masks[view]
            n_tok = m.sum(dim=1)
            n = int(n_tok[0])
            g = int(round(n ** 0.5))
            if bool((n_tok != n).any()) or g * g != n or g % 2:
                raise RuntimeError(f"fusion: camera {view} tokens {n_tok.tolist()} are not an even square grid")
            dropped = torch.tensor([dropped_images is not None and cam in (dropped_images[i] or [])
                                    for i in range(batch)], device=device, dtype=torch.bool)
            tok = target_hidden.float()[m].reshape(batch, n, -1)
            head = self.fusion_predictor[view]
            centred = self.fusion_mode == "masked_query" and getattr(self, "fusion_center", "none") != "none"
            if self.fusion_mode == "registers":
                grid = tok.reshape(batch, g, g, -1)
                h2 = g // 2
                tgt = torch.stack([grid[:, :h2, :h2].mean((1, 2)), grid[:, :h2, h2:].mean((1, 2)),
                                   grid[:, h2:, :h2].mean((1, 2)), grid[:, h2:, h2:].mean((1, 2))], dim=1)
                src = reg_h[:, vi * self.fusion_k:(vi + 1) * self.fusion_k]
                w_row = torch.where(dropped, torch.ones(batch, device=device),
                                    torch.full((batch,), self.fusion_present_weight, device=device))
            else:
                mu = self._fusion_center_prior(view, tok, instructions) if centred else None
                if not bool(dropped.any()):
                    continue
                if getattr(self, "ta_replace_camera_ema", False):
                    continue  # teacher_align mode masked (T1): the teacher replaces the EMA camera target
                tgt = tok
                src = online_hidden.float()[on_masks[view]].reshape(batch, n, -1)
                w_row = dropped.float()
            if centred:
                tgt_u = (tok - tok.mean(dim=(0, 1), keepdim=True)).detach()  # the uncentred target, for metrics
                rms_u = tgt_u.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                tgt = (tok - mu).detach()
            else:
                tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
            rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
            pred = head(src.to(head[1].weight.dtype)).float()
            err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))  # per row
            if centred:
                with torch.no_grad():
                    err_u = ((pred + mu - tok) / rms_u).pow(2).mean(dim=(1, 2))
                    err_p = ((mu - tok) / rms_u).pow(2).mean(dim=(1, 2))
                metrics[f"structured_aux/fusion_r2_uncentred_{view}"] = (1.0 - err_u[dropped].mean()).detach()
                metrics[f"structured_aux/fusion_prior_r2_{view}"] = (1.0 - err_p.mean()).detach()
                metrics["structured_aux/fusion_center_tasks"] = torch.tensor(
                    float(len(self.__dict__["_fusion_center_state"][view]["mean"])), device=device)
            losses.append((w_row * err).sum())
            weights.append(w_row.sum())
            if bool(dropped.any()):
                metrics[f"structured_aux/fusion_r2_dropped_{view}"] = (1.0 - err[dropped].mean()).detach()
            if self.fusion_mode == "registers" and bool((~dropped).any()):
                metrics[f"structured_aux/fusion_r2_present_{view}"] = (1.0 - err[~dropped].mean()).detach()
        if getattr(self, "fusion_state", False):
            smask = online_input_ids == self._state_mask_id
            srows = smask.any(dim=1)
            if bool(srows.any()):
                n_s = int(smask[srows][0].sum())
                tmask = self._fusion_align_mask(smask, online_input_ids, target_inputs)
                if not getattr(self, "_fusion_state_checked", False):
                    self._fusion_state_checked = True
                    tid = set(target_inputs["input_ids"][tmask].tolist())
                    if not tid <= set(self._state_bin_ids + self._state_group_ids):
                        raise RuntimeError("fusion state: target positions do not hold state ids")
                    print(f"__FUSION_STATE__ rows={int(srows.sum())} tokens_per_row={n_s}", flush=True)
                tgt = target_hidden.float()[tmask].reshape(int(srows.sum()), n_s, -1)
                src = online_hidden.float()[smask].reshape(int(srows.sum()), n_s, -1)
                tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                head = self.fusion_predictor["state"]
                pred = head(src.to(head[1].weight.dtype)).float()
                err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                losses.append(err.sum())
                weights.append(torch.tensor(float(err.numel()), device=device))
                metrics["structured_aux/fusion_r2_dropped_state"] = (1.0 - err.mean()).detach()
            metrics["structured_aux/fusion_rows_state_masked"] = torch.tensor(float(srows.float().mean()), device=device)
            if getattr(self, "state_follows_instruction", False):
                im = getattr(self, "_instr_mask", None)
                irows = im.any(1).to(srows.device) if im is not None else torch.zeros_like(srows)
                metrics["structured_aux/fusion_rows_instr_masked"] = irows.float().mean().detach()
                # 0 unless an instruction could not be located (fallback): then its state is masked alone
                metrics["structured_aux/state_instr_row_mismatch"] = (srows != irows).float().sum().detach()
        if getattr(self, "fusion_instruction", False) and getattr(self, "_instr_mask", None) is not None:
            imask = self._instr_mask.to(online_hidden.device)
            tmask = self._fusion_align_mask(imask, online_input_ids, target_inputs)
            if not getattr(self, "_fusion_instr_checked", False):
                self._fusion_instr_checked = True
                if not torch.equal(target_inputs["input_ids"][tmask].cpu(), self._instr_mask_orig.cpu()):
                    raise RuntimeError("fusion instruction: target positions do not hold the original instruction ids")
                print(f"__FUSION_INSTR__ rows={int(imask.any(1).sum())} tokens={int(imask.sum())}", flush=True)
            tgt = target_hidden.float()[tmask]
            src = online_hidden.float()[imask]
            tgt = (tgt - tgt.mean(dim=0, keepdim=True)).detach()
            rms = tgt.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            head = self.fusion_predictor["instr"]
            pred = head(src.to(head[1].weight.dtype)).float()
            err_tok = ((pred - tgt) / rms).pow(2).mean(dim=1)
            rows_idx = imask.nonzero(as_tuple=True)[0]
            n_rows = int(imask.any(1).sum())
            row_err = torch.zeros(imask.shape[0], device=err_tok.device).index_add_(0, rows_idx, err_tok)
            row_cnt = torch.zeros(imask.shape[0], device=err_tok.device).index_add_(0, rows_idx, torch.ones_like(err_tok))
            per_row = row_err[row_cnt > 0] / row_cnt[row_cnt > 0]
            losses.append(per_row.sum())
            weights.append(torch.tensor(float(n_rows), device=device))
            metrics["structured_aux/fusion_r2_dropped_instr"] = (1.0 - per_row.mean()).detach()
        if getattr(self, "_fusion_loop_mode", False) and self._last_mq_mask is not None:
            passes = getattr(self, "_loop_pass_out", None)
            cur_f = self._fu_target  # EMA loop-end block on the current frames (all rows)
            fut_multi = [ex.get("future_images_multi") for ex in (examples or [{}] * batch)]
            rows = [i for i in range(batch) if dropped_images is not None and dropped_images[i]
                    and fut_multi[i] is not None]
            metrics["structured_aux/fusion_loop_rows"] = torch.tensor(float(len(rows)) / batch, device=device)
            if rows and passes is not None and cur_f is not None:
                fut_cache = {}  # one EMA pass per distinct offset per step
                for j, off in enumerate(self.fusion_loop_offsets):
                    if j >= len(passes) or not passes[j].requires_grad:
                        continue
                    if off not in fut_cache:
                        imgs = [fut_multi[i][off] for i in rows]
                        _, fut_inputs = self._fusion_target_pass(
                            imgs, [instructions[i] for i in rows],
                            None if state_bins is None else [state_bins[i] for i in rows], update=False,
                            capture_main=False)
                        fut_cache[off] = (self._fu_target, self._aux_camera_masks(fut_inputs["input_ids"], self.fusion_views))
                    fut_f, f_masks = fut_cache[off]
                    for vi, view in enumerate(self.fusion_views):
                        cam = _VIEW_INDEX[view]
                        sel = [r for r, i in enumerate(rows) if cam in (dropped_images[i] or [])]
                        if not sel:
                            continue
                        src = torch.stack([passes[j][rows[r]][on_masks[view][rows[r]]] for r in sel]).float()
                        tgt = torch.stack([fut_f[r][f_masks[view][r]].float() - cur_f[rows[r]][cam_masks[view][rows[r]]].float()
                                           for r in sel])
                        tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                        rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                        head = self.fusion_future_predictor[view if self.fusion_loop_same else f"{view}_{j}"]
                        pred = head(src.to(head[1].weight.dtype)).float()
                        err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                        losses.append(self.fusion_future_weight * err.sum())
                        weights.append(torch.tensor(float(err.numel()), device=device))
                        metrics[f"structured_aux/fusion_loop_r2_{view}_p{j + 1}_t{off}"] = (1.0 - err.mean()).detach()
        if getattr(self, "fusion_future_layer", None) is not None and not getattr(self, "_fusion_loop_mode", False) \
                and self._last_mq_mask is not None:
            cur_f = self._fu_target  # EMA block-future_layer output on the current frames (all rows)
            fut_imgs = [ex.get("future_image") for ex in (examples or [{}] * batch)]
            rows = [i for i in range(batch) if dropped_images is not None and dropped_images[i]
                    and fut_imgs[i] is not None and len(fut_imgs[i]) == len(batch_images[i])]
            metrics["structured_aux/fusion_future_rows"] = torch.tensor(float(len(rows)) / batch, device=device)
            if rows and online_future is not None and cur_f is not None:
                _, fut_inputs = self._fusion_target_pass(
                    [fut_imgs[i] for i in rows], [instructions[i] for i in rows],
                    None if state_bins is None else [state_bins[i] for i in rows], update=False, capture_main=False)
                fut_f = self._fu_target
                f_masks = self._aux_camera_masks(fut_inputs["input_ids"], self.fusion_views)
                for vi, view in enumerate(self.fusion_views):
                    cam = _VIEW_INDEX[view]
                    sel = [j for j, i in enumerate(rows) if cam in (dropped_images[i] or [])]
                    if not sel:
                        continue
                    src = torch.stack([online_future[rows[j]][on_masks[view][rows[j]]] for j in sel]).float()
                    tgt = torch.stack([fut_f[j][f_masks[view][j]].float() - cur_f[rows[j]][cam_masks[view][rows[j]]].float()
                                       for j in sel])
                    tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                    rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                    head = self.fusion_future_predictor[view]
                    pred = head(src.to(head[1].weight.dtype)).float()
                    err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                    losses.append(self.fusion_future_weight * err.sum())
                    weights.append(torch.tensor(float(err.numel()), device=device))
                    metrics[f"structured_aux/fusion_future_r2_{view}"] = (1.0 - err.mean()).detach()
                    metrics[f"structured_aux/fusion_future_pred_rel_{view}"] = (
                        pred.detach().norm() / (tgt.norm() + 1e-6)).float()
        if getattr(self, "fusion_future_layer", None) is not None and not getattr(self, "_fusion_loop_mode", False):  # diagnostics
            for v in self.fusion_views:
                metrics[f"structured_aux/fusion_headw_cur_{v}"] = self.fusion_predictor[v][-1].weight.detach().float().norm()
                metrics[f"structured_aux/fusion_headw_fut_{v}"] = self.fusion_future_predictor[v][-1].weight.detach().float().norm()
        metrics["structured_aux/fusion_rows_dropped"] = torch.tensor(
            float(sum(1 for d in (dropped_images or []) if d)) / batch, device=device)
        if not losses:
            return zero, metrics
        loss = torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)
        return loss * self.fusion_weight + zero, metrics
