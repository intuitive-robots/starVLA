"""Future copies of the camera tokens (F1, default off).

``framework.future_copies`` appends, after the text and state, a second copy of every camera image of the row
(full token grid, the same current image), marks those tokens with a learned per-position "future" vector
(added at block-0 input, zero-initialised), and trains them, at block ``layer``, to predict that camera's change
over ``offset`` frames as seen by the EMA copy of the backbone: EMA_L(t + offset) - EMA_L(t), per token.
The target is applied on every row that has a future frame, whether the camera itself is present or dropped
(Flex-pi's cross-modality forcing): a dropped camera's copy is grayed with it, so its future must come from the
other inputs. The original tokens keep the present; the action head reads both (its context is all tokens).
At inference the copies are always built (one forward pass, a longer sequence).

Needs fusion_registers (mode masked_query) for the EMA copy, and the dataloader's future frames at ``offset``
(datasets.vla_data.future_offsets containing it, or shared_z_future_offset equal to it).

``state: true`` (2026-10-07) adds future-STATE copies: ``[FSTATE] <n state slots> [/FSTATE]`` after the state block,
every slot holding one learned query id, trained at the same block to predict the state tokens' change as seen by the
EMA copy: EMA_L(state tokens | s(t + offset)) - EMA_L(state tokens | s(t)) (``state_target: delta``, default; ``abs``
predicts EMA_L(state tokens | s(t + offset))). Needs the reserved-id state tokens and the dataloader's
``datasets.vla_data.future_state_offset`` equal to ``offset``. With it, the future EMA pass reads s(t + offset) on the
rows that have it (future images AND future state), so the camera targets also see the future state there.

Config::
    future_copies:
      enabled: false
      offset: 8
      layer: 24
      weight: 1.0
      state: false
      state_target: delta
      target: ema        # ema: EMA_L(t+offset) - EMA_L(t) of the copy's camera (default);
                         # dynaflip: the frozen DynaFLIP-base patch-feature change F(t+offset) - F(t),
                         # pooled to the camera's token grid (same target as future_latent.target dynaflip)
                         # teacher (2026-10-08, T5): the CURRENT frame's precomputed teacher features of the copy's
                         # camera through framework.teacher_align's head (mode copies); needs no future frames
"""
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.modules.future_latent import _StopAfterFutureLatentLayer
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

FSTATE_PLACEHOLDER = "<|fim_prefix|>"


class FutureCopyMixin:
    def _configure_future_copies(self, config) -> None:
        cfg = dict(config.framework.get("future_copies", {}) or {})
        self.fcopy_enabled = bool(cfg.get("enabled", False))
        self._fc_pending = None
        self._fc_capture = None
        self._fc_n = 0
        if not self.fcopy_enabled:
            return
        self.fcopy_offset = int(cfg.get("offset", 8))
        self.fcopy_layer = int(cfg.get("layer", 24))
        self.fcopy_weight = float(cfg.get("weight", 1.0))
        self.fcopy_tokens = int(cfg.get("tokens_per_camera", 64))
        self.fcopy_target = str(cfg.get("target", "ema")).lower()
        if self.fcopy_target not in ("ema", "dynaflip", "teacher"):
            raise ValueError("framework.future_copies.target must be 'ema', 'dynaflip' or 'teacher'")
        if self.fcopy_target == "teacher":
            ta = dict(config.framework.get("teacher_align", {}) or {})
            if not (bool(ta.get("enabled", False)) and str(ta.get("mode", "")).lower() == "copies"):
                raise ValueError("future_copies.target teacher needs framework.teacher_align {enabled: true, mode: copies}")
        if self.fcopy_target == "dynaflip" and not hasattr(self, "future_latent_dynaflip_dir"):
            self.future_latent_dynaflip_dir = str(cfg.get("dynaflip_dir", "/e/project1/m3/blank4/weights"))
        fr = dict(config.framework.get("fusion_registers", {}) or {})
        if not (bool(fr.get("enabled", False)) and str(fr.get("mode", "")) == "masked_query"):
            raise ValueError("future_copies uses the fusion_registers EMA copy: enable fusion_registers mode masked_query")
        vla = config.datasets.vla_data
        offs = {int(k) for k in (vla.get("future_offsets") or [])}
        if (self.fcopy_target != "teacher" and self.fcopy_offset not in offs
                and int(vla.get("shared_z_future_offset", 0) or 0) != self.fcopy_offset):
            raise ValueError("future_copies.offset must be in datasets.vla_data.future_offsets "
                             "(or equal shared_z_future_offset)")
        self.fcopy_state = bool(cfg.get("state", False))
        self.fcopy_state_target = str(cfg.get("state_target", "delta")).lower()
        if self.fcopy_state:
            if self.fcopy_state_target not in ("delta", "abs"):
                raise ValueError("future_copies.state_target must be 'delta' or 'abs'")
            if getattr(self, "state_token_mode", None) != "reserved_ids":
                raise ValueError("future_copies.state needs the reserved-id state tokens (state_token_mode reserved_ids)")
            if int(vla.get("future_state_offset", 0) or 0) != self.fcopy_offset:
                raise ValueError("future_copies.state needs datasets.vla_data.future_state_offset == future_copies.offset")
            if self.fcopy_target != "ema":
                raise ValueError("future_copies.state is implemented for target: ema")
        self._fc_logged = False
        logger.info("future copies ON: offset=%d layer=%d weight=%.3f target=%s state=%s(%s)", self.fcopy_offset,
                    self.fcopy_layer, self.fcopy_weight, self.fcopy_target, self.fcopy_state, self.fcopy_state_target)

    def _build_future_copies(self, hidden_size: int, n_cameras: int) -> None:
        if not getattr(self, "fcopy_enabled", False):
            return
        h = int(hidden_size)
        self._fc_n_cams = int(n_cameras)
        self.fcopy_marker = nn.Parameter(torch.zeros(self._fc_n_cams, self.fcopy_tokens, h))
        out = 768 if getattr(self, "fcopy_target", "ema") == "dynaflip" else h  # DynaFLIP-base feature width
        self.fcopy_predictor = nn.ModuleDict({
            str(j): nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2048), nn.GELU(), nn.Linear(2048, out))
            for j in range(self._fc_n_cams)
        }) if getattr(self, "fcopy_target", "ema") != "teacher" else nn.ModuleDict()  # teacher: teacher_align's head
        for mlp in self.fcopy_predictor.values():
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        if getattr(self, "fcopy_state", False):
            self.fcopy_state_predictor = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2048), nn.GELU(), nn.Linear(2048, h))
            nn.init.zeros_(self.fcopy_state_predictor[-1].weight)
            nn.init.zeros_(self.fcopy_state_predictor[-1].bias)
            tok = self._resolve_tokenizer()
            pid = tok.convert_tokens_to_ids([FSTATE_PLACEHOLDER])
            if not pid or pid[0] is None or pid[0] == getattr(tok, "unk_token_id", None):
                raise RuntimeError(f"future_copies.state: {FSTATE_PLACEHOLDER!r} is not a single token")
            self._fc_state_placeholder_id = int(pid[0])
            n_state = len(self._state_bin_ids) + len(self._state_group_ids)
            # one learned query id, past the state (n_state), fusion registers, state mask (+64) and instruction
            # mask (+65) rows
            self._fc_state_query_id = int(list(self._rare_token_ids(n_state + 67))[-1])
            taken = set(self._state_bin_ids + self._state_group_ids)
            for name in ("_state_mask_id", "_instr_mask_id"):
                if getattr(self, name, None) is not None:
                    taken.add(int(getattr(self, name)))
            taken |= set(getattr(self, "_fusion_reg_ids", []) or [])
            if self._fc_state_query_id in taken:
                raise RuntimeError("future_copies.state: query id collides with a state / fusion / mask id")
            self._install_trainable_bin_embedding([self._fc_state_query_id], name="fcopy_state_embed")
            self._fc_state_ids = torch.tensor(sorted(taken & set(self._state_bin_ids + self._state_group_ids)))
        blocks = self._aux_blocks()
        if not 0 <= self.fcopy_layer < len(blocks):
            raise ValueError(f"future_copies.layer {self.fcopy_layer} outside the {len(blocks)} blocks")
        first = blocks[0]

        def pre_hook(module, args, kwargs):
            if module is not first:
                return None
            pend = self._fc_pending
            if pend is None:
                return None
            self._fc_pending = None
            mask, add = pend
            if "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                hs = kwargs["hidden_states"].clone()
                hs[mask] = hs[mask] + add.to(hs.dtype)
                kwargs["hidden_states"] = hs
                return args, kwargs
            hs = args[0].clone()
            hs[mask] = hs[mask] + add.to(hs.dtype)
            return (hs,) + tuple(args[1:]), kwargs

        def cap_hook(module, inputs, output):
            cap = self._fc_capture
            if cap is None or getattr(self, "_loop_inside", False):
                return None
            cap["hidden"] = output[0] if isinstance(output, tuple) else output
            if cap.get("stop", False):
                raise _StopAfterFutureLatentLayer()
            return None

        first.register_forward_pre_hook(pre_hook, with_kwargs=True)
        blocks[self.fcopy_layer].register_forward_hook(cap_hook)

    # ── prompt ───────────────────────────────────────────────────────────────
    def _fc_duplicate(self, batch_images: List[list]) -> List[list]:
        """Images + their copies; the copies go after the text (interface._n_tail_images)."""
        if not getattr(self, "fcopy_enabled", False):
            return batch_images
        n = {len(imgs) for imgs in batch_images}
        if len(n) != 1:
            raise ValueError("future_copies needs the same number of cameras on every row")
        self._fc_n = n.pop()
        if self._fc_n != self._fc_n_cams:
            raise ValueError(f"future_copies built for {self._fc_n_cams} cameras, batch has {self._fc_n}")
        self.qwen_vl_interface._n_tail_images = self._fc_n
        return [list(imgs) + list(imgs) for imgs in batch_images]

    def _fc_add_state_slots(self, instructions: List[str]) -> List[str]:
        """future_copies.state: one future-state slot per state token, after the state block (train and eval)."""
        if not getattr(self, "fcopy_state", False):
            return instructions
        n = int(getattr(self, "_state_width_seen", 0) or 0) + len(getattr(self, "state_token_group_sizes", []) or [])
        if n <= 0:
            raise RuntimeError("future_copies.state: no state width seen yet (state conditioning must run first)")
        self._fc_state_n = n
        slots = " ".join([FSTATE_PLACEHOLDER] * n)
        return [f"{instr} [FSTATE] {slots} [/FSTATE]" for instr in instructions]

    def _fc_splice_state_slots(self, inputs: dict) -> None:
        """Every pass (online, EMA target, FC EMA) holds the same query id in the slots (token-aligned)."""
        if not getattr(self, "fcopy_state", False):
            return
        ids = inputs["input_ids"]
        pos = ids == self._fc_state_placeholder_id
        if bool(pos.any()):
            if not bool((pos.sum(1) == self._fc_state_n).all()):
                raise RuntimeError(f"future_copies.state: {pos.sum(1).tolist()} slots per row, expected {self._fc_state_n}")
            ids[pos] = self._fc_state_query_id

    def _fc_reset_prompt(self) -> None:
        if getattr(self, "fcopy_enabled", False):
            self.qwen_vl_interface._n_tail_images = 0

    def _fc_image_masks(self, input_ids: torch.Tensor, index: int) -> torch.Tensor:
        cfg = self.qwen_vl_interface.model.config
        image_token_id = int(getattr(cfg, "image_token_id", 151655))
        cam = torch.cumsum((input_ids == int(cfg.vision_start_token_id)).long(), dim=1) - 1
        return (input_ids == image_token_id) & (cam == int(index))

    def _fc_arm(self, qwen_inputs: dict) -> None:
        """Arm the future marker on the copy tokens (train and eval) and, in training, the block-L capture."""
        self._fc_pending = None
        self._fc_capture = None
        if not getattr(self, "fcopy_enabled", False):
            return
        self._ema_done_this_step = False
        ids = qwen_inputs["input_ids"]
        full = torch.zeros_like(ids, dtype=torch.bool)
        adds = []
        per_cam = [self._fc_image_masks(ids, self._fc_n + j) for j in range(self._fc_n)]
        for row in range(ids.shape[0]):
            for j in range(self._fc_n):
                pos = per_cam[j][row].nonzero(as_tuple=True)[0]
                if pos.numel() > self.fcopy_tokens:
                    raise RuntimeError(f"future copy {j}: {pos.numel()} tokens > tokens_per_camera")
                full[row, pos] = True
                adds.append(self.fcopy_marker[j, :pos.numel()])
        self._fc_pending = (full, torch.cat(adds, dim=0))
        if self.training:
            self._fc_capture = {"hidden": None, "stop": False}
        if not self._fc_logged:
            self._fc_logged = True
            print(f"__FUTURE_COPIES__ cams={self._fc_n} tokens={int(full.sum())} training={self.training}", flush=True)

    def _fc_zero_kwargs(self) -> dict:
        """Arguments for _modality_zero_pixels: dropout picks among the originals, grays the copy too."""
        if not getattr(self, "fcopy_enabled", False):
            return {}
        return {"n_candidates": self._fc_n, "mirror": self._fc_n}

    def _fc_mirror_positions(self, positions) -> list:
        positions = [int(p) for p in (positions or [])]
        if not getattr(self, "fcopy_enabled", False):
            return positions
        return positions + [p + self._fc_n for p in positions]

    def _fc_take_online(self) -> None:
        """Right after the online forward: keep the block-L capture before any EMA pass can overwrite it."""
        cap, self._fc_capture = getattr(self, "_fc_capture", None), None
        self._fc_online_hidden = None if cap is None else cap.get("hidden")

    # ── loss ─────────────────────────────────────────────────────────────────
    def _fc_ema_pass(self, images, instructions, state_bins):
        ema = self.__dict__["_fusion_ema_state"]["ema"]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        if state_bins is not None:
            self._pending_state_bins = list(state_bins)
            self._splice_state_tokens(inputs)
        self._fc_splice_state_slots(inputs)
        self._fusion_pending_query = None
        self._fc_pending = None
        self._fc_capture = {"hidden": None, "stop": True}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    ema(**inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
                except _StopAfterFutureLatentLayer:
                    pass
            hid = self._fc_capture["hidden"]
        finally:
            self._fc_capture = None
        if hid is None:
            raise RuntimeError("future_copies: the EMA pass never reached the target block")
        return hid, inputs

    def _future_copies_loss(self, online_input_ids, examples, batch_images, instructions, state_bins):
        if not getattr(self, "fcopy_enabled", False):
            return None, {}
        online, self._fc_online_hidden = getattr(self, "_fc_online_hidden", None), None
        if online is None or not online.requires_grad:
            raise RuntimeError("future_copies: no online block-L capture with gradient")
        device = online.device
        zero = (sum(p.sum() for p in self.fcopy_predictor.parameters()) + self.fcopy_marker.sum()) * 0.0
        if getattr(self, "fcopy_state", False):
            zero = zero + sum(p.sum() for p in self.fcopy_state_predictor.parameters()) * 0.0
        if getattr(self, "fcopy_target", "ema") == "teacher":  # T5: current-frame teacher target, no future frames
            return self._teacher_copies_loss(online, online_input_ids, examples, zero)
        off = self.fcopy_offset
        futs = []
        for ex in examples:
            multi = ex.get("future_images_multi")
            if multi is not None and off in multi:
                futs.append(multi[off])
            elif ex.get("future_image") is not None and not multi:
                futs.append(ex["future_image"])
            else:
                futs.append(None)
        rows = [i for i, f in enumerate(futs) if f is not None and len(f) == len(batch_images[i])]
        metrics = {"structured_aux/fcopy_rows": torch.tensor(float(len(rows)) / len(examples), device=device)}
        if not rows:
            return zero, metrics
        dyn = getattr(self, "fcopy_target", "ema") == "dynaflip"
        if not dyn:
            if self.__dict__.get("_fusion_ema_state") is None or not getattr(self, "_ema_done_this_step", False):
                self._fusion_ema_update()
                self._ema_done_this_step = True
            sb = None if state_bins is None else [state_bins[i] for i in rows]
            ins = [instructions[i] for i in rows]
            sb_fut, fs_rows = sb, []
            if getattr(self, "fcopy_state", False) and sb is not None:
                sb_fut = list(sb)
                for k, i in enumerate(rows):
                    fs = examples[i].get("future_state")
                    if fs is not None and sb[k] is not None and not isinstance(sb[k], str):
                        sb_fut[k] = self._state_bins(self._order_and_validate_state(fs))
                        fs_rows.append(k)
            cur, cur_in = self._fc_ema_pass([batch_images[i] for i in rows], ins, sb)
            fut, fut_in = self._fc_ema_pass([futs[i] for i in rows], ins, sb_fut)
        losses, n_err = [], 0
        rows_t = torch.as_tensor(rows, device=online.device)
        for j in range(self._fc_n):
            m_on = self._fc_image_masks(online_input_ids, self._fc_n + j)[rows_t]
            n_tok = int(m_on[0].sum())
            src = online[rows_t][m_on].reshape(len(rows), n_tok, -1).float()
            if dyn:
                # true (unmasked) frames: dropout grays the pixel tensor, batch_images keeps the originals
                g = int(round(n_tok ** 0.5))
                if g * g != n_tok:
                    raise RuntimeError(f"future_copies dynaflip: camera {j} has {n_tok} tokens, not a square grid")
                f_now = self._dynaflip_grid([batch_images[i][j] for i in rows], g, device)
                f_fut = self._dynaflip_grid([futs[i][j] for i in rows], g, device)
                tgt = (f_fut - f_now).float()
            else:
                m_cur = self._fc_image_masks(cur_in["input_ids"], j)
                m_fut = self._fc_image_masks(fut_in["input_ids"], j)
                tgt = (fut.float()[m_fut.to(fut.device)] - cur.float()[m_cur.to(cur.device)]).reshape(len(rows), n_tok, -1)
            tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
            rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
            head = self.fcopy_predictor[str(j)]
            pred = head(src.to(head[1].weight.dtype)).float()
            err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
            losses.append(err.sum())
            n_err += err.numel()
            metrics[f"structured_aux/fcopy_r2_cam{j}"] = (1.0 - err.mean()).detach()
        if getattr(self, "fcopy_state", False) and not dyn:
            metrics["structured_aux/fcopy_state_rows"] = torch.tensor(float(len(fs_rows)) / len(examples), device=device)
            if fs_rows:
                k_t = torch.as_tensor(fs_rows, device=online.device)
                q_on = (online_input_ids[rows_t] == self._fc_state_query_id)[k_t]
                sid = self._fc_state_ids.to(fut_in["input_ids"].device)
                m_fut = torch.isin(fut_in["input_ids"], sid)[k_t.to(fut_in["input_ids"].device)]
                m_cur = torch.isin(cur_in["input_ids"], sid)[k_t.to(cur_in["input_ids"].device)]
                n_s = self._fc_state_n
                if not (bool((q_on.sum(1) == n_s).all()) and bool((m_fut.sum(1) == n_s).all())
                        and bool((m_cur.sum(1) == n_s).all())):
                    raise RuntimeError("future_copies.state: query / state token counts do not match")
                src = online[rows_t][k_t][q_on].reshape(len(fs_rows), n_s, -1).float()
                f_h = fut.float()[k_t.to(fut.device)][m_fut].reshape(len(fs_rows), n_s, -1)
                c_h = cur.float()[k_t.to(cur.device)][m_cur].reshape(len(fs_rows), n_s, -1)
                tgt = f_h - c_h if self.fcopy_state_target == "delta" else f_h
                tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                head = self.fcopy_state_predictor
                pred = head(src.to(head[1].weight.dtype)).float()
                err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                losses.append(err.sum())
                n_err += err.numel()
                metrics["structured_aux/fcopy_r2_state"] = (1.0 - err.mean()).detach()
                if not getattr(self, "_fc_state_logged", False):
                    self._fc_state_logged = True
                    print(f"__FUTURE_STATE__ rows={len(fs_rows)} slots={n_s} target={self.fcopy_state_target}", flush=True)
        loss = self.fcopy_weight * torch.stack(losses).sum() / max(n_err, 1) + zero
        return loss, metrics
