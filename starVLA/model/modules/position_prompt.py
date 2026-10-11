"""Position prompt: simulator object masks / 3D boxes as a droppable input and as a target (default off).

Needs per-row annotations (``datasets.vla_data.annot: true`` on lerobot_3_0/libero_lerobot_annot): a 4-group
role mask per camera (0 rest, 1 robot, 2 manipulated object, 3 target location; 64 x 64, moved with the image
crop / rotation by AnnotVideoAugment), 3D boxes of up to two manipulated objects and two targets in the robot
frame, and per-camera validity (our simulator replay matches the training frame).

``input: true`` appends, after the text and state, a duplicate of every camera image of the row (full token
grid, the same image), marks those tokens with a learned per-position vector, and adds at block-0 input either
  - the encoded annotation: a SAM-style dense prompt (one-hot mask -> 3 stride-2 convs -> one vector per token
    of the camera's 8 x 8 grid) plus an MLP embedding of the 3D boxes broadcast over the camera's tokens, or
  - a learned per-position query, when the annotation is dropped (probability ``input_dropout`` per row),
    invalid for that camera, or at inference (never available there).
The original image tokens are never touched. A dropped camera (modality dropout) grays its duplicate too.

``target: true`` reads block ``layer`` and predicts the mask (per token: its 8 x 8 cell of the 64 x 64 map,
4-way cross-entropy) and the 3D boxes (camera-pooled MLP, L1 on present slots): from the duplicates when
``input`` is on (only on rows whose annotation was dropped: otherwise it is a copy), from the original image
tokens when it is off. Losses are added to structured_aux (scaled by trainer.loss_scale.structured_aux).

``latent_target: true`` (needs ``input`` and fusion_registers at the same layer; 2026-10-05) predicts the mask's
*latent* instead of decoding it: the EMA target pass of fusion_registers also carries the duplicates, with the
encoded annotation added to every valid camera, and on rows whose annotation was dropped the online block-L
duplicate tokens regress the EMA block-L duplicate tokens (centred, rms-normalised MSE, as the camera / state /
instruction reconstruction). ``blind_given_rate`` grays every camera (originals and duplicates) of that fraction of
annotation-given rows, so on those rows the actions can only come from the mask, state and instruction: the head is
pushed to read the grounding pathway. The grounding-subspace ablation (EXPERIMENTS_2026-10-05 08:21) showed the
decoded-mask arms' heads ignore their grounding.

Config::
    position_prompt:
      enabled: false
      input: true
      target: true
      input_dropout: 0.85
      layer: 16
      mask_weight: 1.0
      box_weight: 1.0
      latent_target: false
      latent_weight: 1.0
      blind_given_rate: 0.0
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

GRID = 8          # 8 x 8 image tokens per camera (64 tokens)
MASK = 64         # annotation mask resolution
CELL = MASK // GRID
N_GROUPS = 4
N_SLOTS = 4


class PositionPromptMixin:
    def _configure_position_prompt(self, config) -> None:
        cfg = dict(config.framework.get("position_prompt", {}) or {})
        self.pp_enabled = bool(cfg.get("enabled", False))
        self._pp_pending = None
        self._pp_capture = None
        self._pp_n = 0
        self.pp_input = self.pp_enabled and bool(cfg.get("input", True))
        self.pp_target = self.pp_enabled and bool(cfg.get("target", True))
        self.pp_latent = self.pp_enabled and bool(cfg.get("latent_target", False))
        self.pp_blind_given = float(cfg.get("blind_given_rate", 0.0)) if self.pp_enabled else 0.0
        self.pp_latent_weight = float(cfg.get("latent_weight", 1.0))
        self._pp_pending_target = None
        self._pp_target_hidden = None
        if not self.pp_enabled:
            return
        if not (self.pp_input or self.pp_target):
            raise ValueError("position_prompt: enable input, target or both")
        if (self.pp_latent or self.pp_blind_given > 0) and not self.pp_input:
            raise ValueError("position_prompt: latent_target / blind_given_rate need input: true")
        fr = dict(config.framework.get("fusion_registers", {}) or {})
        if not bool(fr.get("enabled", False)):
            raise ValueError("position_prompt's loss runs in the structured-aux block: enable fusion_registers")
        if not bool(config.datasets.vla_data.get("annot", False)):
            raise ValueError("position_prompt needs datasets.vla_data.annot: true (per-row simulator annotations)")
        if bool(dict(config.framework.get("future_copies", {}) or {}).get("enabled", False)) and self.pp_input:
            raise ValueError("position_prompt.input and future_copies both append image copies; enable one")
        self.pp_dropout = float(cfg.get("input_dropout", 0.85))
        self.pp_layer = int(cfg.get("layer", 16))
        if self.pp_latent and int(fr.get("layer", 16)) != self.pp_layer:
            raise ValueError(f"position_prompt.latent_target reads the fusion target pass: layer {self.pp_layer} must "
                             f"equal fusion_registers.layer {int(fr.get('layer', 16))}")
        if self.pp_latent and str(fr.get("mode", "registers")) != "masked_query":
            raise ValueError("position_prompt.latent_target needs fusion_registers.mode masked_query")
        self.pp_mask_weight = float(cfg.get("mask_weight", 1.0))
        self.pp_box_weight = float(cfg.get("box_weight", 1.0))
        self._pp_logged = False
        logger.info("position prompt ON: input=%s target=%s latent_target=%s blind_given=%.2f dropout=%.2f layer=%d",
                    self.pp_input, self.pp_target, self.pp_latent, self.pp_blind_given, self.pp_dropout, self.pp_layer)

    def _build_position_prompt(self, hidden_size: int, n_cameras: int) -> None:
        if not getattr(self, "pp_enabled", False):
            return
        h = int(hidden_size)
        self._pp_n_cams = int(n_cameras)
        if self.pp_input:
            self.pp_marker = nn.Parameter(torch.zeros(self._pp_n_cams, GRID * GRID, h))
            self.pp_query = nn.Parameter(torch.zeros(self._pp_n_cams, GRID * GRID, h))
            self.pp_mask_encoder = nn.Sequential(
                nn.Conv2d(N_GROUPS, 64, 4, 2, 1), nn.GELU(),
                nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
                nn.Conv2d(128, 256, 4, 2, 1), nn.GELU(),
            )
            self.pp_mask_proj = nn.Linear(256, h)
            self.pp_box_encoder = nn.Sequential(nn.Linear(N_SLOTS * 7, 512), nn.GELU(), nn.Linear(512, h))
            for lin in (self.pp_mask_proj, self.pp_box_encoder[-1]):  # start as "nothing added"
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        if self.pp_target:
            self.pp_mask_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, CELL * CELL * N_GROUPS))
            self.pp_box_head = nn.Sequential(nn.LayerNorm(h * self._pp_n_cams), nn.Linear(h * self._pp_n_cams, 1024),
                                             nn.GELU(), nn.Linear(1024, N_SLOTS * 6))
        if self.pp_latent:
            m = int(getattr(self, "fusion_mlp_hidden", 2048))
            self.pp_latent_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
        blocks = self._aux_blocks()
        if not 0 <= self.pp_layer < len(blocks):
            raise ValueError(f"position_prompt.layer {self.pp_layer} outside the {len(blocks)} blocks")
        first, target_block = blocks[0], blocks[self.pp_layer]

        def pre_hook(module, args, kwargs):
            if module is first:
                pend, self._pp_pending = self._pp_pending, None
            else:  # the EMA copy carries a copied hook: inert unless the latent-target pass armed it
                pend, self._pp_pending_target = self._pp_pending_target, None
            if pend is None:
                return None
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
            if module is not target_block:
                return None
            cap = self._pp_capture
            if cap is None or getattr(self, "_loop_inside", False):
                return None
            cap["hidden"] = output[0] if isinstance(output, tuple) else output
            return None

        if self.pp_input:
            first.register_forward_pre_hook(pre_hook, with_kwargs=True)
        if self.pp_target or self.pp_latent:
            target_block.register_forward_hook(cap_hook)

    # ── prompt ───────────────────────────────────────────────────────────────
    def _pp_duplicate(self, batch_images: List[list]) -> List[list]:
        if not getattr(self, "pp_input", False):
            return batch_images
        n = {len(imgs) for imgs in batch_images}
        if len(n) != 1:
            raise ValueError("position_prompt needs the same number of cameras on every row")
        self._pp_n = n.pop()
        if self._pp_n != self._pp_n_cams:
            raise ValueError(f"position_prompt built for {self._pp_n_cams} cameras, batch has {self._pp_n}")
        self.qwen_vl_interface._n_tail_images = self._pp_n
        return [list(imgs) + list(imgs) for imgs in batch_images]

    def _pp_reset_prompt(self) -> None:
        if getattr(self, "pp_input", False):
            self.qwen_vl_interface._n_tail_images = 0

    def _pp_zero_kwargs(self) -> dict:
        """Modality dropout picks among the originals and grays the duplicate with its camera."""
        if not getattr(self, "pp_input", False):
            return {}
        return {"n_candidates": self._pp_n, "mirror": self._pp_n}

    def _pp_mirror_positions(self, positions) -> list:
        positions = [int(p) for p in (positions or [])]
        if not getattr(self, "pp_input", False):
            return positions
        return positions + [p + self._pp_n for p in positions]

    def _pp_cam_masks(self, input_ids: torch.Tensor, index: int) -> torch.Tensor:
        cfg = self.qwen_vl_interface.model.config
        image_token_id = int(getattr(cfg, "image_token_id", 151655))
        cam = torch.cumsum((input_ids == int(cfg.vision_start_token_id)).long(), dim=1) - 1
        return (input_ids == image_token_id) & (cam == int(index))

    @staticmethod
    def _pp_annot(examples, key, dtype=np.float32):
        return np.stack([np.asarray(ex["annot"][key], dtype=dtype) for ex in examples], 0)

    def _pp_box_features(self, examples, device) -> torch.Tensor:
        box = torch.as_tensor(self._pp_annot(examples, "box3d"), device=device)            # [B, 4, 6]
        ok = torch.as_tensor(self._pp_annot(examples, "slotok"), device=device)[..., None]  # [B, 4, 1]
        return torch.cat([box * ok, ok], -1).flatten(1)                                    # [B, 28]

    def _pp_arm(self, qwen_inputs: dict, examples, has_annot: bool) -> None:
        """Arm the block-0 additions on the duplicates (train and eval) and, in training, the block-L capture."""
        self._pp_pending = None
        self._pp_capture = None
        self._pp_given = None
        self._pp_blind = None
        self._pp_examples = None
        if not getattr(self, "pp_enabled", False):
            return
        if (self.pp_target or self.pp_latent) and self.training:
            self._pp_capture = {"hidden": None}
        if not self.pp_input:
            return
        ids = qwen_inputs["input_ids"]
        B, dev = ids.shape[0], ids.device
        cams = range(self._pp_n)
        per_cam = [self._pp_cam_masks(ids, self._pp_n + j) for j in cams]
        given = torch.zeros(B, self._pp_n, dtype=torch.bool, device=dev)
        add_full = torch.zeros(*ids.shape, self.pp_query.shape[-1], device=dev, dtype=self.pp_query.dtype)
        if self.training and has_annot:
            row_on = torch.as_tensor(np.random.random(B) >= self.pp_dropout, device=dev)
            valid = torch.as_tensor(self._pp_annot(examples, "valid", np.uint8), device=dev).bool()[:, :self._pp_n]
            given = row_on[:, None] & valid
            if bool(given.any()):
                m = torch.as_tensor(self._pp_annot(examples, "mask", np.int64), device=dev)[:, :self._pp_n]
                onehot = F.one_hot(m.clamp(0, N_GROUPS - 1), N_GROUPS).permute(0, 1, 4, 2, 3).float()
                feat = self.pp_mask_encoder(onehot.flatten(0, 1).to(self.pp_mask_proj.weight.dtype))  # [B*C,256,8,8]
                tok = self.pp_mask_proj(feat.flatten(2).transpose(1, 2)).view(B, self._pp_n, GRID * GRID, -1)
                box = self.pp_box_encoder(self._pp_box_features(examples, dev).to(self.pp_mask_proj.weight.dtype))
                enc = tok + box[:, None, None, :]                                                # [B, C, 64, h]
        full = torch.zeros_like(ids, dtype=torch.bool)
        for row in range(B):
            for j in cams:
                pos = per_cam[j][row].nonzero(as_tuple=True)[0]
                if pos.numel() != GRID * GRID:
                    raise RuntimeError(f"position_prompt: duplicate {j} has {pos.numel()} tokens, expected {GRID * GRID}")
                full[row, pos] = True
                vec = self.pp_marker[j] + (enc[row, j] if bool(given[row, j]) else self.pp_query[j])
                add_full[row, pos] = vec.to(add_full.dtype)
        self._pp_pending = (full, add_full[full])
        self._pp_given = given
        if self.training and has_annot and self.pp_latent:
            self._pp_examples = examples  # the latent-target pass encodes every valid camera's annotation
        if self.training and self.pp_blind_given > 0 and bool(given.any()):
            # blind rows: gray every camera (originals and duplicates) so the row's actions must come from the
            # mask, state and instruction (pixels are zeroed in place, like modality dropout's removal: pixels)
            per_row = [2 * self._pp_n] * B
            pixel_values, offsets = self._pixel_patch_offsets(qwen_inputs, per_row)
            pick = torch.as_tensor(np.random.random(B) < self.pp_blind_given, device=dev) & given.any(1)
            for row in pick.nonzero(as_tuple=True)[0].tolist():
                lo, hi = offsets[row * 2 * self._pp_n], offsets[(row + 1) * 2 * self._pp_n]
                pixel_values[lo:hi] = 0
            self._pp_blind = pick
        if not self._pp_logged:
            self._pp_logged = True
            print(f"__POSITION_PROMPT__ cams={self._pp_n} tokens={int(full.sum())} given={int(given.sum())}/"
                  f"{given.numel()} blind_rows={0 if self._pp_blind is None else int(self._pp_blind.sum())} "
                  f"latent={self.pp_latent} training={self.training}", flush=True)

    def _pp_latent_duplicate(self, batch_images: List[list]) -> List[list]:
        """Target pass of fusion_registers: duplicate the images too (latent_target only)."""
        if not (getattr(self, "pp_latent", False) and self.training and self._pp_examples is not None):
            return batch_images
        return self._pp_duplicate(batch_images)

    def _pp_arm_target(self, target_inputs: dict) -> None:
        """Arm the EMA block-0 addition: encoded annotation on every valid camera's duplicate (no grad)."""
        if not (getattr(self, "pp_latent", False) and self.training and self._pp_examples is not None):
            return
        self._pp_reset_prompt()
        examples = self._pp_examples
        ids = target_inputs["input_ids"]
        B, dev = ids.shape[0], ids.device
        valid = torch.as_tensor(self._pp_annot(examples, "valid", np.uint8), device=dev).bool()[:, :self._pp_n]
        with torch.no_grad():
            m = torch.as_tensor(self._pp_annot(examples, "mask", np.int64), device=dev)[:, :self._pp_n]
            onehot = F.one_hot(m.clamp(0, N_GROUPS - 1), N_GROUPS).permute(0, 1, 4, 2, 3).float()
            feat = self.pp_mask_encoder(onehot.flatten(0, 1).to(self.pp_mask_proj.weight.dtype))
            tok = self.pp_mask_proj(feat.flatten(2).transpose(1, 2)).view(B, self._pp_n, GRID * GRID, -1)
            box = self.pp_box_encoder(self._pp_box_features(examples, dev).to(self.pp_mask_proj.weight.dtype))
            enc = tok + box[:, None, None, :]
            full = torch.zeros_like(ids, dtype=torch.bool)
            add_full = torch.zeros(*ids.shape, self.pp_query.shape[-1], device=dev, dtype=self.pp_query.dtype)
            for j in range(self._pp_n):
                cm = self._pp_cam_masks(ids, self._pp_n + j)
                for row in range(B):
                    pos = cm[row].nonzero(as_tuple=True)[0]
                    if pos.numel() != GRID * GRID:
                        raise RuntimeError(f"position_prompt target: duplicate {j} has {pos.numel()} tokens")
                    full[row, pos] = True
                    vec = self.pp_marker[j] + (enc[row, j] if bool(valid[row, j]) else self.pp_query[j])
                    add_full[row, pos] = vec.to(add_full.dtype)
        self._pp_pending_target = (full, add_full[full].detach())
        self._pp_target_ids = ids

    def _pp_set_target_hidden(self, hidden) -> None:
        if getattr(self, "pp_latent", False) and getattr(self, "_pp_pending_target", None) is None \
                and getattr(self, "_pp_target_ids", None) is not None:
            self._pp_target_hidden = hidden

    def _pp_take_online(self) -> None:
        cap, self._pp_capture = getattr(self, "_pp_capture", None), None
        self._pp_online_hidden = None if cap is None else cap.get("hidden")

    # ── loss ─────────────────────────────────────────────────────────────────
    def _position_prompt_loss(self, online_input_ids, examples):
        dev = online_input_ids.device
        zero = sum(p.sum() for n, p in self.named_parameters() if n.startswith("pp_")) * 0.0
        hid, self._pp_online_hidden = getattr(self, "_pp_online_hidden", None), None
        metrics = {}
        lat_loss = None
        if self.pp_latent:
            lat_loss = self._pp_latent_loss(hid, online_input_ids, examples, metrics)
        if self.pp_blind_given > 0 and getattr(self, "_pp_blind", None) is not None:
            metrics["structured_aux/pp_blind_frac"] = self._pp_blind.float().mean().detach()
        if not self.pp_target or hid is None:
            if self.pp_input and self._pp_given is not None:
                metrics["structured_aux/pp_given_frac"] = torch.tensor(float(self._pp_given.float().mean()), device=dev)
            return zero + (lat_loss if lat_loss is not None else 0.0), metrics
        B = online_input_ids.shape[0]
        n = self._pp_n if self.pp_input else len(examples[0]["image"])
        base = self._pp_n if self.pp_input else 0
        toks = []
        for j in range(n):
            m = self._pp_cam_masks(online_input_ids, base + j)
            if not bool((m.sum(1) == GRID * GRID).all()):
                raise RuntimeError(f"position_prompt target: camera {j} is not a full {GRID}x{GRID} grid")
            toks.append(hid[m].view(B, GRID * GRID, -1))
        valid = torch.as_tensor(self._pp_annot(examples, "valid", np.uint8), device=dev).bool()[:, :n]
        use = valid.clone()
        if self.pp_input and self._pp_given is not None:
            use &= ~self._pp_given                    # given rows: predicting the input back is a copy
        gt = torch.as_tensor(self._pp_annot(examples, "mask", np.int64), device=dev)[:, :n]   # [B, C, 64, 64]
        losses = []
        if bool(use.any()):
            ce = []
            for j in range(n):
                logits = self.pp_mask_head(toks[j].to(self.pp_mask_head[1].weight.dtype)).float()  # [B, 64, CELL*CELL*4]
                logits = logits.view(B, GRID, GRID, CELL, CELL, N_GROUPS).permute(0, 5, 1, 3, 2, 4)
                logits = logits.reshape(B, N_GROUPS, MASK, MASK)
                l = F.cross_entropy(logits, gt[:, j], reduction="none").mean((1, 2))           # [B]
                ce.append(l[use[:, j]])
                with torch.no_grad():
                    pred = logits.argmax(1)
                    for g, name in ((2, "manip"), (3, "target")):
                        inter = ((pred == g) & (gt[:, j] == g))[use[:, j]].sum()
                        union = ((pred == g) | (gt[:, j] == g))[use[:, j]].sum().clamp_min(1)
                        metrics[f"structured_aux/pp_iou_{name}_cam{j}"] = (inter.float() / union.float()).detach()
            ce = torch.cat(ce)
            losses.append(self.pp_mask_weight * ce.mean())
            metrics["structured_aux/pp_mask_ce"] = ce.mean().detach()
        rows = use.any(1)
        if bool(rows.any()):
            pooled = torch.cat([t.mean(1) for t in toks], -1).to(self.pp_box_head[1].weight.dtype)   # [B, C*h]
            pred = self.pp_box_head(pooled).float().view(B, N_SLOTS, 6)
            box = torch.as_tensor(self._pp_annot(examples, "box3d"), device=dev)
            ok = torch.as_tensor(self._pp_annot(examples, "slotok"), device=dev).bool() & rows[:, None]
            if bool(ok.any()):
                l1 = (pred - box).abs().mean(-1)[ok]
                losses.append(self.pp_box_weight * l1.mean())
                metrics["structured_aux/pp_box_l1_m"] = l1.mean().detach()
        metrics["structured_aux/pp_target_rows"] = torch.tensor(float(rows.float().mean()), device=dev)
        if self.pp_input and self._pp_given is not None:
            metrics["structured_aux/pp_given_frac"] = torch.tensor(float(self._pp_given.float().mean()), device=dev)
        if lat_loss is not None:
            losses.append(lat_loss)
        loss = zero + (sum(losses) if losses else 0.0)
        return loss, metrics

    def _pp_latent_loss(self, hid, online_input_ids, examples, metrics):
        """Rows whose annotation was dropped: online block-L duplicates regress the EMA block-L duplicates that saw
        the encoded annotation (centred, rms-normalised MSE over tokens, as fusion_registers)."""
        tgt_h, self._pp_target_hidden = getattr(self, "_pp_target_hidden", None), None
        tgt_ids, self._pp_target_ids = getattr(self, "_pp_target_ids", None), None
        if hid is None or tgt_h is None or tgt_ids is None or self._pp_given is None:
            metrics["structured_aux/pp_latent_rows"] = torch.tensor(0.0, device=online_input_ids.device)
            return None
        dev = online_input_ids.device
        B, n = online_input_ids.shape[0], self._pp_n
        valid = torch.as_tensor(self._pp_annot(examples, "valid", np.uint8), device=dev).bool()[:, :n]
        use = valid & ~self._pp_given                                                     # [B, C]
        metrics["structured_aux/pp_latent_rows"] = use.any(1).float().mean().detach()
        if not bool(use.any()):
            return None
        src, tgt = [], []
        for j in range(n):
            om = self._pp_cam_masks(online_input_ids, n + j)
            tm = self._pp_cam_masks(tgt_ids.to(dev), n + j)
            if not (bool((om.sum(1) == GRID * GRID).all()) and bool((tm.sum(1) == GRID * GRID).all())):
                raise RuntimeError(f"position_prompt latent: duplicate {j} is not a full {GRID}x{GRID} grid")
            s = hid.float()[om].view(B, GRID * GRID, -1)
            t = tgt_h.float()[tm].view(B, GRID * GRID, -1)
            src.append(s[use[:, j]])
            tgt.append(t[use[:, j]])
        src, tgt = torch.cat(src), torch.cat(tgt)                                         # [R, 64, h]
        tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
        rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
        pred = self.pp_latent_head(src.to(self.pp_latent_head[1].weight.dtype)).float()
        err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))                               # per (row, camera)
        metrics["structured_aux/pp_latent_r2"] = (1.0 - err.mean()).detach()
        return self.pp_latent_weight * err.mean()
