"""The E7f + future-copies robustness recipe on the OpenPI PI0 / PI05 frameworks (default off).

Port of the Qwen-GR00T recipe (modality_dropout.py 'independent' + 'pixels', state_tokens.py
state_dropout_mask, instruction_mask.py, fusion_registers.py masked_query with use_queries false,
future_copies.py) onto pi0.5's prefix, reading the SAME config keys, so a launch line carries the
same flags as the Qwen arm:

    framework.modality_dropout   {enabled, mode: independent|camera|all, removal: pixels, vision_rate, protect_state}
    framework.state_dropout_rate / state_dropout_mask
    framework.instruction_dropout_rate
    framework.fusion_registers   {enabled, mode: masked_query, use_queries: false, views, layer, state, instruction,
                                  ema, mlp_hidden, weight}
    framework.future_copies      {enabled, offset, layer, weight}

pi0.5 prefix layout used here: [camera tokens (256 per camera, SigLIP) | text "Task: <instruction>, State: <bins>;
\\nAction: " | future copies of the cameras (256 per camera)]; full bidirectional attention over the valid prefix.

What maps to what (Qwen reference -> pi0.5):
  * camera dropout: the camera's [-1, 1] image tensor is set to 0 (uniform mid-gray), exactly what zeroing the
    Qwen pixel_values (mean = std = 0.5) does; the image tokens stay in the sequence and are attended.
  * state mask: the reference state is 8 one-token slots. pi0.5 writes each bin as "▁" + one token per digit, so a
    masked row would leak the digit count of every bin if masked in place. A masked row therefore carries exactly
    n_dims mask tokens (one per state dimension) in place of the whole numeric span; the reconstruction target of
    mask token i is the EMA copy's layer-L state of dimension i, averaged over that dimension's tokens.
  * instruction mask: every instruction token is replaced in place by one learned mask embedding (as the reference).
  * mask embeddings are raw token rows (scaled by sqrt(width) like every pi0.5 text token), initialised from the
    <seg000> / <seg001> rows of the PaliGemma table.
  * EMA target: a copy of the PaliGemma prefix stream (SigLIP + projector + Gemma, ``self.vlm``) kept outside the
    module tree, fp32 shadow lerped at decay ``ema`` before every target pass (fusion_registers._fusion_ema_update).
    The prefix never attends to the action expert, so the target pass runs the prefix alone and stops at layer L.
  * losses, normalisation, weights and metrics follow fusion_registers._fusion_register_loss and
    future_copies._future_copies_loss line by line (comments name the reference lines).
"""

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.modules.modality_dropout import _resolve_scheduled_rate
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_VIEW_INDEX = {"external": 0, "wrist": 1, "cam0": 0, "cam1": 1, "cam2": 2}
_MASK_INIT_TOKENS = {"state": "<seg000>", "instr": "<seg001>"}


def _cfg_dict(config, key):
    value = config.framework.get(key, {}) if hasattr(config.framework, "get") else getattr(config.framework, key, {})
    return dict(value or {})


class PIRecipeMixin:
    # ── config ───────────────────────────────────────────────────────────────
    def _configure_pi_recipe(self, config) -> None:
        fw = config.framework
        md = _cfg_dict(config, "modality_dropout")
        self.pi_md_enabled = bool(md.get("enabled", False))
        self.pi_md_rate = md.get("vision_rate", 0.0)
        self.pi_md_mode = str(md.get("mode", "all")).lower()
        self.pi_md_protect_state = bool(md.get("protect_state", True))
        self._modality_step = 0  # published by the trainer (_publish_train_step)
        if self.pi_md_enabled:
            if str(md.get("removal", "pixels")).lower() != "pixels":
                raise ValueError("PI recipe: modality_dropout.removal must be 'pixels'")
            if self.pi_md_mode not in ("independent", "camera", "all"):
                raise ValueError("PI recipe: modality_dropout.mode must be independent, camera or all")
            if float(md.get("inference_vision_rate", 0.0) or 0.0) != 0.0:
                raise ValueError("PI recipe: inference_vision_rate > 0 is not implemented")
        elif max(_resolve_scheduled_rate(self.pi_md_rate, s, "x") for s in (0, 10**9)) > 0.0:
            raise ValueError("framework.modality_dropout has a non-zero rate but enabled=false")
        # modality_dropout.cameras (default: every image slot): the slots camera dropout may pick. A padded slot
        # (e.g. pi0.5-DROID's masked right_wrist_0_rgb) must be left out, or independent dropout can remove both
        # real cameras of a row (the all-cameras guard would only see the padded one survive).
        cams = md.get("cameras", None)
        if cams in (None, "null", "", []):
            self.pi_md_cameras = list(range(len(self.image_keys)))
        else:
            self.pi_md_cameras = sorted({int(c) for c in ([cams] if isinstance(cams, (int, str)) else cams)})
            if not all(0 <= c < len(self.image_keys) for c in self.pi_md_cameras):
                raise ValueError(f"modality_dropout.cameras {self.pi_md_cameras} outside image_keys {self.image_keys}")

        self.pi_state_rate = float(fw.get("state_dropout_rate", 0.0) or 0.0)
        self.pi_state_mask = bool(fw.get("state_dropout_mask", False))
        if self.pi_state_rate > 0.0 and not self.pi_state_mask:
            raise ValueError("PI recipe: state_dropout_rate needs state_dropout_mask true (omitting the state is not ported)")
        self.pi_instr_rate = float(fw.get("instruction_dropout_rate", 0.0) or 0.0)
        if not 0.0 <= self.pi_instr_rate < 1.0 or not 0.0 <= self.pi_state_rate <= 1.0:
            raise ValueError("PI recipe: dropout rates out of range")
        if (self.pi_state_rate > 0.0 or self.pi_instr_rate > 0.0) and not self.discrete_state_input:
            raise ValueError("PI recipe: state / instruction masking needs discrete_state_input (state in the prompt)")

        fr = _cfg_dict(config, "fusion_registers")
        self.pi_fusion = bool(fr.get("enabled", False))
        if self.pi_fusion:
            if str(fr.get("mode", "registers")).lower() != "masked_query" or bool(fr.get("use_queries", True)):
                raise ValueError("PI recipe: only fusion_registers mode masked_query with use_queries false is ported")
            for key in ("future_layer", "loop_future_offsets", "loop_target_offset"):
                if fr.get(key, None) not in (None, "null", "", []):
                    raise ValueError(f"PI recipe: fusion_registers.{key} is not ported")
            if str(fr.get("target_center", "none") or "none").lower() != "none":
                raise ValueError("PI recipe: fusion_registers.target_center is not ported")
            views = fr.get("views", ["external", "wrist"])
            self.pi_views = [str(v).lower() for v in ([views] if isinstance(views, str) else views)]
            if any(v not in _VIEW_INDEX or _VIEW_INDEX[v] >= len(self.image_keys) for v in self.pi_views):
                raise ValueError(f"PI recipe: fusion views {self.pi_views} do not fit image_keys {self.image_keys}")
            self.pi_fusion_layer = int(fr.get("layer", 10))
            self.pi_fusion_weight = float(fr.get("weight", 1.0))
            self.pi_ema_decay = float(fr.get("ema", 0.995))
            self.pi_mlp_hidden = int(fr.get("mlp_hidden", 2048))
            self.pi_fusion_state = bool(fr.get("state", False))
            self.pi_fusion_instr = bool(fr.get("instruction", False))
            if self.pi_fusion_state and not (self.pi_state_mask and self.pi_state_rate > 0.0):
                raise ValueError("fusion_registers.state needs state_dropout_mask true and state_dropout_rate > 0")
            if self.pi_fusion_instr and self.pi_instr_rate <= 0.0:
                raise ValueError("fusion_registers.instruction needs instruction_dropout_rate > 0")
            if not (self.pi_md_enabled and self.pi_md_mode in ("camera", "independent")):
                raise ValueError("fusion_registers needs modality_dropout {enabled, mode: camera | independent, removal: pixels}")
        fc = _cfg_dict(config, "future_copies")
        self.pi_fc = bool(fc.get("enabled", False))
        if self.pi_fc:
            if not self.pi_fusion:
                raise ValueError("future_copies uses the fusion_registers EMA copy: enable fusion_registers masked_query")
            if str(fc.get("target", "ema")).lower() != "ema":
                raise ValueError("PI recipe: future_copies.target must be ema")
            self.pi_fc_offset = int(fc.get("offset", 8))
            self.pi_fc_layer = int(fc.get("layer", 10))
            self.pi_fc_weight = float(fc.get("weight", 1.0))
            vla = config.datasets.vla_data
            offs = {int(k) for k in (vla.get("future_offsets") or [])}
            if self.pi_fc_offset not in offs and int(vla.get("shared_z_future_offset", 0) or 0) != self.pi_fc_offset:
                raise ValueError("future_copies.offset must be in datasets.vla_data.future_offsets")
        depth = int(self.config.framework.paligemma_config.depth)
        for name in ("pi_fusion_layer", "pi_fc_layer"):
            if getattr(self, name, None) is not None and not 0 <= getattr(self, name) < depth:
                raise ValueError(f"PI recipe: {name} outside the {depth} Gemma layers")
        self.pi_recipe_on = bool(self.pi_md_enabled or self.pi_state_rate > 0 or self.pi_instr_rate > 0
                                 or self.pi_fusion or self.pi_fc)
        self._pi_counts = {"instr_rows": 0, "instr_fallback": 0}
        self._pi_logged = set()
        if self.pi_recipe_on:
            logger.info(
                "PI recipe ON: camera dropout=%s mode=%s rate=%s | state mask %.2f | instruction mask %.2f | "
                "reconstruction=%s layer=%s views=%s | future copies=%s offset=%s layer=%s",
                self.pi_md_enabled, self.pi_md_mode, self.pi_md_rate, self.pi_state_rate, self.pi_instr_rate,
                self.pi_fusion, getattr(self, "pi_fusion_layer", None), getattr(self, "pi_views", None),
                self.pi_fc, getattr(self, "pi_fc_offset", None), getattr(self, "pi_fc_layer", None))

    def _pi_log_once(self, key: str, msg: str) -> None:
        if key not in self._pi_logged:
            self._pi_logged.add(key)
            print(msg, flush=True)

    # ── modules ──────────────────────────────────────────────────────────────
    def _build_pi_recipe_modules(self) -> None:
        if not self.pi_recipe_on:
            return
        h = int(self.config.framework.paligemma_config.width)
        table = self.vlm.language_model.embed_tokens.weight
        self._pi_mask_init_ids = {}
        for name, enabled in (("state", self.pi_state_rate > 0.0), ("instr", self.pi_instr_rate > 0.0)):
            if not enabled:
                continue
            emb = nn.Embedding(1, h)
            init_id = self.tokenizer.piece_id(_MASK_INIT_TOKENS[name])
            with torch.no_grad():
                emb.weight.copy_(table[init_id].detach().float()[None])
            self._pi_mask_init_ids[name] = init_id
            setattr(self, f"{name}_mask_embed", emb)
        if self.pi_fusion:
            m = self.pi_mlp_hidden

            def mlp():
                net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, m), nn.GELU(), nn.Linear(m, h))
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                return net

            heads = {v: mlp() for v in self.pi_views}
            if self.pi_fusion_instr:
                heads["instr"] = mlp()
            if self.pi_fusion_state:
                heads["state"] = mlp()
            self.fusion_predictor = nn.ModuleDict(heads)
        if self.pi_fc:
            n_cams = len(self.image_keys)
            self.pi_n_img_tokens = self._num_image_tokens()
            self.fcopy_marker = nn.Parameter(torch.zeros(n_cams, self.pi_n_img_tokens, h))
            preds = {}
            for j in range(n_cams):
                net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 2048), nn.GELU(), nn.Linear(2048, h))
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
                preds[str(j)] = net
            self.fcopy_predictor = nn.ModuleDict(preds)

    def _num_image_tokens(self) -> int:
        vc = self.vlm.config.vision_config
        return (int(self.image_resolution[0]) // int(vc.patch_size)) * (int(self.image_resolution[1]) // int(vc.patch_size))

    def _pi_aux_active(self) -> bool:
        return bool((self.pi_fusion or self.pi_fc) and self.training and torch.is_grad_enabled())

    # ── random draws (same order and semantics as the Qwen reference) ────────
    def _pi_current_vision_rate(self) -> float:
        if not self.pi_md_enabled or not self.training:
            return 0.0
        return _resolve_scheduled_rate(self.pi_md_rate, int(getattr(self, "_modality_step", 0)), "modality_dropout.vision_rate")

    def _pi_draw(self, batch: int) -> Dict:
        """state keep (StateTokenMixin._add_state_conditioning) -> cameras (_modality_vision_keep +
        _modality_zero_pixels) -> instruction (_mask_instruction_tokens), each a numpy draw as in the reference."""
        if self.training and self.pi_state_rate > 0.0:
            state_keep = np.random.random(batch) >= self.pi_state_rate
        else:
            state_keep = np.ones(batch, dtype=bool)
        dropped = [[] for _ in range(batch)]
        rate = self._pi_current_vision_rate()
        if rate > 0.0:
            if self.pi_md_mode == "independent":
                vision_keep = np.zeros(batch, dtype=bool)
            else:
                vision_keep = np.random.random(batch) >= rate
            if self.pi_md_protect_state:
                vision_keep = vision_keep | ~state_keep
            for row in range(batch):
                if vision_keep[row]:
                    continue
                cams = list(self.pi_md_cameras)
                if self.pi_md_mode == "camera":
                    chosen = [cams[int(np.random.randint(len(cams)))]]
                elif self.pi_md_mode == "independent":
                    chosen = [i for i in cams if np.random.random() < rate]
                    if len(chosen) == len(cams):
                        chosen.pop(int(np.random.randint(len(chosen))))
                else:
                    chosen = cams
                dropped[row] = chosen
        else:
            vision_keep = np.ones(batch, dtype=bool)
        if self.training and self.pi_instr_rate > 0.0:
            instr_draw = np.random.random(batch) < self.pi_instr_rate
        else:
            instr_draw = np.zeros(batch, dtype=bool)
        return {"state_keep": state_keep, "dropped": dropped, "vision_keep": vision_keep, "instr_draw": instr_draw}

    # ── EMA copy of the prefix stream (fusion_registers._fusion_ema_update) ──
    @torch.no_grad()
    def _pi_ema_update(self):
        st = self.__dict__.get("_pi_ema_state")
        if st is None:
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

            for mod in self.vlm.modules():
                for k, v in vars(mod).items():
                    if k not in ("_parameters", "_buffers", "_modules"):
                        _scan(v)
                for v in list(mod._buffers.values()):
                    _scan(v)
            ema = copy.deepcopy(self.vlm, memo)
            ema.eval()
            # The target pass stops after the deepest recipe layer, so the copy keeps only Gemma layers 0..that
            # layer (same outputs, ~5 GB less with the fp32 shadow at layer 10 of 18).
            used = [self.pi_fusion_layer] if self.pi_fusion else []
            used += [self.pi_fc_layer] if self.pi_fc else []
            lm = ema.language_model
            n_keep = max(used) + 1 if used else len(lm.layers)
            if n_keep < len(lm.layers):
                lm.layers = nn.ModuleList(list(lm.layers)[:n_keep])
            src = dict(self.vlm.named_parameters())
            pairs = []
            for n2, pe in ema.named_parameters():
                if n2 not in src:
                    raise RuntimeError(f"EMA parameter {n2} has no online counterpart")
                p = src[n2]
                pe.requires_grad_(False)
                if p.requires_grad:
                    pairs.append((p, pe))
                else:
                    pe.data = p.data
            shadow = [p.detach().float().clone() for p, _ in pairs]
            st = {"ema": ema, "pairs": pairs, "shadow": shadow}
            self.__dict__["_pi_ema_state"] = st
            logger.info("PI recipe: EMA prefix stream created (%d trainable tensors, %.1fM params in fp32 shadow, "
                        "%d non-leaf attributes detached)", len(pairs), sum(s.numel() for s in shadow) / 1e6, n_nonleaf)
            return ema
        w = 1.0 - self.pi_ema_decay
        for (p, pe), sh in zip(st["pairs"], st["shadow"]):
            sh.lerp_(p.detach().float(), w)
            pe.data.copy_(sh)
        return st["ema"]

    # ── losses ───────────────────────────────────────────────────────────────
    def _pi_recipe_losses(self, ctx: Dict) -> tuple:
        """ctx: online captures and the token layout of this batch (built in PI0._forward_split)."""
        device = ctx["device"]
        batch = ctx["batch"]
        params = []
        for name in ("fusion_predictor", "fcopy_predictor"):
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
        for name in ("fcopy_marker",):
            if hasattr(self, name):
                params.append(getattr(self, name))
        for name in ("state_mask_embed", "instr_mask_embed"):
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
        zero = sum(p.sum() for p in params) * 0.0  # DeepSpeed: every recipe parameter in every graph
        metrics: Dict[str, torch.Tensor] = {}
        dropped = ctx["dropped"]
        state_rows = ctx["state_masked"]          # [B] bool numpy
        instr_mask = ctx["instr_mask_online"]     # [B, Lp_online] bool tensor or None
        any_cam = any(len(d) > 0 for d in dropped)
        any_state = bool(self.pi_fusion_state and state_rows.any())
        any_instr = bool(self.pi_fusion_instr and instr_mask is not None and bool(instr_mask.any()))
        fc_rows = ctx["fc_rows"] if self.pi_fc else []
        need_target = self.pi_fusion and (any_cam or any_state or any_instr)
        ema_done = False
        target_h = None
        if need_target or fc_rows:
            ema = self._pi_ema_update()
            ema_done = True
        total = None
        # ---- masked-latent reconstruction (fusion_registers._fusion_register_loss, masked_query) ----
        if self.pi_fusion:
            online = ctx["online_capture"][self.pi_fusion_layer]
            if not need_target:
                metrics["structured_aux/fusion_rows_dropped"] = torch.tensor(0.0, device=device)
                fusion_loss = zero
            else:
                target_h = self._pi_target_pass(ema, ctx["target_inputs"], self.pi_fusion_layer)
                losses, weights = [], []
                n_img = ctx["n_img_tokens"]
                for view in self.pi_views:
                    cam = _VIEW_INDEX[view]
                    sl = slice(cam * n_img, (cam + 1) * n_img)
                    dmask = torch.tensor([cam in dropped[i] for i in range(batch)], device=device, dtype=torch.bool)
                    if not bool(dmask.any()):
                        continue
                    tok = target_h[:, sl].float()                       # EMA camera tokens, every row
                    tgt = (tok - tok.mean(dim=(0, 1), keepdim=True)).detach()
                    rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                    src = online[:, sl].float()
                    head = self.fusion_predictor[view]
                    pred = head(src.to(head[1].weight.dtype)).float()
                    err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                    w_row = dmask.float()
                    losses.append((w_row * err).sum())
                    weights.append(w_row.sum())
                    metrics[f"structured_aux/fusion_r2_dropped_{view}"] = (1.0 - err[dmask].mean()).detach()
                if self.pi_fusion_state:
                    if any_state:
                        rows = np.nonzero(state_rows)[0].tolist()
                        src = torch.stack([online[r, ctx["state_pos_online"][r]] for r in rows]).float()  # [R, n_dims, D]
                        tgt = torch.stack([
                            torch.stack([target_h[r, a:b].float().mean(dim=0) for a, b in ctx["state_dim_spans_target"][r]])
                            for r in rows])
                        tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                        rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                        head = self.fusion_predictor["state"]
                        pred = head(src.to(head[1].weight.dtype)).float()
                        err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                        losses.append(err.sum())
                        weights.append(torch.tensor(float(err.numel()), device=device))
                        metrics["structured_aux/fusion_r2_dropped_state"] = (1.0 - err.mean()).detach()
                    metrics["structured_aux/fusion_rows_state_masked"] = torch.tensor(float(state_rows.mean()), device=device)
                if any_instr:
                    imask = instr_mask.to(device)
                    tmask = ctx["instr_mask_target"].to(device)
                    tgt = target_h[tmask].float()
                    src = online[imask].float()
                    if tgt.shape != src.shape:
                        raise RuntimeError(f"PI recipe: instruction target {tuple(tgt.shape)} vs online {tuple(src.shape)}")
                    tgt = (tgt - tgt.mean(dim=0, keepdim=True)).detach()
                    rms = tgt.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
                    head = self.fusion_predictor["instr"]
                    pred = head(src.to(head[1].weight.dtype)).float()
                    err_tok = ((pred - tgt) / rms).pow(2).mean(dim=1)
                    rows_idx = imask.nonzero(as_tuple=True)[0]
                    n_rows = int(imask.any(1).sum())
                    row_err = torch.zeros(imask.shape[0], device=device).index_add_(0, rows_idx, err_tok)
                    row_cnt = torch.zeros(imask.shape[0], device=device).index_add_(0, rows_idx, torch.ones_like(err_tok))
                    per_row = row_err[row_cnt > 0] / row_cnt[row_cnt > 0]
                    losses.append(per_row.sum())
                    weights.append(torch.tensor(float(n_rows), device=device))
                    metrics["structured_aux/fusion_r2_dropped_instr"] = (1.0 - per_row.mean()).detach()
                metrics["structured_aux/fusion_rows_dropped"] = torch.tensor(
                    float(sum(1 for d in dropped if d)) / batch, device=device)
                if losses:
                    fusion_loss = torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)
                    fusion_loss = fusion_loss * self.pi_fusion_weight + zero
                else:
                    fusion_loss = zero
            total = fusion_loss
        # ---- future copies (future_copies._future_copies_loss, target ema) ----
        if self.pi_fc:
            metrics["structured_aux/fcopy_rows"] = torch.tensor(float(len(fc_rows)) / batch, device=device)
            online = ctx["online_capture"][self.pi_fc_layer]
            if fc_rows:
                if not ema_done:
                    ema = self._pi_ema_update()
                rows_t = torch.as_tensor(fc_rows, device=device)
                # current frames: the fusion target pass when it ran at this layer on every row, else a pass of its own
                if target_h is not None and self.pi_fc_layer == self.pi_fusion_layer:
                    cur = target_h[rows_t]
                else:
                    cur = self._pi_target_pass(ema, ctx["target_inputs"], self.pi_fc_layer)[rows_t]
                fut = self._pi_target_pass(ema, ctx["future_inputs"], self.pi_fc_layer)
                n_img = ctx["n_img_tokens"]
                losses, n_err = [], 0
                for j in range(len(self.image_keys)):
                    on_sl = slice(ctx["copy_offset"] + j * n_img, ctx["copy_offset"] + (j + 1) * n_img)
                    sl = slice(j * n_img, (j + 1) * n_img)
                    src = online[rows_t][:, on_sl].float()
                    tgt = fut[:, sl].float() - cur[:, sl].float()
                    tgt = (tgt - tgt.mean(dim=(0, 1), keepdim=True)).detach()
                    rms = tgt.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
                    head = self.fcopy_predictor[str(j)]
                    pred = head(src.to(head[1].weight.dtype)).float()
                    err = ((pred - tgt) / rms).pow(2).mean(dim=(1, 2))
                    losses.append(err.sum())
                    n_err += err.numel()
                    metrics[f"structured_aux/fcopy_r2_cam{j}"] = (1.0 - err.mean()).detach()
                fc_loss = self.pi_fc_weight * torch.stack(losses).sum() / max(n_err, 1) + zero
            else:
                fc_loss = zero
            total = fc_loss if total is None else total + fc_loss
        return total, metrics

    @torch.no_grad()
    def _pi_target_pass(self, ema, inputs: Dict, layer: int) -> torch.Tensor:
        """EMA prefix pass on full information (true images, instruction and state, no copies), stopped after
        Gemma layer `layer`; returns that layer's output [B, Lp, D]."""
        embs, pad = self._embed_prefix_tensors(ema, inputs["images"], inputs["img_masks"], inputs["tokens"],
                                               inputs["token_masks"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, caps, _ = self._run_prefix(ema, embs, pad, capture=(layer,), stop_after=layer)
        return caps[layer]
