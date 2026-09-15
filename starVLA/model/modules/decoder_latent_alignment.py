"""Privileged frozen-decoder targets for image-only, PI-consumed reasoning slots.

The teacher is deliberately NOT an nn.Module child: it is neither optimized nor
saved in student checkpoints. At rollout it is never constructed. Targets use the
same augmented views and rewritten annotations as the student training sample.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def content_token_spans(tokenizer, text, actual_ids, patterns):
    """Map field characters to response tokens; fail rather than silently misalign."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if actual_ids[:len(ids)] != ids:
        raise ValueError("Teacher response tokenization differs from chat template")
    spans = {}
    for field, pattern in patterns.items():
        match = pattern.search(text)
        if match is None or not match.group(1).strip():
            spans[field] = []
            continue
        start, end = match.span(1)
        # Decoder position zero is BOS; position one holds response token zero.
        spans[field] = [1 + i for i, (a, b) in enumerate(encoded["offset_mapping"])
                        if b > start and a < end and b > a]
    return spans


class _TargetsCaptured(Exception):
    """Stop the frozen teacher before unused layers and vocabulary logits."""


class FrozenDecoderTeacher:
    def __init__(self, checkpoint, layer=23, micro_batch_size=4, *, layers=None, benchmark_batches=False):
        self.checkpoint = str(checkpoint)
        self.layer = int(layer)
        self.multilayer = layers is not None
        self.layers = tuple(int(i) for i in layers) if self.multilayer else (self.layer,)
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 0:
            raise ValueError("Teacher layers must be distinct nonnegative indices")
        self.benchmark_batches = bool(benchmark_batches)
        self.micro_batch_size = int(micro_batch_size)
        if self.micro_batch_size < 1:
            raise ValueError("Teacher microbatch must be positive")
        self.model = None

    @torch.no_grad()
    def audit(self, images, instructions, conversations, fields, baseline, device):
        """Paired target-sensitivity check, not proof of grounding or invariance."""
        import copy
        import numpy as np
        from PIL import Image
        rows = [i for i, c in enumerate(conversations) if c is not None][:4]
        count = len(rows)
        if count < 2:
            return {}
        images = [images[i] for i in rows]
        instructions = [instructions[i] for i in rows]
        conversations = [conversations[i] for i in rows]
        saved_microbatch = self.micro_batch_size
        try:
            self.micro_batch_size = 1
            singleton, valid_singleton = self.targets(images, instructions, conversations, fields, device)
        finally:
            self.micro_batch_size = saved_microbatch
        blank = [[Image.fromarray(np.zeros_like(np.asarray(view))) for view in views] for views in images]
        no_vision, valid_zero = self.targets(blank, instructions, conversations, fields, device)
        swapped = copy.deepcopy(conversations)
        for row in range(count):
            swapped[row][1]["value"] = conversations[(row + 1) % count][1]["value"]
        wrong, valid_wrong = self.targets(images, instructions, swapped, fields, device)
        ref, valid = baseline[0][rows], baseline[1][rows]
        if self.multilayer:
            valid = valid[:, None, :].expand(ref.shape[:-1])
            valid_singleton = valid_singleton[:, None, :].expand_as(valid)
            valid_zero = valid_zero[:, None, :].expand_as(valid)
            valid_wrong = valid_wrong[:, None, :].expand_as(valid)
        metrics = {}
        keep = valid & valid_singleton
        batch_delta = (ref - singleton).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-8)
        metrics['structured_aux/latent_teacher_audit/batch_vs_singleton_max_relative'] = batch_delta[keep].max()
        if float(batch_delta[keep].max()) > 0.03:
            raise RuntimeError("Teacher targets depend excessively on padding/batch composition")
        for j, name in enumerate(fields):
            for mode, target, ok in [('no_vision', no_vision, valid_zero), ('swapped_answer', wrong, valid_wrong)]:
                keep = valid[..., j] & ok[..., j]
                metrics[f"structured_aux/latent_teacher_audit/{name}_{mode}_count"] = keep.sum().float()
                cosine = F.cosine_similarity(ref[..., j, :], target[..., j, :], dim=-1)
                relative = (ref[..., j, :] - target[..., j, :]).norm(dim=-1) / ref[..., j, :].norm(dim=-1).clamp_min(1e-8)
                metrics[f"structured_aux/latent_teacher_audit/{name}_{mode}_cos"] = (cosine * keep).sum() / keep.sum().clamp_min(1)
                metrics[f"structured_aux/latent_teacher_audit/{name}_{mode}_relative_change"] = (relative * keep).sum() / keep.sum().clamp_min(1)
        print("DECODER_TEACHER_AUDIT", {k: float(v) for k, v in metrics.items()}, flush=True)
        return metrics

    @torch.no_grad()
    def benchmark(self, images, instructions, conversations, fields, baseline, device):
        """Smoke-only full-local-batch timing and target parity across chunk sizes."""
        import json
        import time
        reference, valid = baseline
        valid = valid[:, None, :].expand(reference.shape[:-1]) if self.multilayer else valid
        saved = self.micro_batch_size
        metrics = {}
        try:
            for batch_size in (4, 8, 16):
                self.micro_batch_size = batch_size
                timings = []
                worst_delta = 0.0
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(2):
                    start = time.perf_counter()
                    targets, present = self.targets(images, instructions, conversations, fields, device)
                    torch.cuda.synchronize(device)
                    timings.append(time.perf_counter() - start)
                    if not torch.equal(present, baseline[1]):
                        raise RuntimeError("Teacher batch size changed field coverage")
                    delta = (reference - targets).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-8)
                    worst_delta = max(worst_delta, float(delta[valid].max()))
                    del targets
                if worst_delta > 0.03:
                    raise RuntimeError(f"Teacher batch {batch_size} target drift {worst_delta} exceeds 3%")
                record = dict(batch_size=batch_size, samples=len(images), layers=list(self.layers),
                              seconds=timings, max_relative_difference=worst_delta,
                              peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
                print("DECODER_TEACHER_BATCH_BENCHMARK", json.dumps(record), flush=True)
                metrics[f"structured_aux/latent_teacher_benchmark/batch{batch_size}_seconds"] = sum(timings) / len(timings)
        finally:
            self.micro_batch_size = saved
        return metrics

    def _load(self, device):
        if self.model is None:
            from omegaconf import OmegaConf
            from starVLA.model.modules.vlm import get_vlm_model
            checkpoint = Path(self.checkpoint)
            config = OmegaConf.load(checkpoint.parent.parent / "config.full.yaml")
            if not config.framework.qwenvl.get("separate_cross_attention", False):
                raise ValueError("Latent teacher requires separate-cross encoder/decoder")
            config.framework.qwenvl.skip_decoder = False
            config.framework.qwenvl.collect_encoder_layers = False
            # Teacher initialization must not change student diffusion/dropout RNG.
            with torch.random.fork_rng(devices=[]):
                model = get_vlm_model(config=config)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            prefix = "qwen_vl_interface."
            weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            if not weights:
                raise ValueError("No VLM weights in decoder teacher checkpoint")
            model.load_state_dict(weights, strict=True)
            del weights
            model.requires_grad_(False)
            model.eval()
            model.model.gradient_checkpointing_disable()
            self.model = model.to(device=device, dtype=torch.bfloat16)
            if any(p.requires_grad for p in self.model.parameters()):
                raise RuntimeError("Decoder teacher is not frozen")
        self.model.eval()
        return self.model

    @torch.no_grad()
    def targets(self, images, instructions, conversations, fields, device):
        from starVLA.model.modules.vlm.QWen3_EncDec import _ENCODER_MLM_PATTERNS
        teacher = self._load(device)
        adapters = teacher._text_model().cross_attn_adapters
        if max(self.layers) >= len(adapters):
            raise ValueError("Teacher decoder layer out of range")
        targets, present = [], []
        # Small chunks bound frozen-teacher activation memory independently of
        # the student's microbatch (16). No decoded videos or targets are cached.
        for start in range(0, len(images), self.micro_batch_size):
            stop = start + self.micro_batch_size
            chunk_conversations = conversations[start:stop]
            inputs = teacher.build_qwenvl_inputs(
                images=images[start:stop], instructions=instructions[start:stop],
                cot_conversations=chunk_conversations, cot_modes=["cot"] * len(chunk_conversations))
            all_spans = []
            # Transfer metadata once per chunk, not once per row after GPU upload.
            prefixes = inputs["_cot_encoder_prefix_length"].cpu().tolist()
            token_rows = inputs["input_ids"].cpu().tolist()
            for row, conversation in enumerate(chunk_conversations):
                prefix = prefixes[row]
                actual = token_rows[row][prefix:]
                all_spans.append(content_token_spans(
                    teacher.processor.tokenizer, conversation[1]["value"] if conversation is not None else "", actual,
                    {name: _ENCODER_MLM_PATTERNS[name] for name in fields}))
            captured = {}

            def capture(layer, _module, _args, hidden):
                rows = []
                for row, spans in enumerate(all_spans):
                    vectors = []
                    for name in fields:
                        indices = spans[name]
                        if indices and max(indices) >= hidden.shape[1]:
                            raise ValueError("Decoder field exceeds response activations")
                        vector = (hidden[row, indices].float().mean(0) if indices
                                  else hidden[row, 0].float() * 0)
                        vectors.append(vector)
                    rows.append(torch.stack(vectors))
                captured[layer] = torch.stack(rows)
                if layer == max(self.layers):
                    raise _TargetsCaptured()

            from functools import partial
            hooks = [adapters[layer].register_forward_hook(partial(capture, layer))
                     for layer in self.layers]
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
            except _TargetsCaptured:
                pass
            finally:
                for hook in hooks:
                    hook.remove()
                # The backbone retains encoder memory for optional generation.
                teacher._text_model()._cross_encoder_cache = None
                teacher._text_model()._aux_ctx.clear()
                if hasattr(teacher, "_layer_out"):
                    teacher._layer_out.clear()
            if set(captured) != set(self.layers):
                raise RuntimeError("Teacher did not reach every requested decoder layer")
            targets.append(torch.stack([captured[layer] for layer in self.layers], dim=1)
                           if self.multilayer else captured[self.layer])
            present.extend([[bool(spans[name]) for name in fields] for spans in all_spans])
        return torch.cat(targets), torch.tensor(present, device=device, dtype=torch.bool)


class DecoderLatentAlignment(nn.Module):
    def __init__(self, hidden_dim, teacher_dim, fields, layers=(4, 8, 12)):
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(i < 0 or i % 2 for i in self.layers):
            raise ValueError("Alignment must use distinct, consumed even PI indices")
        self.offsets, offset = {}, 0
        for name, count in fields:
            self.offsets[name] = (offset, offset + count)
            offset += count
        self.num_slots = offset
        # One shallow linear map per depth, shared across semantic fields.
        # No predictor transformer or token-generation decoder can route around it.
        self.heads = nn.ModuleDict({str(i): nn.Linear(hidden_dim, teacher_dim, bias=False)
                                   for i in self.layers})

    def forward(self, states, slot_mask, target, present):
        if max(self.layers) >= len(states) or slot_mask is None:
            raise ValueError("Missing student alignment states/slots")
        if slot_mask.shape != states[0].shape[:2] or not bool(slot_mask.sum(1).eq(self.num_slots).all()):
            raise ValueError("Alignment mask/template mismatch")
        if target.requires_grad:
            raise ValueError("Teacher targets must be stop-gradient")
        target = target.detach().float()
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Nonfinite teacher target")
        if target.ndim == 3:
            # Legacy checkpoints used one common target for all student layers.
            layer_targets = [target] * len(self.layers)
        elif target.ndim == 4 and target.shape[1] == len(self.layers):
            layer_targets = list(target.unbind(dim=1))
        else:
            raise ValueError("Teacher target layer count/shape differs from student mapping")
        if any(t.shape[:2] != present.shape for t in layer_targets):
            raise ValueError("Teacher field coverage shape mismatch")
        losses, metrics = [], {}
        for layer, layer_target in zip(self.layers, layer_targets):
            slots = states[layer][slot_mask.bool()].reshape(target.shape[0], self.num_slots, -1)
            pooled = torch.stack([slots[:, a:b].mean(1) for a, b in self.offsets.values()], dim=1)
            prediction = self.heads[str(layer)](pooled).float()
            loss = 1 - F.cosine_similarity(prediction, layer_target, dim=-1)
            losses.append((loss * present).sum() / present.sum().clamp_min(1))
            for j, field in enumerate(self.offsets):
                valid = present[:, j]
                metrics[f"structured_aux/latent_align/{field}_l{layer}"] = (
                    (loss[:, j] * valid).sum() / valid.sum().clamp_min(1)).detach()
        metrics["structured_aux/latent_align/coverage"] = present.float().mean()
        stacked_targets = torch.stack(layer_targets, dim=1)
        valid_targets = present[:, None, :].expand(stacked_targets.shape[:-1])
        metrics["structured_aux/latent_align/teacher_norm"] = (
            (stacked_targets.norm(dim=-1) * valid_targets).sum() / valid_targets.sum().clamp_min(1))
        # A large same-field cosine can reveal generic text/template-dominated targets.
        if target.shape[0] > 1:
            pair_valid = valid_targets & valid_targets.roll(1, 0)
            pair_cos = F.cosine_similarity(stacked_targets, stacked_targets.roll(1, 0), dim=-1)
            metrics["structured_aux/latent_align/teacher_shuffled_cos"] = (
                (pair_cos * pair_valid).sum() / pair_valid.sum().clamp_min(1)).detach()
        return torch.stack(losses).mean(), metrics

    def gradient_metrics(self, action_loss, aux_loss, states, scale):
        inputs = [states[i] for i in self.layers]
        ga = torch.autograd.grad(action_loss, inputs, retain_graph=True)
        gx = torch.autograd.grad(aux_loss, inputs, retain_graph=True)
        metrics = {}
        for layer, action, aux in zip(self.layers, ga, gx):
            action, aux = action.detach().float(), aux.detach().float()
            na, nx = action.norm(), aux.norm()
            prefix = f"structured_aux/latent_align/grad_l{layer}"
            metrics[prefix + "_action_norm"] = na
            metrics[prefix + "_aux_raw_norm"] = nx
            metrics[prefix + "_weighted_ratio"] = float(scale) * nx / na.clamp_min(1e-12)
            metrics[prefix + "_cosine"] = (action * aux).sum() / (na * nx).clamp_min(1e-12)
        return metrics
