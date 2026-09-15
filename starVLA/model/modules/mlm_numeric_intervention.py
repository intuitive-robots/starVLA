"""Opt-in evaluation-only erasure of fitted numeric MLM-slot directions."""
import logging
import os
from pathlib import Path

import torch


@torch.no_grad()
def erase_numeric_slots(states, slot_mask, payload, mode):
    if mode not in {'identity', 'numeric', 'random_energy'}:
        raise ValueError(f'Unknown numeric-slot intervention: {mode}')
    count = int(payload['total_slots'])
    if slot_mask is None or slot_mask.shape != states[0].shape[:2]:
        raise ValueError('Missing or incompatible MLM slot mask')
    slot_mask = slot_mask.bool()
    if not bool(slot_mask.sum(1).eq(count).all()):
        raise ValueError('Numeric intervention slot count mismatch')
    result = list(states)
    # Match the original probe: erasure arithmetic is FP32 even when the
    # surrounding encoder/projector forward is under BF16 autocast.
    with torch.autocast(device_type=states[0].device.type, enabled=False):
        for layer, groups in payload['bank'].items():
            original = states[int(layer)]
            slots = original[slot_mask].reshape(original.shape[0], count, -1).float()
            edited = slots.clone()
            for field in ['trajectory2d', 'trajectory3d']:
                a, b = payload['offsets'][field]
                entry = groups[field]
                mean = entry['mean'].to(device=slots.device, dtype=torch.float32)
                basis = entry['basis'].to(device=slots.device, dtype=torch.float32)
                centered = slots[:, a:b] - mean
                numeric = (centered @ basis) @ basis.T
                if mode == 'identity':
                    delta = torch.zeros_like(numeric)
                elif mode == 'numeric':
                    delta = numeric
                else:
                    # Predeclared random basis 0; no basis selected on rollout results.
                    random = entry['random_0'].to(device=slots.device, dtype=torch.float32)
                    delta = (centered @ random) @ random.T
                    energy = numeric.square().sum((1, 2), keepdim=True)
                    delta *= (energy / delta.square().sum((1, 2), keepdim=True).clamp_min(1e-20)).sqrt()
                    if not torch.allclose(delta.square().sum((1, 2)), energy.flatten(), rtol=2e-4, atol=1e-6):
                        raise RuntimeError('Random intervention failed energy matching')
                edited[:, a:b] -= delta
            output = original.clone()
            output[slot_mask] = edited.reshape(-1, original.shape[-1]).to(original.dtype)
            result[int(layer)] = output
    return result


def maybe_erase_numeric_slots(model, states):
    path = os.environ.get('STARVLA_MLM_NUMERIC_PATH', '').strip()
    if not path:
        return states
    if model.training or torch.is_grad_enabled():
        raise RuntimeError('Numeric slot intervention is evaluation-only')
    if os.environ.get('STARVLA_PI_COT_SUBSPACE_PATH') or os.environ.get('STARVLA_PI_ALIGNMENT_PATH'):
        raise RuntimeError('Do not stack numeric erasure with other subspace interventions')
    mode = os.environ.get('STARVLA_MLM_NUMERIC_MODE', '')
    key = (path, mode)
    if getattr(model, '_numeric_slot_key', None) != key:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if str(model.config.run_id) != Path(payload['checkpoint']).parents[1].name:
            raise ValueError('Numeric-probe checkpoint/run mismatch')
        numeric_head = getattr(model, 'projected_numeric_aux', None)
        expected_layers = sorted(numeric_head.layers) if numeric_head is not None else [12, 14, 16, 18, 20]
        if sorted(map(int, payload['bank'])) != expected_layers:
            raise ValueError('Unexpected numeric-probe layers')
        if int(payload['total_slots']) != sum(n for _, n in model.qwen_vl_interface.encoder_mlm_fields):
            raise ValueError('Numeric-probe template mismatch')
        model._numeric_slot_payload = payload
        model._numeric_slot_key = key
        logging.info('MLM numeric rollout intervention ready: mode=%s path=%s', mode, path)
    return erase_numeric_slots(states, model.qwen_vl_interface._last_encoder_mlm_slot_mask,
                               model._numeric_slot_payload, mode)
