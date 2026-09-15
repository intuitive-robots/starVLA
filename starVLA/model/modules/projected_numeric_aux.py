"""Small numeric heads on exactly the intermediate slot states consumed by PI."""
import torch
from torch import nn
from torch.nn import functional as F


class ProjectedNumericAux(nn.Module):
    def __init__(self, hidden_dim, fields, layers=(12, 16, 20), beta=0.1):
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        if len(set(self.layers)) != len(self.layers) or any(i < 0 or i % 2 for i in self.layers):
            raise ValueError('Numeric heads require distinct consumed even PI indices')
        self.offsets, offset = {}, 0
        for name, count in fields:
            self.offsets[name] = (offset, offset + count)
            offset += count
        self.num_slots = offset
        self.beta = float(beta)
        self.dims = {'trajectory2d': 10, 'trajectory3d': 12}
        if not set(self.dims).issubset(self.offsets) or self.beta <= 0:
            raise ValueError('Invalid numeric fields/beta')
        self.heads = nn.ModuleDict({f'{i}_{f}': nn.Linear(hidden_dim, d)
                                   for i in self.layers for f, d in self.dims.items()})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)

    def forward(self, states, slot_mask, targets):
        if max(self.layers) >= len(states):
            raise ValueError('Numeric head layer exceeds PI input depth')
        if slot_mask is None or slot_mask.shape != states[0].shape[:2]:
            raise ValueError('Missing/mismatched MLM slot mask')
        slot_mask = slot_mask.bool()
        if not bool(slot_mask.sum(1).eq(self.num_slots).all()):
            raise ValueError('Numeric slot template mismatch')
        losses, metrics = [], {}
        for layer in self.layers:
            slots = states[layer][slot_mask].reshape(len(targets), self.num_slots, -1)
            for field in self.dims:
                a, b = self.offsets[field]
                prediction = self.heads[f'{layer}_{field}'](slots[:, a:b].mean(1)).float()
                expected = 10 if field == 'trajectory2d' else 15
                present = [field in t and len(t[field]) == expected for t in targets]
                valid = torch.tensor(present, device=prediction.device, dtype=torch.bool)
                if any(present):
                    values = torch.tensor([t[field] for t, keep in zip(targets, present) if keep],
                                          device=prediction.device, dtype=torch.float32)
                    # Worker parser already normalized XY by 1000 and XYZ by 20cm.
                    # Center XY; omit the identically zero 3D origin (not a target).
                    values = values * 2 - 1 if field == 'trajectory2d' else values[:, 3:]
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError('Nonfinite numeric annotation')
                    loss = F.smooth_l1_loss(prediction[valid], values, beta=self.beta)
                    unit_scale = 500 if field == 'trajectory2d' else 20
                    rmse = (prediction[valid] - values).square().mean().sqrt() * unit_scale
                    metrics[f'structured_aux/{field}_l{layer}_rmse'] = rmse.detach()
                else:
                    # Keep every head and its upstream path in the distributed graph.
                    loss = prediction.sum() * 0
                losses.append(loss)
                metrics[f'structured_aux/{field}_l{layer}_loss'] = loss.detach()
                metrics[f'structured_aux/{field}_l{layer}_coverage'] = valid.float().mean()
        return torch.stack(losses).mean(), metrics

    def gradient_metrics(self, action_loss, aux_loss, states, scale):
        inputs = [states[i] for i in self.layers]
        ga = torch.autograd.grad(action_loss, inputs, retain_graph=True)
        gx = torch.autograd.grad(aux_loss, inputs, retain_graph=True)
        metrics = {}
        for i, a, x in zip(self.layers, ga, gx):
            a, x = a.detach().float(), x.detach().float()
            na, nx = a.norm(), x.norm()
            prefix = f'structured_aux/grad_l{i}'
            metrics[prefix + '_action_norm'] = na
            metrics[prefix + '_aux_raw_norm'] = nx
            metrics[prefix + '_weighted_ratio'] = float(scale) * nx / na.clamp_min(1e-12)
            metrics[prefix + '_cosine'] = (a * x).sum() / (na * nx).clamp_min(1e-12)
        return metrics
