from types import SimpleNamespace
from unittest import TestCase, main

import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.modules.shared_z import SharedZMixin


class _Host(nn.Module, SharedZMixin):
    def __init__(self):
        super().__init__()


class SharedZConditioningTest(TestCase):
    def test_token_only_mode_does_not_size_adaln_conditioning(self):
        host = _Host()
        diffusion = {}
        cfg = OmegaConf.create({
            "framework": {
                "shared_z": {
                    "enabled": True,
                    "dim": 128,
                    "cross_memory_tokens": 4,
                    "adaln_conditioning": False,
                }
            }
        })
        host._configure_shared_z(cfg, diffusion)
        self.assertEqual(diffusion["extra_conditioning_dim"], 0)
        self.assertIsNone(host._shared_z_action_conditioning(torch.randn(2, 128)))

    def test_camera_dropout_masks_one_image_span_and_preserves_text(self):
        host = _Host().train()
        host.shared_z_memory_dropout_mode = "camera"
        host.shared_z_memory_dropout_rate = 1.0
        host.qwen_vl_interface = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(image_token_id=99))
        )
        input_ids = torch.tensor([
            [1, 99, 99, 2, 99, 99, 3],
            [4, 99, 99, 5, 99, 99, 6],
        ])
        valid = torch.ones_like(input_ids, dtype=torch.bool)

        keep = host._shared_z_encoder_memory_keep(valid, input_ids)

        self.assertTrue(torch.all(keep[:, [0, 3, 6]]))
        for row in range(2):
            first_kept = bool(keep[row, 1:3].all())
            second_kept = bool(keep[row, 4:6].all())
            self.assertNotEqual(first_kept, second_kept)

    def test_camera_dropout_is_disabled_at_evaluation(self):
        host = _Host().eval()
        host.shared_z_memory_dropout_mode = "camera"
        host.shared_z_memory_dropout_rate = 1.0
        valid = torch.tensor([[True, True, False]])
        keep = host._shared_z_encoder_memory_keep(valid, None)
        self.assertTrue(torch.all(keep))

    def test_token_mask_keeps_z_prefix_visible(self):
        host = _Host()
        host.shared_z_dim = 2
        host.shared_z_cross_memory_tokens = 1
        host.shared_z_memory_projector = nn.Linear(2, 4, bias=False)
        encoder = torch.randn(1, 3, 4)
        valid = torch.ones(1, 3, dtype=torch.bool)
        keep = torch.tensor([[True, False, True]])
        memory, memory_valid = host._augment_shared_z_cross_memory(
            encoder, valid, torch.randn(1, 2), keep
        )
        self.assertEqual(tuple(memory.shape), (1, 4, 4))
        self.assertEqual(memory_valid.tolist(), [[True, True, False, True]])


if __name__ == "__main__":
    main()
