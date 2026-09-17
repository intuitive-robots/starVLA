from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

try:
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader_v4 import (
        LayerwiseFlowmatchingActionHeadV4,
    )
    from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit_v4 import (
        QwenPIv4DiT,
    )
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
    from starVLA.model.framework.VLM4A.QwenPI_v4 import Qwen_PI_v4
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error)) from error


def _tiny_dit(num_layers=2):
    return QwenPIv4DiT(
        num_attention_heads=4,
        attention_head_dim=4,
        output_dim=16,
        num_layers=num_layers,
        dropout=0.0,
        final_dropout=False,
        norm_type="ada_norm",
        positional_embeddings=None,
        input_embedding_dim=16,
        cross_attention_dim=16,
    ).eval()


class QwenPIv4AttentionTest(unittest.TestCase):
    def test_framework_reuses_complete_v3_constructor(self):
        with patch.object(Qwen_PI_v3, "__init__", autospec=True) as parent_init:
            Qwen_PI_v4(OmegaConf.create({"framework": {}}))

        parent_init.assert_called_once()
        merged_config = parent_init.call_args.kwargs["config"]
        self.assertEqual(merged_config.framework.name, "QwenPI_v4")
        self.assertEqual(
            merged_config.framework.action_model.action_model_type,
            "LayerwiseFM_v4",
        )

    def test_action_positions_exchange_information(self):
        torch.manual_seed(0)
        model = _tiny_dit(num_layers=1)
        actions = torch.randn(1, 4, 16, requires_grad=True)
        memory = torch.randn(1, 5, 16)

        output = model(actions, memory, timestep=torch.tensor([3]), return_pre_output=True)
        gradient = torch.autograd.grad(output[0, 0].sum(), actions)[0]

        self.assertGreater(gradient[0, 1:].abs().sum().item(), 0.0)

    def test_additive_mask_hides_vlm_padding_tokens(self):
        torch.manual_seed(1)
        model = _tiny_dit(num_layers=1)
        actions = torch.randn(1, 4, 16)
        memory = torch.randn(1, 2, 16)
        changed_memory = memory.clone()
        changed_memory[:, 1] += 1_000.0
        additive_bias = torch.tensor([[[0.0, -10_000.0]]])

        first = model(
            actions,
            memory,
            timestep=torch.tensor([2]),
            encoder_attention_mask=additive_bias,
            return_pre_output=True,
        )
        second = model(
            actions,
            changed_memory,
            timestep=torch.tensor([2]),
            encoder_attention_mask=additive_bias,
            return_pre_output=True,
        )

        torch.testing.assert_close(first, second)

    def test_state_token_receives_action_loss_gradient(self):
        action = SimpleNamespace(
            action_dim=3,
            state_dim=5,
            action_horizon=4,
            num_inference_timesteps=2,
            num_target_vision_tokens=0,
            add_pos_embed=True,
            max_seq_len=16,
            noise_beta_alpha=1.5,
            noise_beta_beta=1.0,
            noise_s=0.999,
            num_timestep_buckets=100,
            diffusion_model_cfg={
                "num_layers": 2,
                "input_embedding_dim": 16,
                "cross_attention_dim": 16,
                "output_dim": 16,
                "num_attention_heads": 4,
                "attention_head_dim": 4,
                "dropout": 0.0,
                "final_dropout": False,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
        )
        config = SimpleNamespace(framework=SimpleNamespace(action_model=action))
        head = LayerwiseFlowmatchingActionHeadV4(config)
        memory = [torch.randn(2, 5, 16) for _ in range(2)]
        actions = torch.randn(2, 4, 3)
        state = torch.randn(2, 1, 5)
        attention_bias = torch.zeros(2, 1, 5)

        loss = head(memory, actions, state, encoder_attention_mask=attention_bias)
        loss.backward()

        total_state_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in head.state_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(total_state_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
