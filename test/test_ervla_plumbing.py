"""Focused regressions for ERVLA CoT dropout and DiT attention-mask plumbing."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.lerobot_datasets import collate_fn
from starVLA.dataloader.lerobot_datasets import _apply_trajectory_split
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.cot_augmentation import _rewrite_coordinate_text
from starVLA.dataloader.cot_resolver import extract_structured_cot_targets
from starVLA.model.framework.VLM4A.QwenGR00T import (
    ERVLAChoiceHead,
    Qwen_GR00T,
    StructuredEncoderRegressionHead,
)
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3, SharedZPooler
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.vlm.QWen3_EncDec import (
    DEFAULT_ENCODER_MLM_FIELDS,
    build_encoder_mlm_template,
    extract_encoder_mlm_fields,
)
from starVLA.training.train_starvla import VLATrainer
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class _Resolver:
    conversation = [
        {"from": "human", "value": "Do {instruction}."},
        {"from": "gpt", "value": "<movement>move left</movement>"},
    ]

    def resolve(self, trajectory_name, frame_index):
        return self.conversation


def _dataset(*, training=True, dropout_rate=0.5):
    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset._dataset_name = "libero_spatial"
    dataset._chunk_size = 1000
    dataset._cot_resolver = _Resolver()
    dataset._cot_source = "mapping"
    dataset._cot_dropout_enabled = True
    dataset._cot_dropout_rate = dropout_rate
    dataset._is_training_mode = lambda: training
    return dataset


class CoTDropoutTest(TestCase):
    def test_resolution_preserves_target_until_collation(self):
        data = _dataset()._attach_cot({}, trajectory_id=3, base_index=7)
        self.assertTrue(data["_cot_available"])
        self.assertIsNotNone(data["_cot_conversation"])
        self.assertEqual(data["_cot_mode"], "cot")


    @patch("numpy.random.randint", return_value=1)
    @patch("numpy.random.random", return_value=0.0)
    def test_collator_repairs_all_dropped_batch(self, _random, _randint):
        conversation = _Resolver.conversation
        batch = [
            {
                "cot_conversation": conversation,
                "cot_available": True,
                "cot_mode": "cot",
                "_cot_dropout_enabled": True,
                "_cot_dropout_rate": 0.5,
            }
            for _ in range(3)
        ]
        result = collate_fn(batch)
        self.assertEqual(sum(x["cot_conversation"] is not None for x in result), 1)
        self.assertEqual([x["cot_mode"] for x in result], ["no_cot", "cot", "no_cot"])
        self.assertNotIn("_cot_dropout_enabled", result[0])

    def test_collator_fails_fast_without_mapped_target(self):
        batch = [{
            "cot_conversation": None,
            "cot_available": False,
            "cot_mode": "no_cot",
            "_cot_dropout_enabled": True,
            "_cot_dropout_rate": 0.5,
        }]
        with self.assertRaisesRegex(RuntimeError, "no mapped targets"):
            collate_fn(batch)

    @patch("numpy.random.random", return_value=0.0)
    def test_eval_never_drops(self, _random):
        data = _dataset(training=False)._attach_cot({}, trajectory_id=3, base_index=7)
        self.assertIsNotNone(data["_cot_conversation"])
        self.assertEqual(data["_cot_mode"], "cot")


class EncoderMaskedReasoningTest(TestCase):
    def test_template_has_fixed_all_masked_field_slots(self):
        token = "<MASK>"
        template = build_encoder_mlm_template(token)
        self.assertEqual(
            template.count(token),
            sum(slots for _, slots in DEFAULT_ENCODER_MLM_FIELDS),
        )
        for name, _ in DEFAULT_ENCODER_MLM_FIELDS:
            self.assertIn(f"<{name}>", template)
            self.assertIn(f"</{name}>", template)

    def test_cam3d_values_are_extracted_without_structural_markup(self):
        text = (
            "<subtask>lift mug</subtask>\n"
            "<cam1><target>mug <point>(100,200)</point></target>"
            "<trajectory>(1,2) (3,4)</trajectory></cam1>\n"
            "<cam2><trajectory3d frame=\"cam2_t0\">(0,0,0) (1,2,3)"
            "</trajectory3d></cam2>\n"
            "<movement>move left 2 cm</movement>\n"
            "<reasoning>approach the handle</reasoning>"
        )
        fields = extract_encoder_mlm_fields(text)
        self.assertEqual(fields["subtask"], "lift mug")
        self.assertEqual(fields["cam1_target"], "mug <point>(100,200)</point>")
        self.assertEqual(fields["trajectory2d"], "(1,2) (3,4)")
        self.assertEqual(fields["trajectory3d"], "(0,0,0) (1,2,3)")
        self.assertEqual(fields["movement"], "move left 2 cm")
        self.assertEqual(fields["reasoning"], "approach the handle")


class AttentionMaskTest(TestCase):
    def test_valid_mask_becomes_additive_bias(self):
        valid = torch.tensor([[True, False, True], [False, True, True]])
        bias = Qwen_GR00T._dit_attention_bias(valid, torch.float32)
        self.assertEqual(tuple(bias.shape), (2, 1, 3))
        self.assertTrue(torch.equal(bias[0, 0], torch.tensor([0.0, -10_000.0, 0.0])))

    def test_pi_valid_mask_becomes_additive_bias(self):
        valid = torch.tensor([[True, False, True], [False, True, True]])
        bias = Qwen_PI_v3._dit_attention_bias(valid, torch.float32)
        self.assertEqual(tuple(bias.shape), (2, 1, 3))
        self.assertTrue(torch.equal(bias[1, 0], torch.tensor([-10_000.0, 0.0, 0.0])))

    @patch("torch.rand", return_value=torch.tensor([0.1, 0.9]))
    def test_pi_mlm_action_slot_dropout_restricts_only_sampled_rows(self, _rand):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.encoder_mlm_action_slot_dropout_rate = 0.15
        pi.is_inference = False
        pi.qwen_vl_interface = SimpleNamespace(
            _last_encoder_mlm_slot_mask=torch.tensor(
                [[False, True, False, True], [False, True, False, True]]
            )
        )
        pi.train()
        valid = torch.tensor([[True, True, True, True], [True, True, False, True]])

        selected = pi._action_encoder_valid_mask(valid)

        self.assertTrue(
            torch.equal(
                selected,
                torch.tensor(
                    [[False, True, False, True], [True, True, False, True]]
                ),
            )
        )
        self.assertEqual(pi._last_encoder_mlm_action_slot_only_rate, 0.5)

    def test_pi_mlm_action_slot_dropout_is_disabled_during_eval(self):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.encoder_mlm_action_slot_dropout_rate = 0.15
        pi.is_inference = False
        pi.qwen_vl_interface = SimpleNamespace(_last_encoder_mlm_slot_mask=None)
        pi.eval()
        valid = torch.tensor([[True, False, True]])

        self.assertTrue(torch.equal(pi._action_encoder_valid_mask(valid), valid))
        self.assertEqual(pi._last_encoder_mlm_action_slot_only_rate, 0.0)

    def test_pi_honors_action_model_diffusion_repeats(self):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.config = SimpleNamespace(
            framework=SimpleNamespace(
                action_model={"repeated_diffusion_steps": 8}
            )
        )
        self.assertEqual(pi._repeated_diffusion_steps(), 8)
        pi.config.framework.action_model["repeated_diffusion_steps"] = 0
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            pi._repeated_diffusion_steps()

    def test_layerwise_pi_constructor_omits_zero_sized_future_parameter(self):
        config = OmegaConf.create({
            "framework": {
                "action_model": {
                    "action_dim": 1,
                    "state_dim": 0,
                    "action_horizon": 2,
                    "num_inference_timesteps": 1,
                    "num_target_vision_tokens": 0,
                    "add_pos_embed": False,
                    "max_seq_len": 8,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "diffusion_model_cfg": {
                        "num_layers": 2,
                        "input_embedding_dim": 4,
                        "attention_head_dim": 4,
                        "num_attention_heads": 1,
                        "output_dim": 4,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "positional_embeddings": None,
                        "interleave_self_attention": True,
                        "cross_attention_dim": 4,
                    },
                }
            }
        })

        head = LayerwiseFlowmatchingActionHead(config)

        self.assertIsNone(head.future_tokens)
        self.assertNotIn("future_tokens.weight", head.state_dict())

    @patch("numpy.random.random", return_value=np.array([0.1, 0.9]))
    def test_pi_v3_state_dropout_omits_complete_state_suffix_per_example(self, _random):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.state_dropout_rate = 0.5
        instructions = ["first task", "second task"]
        states = [
            np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
            np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
        ]

        conditioned, keep_rate = pi._add_state_conditioning(instructions, states)

        self.assertEqual(conditioned[0], "first task")
        self.assertEqual(conditioned[1], "second task [STATE] 0 128 255 [ACTION]")
        self.assertEqual(keep_rate, 0.5)

    @patch("numpy.random.random")
    def test_pi_v3_state_dropout_is_disabled_during_inference(self, random_mock):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.state_dropout_rate = 1.0
        pi.eval()

        conditioned, keep_rate = pi._add_state_conditioning(
            ["task"], [np.array([[0.0]], dtype=np.float32)]
        )

        random_mock.assert_not_called()
        self.assertEqual(conditioned, ["task [STATE] 128 [ACTION]"])
        self.assertEqual(keep_rate, 1.0)

    def test_layerwise_pi_delegates_to_dit_and_supports_no_future_tokens(self):
        class ZeroActionEncoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, actions, timesteps):
                return torch.zeros(
                    *actions.shape[:2], self.hidden_size,
                    device=actions.device, dtype=actions.dtype,
                )

        class DummyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["hidden_states"]

        hidden_size = 4
        head = LayerwiseFlowmatchingActionHead.__new__(LayerwiseFlowmatchingActionHead)
        nn.Module.__init__(head)
        head.model = DummyDiT()
        head.action_encoder = ZeroActionEncoder(hidden_size)
        head.action_decoder = nn.Linear(hidden_size, 1)
        head.future_tokens = None
        head.layerwise_attention_layout = "alternating"
        head.state_encoder = None
        head.position_embedding = None
        head.action_horizon = 2
        head.action_dim = 1
        head.input_embedding_dim = hidden_size
        head.num_inference_timesteps = 1
        head.num_timestep_buckets = 1000
        head.beta_dist = torch.distributions.Beta(1.5, 1.0)
        head.config = SimpleNamespace(noise_s=0.999, add_pos_embed=False)

        vl_embs = [torch.randn(2, 3, hidden_size) for _ in range(2)]
        actions = torch.randn(2, 2, 1)
        mask = torch.tensor(
            [[[0.0, -10_000.0, 0.0]], [[-10_000.0, 0.0, 0.0]]]
        )
        head(vl_embs, actions, encoder_attention_mask=mask)
        head.predict_action(vl_embs, encoder_attention_mask=mask)

        self.assertEqual(len(head.model.calls), 2)
        for call in head.model.calls:
            self.assertIs(call["encoder_hidden_states"], vl_embs)
            self.assertIs(call["encoder_attention_mask"], mask)
            self.assertTrue(call["return_pre_output"])
            # With zero target tokens, only the two action positions remain.
            self.assertEqual(tuple(call["hidden_states"].shape), (2, 2, hidden_size))
            self.assertFalse(call["force_layerwise_all_cross"])
        self.assertNotIn("future_tokens.weight", head.state_dict())

    @patch("torch.randn")
    def test_layerwise_pi_can_return_differentiable_clean_action_estimate(self, randn):
        class IdentityActionEncoder(nn.Module):
            def forward(self, actions, timesteps):
                return actions

        class IdentityDiT(nn.Module):
            def forward(self, **kwargs):
                return kwargs["hidden_states"]

        head = LayerwiseFlowmatchingActionHead.__new__(LayerwiseFlowmatchingActionHead)
        nn.Module.__init__(head)
        head.model = IdentityDiT()
        head.action_encoder = IdentityActionEncoder()
        head.action_decoder = nn.Identity()
        head.future_tokens = None
        head.layerwise_attention_layout = "alternating"
        head.state_encoder = None
        head.position_embedding = None
        head.action_horizon = 2
        head.action_dim = 1
        head.num_timestep_buckets = 1000
        head.config = SimpleNamespace(add_pos_embed=False)
        head.sample_time = lambda batch_size, device, dtype: torch.full(
            (batch_size,), 0.25, device=device, dtype=dtype
        )
        randn.return_value = torch.zeros(1, 2, 1)
        target = torch.tensor([[[1.0], [2.0]]])

        loss, clean = head(
            [torch.zeros(1, 1, 1)], target, return_clean_actions=True
        )

        # Dummy prediction is x_t=0.25*action, so x_t+(1-t)*prediction.
        self.assertTrue(torch.allclose(clean, 0.4375 * target))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(clean.requires_grad is False)

    def test_pi_tied_dynamics_uses_cumulative_xyz_at_requested_horizons(self):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        pi.tied_dynamics_position_dims = 3
        pi.tied_dynamics_horizons = [2, 4]
        pi.tied_dynamics_beta = 1.0
        target = torch.zeros(1, 4, 7)
        predicted = target.clone()
        predicted[:, :, 0] = 1.0

        loss = pi._tied_dynamics_forward(predicted, target)

        # x errors are 2 and 4 at selected horizons; y/z are zero.
        expected = torch.tensor(((2.0 - 0.5) + (4.0 - 0.5)) / 6.0)
        self.assertTrue(torch.allclose(loss, expected))

    def test_pi_anchor_teacher_is_not_registered_or_checkpointed(self):
        pi = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(pi)
        teacher = nn.Linear(2, 2)
        object.__setattr__(pi, "_representation_anchor_teacher", teacher)

        self.assertNotIn("_representation_anchor_teacher", pi._modules)
        self.assertEqual(pi.state_dict(), {})

    def test_dit_interleaves_layerwise_cross_and_self_attention(self):
        class ZeroTimeEncoder(nn.Module):
            def forward(self, timesteps):
                return torch.zeros(timesteps.shape[0], 4, device=timesteps.device)

        class RecorderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_states = []
                self.encoder_masks = []

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=None,
            ):
                self.encoder_states.append(encoder_hidden_states)
                self.encoder_masks.append(encoder_attention_mask)
                return hidden_states

        dit = DiT.__new__(DiT)
        nn.Module.__init__(dit)
        dit._interleave_self_attention = True
        dit.timestep_encoder = ZeroTimeEncoder()
        dit.transformer_blocks = nn.ModuleList([RecorderBlock() for _ in range(4)])

        hidden = torch.randn(2, 2, 4)
        vl_embs = [torch.randn(2, 3, 4) for _ in range(4)]
        mask = torch.tensor([[[0.0, -10_000.0, 0.0]]] * 2)
        output = dit(
            hidden_states=hidden,
            encoder_hidden_states=vl_embs,
            timestep=torch.zeros(2, dtype=torch.long),
            encoder_attention_mask=mask,
            return_pre_output=True,
        )

        self.assertIs(output, hidden)
        for idx, block in enumerate(dit.transformer_blocks):
            if idx % 2:
                self.assertIsNone(block.encoder_states[0])
                self.assertIsNone(block.encoder_masks[0])
            else:
                self.assertIs(block.encoder_states[0], vl_embs[idx])
                self.assertIs(block.encoder_masks[0], mask)

    def test_dit_rejects_wrong_number_of_layerwise_states(self):
        class ZeroTimeEncoder(nn.Module):
            def forward(self, timesteps):
                return torch.zeros(timesteps.shape[0], 4, device=timesteps.device)

        dit = DiT.__new__(DiT)
        nn.Module.__init__(dit)
        dit._interleave_self_attention = True
        dit.timestep_encoder = ZeroTimeEncoder()
        dit.transformer_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

        with self.assertRaisesRegex(ValueError, "must match the number of DiT blocks"):
            dit(
                hidden_states=torch.randn(1, 2, 4),
                encoder_hidden_states=[torch.randn(1, 3, 4)],
                timestep=torch.zeros(1, dtype=torch.long),
                return_pre_output=True,
            )

    def test_dit_can_reproduce_legacy_layerwise_all_cross_checkpoints(self):
        class ZeroTimeEncoder(nn.Module):
            def forward(self, timesteps):
                return torch.zeros(timesteps.shape[0], 4, device=timesteps.device)

        class RecorderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_states = []

            def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
                self.encoder_states.append(encoder_hidden_states)
                return hidden_states

        dit = DiT.__new__(DiT)
        nn.Module.__init__(dit)
        dit._interleave_self_attention = True
        dit.timestep_encoder = ZeroTimeEncoder()
        dit.transformer_blocks = nn.ModuleList([RecorderBlock() for _ in range(4)])
        vl_embs = [torch.randn(1, 3, 4) for _ in range(4)]
        dit(
            hidden_states=torch.randn(1, 2, 4),
            encoder_hidden_states=vl_embs,
            timestep=torch.zeros(1, dtype=torch.long),
            return_pre_output=True,
            force_layerwise_all_cross=True,
        )
        for idx, block in enumerate(dit.transformer_blocks):
            self.assertIs(block.encoder_states[0], vl_embs[idx])


class ChoiceHeadTest(TestCase):
    def test_candidate_and_score_losses_mask_padded_timesteps(self):
        head = ERVLAChoiceHead(
            hidden_size=4, action_horizon=2, action_dim=1, num_choices=3
        )
        with torch.no_grad():
            head.action_head.weight.zero_()
            head.action_head.bias.zero_()
            head.score_head.weight.zero_()
            head.score_head.bias.zero_()
        query = torch.zeros(1, 3, 4, requires_grad=True)
        target = torch.tensor([[[1.0], [100.0]]])
        output = head(query, target, time_mask=torch.tensor([[True, False]]))
        self.assertAlmostEqual(output["choice_loss"].item(), 1.0, places=6)
        self.assertAlmostEqual(output["score_loss"].item(), 3.0, places=6)
        self.assertEqual(tuple(output["choice_winner_histogram"].shape), (3,))
        (output["choice_loss"] + output["score_loss"]).backward()
        self.assertIsNotNone(head.action_head.bias.grad)
        self.assertIsNotNone(head.score_head.bias.grad)


class TrajectorySplitCacheTest(TestCase):
    def test_holdout_split_refreshes_episode_index_cache(self):
        class DummyDataset:
            dataset_name = "dummy"

            def __init__(self):
                self._trajectory_ids = np.asarray([10, 20, 30])
                self._trajectory_lengths = np.asarray([2, 3, 4])
                self._trajectory_id_to_index = {10: 0, 20: 1, 30: 2}

            @property
            def trajectory_ids(self):
                return self._trajectory_ids

            @property
            def trajectory_lengths(self):
                return self._trajectory_lengths

            def _refresh_trajectory_index_cache(self):
                return LeRobotSingleDataset._refresh_trajectory_index_cache(self)

            def get_trajectory_index(self, trajectory_id):
                return LeRobotSingleDataset.get_trajectory_index(self, trajectory_id)

            def _get_all_steps_single_process(self):
                return [
                    (int(trajectory_id), step)
                    for trajectory_id, length in zip(
                        self._trajectory_ids, self._trajectory_lengths
                    )
                    for step in range(int(length))
                ]

            def _build_valid_base_indices_by_trajectory(self):
                return {
                    int(trajectory_id): list(range(int(length)))
                    for trajectory_id, length in zip(
                        self._trajectory_ids, self._trajectory_lengths
                    )
                }

        dataset = _apply_trajectory_split(
            DummyDataset(), split="eval", holdout_trajectories_per_dataset=1
        )
        self.assertEqual(dataset.trajectory_ids.tolist(), [30])
        self.assertEqual(dataset.trajectory_lengths.tolist(), [4])
        self.assertEqual(dataset.get_trajectory_index(30), 0)
        self.assertEqual(dataset.trajectory_lengths[dataset.get_trajectory_index(30)], 4)


class MultiViewCoordinateAugmentationTest(TestCase):
    def test_cam3d_annotation_parses_to_normalized_structured_targets(self):
        conversation = [{
            "from": "gpt",
            "value": (
                "<cam1><target>mug <box>(100,200,300,400)</box> "
                "<point>(250,350)</point></target>"
                "<trajectory>(1,2) (3,4) (5,6) (7,8) (9,10)</trajectory></cam1>"
                "<cam2><trajectory3d units=\"cm\">(0,0,0) (2,-4,6) "
                "(4,-8,12) (6,-12,18) (8,-16,20)</trajectory3d></cam2>"
            ),
        }]
        targets = extract_structured_cot_targets(conversation)
        self.assertEqual(targets["target_point"], [0.25, 0.35])
        self.assertEqual(targets["object_box"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(targets["trajectory2d"]), 10)
        self.assertEqual(len(targets["trajectory3d"]), 15)
        self.assertEqual(targets["trajectory3d"][-1], 1.0)
        self.assertEqual(targets["ground_visibility"], [1.0, 1.0])

    def test_shared_z_phase_and_visibility_targets(self):
        conversation = [{
            "from": "gpt",
            "value": (
                "<cam1><target>mug <point>(200,300)</point></target></cam1>"
                "<movement>move forward 2 cm, keep gripper closed</movement>"
            ),
        }]
        targets = extract_structured_cot_targets(conversation)
        self.assertEqual(targets["ground_visibility"], [1.0, 0.0])
        self.assertEqual(targets["phase"], 2)

    def test_shared_z_pooler_is_compact_and_backpropagates_to_encoder(self):
        pooler = SharedZPooler(
            input_dim=16, z_dim=8, num_queries=2, query_dim=8, num_heads=2
        )
        hidden = torch.randn(3, 5, 16, requires_grad=True)
        valid = torch.tensor([
            [True, True, True, True, True],
            [False, True, True, True, True],
            [False, False, True, True, True],
        ])
        z = pooler(hidden, valid)
        self.assertEqual(tuple(z.shape), (3, 8))
        z.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertGreater(hidden.grad.abs().sum().item(), 0.0)

    def test_structured_head_masks_padding_and_backpropagates(self):
        head = StructuredEncoderRegressionHead(8, 2)
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        valid = torch.tensor([[False, True, True, True], [True, True, False, False]])
        prediction = head(hidden, valid)
        self.assertEqual(tuple(prediction.shape), (2, 2))
        prediction.sum().backward()
        self.assertIsNotNone(hidden.grad)

    def test_all_view_indexed_coordinate_blocks_are_rewritten(self):
        text = (
            "<view>front_external</view>\n"
            "<object>x <box>(100,200,300,400)</box></object>\n"
            "<target>y <point>(500,600)</point></target>\n"
            "<view>wrist</view>\n"
            "<object>x <box>(200,300,400,500)</box></object>\n"
            "<target>y <point>(600,700)</point></target>\n"
            "<trajectory>front_external: (100,200) (300,400)</trajectory>"
        )
        rewritten = _rewrite_coordinate_text(
            text,
            left=12,
            top=8,
            crop_width=456,
            crop_height=456,
            image_width=480,
            image_height=480,
            angle_degrees=3,
        )
        self.assertEqual(rewritten.count("<box>"), 2)
        self.assertEqual(rewritten.count("<point>"), 2)
        self.assertNotIn("<box>(100,200,300,400)</box>", rewritten)
        self.assertNotIn("<box>(200,300,400,500)</box>", rewritten)
        self.assertNotIn("front_external: (100,200)", rewritten)

    def test_cam2_metric_trajectory_ignores_crop_but_rotates_xy_axes(self):
        text = (
            '<cam2><trajectory3d frame="cam2_t0" units="cm">'
            '(0,0,0) (10,20,-3)</trajectory3d></cam2>'
        )
        cropped = _rewrite_coordinate_text(
            text, left=12, top=8, crop_width=456, crop_height=456,
            image_width=480, image_height=480, angle_degrees=0)
        self.assertEqual(cropped, text)
        rotated = _rewrite_coordinate_text(
            text, left=0, top=0, crop_width=480, crop_height=480,
            image_width=480, image_height=480, angle_degrees=90)
        self.assertIn('(20,-10,-3)', rotated)
        self.assertIn('frame="cam2_t0" units="cm"', rotated)


class FrozenDecoderTest(TestCase):
    def test_optimizer_lr_override_skips_disabled_optional_module(self):
        model = nn.Module()
        model.encoder = nn.Linear(4, 4)
        model.optional_head = None
        cfg = OmegaConf.create({
            "trainer": {
                "freeze_modules": "",
                "learning_rate": {
                    "base": 1e-4,
                    "optional_head": 2e-4,
                },
            }
        })

        groups = build_param_lr_groups(model, cfg)
        optimized = {id(p) for group in groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in model.encoder.parameters()})

    def test_fixed_decoder_is_out_of_optimizer_but_passes_input_gradient(self):
        model = nn.Module()
        model.qwen_vl_interface = nn.Module()
        model.qwen_vl_interface.model = nn.Module()
        model.qwen_vl_interface.model.model = nn.Module()
        model.qwen_vl_interface.model.model.language_model = nn.Module()
        decoder = nn.Linear(4, 4, bias=False)
        model.qwen_vl_interface.model.model.language_model.layers = decoder
        model.encoder = nn.Linear(4, 4, bias=False)
        path = "qwen_vl_interface.model.model.language_model.layers"
        cfg = OmegaConf.create({
            "trainer": {
                "freeze_modules": path,
                "learning_rate": {"base": 1e-4},
            }
        })

        groups = build_param_lr_groups(model, cfg)
        optimized = {id(p) for group in groups for p in group["params"]}
        self.assertFalse(any(id(p) in optimized for p in decoder.parameters()))
        TrainerUtils.freeze_backbones(model, path)

        encoder_output = model.encoder(torch.ones(1, 4))
        decoder(encoder_output).sum().backward()
        self.assertIsNotNone(model.encoder.weight.grad)
        self.assertIsNone(decoder.weight.grad)


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self._last_encoder_attention_mask = torch.tensor(
            [[False, True, True], [True, True, True]]
        )
        self.seen_modes = None
        self._last_choice_hidden = None

    def build_qwenvl_inputs(self, **kwargs):
        self.seen_modes = kwargs["cot_modes"]
        return {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": self._last_encoder_attention_mask.long(),
        }

    def forward(self, **kwargs):
        hidden = self.anchor * torch.ones(2, 3, 4)
        self._last_choice_hidden = (
            self.anchor * torch.ones(2, 3, 4)
            if kwargs.get("_run_choice_queries", False)
            else None
        )
        return SimpleNamespace(hidden_states=(hidden,), loss=self.anchor * 2.0)


class _FakeAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_mask = None

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None):
        self.seen_mask = encoder_attention_mask
        return vl_embs.mean() + actions.mean() * 0.0


class FrameworkPlumbingTest(TestCase):
    def test_pi_structured_aux_uses_raw_captured_encoder_states(self):
        model = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(model)
        model.structured_aux_enabled = True
        model.structured_aux_specs = {"target_point": {"layer": 1, "dim": 2}}
        model.structured_aux_heads = nn.ModuleDict({
            "target_point": StructuredEncoderRegressionHead(4, 2)
        })
        raw_hidden = torch.randn(2, 3, 4, requires_grad=True)
        model.qwen_vl_interface = SimpleNamespace(
            _structured_layer_out={1: raw_hidden},
            _last_encoder_attention_mask=torch.tensor(
                [[False, True, True], [True, True, True]]
            ),
        )
        examples = [
            {"cot_structured_targets": {"target_point": [0.1, 0.2]}},
            {"cot_structured_targets": {"target_point": [0.3, 0.4]}},
        ]

        loss, metrics = model._structured_aux_forward(examples, [None, None])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["structured_aux/target_point_coverage"].item(), 1.0)
        loss.backward()
        self.assertIsNotNone(raw_hidden.grad)

    def test_pi_structured_aux_supports_normalized_per_target_weights(self):
        model = Qwen_PI_v3.__new__(Qwen_PI_v3)
        nn.Module.__init__(model)
        model.structured_aux_enabled = True
        model.structured_aux_specs = {
            "target_point": {"layer": 1, "dim": 2, "weight": 0.25},
            "object_box": {"layer": 1, "dim": 4, "weight": 0.75},
        }
        model.structured_aux_heads = nn.ModuleDict({
            "target_point": StructuredEncoderRegressionHead(4, 2),
            "object_box": StructuredEncoderRegressionHead(4, 4),
        })
        model.qwen_vl_interface = SimpleNamespace(
            _structured_layer_out={1: torch.randn(2, 3, 4, requires_grad=True)},
            _last_encoder_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        examples = [
            {"cot_structured_targets": {
                "target_point": [0.1, 0.2], "object_box": [0.1, 0.2, 0.3, 0.4]
            }},
            {"cot_structured_targets": {
                "target_point": [0.3, 0.4], "object_box": [0.2, 0.3, 0.4, 0.5]
            }},
        ]
        loss, metrics = model._structured_aux_forward(examples, [None, None])
        expected = (
            0.25 * metrics["structured_aux/target_point_loss"]
            + 0.75 * metrics["structured_aux/object_box_loss"]
        )
        self.assertTrue(torch.allclose(loss.detach(), expected))

    def test_modes_and_mask_reach_action_head(self):
        model = Qwen_GR00T.__new__(Qwen_GR00T)
        nn.Module.__init__(model)
        model.qwen_vl_interface = _FakeVLM()
        model.action_model = _FakeAction()
        model.readout_projector = None
        model.cot_resolver = None
        model.cot_dropout_enabled = True
        model.cot_dropout_rate = 0.5
        model.action_horizon = 2
        model.choice_head = None
        model.config = OmegaConf.create({
            "framework": {"action_model": {"repeated_diffusion_steps": 2}},
            "datasets": {"vla_data": {"augmentation": "none"}},
        })
        conversation = _Resolver.conversation
        examples = [
            {
                "image": [], "lang": "a", "action": np.zeros((2, 1), np.float32),
                "cot_conversation": conversation, "cot_available": True, "cot_mode": "cot",
            },
            {
                "image": [], "lang": "b", "action": np.zeros((2, 1), np.float32),
                "cot_conversation": None, "cot_available": True, "cot_mode": "no_cot",
            },
        ]

        result = model(examples)

        self.assertEqual(model.qwen_vl_interface.seen_modes, ["cot", "no_cot"])
        self.assertEqual(result["cot_coverage"], 1.0)
        self.assertEqual(result["cot_keep_rate"], 0.5)
        self.assertEqual(tuple(model.action_model.seen_mask.shape), (4, 1, 3))
        self.assertEqual(model.action_model.seen_mask[0, 0, 0].item(), -10_000.0)
        self.assertEqual(model.action_model.seen_mask[0, 0, 1].item(), 0.0)

    def test_choice_losses_are_returned_without_entering_dit_context(self):
        model = Qwen_GR00T.__new__(Qwen_GR00T)
        nn.Module.__init__(model)
        model.qwen_vl_interface = _FakeVLM()
        model.action_model = _FakeAction()
        model.readout_projector = None
        model.cot_resolver = None
        model.cot_dropout_enabled = True
        model.cot_dropout_rate = 0.5
        model.action_horizon = 2
        model.choice_head = ERVLAChoiceHead(4, 2, 1, 3)
        model.config = OmegaConf.create({
            "framework": {"action_model": {"repeated_diffusion_steps": 1}},
            "datasets": {"vla_data": {"augmentation": "none"}},
        })
        conversation = _Resolver.conversation
        examples = [
            {
                "image": [], "lang": str(i),
                "action": np.zeros((2, 1), np.float32),
                "action_time_mask": np.array([True, i == 0]),
                "cot_conversation": conversation if i == 0 else None,
                "cot_available": True,
                "cot_mode": "cot" if i == 0 else "no_cot",
            }
            for i in range(2)
        ]
        output = model(examples)
        self.assertIn("choice_loss", output)
        self.assertIn("score_loss", output)
        self.assertEqual(tuple(output["choice_winner_histogram"].shape), (3,))
        # DiT still received only the three semantic VLM positions.
        self.assertEqual(tuple(model.action_model.seen_mask.shape), (2, 1, 3))


class PreparedConfigTest(TestCase):
    def test_encoder_mlm_matrix_is_matched_and_decoderless(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        names = (
            "ervla_mlm_pi_encoder_cam3d.yaml",
            "ervla_mlm_pi_encoder_cam3d_rand.yaml",
            "ervla_mlm_pi_encoder_control.yaml",
        )
        configs = [yaml.safe_load((config_dir / name).read_text()) for name in names]
        for name, cfg in zip(names, configs):
            qwen = cfg["framework"]["qwenvl"]
            action = cfg["framework"]["action_model"]
            self.assertTrue(qwen["skip_decoder"], name)
            self.assertTrue(qwen["collect_encoder_layers"], name)
            self.assertTrue(qwen["encoder_mlm"]["enabled"], name)
            self.assertEqual(action["layerwise_attention_layout"], "alternating", name)
            self.assertEqual(action["num_target_vision_tokens"], 0, name)
            self.assertEqual(cfg["datasets"]["vla_data"]["per_device_batch_size"], 16)
            self.assertEqual(cfg["trainer"]["gradient_accumulation_steps"], 1)
            self.assertEqual(cfg["trainer"]["max_train_steps"], 20000)
            self.assertEqual(cfg["trainer"]["save_interval"], 10000)
            self.assertIn("qwen_vl_interface.model.lm_head", cfg["trainer"]["freeze_modules"])
        self.assertTrue(configs[0]["framework"]["qwenvl"]["encoder_mlm"]["loss_enabled"])
        self.assertTrue(configs[1]["framework"]["qwenvl"]["encoder_mlm"]["loss_enabled"])
        self.assertFalse(configs[2]["framework"]["qwenvl"]["encoder_mlm"]["loss_enabled"])
        self.assertIn("cot_d_cam3d_rand.jsonl", configs[1]["datasets"]["vla_data"]["cot"]["mapping_path"])

        correct = configs[0]
        self.assertEqual(
            correct["framework"]["qwenvl"]["encoder_mlm"]["action_slot_dropout_rate"],
            0.15,
        )
        self.assertFalse(correct["datasets"]["vla_data"]["cot"]["dropout_enabled"])
        self.assertEqual(correct["datasets"]["vla_data"]["cot"]["dropout_rate"], 0.0)
        diffusion = correct["framework"]["action_model"]["diffusion_model_cfg"]
        self.assertEqual(diffusion["dropout"], 0.0)
        self.assertFalse(diffusion["final_dropout"])

    def test_delayed_loss_scale_schedule_and_shared_z_decoder_config(self):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config = OmegaConf.create({
            "trainer": {"loss_scale": {
                "cot": {
                    "start": 0.0013,
                    "end": 0.025,
                    "start_step": 100,
                    "end_step": 500,
                }
            }}
        })
        expected = {
            0: 0.0013,
            100: 0.0013,
            300: 0.01315,
            500: 0.025,
            900: 0.025,
        }
        for step, value in expected.items():
            trainer.completed_steps = step
            self.assertAlmostEqual(
                trainer._scheduled_loss_scale("cot", 0.1), value, places=8
            )

        path = (
            Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
            / "ervla_zsupdec_pi_sharedz_ground_temporal.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        scales = cfg["trainer"]["loss_scale"]
        self.assertEqual(scales["structured_aux"]["start_step"], 100)
        self.assertEqual(scales["structured_aux"]["end_step"], 500)
        self.assertEqual(scales["structured_aux"]["end"], 0.05)
        self.assertEqual(scales["cot"]["start_step"], 100)
        self.assertEqual(scales["cot"]["end_step"], 500)
        self.assertEqual(scales["cot"]["end"], 0.025)

    def test_droid_pi_delta_eef_config_matches_pi05_like_head(self):
        path = (
            Path(__file__).parents[1]
            / "examples" / "DROID" / "train_files"
            / "train_droid_pi_delta_eef_nostate.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        framework = cfg["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        data = cfg["datasets"]["vla_data"]

        self.assertEqual(framework["state_dropout_rate"], 0.0)
        self.assertEqual(action["action_horizon"], 16)
        self.assertEqual(data["action_horizon"], 16)
        self.assertEqual(action["num_target_vision_tokens"], 0)
        self.assertEqual(action["repeated_diffusion_steps"], 1)
        self.assertEqual(action["num_inference_timesteps"], 10)
        self.assertTrue(action["add_pos_embed"])
        self.assertTrue(diffusion["interleave_self_attention"])
        self.assertEqual(diffusion["dropout"], 0.0)
        self.assertFalse(diffusion["final_dropout"])
        self.assertEqual(data["marigold_action_representation"], "delta_eef_abs_gripper")
        self.assertFalse(data["include_state"])
        self.assertEqual(data["obs_image_size"], [224, 224])
        self.assertEqual(cfg["trainer"]["eval_interval"], 5000)
        self.assertEqual(data["per_device_batch_size"], 32)

    def test_pi_no_latent_libero_configs_use_corrected_one_node_recipe(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        pairs = (
            ("ervla_k_pi_cam3d_cot05.yaml", "ervla_k_pi_cam3d_cot05_nolatent.yaml"),
            ("ervla_w_pi_encoder_actiononly.yaml", "ervla_w_pi_encoder_actiononly_nolatent.yaml"),
        )
        for baseline_name, no_latent_name in pairs:
            baseline = yaml.safe_load((config_dir / baseline_name).read_text())
            no_latent = yaml.safe_load((config_dir / no_latent_name).read_text())
            baseline.pop("run_id")
            no_latent.pop("run_id")
            baseline["framework"]["action_model"]["num_target_vision_tokens"] = 0
            baseline["framework"]["action_model"]["layerwise_attention_layout"] = (
                "alternating"
            )
            baseline["datasets"]["vla_data"]["per_device_batch_size"] = 16
            baseline["trainer"]["gradient_accumulation_steps"] = 1
            self.assertEqual(no_latent, baseline, no_latent_name)
            self.assertEqual(
                no_latent["framework"]["action_model"]["num_target_vision_tokens"], 0
            )
            self.assertEqual(
                no_latent["framework"]["action_model"]["layerwise_attention_layout"],
                "alternating",
            )
            self.assertEqual(
                no_latent["datasets"]["vla_data"]["per_device_batch_size"], 16
            )
            self.assertEqual(
                no_latent["trainer"]["gradient_accumulation_steps"], 1
            )

    def test_droid_pi_abs_joint_config_matches_requested_action_head(self):
        path = (
            Path(__file__).parents[1]
            / "examples" / "DROID" / "train_files"
            / "train_droid_pi_abs_joint_state_dropout05.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        framework = cfg["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        data = cfg["datasets"]["vla_data"]

        self.assertEqual(framework["state_dropout_rate"], 0.5)
        self.assertEqual(action["action_horizon"], 16)
        self.assertEqual(data["action_horizon"], 16)
        self.assertEqual(action["num_target_vision_tokens"], 0)
        self.assertEqual(action["repeated_diffusion_steps"], 1)
        self.assertEqual(action["num_inference_timesteps"], 10)
        self.assertTrue(action["add_pos_embed"])
        self.assertTrue(diffusion["interleave_self_attention"])
        self.assertEqual(diffusion["dropout"], 0.0)
        self.assertFalse(diffusion["final_dropout"])
        self.assertEqual(data["obs_image_size"], [224, 224])
        self.assertEqual(cfg["trainer"]["eval_interval"], 5000)
        self.assertEqual(data["per_device_batch_size"], 32)

    @patch("torch.autocast", return_value=nullcontext())
    def test_deepspeed_accumulation_includes_scaled_cot_loss(self, _autocast):
        class FakeModel:
            def forward(self, _batch):
                return {
                    "action_loss": torch.tensor(2.0, requires_grad=True),
                    "cot_loss": torch.tensor(4.0, requires_grad=True),
                    "cot_coverage": 0.5,
                    "cot_keep_rate": 0.75,
                }

        class FakeAccelerator:
            gradient_accumulation_steps = 2

            def __init__(self):
                self.backward_losses = []

            def backward(self, loss):
                self.backward_losses.append(float(loss.detach()))
                loss.backward()

        trainer = VLATrainer.__new__(VLATrainer)
        trainer.accelerator = FakeAccelerator()
        trainer.model = FakeModel()
        trainer.completed_steps = 1
        trainer.config = SimpleNamespace(
            use_deepspeed=True,
            trainer=SimpleNamespace(
                loss_scale=SimpleNamespace(cot=0.02, choice=1.0, score=1.0),
                logging_frequency=0,
                gradient_clipping=None,
            ),
        )

        metrics = trainer._train_step([{}])

        self.assertAlmostEqual(trainer.accelerator.backward_losses[0], 1.04, places=6)
        self.assertEqual(metrics["action_dit_loss"], 2.0)
        self.assertEqual(metrics["cot_loss"], 4.0)
        self.assertEqual(metrics["cot_coverage"], 0.5)
        self.assertEqual(metrics["cot_keep_rate"], 0.75)

    def test_droid_g_config_preserves_production_recipe_and_g_isolation(self):
        path = (
            Path(__file__).parents[1]
            / "examples" / "DROID" / "train_files" / "ervla_droid_g_sparc.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        qwen = cfg["framework"]["qwenvl"]
        data = cfg["datasets"]["vla_data"]
        trainer = cfg["trainer"]
        cot = data["cot"]

        self.assertTrue(qwen["enc_dec"])
        self.assertTrue(qwen["separate_cross_attention"])
        self.assertFalse(qwen["use_merged_attention"])
        self.assertFalse(qwen["skip_decoder"])
        self.assertEqual(data["data_mix"], "droid_lerobot_180x320_delta_eef")
        self.assertEqual(data["obs_image_size"], [180, 320])
        self.assertEqual(data["action_horizon"], 20)
        self.assertEqual(data["action_dim"], 7)
        self.assertEqual(data["state_dim"], 7)
        self.assertTrue(data["include_state"])
        self.assertEqual(data["per_device_batch_size"], 16)
        self.assertEqual(trainer["gradient_accumulation_steps"], 2)
        self.assertEqual(cot["source"], "sparc_sqlite")
        self.assertEqual(cot["min_selection_score"], 0.94)
        self.assertTrue(cot["immutable"])
        self.assertEqual(cot["dropout_rate"], 0.5)
        self.assertEqual(trainer["loss_scale"]["cot"], 0.02)

        frozen = set(trainer["freeze_modules"].split(","))
        self.assertIn("qwen_vl_interface.model.model.language_model.layers", frozen)
        self.assertIn(
            "qwen_vl_interface.model.model.language_model.decoder_embed_tokens", frozen
        )
        self.assertIn("qwen_vl_interface.model.lm_head", frozen)
        self.assertNotIn(
            "qwen_vl_interface.model.model.language_model.cross_attn_adapters", frozen
        )

    def test_ervla_configs_have_no_readout_and_reasoning_arms_use_dropout(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        names = (
            "ervla_a_causal.yaml",
            "ervla_b_bidir.yaml",
            "ervla_c_action.yaml",
            "ervla_crand.yaml",
            "ervla_d_ground.yaml",
            "ervla_d_ground_state.yaml",
            "ervla_e_staged.yaml",
            "ervla_f_decoder_frozen.yaml",
            "ervla_g_cross_only.yaml",
            "ervla_h_enc_choice_cam3d.yaml",
            "ervla_i_dec_choice_cam3d.yaml",
            "ervla_j_dec_choice_staged_cam3d.yaml",
        )
        reasoning = set(names[2:])
        for name in names:
            cfg = yaml.safe_load((config_dir / name).read_text())
            self.assertNotIn("readout_tokens", cfg["framework"], name)
            action_cfg = cfg["framework"]["action_model"]
            dit_cfg = action_cfg["diffusion_model_cfg"]
            self.assertEqual(action_cfg["hidden_size"], 1024, name)
            self.assertNotIn("action_hidden_dim", action_cfg, name)
            self.assertEqual(dit_cfg["num_layers"], 16, name)
            self.assertEqual(dit_cfg["output_dim"], 1024, name)
            self.assertEqual(dit_cfg["dropout"], 0.2, name)
            is_state_arm = name == "ervla_d_ground_state.yaml"
            self.assertEqual(action_cfg["state_dim"], 8 if is_state_arm else 0, name)
            self.assertEqual(cfg["datasets"]["vla_data"]["include_state"], is_state_arm, name)
            cot = cfg["datasets"]["vla_data"].get("cot")
            if name in reasoning:
                self.assertTrue(cot["dropout_enabled"], name)
                self.assertEqual(cot["dropout_rate"], 0.5, name)
            self.assertTrue(cfg["datasets"]["vla_data"]["drop_last"], name)
            if name == "ervla_f_decoder_frozen.yaml":
                self.assertEqual(
                    cfg["trainer"]["freeze_modules"],
                    "qwen_vl_interface.model.model.language_model.layers",
                )
            if name == "ervla_g_cross_only.yaml":
                self.assertTrue(cfg["framework"]["qwenvl"]["separate_cross_attention"])
                self.assertFalse(cfg["framework"]["qwenvl"]["use_merged_attention"])
                frozen = set(cfg["trainer"]["freeze_modules"].split(","))
                self.assertIn("qwen_vl_interface.model.model.language_model.layers", frozen)
                self.assertIn(
                    "qwen_vl_interface.model.model.language_model.decoder_embed_tokens", frozen)
                self.assertIn("qwen_vl_interface.model.lm_head", frozen)
                self.assertNotIn(
                    "qwen_vl_interface.model.model.language_model.cross_attn_adapters", frozen)
            if name in {
                "ervla_h_enc_choice_cam3d.yaml", "ervla_i_dec_choice_cam3d.yaml",
                "ervla_j_dec_choice_staged_cam3d.yaml",
            }:
                choice = cfg["framework"]["choice_policy"]
                expected_location = "encoder" if "_h_" in name else "decoder"
                self.assertTrue(choice["enabled"])
                self.assertEqual(choice["location"], expected_location)
                self.assertEqual(choice["num_action_queries"], 16)
                self.assertEqual(choice["num_choices"], 5)
                self.assertEqual(cfg["trainer"]["loss_scale"]["choice"], 0.1)
                self.assertEqual(cfg["trainer"]["loss_scale"]["score"], 0.002)
                self.assertIn("cot_d_cam3d.jsonl", cot["mapping_path"])
            if name == "ervla_j_dec_choice_staged_cam3d.yaml":
                self.assertEqual(cfg["datasets"]["vla_data"]["per_device_batch_size"], 8)
                self.assertEqual(cot["decoder_unfreeze_step"], 1000)
                self.assertTrue(cot["stage_private_decoder_io"])
                self.assertEqual(cfg["trainer"]["loss_scale"]["cot"], 1.0)
                self.assertEqual(cfg["trainer"]["freeze_modules"], "")

    def test_layer_split_and_overlap_action_configs_are_matched(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        names = (
            "ervla_l_split14_gr00t_action.yaml",
            "ervla_m_overlap18_gr00t_action.yaml",
            "ervla_n_overlap18_structaux_cot.yaml",
            "ervla_o_split14_pi_cot.yaml",
        )
        configs = [yaml.safe_load((config_dir / name).read_text()) for name in names]
        for cfg in configs:
            self.assertEqual(cfg["trainer"]["max_train_steps"], 20000)
            self.assertEqual(cfg["trainer"]["save_interval"], 10000)
            self.assertEqual(cfg["datasets"]["vla_data"]["per_device_batch_size"], 16)
        overlap = configs[1]["framework"]["qwenvl"]
        self.assertEqual(
            (overlap["n_encoder_layers"], overlap["n_decoder_layers"], overlap["n_overlap_layers"]),
            (18, 10, 2),
        )
        structured = configs[2]
        self.assertTrue(structured["framework"]["structured_aux"]["enabled"])
        self.assertTrue(structured["framework"]["qwenvl"]["freeze_decoder_blocks"])
        self.assertEqual(structured["trainer"]["loss_scale"]["structured_aux"], 0.1)
        pi = configs[3]
        self.assertEqual(pi["framework"]["name"], "QwenPI_v3")
        self.assertTrue(pi["framework"]["qwenvl"]["collect_encoder_layers"])

    def test_base_initialized_pi_split_sweep_is_matched(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        names = (
            "ervla_q_pi_split14_base_cot.yaml",
            "ervla_r_pi_split18_base_cot.yaml",
            "ervla_s_pi_overlap18_base_cot.yaml",
            "ervla_t_pi_overlap18_structaux_cot.yaml",
        )
        configs = [yaml.safe_load((config_dir / name).read_text()) for name in names]
        for name, cfg in zip(names, configs):
            qwen = cfg["framework"]["qwenvl"]
            self.assertEqual(cfg["framework"]["name"], "QwenPI_v3", name)
            self.assertNotIn("encdec_ckpt", qwen, name)
            self.assertTrue(qwen["collect_encoder_layers"], name)
            self.assertTrue(qwen["freeze_decoder_blocks"], name)
            self.assertEqual(cfg["datasets"]["vla_data"]["per_device_batch_size"], 8, name)
            self.assertEqual(cfg["trainer"]["max_train_steps"], 20000, name)
            self.assertEqual(cfg["trainer"]["save_interval"], 10000, name)
            self.assertEqual(cfg["trainer"]["loss_scale"]["cot"], 0.5, name)
        self.assertEqual(configs[0]["framework"]["qwenvl"]["n_encoder_layers"], 14)
        self.assertEqual(configs[1]["framework"]["qwenvl"]["n_encoder_layers"], 18)
        for cfg in configs[2:]:
            qwen = cfg["framework"]["qwenvl"]
            self.assertEqual(
                (qwen["n_encoder_layers"], qwen["n_decoder_layers"], qwen["n_overlap_layers"]),
                (18, 10, 2),
            )
        self.assertTrue(configs[3]["framework"]["structured_aux"]["enabled"])

    def test_k_structured_aux15_config_is_future_heavy_and_constant(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        cfg = yaml.safe_load((config_dir / "ervla_y_pi_k_structaux_cot.yaml").read_text())
        targets = cfg["framework"]["structured_aux"]["targets"]
        self.assertAlmostEqual(sum(spec["weight"] for spec in targets.values()), 1.0)
        self.assertAlmostEqual(
            targets["trajectory2d"]["weight"] + targets["trajectory3d"]["weight"], 0.95
        )
        self.assertEqual(cfg["trainer"]["loss_scale"]["structured_aux"], 15.0)

    def test_k_layerwise_decoder_pi_config_changes_only_decoder_memory_routing(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        baseline = yaml.safe_load((config_dir / "ervla_k_pi_cam3d_cot05.yaml").read_text())
        layerwise = yaml.safe_load(
            (config_dir / "ervla_u_pi_layerwise_decoder_cot.yaml").read_text()
        )
        self.assertEqual(layerwise["framework"]["name"], "QwenPI_v3")
        qwen = layerwise["framework"]["qwenvl"]
        self.assertTrue(qwen["collect_encoder_layers"])
        self.assertTrue(qwen["separate_cross_attention"])
        self.assertTrue(qwen["layerwise_decoder_cross_attention"])
        self.assertEqual(qwen["encdec_ckpt"], baseline["framework"]["qwenvl"]["encdec_ckpt"])
        self.assertEqual(layerwise["framework"]["action_model"], baseline["framework"]["action_model"])
        self.assertEqual(layerwise["datasets"], baseline["datasets"])
        baseline_trainer = dict(baseline["trainer"])
        layerwise_trainer = dict(layerwise["trainer"])
        self.assertEqual(layerwise_trainer, baseline_trainer)

    def test_k_staged_decoder_pi_config_preserves_k_objective(self):
        config_dir = Path(__file__).parents[1] / "examples" / "LIBERO" / "train_files"
        baseline = yaml.safe_load((config_dir / "ervla_k_pi_cam3d_cot05.yaml").read_text())
        staged = yaml.safe_load(
            (config_dir / "ervla_v_pi_decoder_staged_cot.yaml").read_text()
        )
        self.assertEqual(staged["framework"], baseline["framework"])
        baseline_data = baseline["datasets"]
        staged_data = staged["datasets"]
        for section in ("vlm_data",):
            self.assertEqual(staged_data[section], baseline_data[section])
        staged_cot = dict(staged_data["vla_data"]["cot"])
        self.assertEqual(staged_cot.pop("decoder_unfreeze_step"), 1000)
        self.assertTrue(staged_cot.pop("stage_private_decoder_io"))
        self.assertEqual(staged_cot, baseline_data["vla_data"]["cot"])
        staged_vla = dict(staged_data["vla_data"])
        baseline_vla = dict(baseline_data["vla_data"])
        staged_vla.pop("cot")
        baseline_vla.pop("cot")
        self.assertEqual(staged_vla, baseline_vla)
        self.assertEqual(staged["trainer"]["loss_scale"], baseline["trainer"]["loss_scale"])
        self.assertEqual(staged["trainer"]["freeze_modules"], "")
        for path in (
            "qwen_vl_interface.model.model.language_model.layers",
            "qwen_vl_interface.model.model.language_model.decoder_embed_tokens",
            "qwen_vl_interface.model.model.language_model.decoder_norm",
            "qwen_vl_interface.model.lm_head",
        ):
            self.assertEqual(staged["trainer"]["learning_rate"][path], 1.0e-6)


if __name__ == "__main__":
    main()
