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
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
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

    def test_layerwise_pi_forwards_mask_to_every_block(self):
        class RecorderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_masks = []

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=None,
            ):
                self.seen_masks.append(encoder_attention_mask)
                return hidden_states

        class ZeroActionEncoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, actions, timesteps):
                return torch.zeros(
                    *actions.shape[:2], self.hidden_size,
                    device=actions.device, dtype=actions.dtype,
                )

        class ZeroTimeEncoder(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, timesteps):
                return torch.zeros(
                    timesteps.shape[0], self.hidden_size,
                    device=timesteps.device, dtype=torch.float32,
                )

        class DummyDiT(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([RecorderBlock(), RecorderBlock()])
                self.timestep_encoder = ZeroTimeEncoder(hidden_size)

        hidden_size = 4
        head = LayerwiseFlowmatchingActionHead.__new__(LayerwiseFlowmatchingActionHead)
        nn.Module.__init__(head)
        head.model = DummyDiT(hidden_size)
        head.action_encoder = ZeroActionEncoder(hidden_size)
        head.action_decoder = nn.Linear(hidden_size, 1)
        head.future_tokens = nn.Embedding(1, hidden_size)
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
        for block in head.model.transformer_blocks:
            self.assertIs(block.seen_masks[-1], mask)

        head.predict_action(vl_embs, encoder_attention_mask=mask)
        for block in head.model.transformer_blocks:
            self.assertIs(block.seen_masks[-1], mask)


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


if __name__ == "__main__":
    main()
