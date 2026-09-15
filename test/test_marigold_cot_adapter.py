"""Regressions for joint CoT augmentation at the Marigold adapter boundary."""

from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from starVLA.dataloader.cot_augmentation import augment_cot_sample
from starVLA.dataloader.marigold_data_datasets import (
    StarVLAMarigoldDataReader,
    _Q99Normalizer,
    _build_worker_cot_resolver,
    _stats_from_lerobot_meta,
    collate_fn,
)
from starVLA.model.framework.VLM4A.QwenGR00T_CoTrain import QwenGR00T_CoT
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenOFT_CoT import QwenOFT_CoT


class _IdentityNormalizer:
    def actions(self, value, _indices):
        return value

    def state(self, value, _indices):
        return value

    def unnormalize_actions(self, value, _indices):
        return np.asarray(value, dtype=np.float32)


class _Resolver:
    conversation = [
        {"from": "human", "value": "Do {instruction}."},
        {"from": "gpt", "value": "<point>(100,200)</point>"},
    ]

    def __init__(self):
        self.calls = []

    def resolve(self, trajectory_name, frame_index):
        self.calls.append((trajectory_name, frame_index))
        return self.conversation


class _CameraResolver(_Resolver):
    def resolve(self, trajectory_name, frame_index, *, camera_view=None):
        self.calls.append((trajectory_name, frame_index, camera_view))
        return self.conversation


class _FailingResolver:
    def resolve(self, *_args):
        raise AssertionError("worker-resolved CoT must not be resolved again in the model")


def _reader(
    *,
    train=True,
    augmentation="crop_photometric",
    cot_source="mapping",
    cot_dropout_enabled=True,
    include_state=True,
):
    reader = StarVLAMarigoldDataReader.__new__(StarVLAMarigoldDataReader)
    reader.data_cfg = {"marigold_cot_frame_index_key": "frame_index"}
    reader.train = train
    reader.include_state = include_state
    reader.action_representation = "joint_position_abs_gripper"
    reader.action_dim = 2
    reader.state_dim = 2
    reader.frame_index = -1
    reader.camera_groups = [["camera_external_left"], ["camera_wrist"]]
    reader.cot_camera_key_to_view = {
        "camera_external_left": "observation.images.left_external",
        "camera_external_right": "observation.images.right_external",
    }
    reader.augmentation = augmentation
    reader.cot_source = cot_source
    reader.cot_dropout_enabled = cot_dropout_enabled
    reader.cot_dropout_rate = 0.5
    reader.cot_resolver = _Resolver()
    reader.normalizer = _IdentityNormalizer()
    return reader


def _sample():
    return {
        "actions": torch.zeros(3, 24),
        "proprio": torch.zeros(1, 24),
        "trajectory_fields": [("arm", 0, 2)],
        "trajectory_mask": torch.ones(24, dtype=torch.bool),
        "images": {
            "camera_external_left": torch.full((1, 3, 12, 16), 32, dtype=torch.uint8),
            "camera_external_right": torch.full((1, 3, 12, 16), 48, dtype=torch.uint8),
            "camera_wrist": torch.full((1, 3, 12, 16), 64, dtype=torch.uint8),
        },
        "instruction": "pick up the object",
        "embodiment": "franka_robotiq_gripper",
        "uuid": "droid-episode-7",
        "frame_index": 19,
    }


class MarigoldCoTAdapterTest(TestCase):
    def test_q99_action_inverse_leaves_absolute_gripper_in_zero_one_space(self):
        normalizer = _Q99Normalizer.__new__(_Q99Normalizer)
        normalizer.enabled = True
        normalizer.action_stats = {
            "q01": [-2.0, 10.0, 0.0],
            "q99": [2.0, 20.0, 1.0],
        }
        normalized = np.asarray([[-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        physical = normalizer.unnormalize_actions(normalized, [0, 1, 2])

        np.testing.assert_allclose(
            physical,
            np.asarray([[-2.0, 20.0, 0.0], [0.0, 15.0, 1.0]], dtype=np.float32),
        )

    def test_open_loop_protocol_uses_final_nonempty_episode_and_chunk_stride(self):
        sampler = MagicMock()
        sampler.num_actions = 2
        sampler.action_frame_stride = 1
        sampler.num_episodes = 3
        sampler.filter_episode_indices.return_value = [0, 1, 2]
        sampler.get_episode_info.side_effect = lambda idx: {"episode": idx}
        sampler._build_instruction_inventory.side_effect = lambda info: ("inventory", info["episode"])
        sampler.plan_episode_samples.side_effect = lambda episode_idx, **_: [] if episode_idx == 2 else [10, 12]
        sampler.episode_uuid.side_effect = lambda info: f"episode-{info['episode']}"
        sampler._target_frame_offset_to_source_position.side_effect = lambda value: value
        sampler._round_source_position.side_effect = lambda value: value
        state = MagicMock()
        state.sampler = sampler
        state.mode = "local"
        reader = StarVLAMarigoldDataReader.__new__(StarVLAMarigoldDataReader)
        reader.train = False
        reader._get_open_loop_states = lambda: [("droid", state)]

        episodes = reader.build_open_loop_episodes(action_horizon=2)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["episode_idx"], 1)
        self.assertEqual(episodes[0]["anchors"], [10, 12])
        self.assertEqual(episodes[0]["anchor_stride_source_frames"], 2)

    def test_open_loop_materialization_preserves_source_frames_and_valid_mask(self):
        sampler = MagicMock()
        sampler.num_actions = 2
        sampler._output_target_source_offsets.return_value = [0, 2]
        sampler.sample_episode_at.return_value = {
            "actions": torch.zeros(2, 24),
            "trajectory_fields": [("eef", 0, 2)],
            "trajectory_mask": torch.ones(24, dtype=torch.bool),
            "action_valid_mask": torch.tensor([True, False]),
            "frame_index": 5,
        }
        state = MagicMock()
        state.sampler = sampler
        state.cfg = {"root": "/dataset"}
        reader = StarVLAMarigoldDataReader.__new__(StarVLAMarigoldDataReader)
        reader.action_dim = 2
        reader.normalizer = _IdentityNormalizer()
        reader._convert_sample = lambda _raw: {
            "action": np.asarray([[0.25, -0.5], [0.5, 0.75]], dtype=np.float32)
        }
        episode = {
            "state": state,
            "episode_idx": 7,
            "uuid": "episode-7",
            "inventory": object(),
            "anchors": [100],
        }

        samples = reader.materialize_open_loop_episode(episode, seed=9)

        self.assertEqual(len(samples), 1)
        np.testing.assert_array_equal(samples[0]["valid_mask"], [True, False])
        np.testing.assert_array_equal(samples[0]["source_frames"], [5, 7])
        np.testing.assert_allclose(samples[0]["gt_physical"], samples[0]["gt_normalized"])

    def test_delta_eef_stats_use_cartesian_velocity_and_absolute_gripper(self):
        payload = {
            "action.cartesian_velocity": {
                "q01": [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
                "q99": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "action.gripper_position": {"q01": [0.0], "q99": [1.0]},
            "observation.state.cartesian_position": {
                "q01": [0.1, 0.2, 0.3, -3.0, -1.0, -2.0],
                "q99": [0.9, 0.8, 0.7, 3.0, 1.0, 2.0],
            },
            "observation.state.gripper_position": {"q01": [0.0], "q99": [1.0]},
        }
        action_stats, state_stats = _stats_from_lerobot_meta(
            payload, "delta_eef_abs_gripper"
        )
        self.assertEqual(action_stats["q01"][:7], [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
        self.assertEqual(action_stats["q99"][:7], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0])
        self.assertEqual(state_stats["q01"][:7], [0.1, 0.2, 0.3, -3.0, -1.0, -2.0, 0.0])

    def test_relative_joint_stats_keep_gripper_absolute(self):
        payload = {
            "action.relative_joint_position": {
                "q01": [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
                "q99": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            },
            "action.gripper_position": {"q01": [0.0], "q99": [1.0]},
            "observation.state.joint_position": {
                "q01": [-2.0, -1.0, -2.0, -3.0, -2.0, 0.0, -3.0],
                "q99": [2.0, 1.0, 2.0, 0.0, 2.0, 4.0, 3.0],
            },
            "observation.state.gripper_position": {"q01": [0.0], "q99": [1.0]},
        }

        action_stats, state_stats = _stats_from_lerobot_meta(
            payload, "relative_joint_abs_gripper"
        )

        self.assertEqual(action_stats["q01"][:7], payload["action.relative_joint_position"]["q01"])
        self.assertEqual(action_stats["q01"][7], 0.0)
        self.assertEqual(action_stats["q99"][7], 1.0)
        self.assertEqual(state_stats["q99"][:7], payload["observation.state.joint_position"]["q99"])

    def test_relative_joint_actions_share_current_state_reference(self):
        reader = _reader(train=False, cot_source="none")
        reader.action_dim = 8
        reader.state_dim = 8
        reader.action_representation = "relative_joint_abs_gripper"
        sample = _sample()
        sample["trajectory_fields"] = [("arm", 0, 7), ("absolute_gripper", 7, 8)]
        sample["proprio"][0, :7] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        sample["proprio"][0, 7] = 0.25
        sample["actions"][:, :7] = sample["proprio"][0, :7] + torch.tensor(
            [[0.0], [0.1], [0.2]]
        )
        sample["actions"][:, 7] = torch.tensor([0.0, 1.0, 1.0])

        result = reader._convert_sample(sample)

        np.testing.assert_allclose(
            result["action"][:, :7],
            [[0.0] * 7, [0.1] * 7, [0.2] * 7],
            atol=1e-3,
        )
        np.testing.assert_allclose(result["action"][:, 7], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(
            result["state"][0, :7], sample["proprio"][0, :7], atol=1e-3
        )

    def test_include_state_false_omits_marigold_proprio(self):
        reader = _reader(include_state=False)
        result = reader._convert_sample(_sample())
        self.assertNotIn("state", result)

    def test_optional_gripper_inversion_applies_to_action_and_state(self):
        reader = _reader(train=False, cot_source="none")
        reader.invert_gripper = True
        sample = _sample()
        sample["trajectory_fields"] = [("arm", 0, 1), ("gripper", 22, 23)]
        sample["actions"][:, 0] = 0.25
        sample["actions"][:, 22] = 0.8
        sample["proprio"][:, 0] = 0.5
        sample["proprio"][:, 22] = 0.1

        result = reader._convert_sample(sample)

        np.testing.assert_allclose(result["action"][:, 0], 0.25)
        np.testing.assert_allclose(result["action"][:, 1], 0.2)
        np.testing.assert_allclose(result["state"][:, 0], 0.5)
        np.testing.assert_allclose(result["state"][:, 1], 0.9)

    def test_real_joint_transform_preserves_shape_and_rewrites_coordinates(self):
        torch.manual_seed(3)
        images = [
            Image.fromarray(np.full((40, 60, 3), value, dtype=np.uint8))
            for value in (80, 160)
        ]
        augmented, conversation = augment_cot_sample(
            images,
            _Resolver.conversation,
            mode="crop_photometric",
        )

        self.assertEqual([image.size for image in augmented], [(60, 40), (60, 40)])
        self.assertNotEqual(conversation[1]["value"], _Resolver.conversation[1]["value"])

    @patch("starVLA.dataloader.marigold_data_datasets.augment_cot_sample")
    def test_training_resolves_then_jointly_augments(self, augment):
        rewritten = [
            {"from": "human", "value": "Do {instruction}."},
            {"from": "gpt", "value": "<point>(120,220)</point>"},
        ]
        augment.side_effect = lambda images, conversation, mode: (images, rewritten)
        reader = _reader()

        result = reader._convert_sample(_sample())

        self.assertEqual(reader.cot_resolver.calls, [("droid-episode-7", 19)])
        augment.assert_called_once()
        self.assertEqual(augment.call_args.kwargs["mode"], "crop_photometric")
        self.assertIs(augment.call_args.args[1], _Resolver.conversation)
        self.assertEqual(result["cot_conversation"], rewritten)
        self.assertTrue(result["cot_available"])
        self.assertEqual(result["cot_mode"], "cot")
        self.assertTrue(result["_cot_dropout_enabled"])
        self.assertEqual(result["frame_index"], 19)
        self.assertEqual(len(result["image"]), 2)

    @patch("starVLA.dataloader.marigold_data_datasets.augment_cot_sample")
    @patch("starVLA.dataloader.marigold_data_datasets.random.choice")
    def test_selected_external_camera_drives_sparc_geometry_before_crop(
        self, choose, augment
    ):
        choose.side_effect = ["camera_external_right", "camera_wrist"]
        augment.side_effect = lambda images, conversation, mode: (images, conversation)
        reader = _reader(cot_source="sparc_sqlite")
        reader.camera_groups = [
            ["camera_external_left", "camera_external_right"],
            ["camera_wrist"],
        ]
        reader.cot_resolver = _CameraResolver()

        result = reader._convert_sample(_sample())

        self.assertEqual(
            reader.cot_resolver.calls,
            [("droid-episode-7", 19, "observation.images.right_external")],
        )
        augment.assert_called_once()
        self.assertIs(augment.call_args.args[1], _CameraResolver.conversation)
        self.assertEqual(len(result["image"]), 2)

    @patch("starVLA.dataloader.marigold_data_datasets.augment_cot_sample")
    def test_evaluation_resolves_without_augmentation_or_dropout(self, augment):
        reader = _reader(train=False)
        result = reader._convert_sample(_sample())

        augment.assert_not_called()
        self.assertEqual(result["cot_conversation"], _Resolver.conversation)
        self.assertFalse(result["_cot_dropout_enabled"])

    def test_mapping_requires_frame_metadata(self):
        reader = _reader()
        sample = _sample()
        del sample["frame_index"]
        with self.assertRaisesRegex(KeyError, "missing CoT frame key"):
            reader._convert_sample(sample)

    @patch("numpy.random.randint", return_value=1)
    @patch("numpy.random.random", return_value=0.0)
    def test_collator_repairs_an_all_dropped_batch(self, _random, _randint):
        batch = [
            {
                "cot_conversation": _Resolver.conversation,
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

    def test_dropout_disabled_retains_targets_but_keeps_graph_guard(self):
        reader = _reader(cot_dropout_enabled=False)
        samples = [reader._convert_sample(_sample()) for _ in range(2)]
        self.assertTrue(all(sample["_cot_dropout_enabled"] for sample in samples))
        self.assertTrue(all(sample["_cot_dropout_rate"] == 0.0 for sample in samples))

        result = collate_fn(samples)
        self.assertTrue(all(sample["cot_conversation"] is not None for sample in result))

    def test_all_unmapped_batch_uses_zero_loss_decoder_graph_guard(self):
        guard = [
            {"from": "human", "value": "Do {instruction}."},
            {"from": "gpt", "value": "Keep the decoder graph active."},
        ]
        batch = [
            {
                "cot_conversation": None,
                "cot_available": False,
                "cot_mode": "no_cot",
                "_cot_dropout_enabled": True,
                "_cot_dropout_rate": 0.5,
                "_cot_graph_guard_conversation": guard,
            }
            for _ in range(2)
        ]

        result = collate_fn(batch)

        self.assertIs(result[0]["cot_conversation"], guard)
        self.assertEqual(result[0]["cot_mode"], "cot")
        self.assertTrue(result[0]["cot_graph_guard"])
        self.assertFalse(result[0]["cot_available"])
        self.assertIsNone(result[1]["cot_conversation"])
        self.assertNotIn("_cot_graph_guard_conversation", result[0])

        loss = torch.tensor(3.0, requires_grad=True)
        guarded = Qwen_GR00T._apply_cot_graph_guard(
            loss, result, [sample["cot_conversation"] for sample in result]
        )
        guarded.backward()
        self.assertEqual(guarded.item(), 0.0)
        self.assertIsNotNone(loss.grad)
        self.assertEqual(loss.grad.item(), 0.0)

    @patch("starVLA.dataloader.marigold_data_datasets.SparcSQLiteCoTResolver")
    def test_marigold_forwards_sparc_score_gate(self, resolver_cls):
        resolver = MagicMock()
        resolver.human_prompt_template = None
        resolver_cls.return_value = resolver
        _build_worker_cot_resolver(
            {
                "source": "sparc_sqlite",
                "mapping_path": "/tmp/not-opened.sqlite",
                "min_selection_score": 0.94,
            },
            train=True,
        )
        self.assertEqual(
            resolver_cls.call_args.kwargs["min_selection_score"], 0.94
        )

    def test_cot_models_reject_mapping_batch_without_a_target(self):
        examples = [{
            "cot_conversation": None,
            "cot_available": False,
            "cot_mode": "no_cot",
        }]

        groot = QwenGR00T_CoT.__new__(QwenGR00T_CoT)
        groot.training = True
        groot.cot_source = "mapping"
        with self.assertRaisesRegex(RuntimeError, "without a local CoT target"):
            groot._resolve_training_cot(examples)

        oft = QwenOFT_CoT.__new__(QwenOFT_CoT)
        oft.training = True
        oft.cot_source = "mapping"
        with self.assertRaisesRegex(RuntimeError, "without a local CoT target"):
            oft._resolve_cot(examples)

    def test_cot_models_preserve_worker_rewritten_conversation(self):
        rewritten = [
            {"from": "human", "value": "Do {instruction}."},
            {"from": "gpt", "value": "<point>(120,220)</point>"},
        ]
        examples = [{
            "cot_conversation": rewritten,
            "cot_available": True,
            "trajectory_name": "droid-episode-7",
            "frame_index": 19,
        }]

        groot = QwenGR00T_CoT.__new__(QwenGR00T_CoT)
        groot.cot_resolver = _FailingResolver()
        conversations, coverage = groot._resolve_training_cot(examples)
        self.assertIs(conversations[0], rewritten)
        self.assertEqual(coverage, 1.0)

        oft = QwenOFT_CoT.__new__(QwenOFT_CoT)
        oft.cot_resolver = _FailingResolver()
        conversations, coverage = oft._resolve_cot(examples)
        self.assertIs(conversations[0], rewritten)
        self.assertEqual(coverage, 1.0)


if __name__ == "__main__":
    main()
