"""Focused regressions for native Marigold open-loop trainer plumbing."""

from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from starVLA.training.train_starvla import VLATrainer


class _EvalAwareModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.saw_predict_eval = False
        self.saw_forward_eval = False

    def predict_action(self, *, examples, **_kwargs):
        self.saw_predict_eval = not self.training
        shape = np.asarray([example["action"] for example in examples]).shape
        return {"normalized_actions": np.zeros(shape, dtype=np.float32)}

    def forward(self, _examples):
        self.saw_forward_eval = not self.training
        return {}


class _Accelerator:
    is_main_process = False

    @staticmethod
    def unwrap_model(model):
        return model


class _ConfigDict(dict):
    __getattr__ = dict.__getitem__


class _IdentityNormalizer:
    @staticmethod
    def unnormalize_actions(actions, _indices):
        return np.asarray(actions, dtype=np.float32)


class _OpenLoopDataset:
    normalizer = _IdentityNormalizer()

    @staticmethod
    def build_open_loop_episodes(**_kwargs):
        return [{
            "dataset_name": "droid",
            "episode_idx": 9,
            "uuid": "episode-9",
        }]

    @staticmethod
    def materialize_open_loop_episode(_episode, *, seed):
        del seed
        action = np.asarray([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
        return [{
            "example": {"action": action},
            "gt_normalized": action,
            "valid_mask": np.asarray([True, True]),
            "source_frames": np.asarray([4, 5]),
            "action_indices": [0, 1],
        }]


class _PredictModel(torch.nn.Module):
    def predict_action(self, *, examples, **_kwargs):
        return {
            "normalized_actions": np.stack(
                [np.asarray(example["action"]) + 0.1 for example in examples]
            )
        }


class MarigoldOpenLoopTrainerTest(TestCase):
    @patch("starVLA.training.train_starvla.dist.barrier")
    def test_action_eval_disables_dropout_and_restores_training_mode(self, _barrier):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.model = _EvalAwareModel().train()
        trainer.accelerator = _Accelerator()
        trainer._get_next_batch = lambda: [
            {"action": np.zeros((2, 3), dtype=np.float32)}
        ]

        trainer.eval_action_model({})

        self.assertTrue(trainer.model.saw_predict_eval)
        self.assertTrue(trainer.model.saw_forward_eval)
        self.assertTrue(trainer.model.training)

    def test_flash_attention_does_not_disable_supported_open_loop_eval(self):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(open_loop_eval=True),
            framework=SimpleNamespace(
                qwenvl=SimpleNamespace(attn_implementation="flash_attention_3")
            ),
        )

        self.assertFalse(trainer._should_skip_open_loop_eval())

    @patch("starVLA.training.train_starvla.wandb.log")
    @patch("starVLA.training.train_starvla.wandb.Image", side_effect=lambda path: path)
    def test_native_marigold_path_writes_normalized_and_physical_plots(
        self, _image, _log
    ):
        with TemporaryDirectory() as tmpdir:
            trainer = VLATrainer.__new__(VLATrainer)
            trainer.model = _PredictModel().eval()
            trainer.accelerator = _Accelerator()
            trainer.vla_eval_dataset = _OpenLoopDataset()
            trainer.completed_steps = 123
            trainer.config = SimpleNamespace(
                seed=42,
                output_dir=tmpdir,
                trainer=_ConfigDict(
                    open_loop_eval=True,
                    open_loop_num_episodes_per_subdataset=1,
                    open_loop_batch_size=8,
                    open_loop_max_anchors=None,
                ),
                framework=SimpleNamespace(
                    action_model=SimpleNamespace(action_horizon=2),
                ),
            )

            metrics = trainer._eval_open_loop_trajectories()

            self.assertAlmostEqual(metrics["open_loop/mse_mean"], 0.01, places=6)
            eval_dir = Path(tmpdir) / "open_loop_eval" / "step_123"
            self.assertEqual(len(list(eval_dir.glob("*_normalized.png"))), 1)
            self.assertEqual(len(list(eval_dir.glob("*_physical.png"))), 1)
            self.assertEqual(len(list(eval_dir.glob("*.npz"))), 1)


if __name__ == "__main__":
    main()
