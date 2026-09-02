"""Regression tests for deferred Accelerate preparation of Marigold loaders."""

import unittest

from torch.utils.data import DataLoader, IterableDataset

from starVLA.dataloader import DeferredMainProcessBatchLoader


class _Samples(IterableDataset):
    def __iter__(self):
        yield from range(7)


class _Accelerator:
    def __init__(self):
        self.seen = None

    def prepare(self, *components):
        self.seen = components
        return components[0] if len(components) == 1 else components


class MarigoldRoundRobinLoaderTest(unittest.TestCase):
    def test_singleton_inner_loader_is_prepared_before_main_process_batching(self):
        from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

        inner = DataLoader(_Samples(), batch_size=1, collate_fn=lambda batch: batch[0])
        deferred = DeferredMainProcessBatchLoader(
            inner, batch_size=3, collate_fn=list, drop_last=True
        )
        accelerator = _Accelerator()

        (prepared,) = TrainerUtils.setup_distributed_training(accelerator, deferred)

        self.assertIs(accelerator.seen[0], inner)
        self.assertEqual(list(prepared), [[0, 1, 2], [3, 4, 5]])


if __name__ == "__main__":
    unittest.main()
