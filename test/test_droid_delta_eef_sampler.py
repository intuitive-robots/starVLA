"""Regressions for UUID-preserving DROID Cartesian action/state conversion."""

import math
import unittest

import torch

from marigold_data.samplers.droid.geometry import quaternion_wxyz_to_axis_angle


class DroidDeltaEEFSamplerTest(unittest.TestCase):
    def test_identity_delta_quaternion_maps_to_zero(self):
        actual = quaternion_wxyz_to_axis_angle(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(actual, torch.zeros(1, 3))

    def test_wxyz_quaternion_maps_to_axis_angle(self):
        half = math.pi / 4.0
        actual = quaternion_wxyz_to_axis_angle(
            torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
        )
        expected = torch.tensor([[0.0, 0.0, math.pi / 2.0]])
        torch.testing.assert_close(actual, expected)

if __name__ == "__main__":
    unittest.main()
