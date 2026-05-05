import unittest

import numpy as np
import torch

from important_code.training.downsample_lerobot_dataset import (
    build_downsampled_frame,
    compute_stride,
    iter_downsampled_relative_indices,
    to_hwc_uint8,
)


class DownsampleLeRobotDatasetTest(unittest.TestCase):
    def test_compute_stride_requires_integer_downsample(self):
        self.assertEqual(compute_stride(30, 10), 3)

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            compute_stride(30, 12)

    def test_iter_downsampled_relative_indices_keeps_every_stride_frame(self):
        self.assertEqual(list(iter_downsampled_relative_indices(10, stride=3)), [0, 3, 6, 9])
        self.assertEqual(list(iter_downsampled_relative_indices(1, stride=3)), [0])

    def test_to_hwc_uint8_converts_decoded_lerobot_tensor(self):
        chw = torch.tensor(
            [
                [[0.0, 0.5], [1.0, 0.25]],
                [[0.1, 0.2], [0.3, 0.4]],
                [[1.0, 0.0], [0.5, 0.75]],
            ],
            dtype=torch.float32,
        )

        image = to_hwc_uint8(chw)

        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[0, 0], np.array([0, 26, 255], dtype=np.uint8))

    def test_build_downsampled_frame_removes_old_indices_and_keeps_task(self):
        item = {
            "action": torch.ones(7, dtype=torch.float32),
            "observation.state": torch.zeros(7, dtype=torch.float32),
            "observation.images.wrist": torch.zeros((3, 2, 2), dtype=torch.float32),
            "timestamp": torch.tensor(0.3),
            "frame_index": torch.tensor(9),
            "episode_index": torch.tensor(2),
            "index": torch.tensor(123),
            "task_index": torch.tensor(0),
            "task": "pick the grape",
        }
        features = {
            "action": {"dtype": "float32"},
            "observation.state": {"dtype": "float32"},
            "observation.images.wrist": {"dtype": "video"},
            "timestamp": {"dtype": "float32"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "task_index": {"dtype": "int64"},
        }

        frame = build_downsampled_frame(item, features)

        self.assertEqual(set(frame), {"action", "observation.state", "observation.images.wrist", "task"})
        self.assertEqual(frame["task"], "pick the grape")
        self.assertEqual(frame["action"].dtype, np.float32)
        self.assertEqual(frame["observation.images.wrist"].shape, (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
