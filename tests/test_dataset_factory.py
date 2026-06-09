# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from threedgrut import datasets


class MakeTestDatasetTest(unittest.TestCase):
    def test_colmap_test_dataset_uses_image_path_override(self):
        config = SimpleNamespace(
            path="/scene",
            dataset=SimpleNamespace(
                type="colmap",
                downsample_factor=4,
                test_split_interval=8,
            ),
            selected_indices_file="/tmp/selected.json",
            num_selected_indices=6,
            train_test_split_file=None,
            image_path_override="predictions",
        )

        with patch.object(datasets, "ColmapDataset", return_value="dataset") as dataset_cls:
            result = datasets.make_test("colmap", config)

        self.assertEqual(result, "dataset")
        dataset_cls.assert_called_once_with(
            "/scene",
            split="test",
            downsample_factor=4,
            test_split_interval=8,
            selected_indices_file="/tmp/selected.json",
            num_selected_indices=6,
            train_test_split_file=None,
            image_path_override="predictions",
        )


if __name__ == "__main__":
    unittest.main()
