# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from threedgrut.datasets.dataset_colmap import ColmapDataset


def _write_tiny_transforms_scene(scene_dir: Path) -> None:
    image_dir = scene_dir / "images_4"
    image_dir.mkdir()

    Image.new("RGB", (2, 2), color=(16, 32, 48)).save(image_dir / "frame_00001.png")
    (scene_dir / "transforms.json").write_text(
        json.dumps(
            {
                "fl_x": 4.0,
                "fl_y": 4.0,
                "cx": 4.0,
                "cy": 4.0,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "w": 8,
                "h": 8,
                "frames": [
                    {
                        "file_path": "frame_00001.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        )
    )


class ColmapDatasetTest(unittest.TestCase):
    def test_integer_valued_float_downsample_factor_uses_integer_image_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            _write_tiny_transforms_scene(scene_dir)

            dataset = ColmapDataset(
                str(scene_dir),
                device="cpu",
                split="test",
                downsample_factor=4.0,
                test_split_interval=-1,
            )

            self.assertEqual(dataset.get_images_folder(), "images_4")
            self.assertIn(1, dataset.intrinsics)

            sample = dataset[0]
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(
                {
                    "data": sample["data"].unsqueeze(0),
                    "pose": sample["pose"].unsqueeze(0),
                    "intr": torch.tensor([sample["intr"]]),
                    "is_override": torch.tensor([sample["is_override"]]),
                }
            )
            self.assertEqual(gpu_batch.rays_ori.shape, (1, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
