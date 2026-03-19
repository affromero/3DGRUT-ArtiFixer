# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

from threedgrut.render import Renderer
import threedgrut.datasets as datasets
import json

if __name__ == "__main__":
    '''
    python render_trajectory.py \
        --checkpoint /path/to/checkpoint.pt \
        --path /path/to/data \
        --out-dir /path/to/output \
        --trajectory-file /path/to/trajectory.json
    Example:
    python render_trajectory.py \
        --checkpoint /lustre/fsw/portfolios/nvr/users/hturki/3dgrut-outputs/3dgut-mcmc-3view/bicycle/ours_30000/ckpt_30000.pt \
        --trajectory-file /lustre/fsw/portfolios/nvr/users/hturki/datasets/mipnerf360/bicycle/best_trajectory_3.json \
        --out-dir /lustre/fsw/portfolios/nvr/users/hturki/3dgrut-outputs/3dgut-mcmc-3view/bicycle/ours_30000/
    '''
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument(
        "--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt"
    )
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--scale-file", required=True, type=Path, help="Path to the scale info file")
    parser.add_argument("--training-pose-start", type=int, required=True, help="Index of the first training pose to include in the orbit trajectory")
    parser.add_argument("--interp-distance", type=float, default=0.1, help="Distance between consecutive frames on the orbit")
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
    )

    # dataset = datasets.make(name=renderer.conf.dataset.type, config=renderer.conf, ray_jitter=None)[0]
    # with (Path(args.trajectory_file).parent / f"train_poses_{Path(args.out_dir).name[-5]}.json").open("w") as f:
    #     json.dump(dataset.poses.tolist(), f, indent=4)

    # import sys; sys.exit(0)

    with args.scale_file.open() as f:
        scale = float(f.read().split("Scale factor: ")[1])

    print(f"Scale factor: {scale}")
    renderer.render_orbit_trajectory(args.interp_distance, scale, args.training_pose_start)
