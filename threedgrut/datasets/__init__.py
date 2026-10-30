# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
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

import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from .dataset_colmap import ColmapDataset
from .dataset_scannetpp_fisheye import ScannetppFisheyeDataset
from .utils import read_colmap_extrinsics_binary, read_colmap_extrinsics_text


def _load_colmap_exif_exposures(dataset_path: str, downsample_factor: int) -> list[Optional[float]]:
    """Mean-normalized log2 EXIF exposures of all COLMAP images (shared by both splits)."""
    from threedgrut.utils.exif import load_exif_exposures

    try:
        cam_extrinsics = read_colmap_extrinsics_binary(os.path.join(dataset_path, "sparse/0", "images.bin"))
    except Exception:
        cam_extrinsics = read_colmap_extrinsics_text(os.path.join(dataset_path, "sparse/0", "images.txt"))

    downsample_suffix = "" if downsample_factor == 1 else f"_{downsample_factor}"
    images_folder = f"images{downsample_suffix}"
    image_paths = [Path(dataset_path) / images_folder / extr.name for extr in cam_extrinsics]
    return load_exif_exposures(image_paths)


def _dataset_kwargs(config) -> dict:
    return dict(
        downsample_factor=config.dataset.downsample_factor,
        test_split_interval=config.dataset.test_split_interval,
        test_list=config.dataset.get("test_list", None),
        calibrated_domain=bool(OmegaConf.select(config, "render.gofish.calibrated_domain", default=True)),
        camera_domains=list(config.dataset.get("camera_domains", []) or []),
    )


def make(name: str, config, ray_jitter=None):
    """Train and validation splits of one scene."""
    kwargs = _dataset_kwargs(config)
    match name:
        case "colmap":
            exif_exposures = (
                _load_colmap_exif_exposures(config.path, config.dataset.downsample_factor)
                if config.dataset.get("load_exif", True)
                else None
            )
            train_dataset = ColmapDataset(
                config.path, split="train", ray_jitter=ray_jitter, exif_exposures=exif_exposures, **kwargs
            )
            val_dataset = ColmapDataset(config.path, split="val", exif_exposures=exif_exposures, **kwargs)
        case "scannetpp_fisheye":
            train_dataset = ScannetppFisheyeDataset(config.path, split="train", ray_jitter=ray_jitter, **kwargs)
            val_dataset = ScannetppFisheyeDataset(config.path, split="val", **kwargs)
        case _:
            raise ValueError(f'Unsupported dataset type: {name}. Choose between: ["colmap", "scannetpp_fisheye"].')
    return train_dataset, val_dataset


def make_test(name: str, config):
    """Held-out split of one scene."""
    return make(name, config)[1]
