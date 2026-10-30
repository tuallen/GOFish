# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .dataset_colmap import ColmapDataset
from .utils import read_colmap_extrinsics_text, read_colmap_intrinsics_text


class ScannetppFisheyeDataset(ColmapDataset):
    """ScanNet++ DSLR with its native OPENCV_FISHEYE (Kannala-Brandt) calibration.

    Unlike the stock ScanNet++ loader, the distortion coefficients are kept
    and the raw fisheye images are read from ``resized_images_<factor>``
    (or ``resized_images``); intrinsics are rescaled from the actual image
    size.
    """

    def get_images_folder(self):
        factor = int(self.downsample_factor)
        if factor > 1 and os.path.isdir(os.path.join(self.path, f"resized_images_{factor}")):
            return f"resized_images_{factor}"
        return "resized_images"

    def load_intrinsics_and_extrinsics(self):
        self.cam_extrinsics = read_colmap_extrinsics_text(os.path.join(self.path, "colmap", "images.txt"))
        self.cam_intrinsics = read_colmap_intrinsics_text(os.path.join(self.path, "colmap", "cameras.txt"))
