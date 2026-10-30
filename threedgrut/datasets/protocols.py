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

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch


@dataclass
class Batch:
    rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
    rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
    T_to_world: torch.Tensor  # [B, 4, 4] transformation matrix from the ray space to the world space (START pose)
    T_to_world_end: Optional[torch.Tensor] = None  # [B, 4, 4] END pose for rolling shutter
    rays_in_world_space: bool = False  # True if rays are already in world space (no transform needed)
    rgb_gt: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None  # [B, H, W, 1] float learning-validity mask
    # Calibrated camera domain. physical_fov_mask is the camera's pixel
    # support; training_valid_mask additionally removes per-frame invalid
    # pixels (``<image>_mask.png``). A pinhole camera uses its full raster
    # without allocating a mask (physical_fov_full_raster).
    training_valid_mask: Optional[torch.Tensor] = None  # [B, H, W, 1] bool
    semantic_validity_applied: bool = False
    physical_fov_mask: Optional[torch.Tensor] = None  # [B, H, W, 1] bool
    physical_fov_full_raster: bool = False
    physical_fov_key: Optional[str] = None  # stable per-camera support identifier
    physical_fov_exterior_value: float = 0.0
    physical_fov_gradient_gate_installed: bool = False
    intrinsics: Optional[list] = None
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict] = None
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict] = None
    intrinsics_FThetaCameraModelParameters: Optional[dict] = None
    # Camera/frame indices for post-processing
    camera_idx: int = -1  # 0-based camera index
    frame_idx: int = -1  # 0-based frame index within the split
    camera_intrinsic_id: int = -1  # COLMAP camera id
    # Pixel coordinates for post-processing
    pixel_coords: Optional[torch.Tensor] = None  # [B, H, W, 2] (x, y) with +0.5 center offset
    # Exposure prior from EXIF metadata (mean-normalized log2 exposure [1], None if unavailable)
    exposure: Optional[torch.Tensor] = None

    def __post_init__(self):
        batch_size = self.T_to_world.shape[0]
        assert self.rays_ori.shape[0] == batch_size, "rays_ori must have the same batch size"
        assert self.rays_dir.shape[0] == batch_size, "rays_dir must have the same batch size"
        if self.rgb_gt is not None:
            assert self.rgb_gt.ndim == 4, "rgb_gt must be a 4D tensor [B, H, W, 3]"
            assert self.rgb_gt.shape[0] == batch_size, "rgb_gt must have the same batch size"
        if self.mask is not None:
            assert self.mask.ndim == 4, "mask must be a 4D tensor [B, H, W, 1]"
            assert self.mask.shape[0] == batch_size, "mask must have the same batch size"
        if self.physical_fov_mask is not None:
            assert self.physical_fov_mask.ndim == 4 and self.physical_fov_mask.shape[-1] == 1
            assert self.physical_fov_mask.dtype == torch.bool
        if self.training_valid_mask is not None:
            assert self.training_valid_mask.ndim == 4 and self.training_valid_mask.shape[-1] == 1
            assert self.training_valid_mask.dtype == torch.bool
        if self.physical_fov_full_raster:
            assert self.physical_fov_mask is None and self.training_valid_mask is not None
        if self.intrinsics:
            assert isinstance(self.intrinsics, list), "intrinsics must be a list"
            assert len(self.intrinsics) == 4, "intrinsics must have 4 elements [fx, fy, cx, cy]"
        if self.pixel_coords is not None:
            assert self.pixel_coords.ndim == 4, "pixel_coords must be a 4D tensor [B, H, W, 2]"
            assert self.pixel_coords.shape[0] == batch_size, "pixel_coords must have the same batch size"
            assert self.pixel_coords.shape[3] == 2, "pixel_coords last dimension must be 2 (x, y)"

    @property
    def physical_fov_enabled(self) -> bool:
        """True when the batch carries a calibrated camera domain."""
        return self.physical_fov_mask is not None or self.physical_fov_full_raster


class BoundedMultiViewDataset(Protocol):
    """Defines the basic functionality required from all datasets that can be used with the 3dgrut Trainer."""

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the bounding box of the scene as a tuple of vec3 (min,max)"""
        ...

    def get_scene_extent(self) -> float:
        """Robust scene extent of the training cameras."""
        ...

    def get_observer_points(self) -> np.ndarray:
        """Camera centres used to initialize Gaussian scales."""
        ...

    def get_poses(self) -> np.ndarray:
        """Camera-to-world matrices ``[N, 4, 4]`` in the right-down-front convention."""
        ...

    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Add the intrinsics to the batch and move data to GPU."""
        ...

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given train split frame index."""
        ...

    def get_frames_per_camera(self) -> list[int]:
        """Number of frames of each 0-based camera."""
        ...

    def __getitem__(self, index: int) -> dict: ...

    def __len__(self) -> int: ...
