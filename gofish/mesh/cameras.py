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

"""Training cameras of a checkpoint, as consumed by the opacity field.

The field is evaluated over the checkpoint's own training split, in training
order, with each camera's calibrated rays and its physical-domain mask. The
sensor/pose conversion below is the 3DGUT tracer's camera construction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

COLMAP_LIKE_TYPES = frozenset(("colmap", "scannetpp", "scannetpp_fisheye"))


@dataclass
class SensorPose3D:
    T_world_sensors: list  # two tquat [t, q] poses (start, end)
    timestamps_us: list


@dataclass
class TrainingCamera:
    frame_number: int  # index of the frame among all registered frames
    ray_origin: torch.Tensor  # [3] float32, CPU
    ray_direction: torch.Tensor  # [1, H, W, 3] float32, CPU
    physical_mask: torch.Tensor | None  # [H, W] uint8, CPU, or None
    sensor: object
    poses: SensorPose3D


def _so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
    """Rotation matrices [B,3,3] or [3,3] to unit quaternions (XYZW)."""
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)

    R = R.reshape((-1, 3, 3))
    num_rotations, D1, D2 = R.shape
    assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
    quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

    decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
    choices = decision_matrix.argmax(dim=1)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]

    if unbatch:
        quat = quat.squeeze()

    return quat


def _sensor_pose(R_start, T_start, R_end, T_end) -> SensorPose3D:
    T_world_sensor_t_start = torch.from_numpy(T_start).float()
    T_world_sensor_R_start = torch.from_numpy(R_start.transpose()).float()
    T_world_sensor_quat_start = _so3_matrix_to_quat(T_world_sensor_R_start)
    T_world_sensor_tquat_start = torch.hstack([T_world_sensor_t_start.cpu(), T_world_sensor_quat_start.cpu()])

    T_world_sensor_t_end = torch.from_numpy(T_end).float()
    T_world_sensor_R_end = torch.from_numpy(R_end.transpose()).float()
    T_world_sensor_quat_end = _so3_matrix_to_quat(T_world_sensor_R_end)
    T_world_sensor_tquat_end = torch.hstack([T_world_sensor_t_end.cpu(), T_world_sensor_quat_end.cpu()])

    return SensorPose3D(
        T_world_sensors=[T_world_sensor_tquat_start, T_world_sensor_tquat_end],
        timestamps_us=[0, 1],
    )


def _pose_to_world_to_camera(pose: torch.Tensor):
    C2W = np.concatenate((pose[:3, :4].cpu().detach().numpy(), np.zeros((1, 4))))
    C2W[3, 3] = 1.0
    W2C = np.linalg.inv(C2W)
    R = np.transpose(W2C[:3, :3])
    T = W2C[:3, 3]
    return R, T


def _camera_model_parameters(plugin, intrinsics_name: str, K: dict):
    shutter_type = getattr(plugin.ShutterType, K["shutter_type"])
    if intrinsics_name == "OpenCVPinholeCameraModelParameters":
        return plugin.fromOpenCVPinholeCameraModelParameters(
            resolution=K["resolution"],
            shutter_type=shutter_type,
            principal_point=K["principal_point"],
            focal_length=K["focal_length"],
            radial_coeffs=K["radial_coeffs"],
            tangential_coeffs=K["tangential_coeffs"],
            thin_prism_coeffs=K.get("thin_prism_coeffs", np.zeros((4,), dtype=np.float32)),
        )
    if intrinsics_name == "OpenCVFisheyeCameraModelParameters":
        return plugin.fromOpenCVFisheyeCameraModelParameters(
            resolution=K["resolution"],
            shutter_type=shutter_type,
            principal_point=K["principal_point"],
            focal_length=K["focal_length"],
            radial_coeffs=K["radial_coeffs"],
            max_angle=K["max_angle"],
        )
    raise ValueError(f"unsupported camera model for meshing: {intrinsics_name}")


def all_frames_config(checkpoint_conf, dataset_path: str | None = None):
    """Checkpoint config that loads every registered frame (in original order)."""
    conf = copy.deepcopy(checkpoint_conf)
    OmegaConf.set_struct(conf, False)
    with open_dict(conf):
        conf.num_workers = 0
        if dataset_path is not None:
            conf.path = str(dataset_path)
            conf.dataset.path = str(dataset_path)
        # No held-out list and a non-positive interval keep all frames in the "val" split.
        conf.dataset.test_split_interval = 0
        conf.dataset.test_list = None
    return conf


def _test_list(checkpoint_conf, dataset_path: str | None = None) -> str | None:
    """The checkpoint's held-out list, resolved against ``dataset_path`` when given."""
    conf = copy.deepcopy(checkpoint_conf)
    OmegaConf.set_struct(conf, False)
    with open_dict(conf):
        if dataset_path is not None:
            conf.path = str(dataset_path)
            conf.dataset.path = str(dataset_path)
    return OmegaConf.select(conf, "dataset.test_list", default=None)


def training_indices(names: list[str], interval: int, test_list: str | None = None) -> list[int]:
    """The trainer's split over all frames: with ``test_list`` the frames named in it are held out,
    otherwise frame i is held out iff interval > 0 and i % interval == 0."""
    if test_list:
        with open(test_list) as f:
            held_out = {line.strip() for line in f if line.strip()}
        selected = [index for index, name in enumerate(names) if name not in held_out]
    else:
        selected = [index for index in range(len(names)) if interval <= 0 or index % interval != 0]
    if not selected:
        raise RuntimeError("the checkpoint split has no training camera")
    return selected


def _physical_mask(dataset, intrinsic_id: int, height: int, width: int) -> torch.Tensor | None:
    """Physical-domain mask of one camera as uint8 [H, W], or None for the full raster.

    A calibrated fisheye support gives its mask. A pinhole camera declared on
    its full raster gives an all-ones mask, so the field runs the same masked
    code path for every declared domain.
    """
    support = dataset.get_camera_support(intrinsic_id)
    if support is not None:
        mask = support.torch_mask(device="cpu")[0, :, :, 0]
        if tuple(mask.shape) != (height, width):
            raise RuntimeError("physical mask raster differs from the camera raster")
        return mask.to(torch.uint8).contiguous()
    if int(intrinsic_id) in dataset.full_raster_cameras:
        return torch.ones((height, width), dtype=torch.uint8).contiguous()
    return None


def load_training_cameras(checkpoint_conf, plugin, dataset_path: str | None = None):
    """Return the checkpoint's training cameras in training order."""
    import threedgrut.datasets as datasets

    dataset_type = str(OmegaConf.select(checkpoint_conf, "dataset.type"))
    if dataset_type not in COLMAP_LIKE_TYPES:
        raise ValueError(f"meshing supports COLMAP-style datasets, got {dataset_type!r}")
    interval = int(OmegaConf.select(checkpoint_conf, "dataset.test_split_interval", default=8))
    test_list = _test_list(checkpoint_conf, dataset_path)
    conf = all_frames_config(checkpoint_conf, dataset_path)
    dataset = datasets.make_test(name=conf.dataset.type, config=conf)

    cameras = []
    masks: dict[int, torch.Tensor | None] = {}
    names = [extrinsic.name for extrinsic in dataset.cam_extrinsics]
    for index in training_indices(names, interval, test_list):
        intrinsic_id = int(dataset.get_intrinsics_idx(index))
        params, rays_origin, rays_direction, camera_name, _pixels = dataset.intrinsics[intrinsic_id]
        pose = torch.as_tensor(dataset.poses[index], dtype=torch.float32)
        R, T = _pose_to_world_to_camera(pose)
        sensor = _camera_model_parameters(plugin, camera_name, params)
        poses = _sensor_pose(R, T, R, T)

        origins = rays_origin.detach()
        origin = origins[0, 0, 0].float()
        if not torch.equal(origins, origin.view(1, 1, 1, 3).expand_as(origins)):
            raise RuntimeError("the opacity field requires central cameras")
        direction = rays_direction.detach().float().cpu().contiguous()
        height, width = int(direction.shape[1]), int(direction.shape[2])

        if intrinsic_id not in masks:
            masks[intrinsic_id] = _physical_mask(dataset, intrinsic_id, height, width)
        mask = masks[intrinsic_id]

        cameras.append(TrainingCamera(
            frame_number=index,
            ray_origin=origin.cpu().contiguous(),
            ray_direction=direction,
            physical_mask=mask,
            sensor=sensor,
            poses=poses,
        ))
    return cameras
