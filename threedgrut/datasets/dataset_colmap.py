# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from typing import Iterable, Optional

import ncore.sensors
import numpy as np
import torch
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from PIL import Image
from torch.utils.data import Dataset

from gofish.camera.domain import (
    CameraDomainTable,
    CameraSupport,
    aperture_support,
    branch_support,
    fisheye_domain,
    full_raster_support,
)
from gofish.camera.pixel_chart import resize_ratios, scale_camera_parameters
from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset
from .utils import (
    compute_max_radius,
    create_pixel_coords,
    get_3dgut_scene_extent,
    get_worker_id,
    pinhole_camera_rays,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
)


def _get_relative_paths(path_dir: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(path_dir):
        for filename in filenames:
            paths.append(os.path.relpath(os.path.join(dirpath, filename), path_dir))
    return paths


def _opencv_pinhole_intrinsics_from_colmap(
    intr_model: str,
    intr_params: np.ndarray,
    active_ratio_x: float,
    active_ratio_y: float,
):
    """Convert COLMAP distorted pinhole parameters to OpenCVPinholeCameraModelParameters fields."""
    intr_params = np.asarray(intr_params, dtype=np.float64)
    radial_coeffs = np.zeros((6,), dtype=np.float32)
    tangential_coeffs = np.zeros((2,), dtype=np.float32)
    thin_prism_coeffs = np.zeros((4,), dtype=np.float32)

    if intr_model not in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
        raise ValueError(f"Unsupported distorted pinhole camera model: {intr_model}")

    scaled = scale_camera_parameters(intr_model, intr_params, active_ratio_x, active_ratio_y)
    focal_length = np.asarray(scaled[:2], dtype=np.float32)
    principal_point = np.asarray(scaled[2:4], dtype=np.float32)

    if intr_model == "SIMPLE_RADIAL":
        radial_coeffs[0] = intr_params[3]
    elif intr_model == "RADIAL":
        radial_coeffs[:2] = intr_params[3:5]
    elif intr_model == "OPENCV":
        radial_coeffs[:2] = intr_params[4:6]
        tangential_coeffs[:] = intr_params[6:8]
    elif intr_model == "FULL_OPENCV":
        radial_coeffs[:] = intr_params[[4, 5, 8, 9, 10, 11]]
        tangential_coeffs[:] = intr_params[6:8]
    return focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs


class ColmapDataset(Dataset, BoundedMultiViewDataset):
    """COLMAP scene with an optional calibrated camera domain.

    With ``calibrated_domain`` enabled, every fisheye camera is restricted to
    its calibrated support (a measured aperture or an angular cap from
    ``camera_domains``; without a record, the raster pixels on the first
    monotone branch) and its rays are the first-branch inverse of the distortion
    polynomial at exact native pixel centres; pinhole cameras use their full
    raster. Otherwise the stock 3DGUT angular crop and ncore rays are used.
    """

    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        test_list=None,
        ray_jitter=None,
        exif_exposures: Optional[list[Optional[float]]] = None,
        calibrated_domain: bool = True,
        camera_domains: Iterable[str] = (),
    ):
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        self.test_split_interval = test_split_interval
        # Optional explicit held-out set: a text file of COLMAP image names, one
        # per line (FullCircle's tripod test frames).  It replaces the interval.
        self.test_names = None
        if test_list:
            with open(test_list) as f:
                self.test_names = {line.strip() for line in f if line.strip()}
        self._all_exif_exposures = exif_exposures  # Exposure values for all frames (pre-split)
        self.calibrated_domain = bool(calibrated_domain)
        self.camera_domain_table = CameraDomainTable(camera_domains)
        self.physical_fov_exterior_value = 0.0

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}
        self._worker_physical_fov_cache = {}

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        # GPU cache of processed camera intrinsics, per camera ID
        self.intrinsics = {}
        self.camera_supports: dict[int, CameraSupport] = {}  # fisheye cameras with a calibrated support
        self.full_raster_cameras: set[int] = set()  # pinhole cameras on their full raster
        self.camera_max_angles: dict[int, float] = {}  # first-branch angular limit per fisheye camera

        self.load_intrinsics_and_extrinsics()

        # Build mapping from COLMAP camera_id to 0-based contiguous index
        # This is needed for post-processing which expects 0-based camera indices
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        self._camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)}

        self.n_frames = len(self.cam_extrinsics)
        self.load_camera_data()

        # The scene extent is computed from the training cameras only, for
        # both splits; it drives the position learning rate, the clone/split
        # boundary and the world-size prune.
        all_indices = np.arange(self.n_frames)
        if self.test_names is not None:
            names = [extrinsic.name for extrinsic in self.cam_extrinsics]
            missing = self.test_names - set(names)
            if missing:
                raise ValueError(f"test_list names not in the COLMAP model: {sorted(missing)[:5]}")
            validation_indices = np.array([name in self.test_names for name in names], dtype=bool)
            training_indices = ~validation_indices
        elif self.test_split_interval > 0:
            training_indices = np.mod(all_indices, self.test_split_interval) != 0
            validation_indices = ~training_indices
        else:
            training_indices = np.ones(self.n_frames, dtype=bool)
            validation_indices = training_indices.copy()
        self.cameras_extent = get_3dgut_scene_extent(self.camera_centers[training_indices])

        holdout = self.test_names is not None or self.test_split_interval > 0
        indices = training_indices if (self.split == "train" or not holdout) else validation_indices

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in np.where(indices)[0]]
        self.poses = self.poses[indices].astype(np.float32)

        # numpy str array of image paths and mask paths
        self.image_paths = self.image_paths[indices]
        self.mask_paths = self.mask_paths[indices]

        self.camera_centers = self.camera_centers[indices]
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        if self._all_exif_exposures is not None:
            self.exif_exposures: Optional[list[Optional[float]]] = [
                self._all_exif_exposures[i] for i in np.where(indices)[0]
            ]
        else:
            self.exif_exposures = None

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # Clear existing worker caches to force recreation with new intrinsics
        self._worker_gpu_cache.clear()
        self._worker_physical_fov_cache.clear()

    def load_intrinsics_and_extrinsics(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"images{downsample_suffix}"

    def load_camera_data(self):
        """Store camera parameters on CPU; GPU tensors are created per worker."""

        self.camera_active_dimensions = {}

        image_dir = os.path.join(self.path, self.get_images_folder())
        precomputed_png_dir = image_dir + "_png"
        # Some preprocessed releases store the downsampled images as PNG while
        # COLMAP keeps the original names; map them by sorted relative path.
        colmap_to_image = None
        if not os.path.isdir(image_dir) and os.path.isdir(precomputed_png_dir):
            image_dir = precomputed_png_dir
            colmap_files = sorted(_get_relative_paths(os.path.join(self.path, "images")))
            image_files = sorted(_get_relative_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))

        def create_pinhole_camera(focalx, focaly, w, h, cx=None, cy=None):
            cx = cx if cx is not None else w / 2.0
            cy = cy if cy is not None else h / 2.0
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, focalx, focaly, w, h, self.ray_jitter, cx=cx, cy=cy)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_opencv_pinhole_camera(
            focal_length, principal_point, w, h, radial_coeffs, tangential_coeffs, thin_prism_coeffs
        ):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.asarray(principal_point, dtype=np.float32),
                focal_length=np.asarray(focal_length, dtype=np.float32),
                radial_coeffs=np.asarray(radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.asarray(tangential_coeffs, dtype=np.float32),
                thin_prism_coeffs=np.asarray(thin_prism_coeffs, dtype=np.float32),
            )
            camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = camera_model.pixels_to_image_points(int_pixel_coords)
            rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_fisheye_camera(params, w, h, domain=None, support=None):
            """Fisheye camera. ``domain`` selects the calibrated first-branch
            rays; otherwise ncore rays with the stock 3DGUT angular crop."""
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            resolution = np.array([w, h]).astype(np.uint64)
            principal_point = params[2:4].astype(np.float32)
            focal_length = params[0:2].astype(np.float32)
            radial_coeffs = params[4:].astype(np.float32)
            if domain is None:
                max_radius_pixels = compute_max_radius(resolution.astype(np.float64), principal_point)
                fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
                fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
                max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0
            else:
                max_angle = float(domain.max_angle_radians)

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                resolution=resolution,
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            if domain is None:
                camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
                image_points = camera_model.pixels_to_image_points(int_pixel_coords)
                rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            else:
                rays_d_cam, _ = domain.unproject_raster(int_pixel_coords, support.reshape(-1, 1).bool())
            rays_o_cam = torch.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        # One frame per intrinsic is consulted for the raster dimensions.
        cam_id_to_image_name = {}
        for extr in self.cam_extrinsics:
            cam_id_to_image_name.setdefault(extr.camera_id, extr.name)

        for intr in self.cam_intrinsics.values():
            image_name = cam_id_to_image_name[intr.id]
            if colmap_to_image is not None:
                image_path = os.path.join(image_dir, colmap_to_image[image_name])
            else:
                image_name = (
                    os.path.join(os.path.split(image_name)[1], "")
                    if self.get_images_folder() in image_name
                    else image_name
                )
                image_path = os.path.join(self.path, self.get_images_folder(), image_name)

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                logger.error(f"Image {image_path} not found. Cannot determine dimensions for intrinsic ID {intr.id}.")
                continue

            active_ratio_x, active_ratio_y = resize_ratios(intr.width, intr.height, width, height)
            self.camera_active_dimensions[int(intr.id)] = (int(width), int(height))

            if intr.model in ("SIMPLE_PINHOLE", "PINHOLE"):
                focal_length_x, focal_length_y, cx, cy = scale_camera_parameters(
                    intr.model, intr.params, active_ratio_x, active_ratio_y
                )
                self.intrinsics[intr.id] = create_pinhole_camera(
                    focal_length_x, focal_length_y, width, height, cx=cx, cy=cy
                )
                if self.calibrated_domain:
                    self.full_raster_cameras.add(int(intr.id))

            elif intr.model in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
                focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs = (
                    _opencv_pinhole_intrinsics_from_colmap(intr.model, intr.params, active_ratio_x, active_ratio_y)
                )
                self.intrinsics[intr.id] = create_opencv_pinhole_camera(
                    focal_length, principal_point, width, height, radial_coeffs, tangential_coeffs, thin_prism_coeffs
                )

            elif intr.model == "OPENCV_FISHEYE":
                params = np.asarray(
                    scale_camera_parameters(intr.model, intr.params, active_ratio_x, active_ratio_y),
                    dtype=np.float64,
                )
                if not self.calibrated_domain:
                    self.intrinsics[intr.id] = create_fisheye_camera(params, width, height)
                    continue
                record = self.camera_domain_table.find(intr.model, intr.width, intr.height, intr.params)
                if record is not None and record["kind"] == "aperture":
                    support = aperture_support(record, int(width), int(height))
                elif record is not None:
                    support = full_raster_support(width, height, record["max_angle"])
                else:
                    support = None
                domain = fisheye_domain(intr.params, intr.width, intr.height, width, height,
                                        max_angle_cap=None if support is None else support.max_angle_radians)
                if support is None:
                    # No record: the raster pixels on the first monotone branch.
                    support = branch_support(domain, int(width), int(height))
                support_mask = support.torch_mask(device="cpu")
                self.camera_supports[int(intr.id)] = support
                self.camera_max_angles[int(intr.id)] = domain.max_angle_radians
                self.intrinsics[intr.id] = create_fisheye_camera(params, width, height, domain, support_mask)

            else:
                assert False, (
                    f"Colmap camera model '{intr.model}' not handled: supported camera models are "
                    "PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, and OPENCV_FISHEYE."
                )

        # Load poses and paths
        self.poses = []
        self.image_paths = []
        self.mask_paths = []

        cam_centers = []
        for extr in logger.track(self.cam_extrinsics, description=f"Load Dataset ({self.split})", color="salmon1"):
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])

            if colmap_to_image is not None:
                image_path = os.path.join(image_dir, colmap_to_image[extr.name])
            else:
                image_path = os.path.join(self.path, self.get_images_folder(), extr.name)
            self.image_paths.append(image_path)
            self.mask_paths.append(os.path.splitext(image_path)[0] + "_mask.png")

        self.camera_centers = np.array(cam_centers)
        self.poses = np.stack(self.poses)
        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)

    def _lazy_worker_intrinsics_cache(self):
        """Create intrinsics cache for a specific worker."""
        worker_id = get_worker_id()

        if worker_id not in self._worker_gpu_cache:
            worker_intrinsics = {}
            worker_physical_fov = {}
            for intr_id, (params_dict, rays_ori, rays_dir, camera_name, pixel_coords) in self.intrinsics.items():
                worker_intrinsics[intr_id] = (
                    params_dict,
                    rays_ori.to(self.device, non_blocking=True),
                    rays_dir.to(self.device, non_blocking=True),
                    camera_name,
                    pixel_coords.to(self.device, non_blocking=True),
                )
                if intr_id in self.camera_supports:
                    worker_physical_fov[intr_id] = self.camera_supports[intr_id].torch_mask(
                        device=self.device).bool()
            self._worker_gpu_cache[worker_id] = worker_intrinsics
            self._worker_physical_fov_cache[worker_id] = worker_physical_fov

        return self._worker_gpu_cache[worker_id]

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)



    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_poses(self) -> np.ndarray:
        """Camera-to-world matrices (COLMAP is already right-down-front)."""
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        return self.cam_extrinsics[extr_idx].camera_id

    def get_camera_idx(self, frame_idx: int) -> int:
        """0-based camera index of a frame (COLMAP ids may be non-contiguous)."""
        return self._camera_id_to_idx[self.cam_extrinsics[frame_idx].camera_id]

    def get_frames_per_camera(self) -> list[int]:
        counts = [0] * len(self.cam_intrinsics)
        for extr in self.cam_extrinsics:
            counts[self._camera_id_to_idx[extr.camera_id]] += 1
        return counts


    def get_camera_support(self, intrinsic_id: int) -> Optional[CameraSupport]:
        """Calibrated support of a fisheye camera (None for pinhole/stock)."""
        return self.camera_supports.get(int(intrinsic_id))

    def __len__(self) -> int:
        return self.n_frames

    @torch.cuda.nvtx.range("colmap_dataset::_getitem")
    def __getitem__(self, idx) -> dict:
        image_data = np.asarray(Image.open(self.image_paths[idx]))
        actual_h, actual_w = image_data.shape[:2]

        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
            "camera_idx": self.get_camera_idx(idx),
            "frame_idx": idx,
        }

        # Optional per-frame validity mask
        if os.path.exists(mask_path := self.mask_paths[idx]):
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(1, actual_h, actual_w, 1)
            output_dict["mask"] = mask

        if self.exif_exposures is not None and self.exif_exposures[idx] is not None:
            output_dict["exposure"] = torch.tensor(self.exif_exposures[idx], dtype=torch.float32)

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        worker_intrinsics = self._lazy_worker_intrinsics_cache()
        camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = worker_intrinsics[intr]
        physical_fov_mask = self._worker_physical_fov_cache[get_worker_id()].get(intr)
        full_raster = int(intr) in self.full_raster_cameras

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
            "camera_idx": batch["camera_idx"][0].item(),
            "frame_idx": batch["frame_idx"][0].item(),
            "camera_intrinsic_id": int(intr),
            "pixel_coords": pixel_coords,
        }
        if physical_fov_mask is not None or full_raster:
            sample.update({
                "physical_fov_mask": physical_fov_mask,
                "physical_fov_full_raster": full_raster,
                "physical_fov_key": f"camera{int(intr)}:{data.shape[1]}x{data.shape[2]}",
                "physical_fov_exterior_value": self.physical_fov_exterior_value,
            })

        # Learning validity: camera support intersected with the frame mask.
        semantic_valid_mask = None
        if "mask" in batch:
            semantic_valid_mask = batch["mask"][0].to(self.device, non_blocking=True) > 127
        if physical_fov_mask is not None:
            training_valid_mask = physical_fov_mask.bool()
            if semantic_valid_mask is not None:
                training_valid_mask = training_valid_mask & semantic_valid_mask
        elif full_raster:
            training_valid_mask = (
                semantic_valid_mask
                if semantic_valid_mask is not None
                else torch.ones(data.shape[:-1] + (1,), dtype=torch.bool, device=data.device)
            )
        else:
            training_valid_mask = semantic_valid_mask
        if training_valid_mask is not None:
            training_valid_mask = training_valid_mask.contiguous()
            sample["training_valid_mask"] = training_valid_mask
            sample["semantic_validity_applied"] = semantic_valid_mask is not None
            sample["mask"] = training_valid_mask.to(torch.float32)

        if "exposure" in batch and batch["exposure"][0] is not None:
            sample["exposure"] = batch["exposure"].to(self.device)

        return Batch(**sample)
