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

from __future__ import annotations

import collections
import platform
import struct

import numpy as np
import torch

from gofish.camera.pixel_chart import raster_center, torch_raster_centers

DEFAULT_DEVICE = torch.device("cuda")
SCENE_EXTENT_SCALE = 1.1
_CAMERA_SAMPLER_SEED: int | None = None
_CAMERA_SAMPLER_GENERATOR: torch.Generator | None = None


def configure_camera_sampler_seed(seed: int) -> None:
    """Seed the dedicated generator of the training-view shuffle."""
    global _CAMERA_SAMPLER_SEED, _CAMERA_SAMPLER_GENERATOR
    seed = int(seed)
    if seed < 0:
        raise ValueError("camera sampler seed must be non-negative")
    if _CAMERA_SAMPLER_SEED is not None and _CAMERA_SAMPLER_SEED != seed:
        raise RuntimeError("camera sampler seed changed within one process")
    _CAMERA_SAMPLER_SEED = seed
    _CAMERA_SAMPLER_GENERATOR = torch.Generator(device="cpu")
    _CAMERA_SAMPLER_GENERATOR.manual_seed(seed)


def create_pixel_coords(width: int, height: int, device: torch.device = None) -> torch.Tensor:
    """Generate pixel coordinates with +0.5 center offset for post-processing.

    Creates a grid of pixel coordinates where each coordinate represents the center
    of the pixel (hence the +0.5 offset).

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        device: Target device for the tensor. Defaults to None (CPU).

    Returns:
        Pixel coordinates tensor of shape [1, H, W, 2] containing (x, y) coordinates.
    """
    return torch_raster_centers(width, height, device=device, dtype=torch.float32)


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None, cx=None, cy=None):
    """
    return:
        ray_origin (sz_y, sz_x, 3)
        normalized ray_direction (sz_y, sz_x, 3)
    """

    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).numpy()
        jitter_xs = jitter[:, 0]
        jitter_ys = jitter[:, 1]
    else:
        jitter_xs = jitter_ys = None

    if cx is None:
        cx = 0.5 * w
    if cy is None:
        cy = 0.5 * h

    sample_x = raster_center(x) if jitter_xs is None else x + jitter_xs
    sample_y = raster_center(y) if jitter_ys is None else y + jitter_ys
    xs = (sample_x - cx) / f_x
    ys = (sample_y - cy) / f_y

    ray_lookat = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    ray_origin = np.zeros_like(ray_lookat)

    return ray_origin, ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)


def get_center_and_diag(cam_centers):
    """Return 3DGUT's median camera centre and median camera radius."""

    cam_centers = np.asarray(cam_centers)
    if cam_centers.ndim != 2 or cam_centers.shape[1] != 3 or cam_centers.shape[0] == 0:
        raise ValueError(f"Expected non-empty camera centres with shape (N, 3), got {cam_centers.shape}.")
    if (
        not np.issubdtype(cam_centers.dtype, np.number)
        or not np.isrealobj(cam_centers)
        or not np.all(np.isfinite(cam_centers))
    ):
        raise ValueError("Camera centres must be finite real numbers.")

    center = np.median(cam_centers, axis=0)
    diagonal = np.median(
        np.linalg.norm(cam_centers - center[None, :], axis=1)
    )
    return center, float(diagonal)


def get_3dgut_scene_extent(cam_centers) -> float:
    """Return ``1.1 * median ||c_i - median(c)||`` from training cameras."""

    _, diagonal = get_center_and_diag(cam_centers)
    return float(SCENE_EXTENT_SCALE * diagonal)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary.
    """
    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(image_size: np.ndarray, principal_point: np.ndarray) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""
    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_points3D_text(path):
    """
    Read points3D.txt file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    # Single file read
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # Convert directly to numpy arrays while appending
                xyzs.append([float(x) for x in elems[1:4]])
                rgbs.append([int(x) for x in elems[4:7]])
                errors.append(float(elems[7]))

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_intrinsics_text(path):
    """
    Read camera intrinsics from a COLMAP text file.
    Args:
        path: Path to the cameras.txt file
    Returns:
        Dict of Camera objects indexed by camera ID
    """
    cameras = {}
    with open(path, "r") as fid:
        # Skip comment lines at the start
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        for line in lines:
            # Unpack elements directly using split with maxsplit
            camera_id, model, width, height, *params = line.split()
            camera_id = int(camera_id)
            width, height = int(width), int(height)
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=np.array([float(p) for p in params]),
            )
    return cameras


def read_colmap_intrinsics_binary(path_to_model_file):
    """
    Read camera intrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the cameras.bin file
    Returns:
        Dict of Camera objects indexed by camera ID
    Raises:
        ValueError: If the number of cameras read doesn't match the expected count
        KeyError: If an invalid camera model ID is encountered
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # Read fixed-size camera properties
            camera_id, model_id, width, height = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            # Get camera model information
            try:
                camera_model = CAMERA_MODEL_IDS[model_id]
            except KeyError:
                raise KeyError(f"Invalid camera model ID: {model_id}")
            # Read camera parameters
            params = read_next_bytes(
                fid,
                num_bytes=8 * camera_model.num_params,
                format_char_sequence="d" * camera_model.num_params,
            )
            # Create camera object
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model.model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    # Verify camera count
    assert len(cameras) == num_cameras, f"Expected {num_cameras} cameras, but read {len(cameras)}"
    return cameras


def qvec_to_so3(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec_to_so3(self):
        return qvec_to_so3(self.qvec)


def read_colmap_extrinsics_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the images.bin file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If string parsing or data reading fails
    """
    images = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of registered images
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # Read image properties (id, rotation, translation, camera_id)
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id, *qvec_tvec, camera_id = props
            qvec = np.array(qvec_tvec[:4])
            tvec = np.array(qvec_tvec[4:7])
            # Read image name (null-terminated string)
            image_name = ""
            while True:
                current_char = read_next_bytes(fid, 1, "c")[0]
                if current_char == b"\x00":
                    break
                try:
                    image_name += current_char.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Invalid character in image name at position {len(image_name)}")
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            point_data = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            # Parse point data into coordinates and IDs
            xys = np.array([(point_data[i], point_data[i + 1]) for i in range(0, len(point_data), 3)])
            point3D_ids = np.array([int(point_data[i + 2]) for i in range(0, len(point_data), 3)])
            # Create image object
            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return sorted(images, key=lambda x: x.name)


def read_colmap_extrinsics_text(path):
    """
    Read camera extrinsics from a COLMAP text file.
    Args:
        path: Path to the images.txt file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If file format is invalid or data parsing fails
    """
    images = []
    with open(path, "r") as fid:
        # Skip comment lines and get valid lines
        lines = (line.strip() for line in fid)
        # An image without 2D points has only the line
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # followed by an empty POINTS2D[] line (normally (X, Y, POINT3D_ID) triples).
        lines = (line for line in lines if line == "" or not line.startswith("#"))
        # Process lines in pairs (image info + points info)
        try:
            while True:
                # Read image info line
                image_line = next(lines, None)
                if image_line is None:
                    break
                # Parse image properties
                elems = image_line.split()
                if len(elems) < 10:  # Minimum required elements
                    raise ValueError(f"Invalid image line format: {image_line}")
                image_id = int(elems[0])
                qvec = np.array([float(x) for x in elems[1:5]])
                tvec = np.array([float(x) for x in elems[5:8]])
                camera_id = int(elems[8])
                image_name = elems[9]
                # Read points line
                points_line = next(lines, None)
                if points_line is None:
                    raise ValueError(f"Missing points data for image {image_name}")
                # Parse 2D points and 3D point IDs
                point_elems = points_line.split()
                if len(point_elems) % 3 != 0:
                    raise ValueError(f"Invalid points format for image {image_name}")
                xys = np.array(
                    [(float(point_elems[i]), float(point_elems[i + 1])) for i in range(0, len(point_elems), 3)]
                )
                point3D_ids = np.array([int(point_elems[i + 2]) for i in range(0, len(point_elems), 3)])
                # Create image object
                images.append(
                    Image(
                        id=image_id,
                        qvec=qvec,
                        tvec=tvec,
                        camera_id=camera_id,
                        name=image_name,
                        xys=xys,
                        point3D_ids=point3D_ids,
                    )
                )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing extrinsics file: {e}")
    return sorted(images, key=lambda x: x.name)


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader multiprocessing.

    This function ensures that each worker process has a proper CUDA context
    and random number generator state, which is especially important on Windows.
    """
    import random

    import numpy as np
    import torch

    # Set random seeds for reproducibility
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Initialize CUDA context in worker process
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
        # Force CUDA context creation by doing a small operation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def configure_dataloader_for_platform(dataloader_kwargs: dict) -> dict:
    """
    Configure DataLoader kwargs for the current platform.

    Args:
        dataloader_kwargs: Dictionary of DataLoader arguments

    Returns:
        Updated DataLoader kwargs
    """
    kwargs = dataloader_kwargs.copy()
    if bool(kwargs.get("shuffle", False)):
        if _CAMERA_SAMPLER_GENERATOR is None:
            raise RuntimeError("configure_camera_sampler_seed() must be called first")
        if kwargs.get("generator") not in (None, _CAMERA_SAMPLER_GENERATOR):
            raise RuntimeError("training-view sampler has a conflicting generator")
        kwargs["generator"] = _CAMERA_SAMPLER_GENERATOR

    if "num_workers" in kwargs:
        original_num_workers = kwargs["num_workers"]
        kwargs["num_workers"] = original_num_workers

        # Adjust persistent_workers based on actual num_workers
        if "persistent_workers" in kwargs:
            kwargs["persistent_workers"] = kwargs["num_workers"] > 0

        # On Windows with multiprocessing, add worker initialization function
        if platform.system() == "Windows" and kwargs["num_workers"] > 0:
            kwargs["worker_init_fn"] = worker_init_fn

    return kwargs


def get_worker_id():
    """Get current worker ID for thread-local caching."""
    import threading

    # Get worker ID from current process/thread
    try:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            return f"worker_{worker_info.id}"
        else:
            return "main_process"
    except:
        return f"thread_{threading.get_ident()}"
