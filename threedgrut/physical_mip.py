# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Calibrated Mip Filtering, view-independent part: the 3D filter.

The per-view part, the pixel footprint added for each render, is in
``threedgut_tracer.view_observation_filter``.

Each Gaussian receives an isotropic 3-D low-pass variance ``filter_3D^2``
derived from the finest sampling rate at which any training camera observes
its centre: for every training frame whose calibrated support contains the
projected centre, the rate is the smallest singular value of the 2x3 pixel
Jacobian, and ``filter_3D = sqrt(mip_filter_variance) / max_rate``. Gaussians
seen by no training camera receive the largest observed filter. ``filter_dir``
is the unit direction from the owning camera to the Gaussian.

The search runs in float64 in a CUDA extension; ties between frames are
broken by a deterministic hash order of the frame records.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import torch

from gofish.camera.pixel_chart import resize_ratios, scale_camera_parameters


HERE = Path(__file__).resolve().parent
MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "OPENCV_FISHEYE": 2,
    "SIMPLE_RADIAL": 3,
    "RADIAL": 4,
    "OPENCV": 5,
    "FULL_OPENCV": 6,
}
PACKED_PARAMETER_COUNT = 12
_EXTENSION = None


def configured(conf: Any) -> bool:
    """The 3D filter is part of Calibrated Mip Filtering."""
    from omegaconf import OmegaConf

    return bool(OmegaConf.select(conf, "render.gofish.scene_sensor_factorization", default=True))


def filter_variance(conf: Any) -> float:
    from omegaconf import OmegaConf

    return float(OmegaConf.select(conf, "render.gofish.mip_filter_variance", default=0.2))


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        csrc = HERE / "physical_mip_csrc"
        _EXTENSION = load(
            name="gofish_physical_mip",
            sources=[str(csrc / "bindings.cpp"), str(csrc / "fused_refresh_cuda.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--prec-div=true", "--prec-sqrt=true", "--fmad=false"],
            verbose=verbose,
        )
    return _EXTENSION


@dataclass(frozen=True)
class CameraTable:
    """Device-side training-camera table consumed by the CUDA search."""

    rotations: torch.Tensor  # [F, 3, 3] float64 camera-to-world rotation
    centers: torch.Tensor  # [F, 3] float64 camera centre
    frame_camera: torch.Tensor  # [F] int32 camera slot
    frame_index: torch.Tensor  # [F] int32 training frame index
    frame_rank: torch.Tensor  # [F] int64 tie-break rank
    camera_model: torch.Tensor  # [C] int32
    camera_width: torch.Tensor  # [C] int32
    camera_height: torch.Tensor  # [C] int32
    camera_params: torch.Tensor  # [C, 12] float64 active-chart parameters
    camera_max_angle: torch.Tensor  # [C] float64, -1 for pinhole
    row_lo: torch.Tensor  # [C, Hmax] int32 support row runs [lo, hi)
    row_hi: torch.Tensor

    def arguments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.rotations, self.centers, self.frame_camera, self.frame_index, self.frame_rank,
            self.camera_model, self.camera_width, self.camera_height, self.camera_params,
            self.camera_max_angle, self.row_lo, self.row_hi,
        )


def _tie_break_key(frame_id: str, camera_id: int, source_index: int, relative_path: str, pose_values) -> str:
    pose_sha256 = hashlib.sha256(struct.pack("<16f", *pose_values)).hexdigest()
    return _canonical_json_sha256({
        "frame_id": frame_id,
        "camera_id": int(camera_id),
        "source_frame_index": int(source_index),
        "relative_image_path": relative_path,
        "pose_sha256": pose_sha256,
    })


def build_camera_table(train_dataset, device) -> CameraTable:
    """Training cameras, their calibrated supports and their poses."""
    if str(getattr(train_dataset, "split", "")).lower() != "train":
        raise RuntimeError("the Mip filter is built from the training split")
    poses = torch.as_tensor(train_dataset.poses, device="cpu").contiguous()
    extrinsics = list(train_dataset.cam_extrinsics)
    used_ids = sorted({int(item.camera_id) for item in extrinsics})

    cameras = []
    for camera_id in used_ids:
        intrinsic = train_dataset.cam_intrinsics[camera_id]
        width, height = train_dataset.camera_active_dimensions[camera_id]
        ratio_x, ratio_y = resize_ratios(intrinsic.width, intrinsic.height, width, height)
        params = scale_camera_parameters(str(intrinsic.model), list(intrinsic.params), ratio_x, ratio_y)
        support = train_dataset.get_camera_support(camera_id)
        if support is not None:
            rows = [[int(lo), int(hi)] if hi > lo else [0, 0] for lo, hi in support.rows]
            max_angle = float(support.max_angle_radians)
        else:
            rows = [[0, int(width)] for _ in range(int(height))]
            max_angle = None
            if str(intrinsic.model) == "OPENCV_FISHEYE":
                max_angle = _stock_max_angle(train_dataset, camera_id)
        cameras.append({
            "camera_id": camera_id,
            "model": str(intrinsic.model),
            "width": int(width),
            "height": int(height),
            "params": params,
            "max_angle": max_angle,
            "rows": rows,
        })

    # Lexical path below the dataset root, so symlinked images (prepared FullCircle scenes) keep their name.
    dataset_root = os.path.abspath(str(train_dataset.path))
    frames = []
    for index, (extrinsic, pose) in enumerate(zip(extrinsics, poses)):
        image_path = os.path.abspath(str(train_dataset.image_paths[index]))
        name = Path(os.path.relpath(image_path, dataset_root)).as_posix()
        source_index = int(getattr(extrinsic, "id", index))
        values = [float(value) for value in pose.reshape(-1).tolist()]
        frames.append({
            "frame_index": index,
            "camera_id": int(extrinsic.camera_id),
            "c2w": values,
            "tie_break": _tie_break_key(f"train:{source_index}:{name}", extrinsic.camera_id, source_index, name,
                                        values),
        })
    return pack_camera_table(cameras, frames, device)


def _stock_max_angle(train_dataset, camera_id: int) -> float:
    """Angular limit of a fisheye camera without calibrated support."""
    if camera_id in getattr(train_dataset, "camera_max_angles", {}):
        return float(train_dataset.camera_max_angles[camera_id])
    payload = train_dataset.intrinsics[camera_id][0]
    return float(torch.as_tensor(payload["max_angle"]).reshape(-1)[0])


def pack_camera_table(cameras: list[dict], frames: list[dict], device) -> CameraTable:
    """Pack camera and frame records for ``compute_filter``.

    ``cameras``: one dict per COLMAP camera,
    ``{"camera_id": int, "model": "OPENCV_FISHEYE" | "PINHOLE" | ..., "width": int, "height": int,
    "params": [COLMAP parameters at the active raster], "max_angle": float | None,
    "rows": [[begin, end) valid pixel columns of every raster row]}``.
    ``frames``: one dict per training frame,
    ``{"frame_index": int, "camera_id": int, "c2w": the camera-to-world matrix flattened row-major (3x4 or 4x4),
    "tie_break": str}``; equal sampling rates are resolved by ``tie_break`` order.

    Example, a 64 x 48 pinhole camera at the origin::

        cameras = [{"camera_id": 1, "model": "PINHOLE", "width": 64, "height": 48,
                    "params": [50.0, 50.0, 32.0, 24.0], "max_angle": None, "rows": [[0, 64]] * 48}]
        frames = [{"frame_index": 0, "camera_id": 1, "tie_break": "f0",
                   "c2w": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}]
        filter_3d, filter_dir = compute_filter(points, pack_camera_table(cameras, frames, "cuda"), 0.2)
    """
    cameras = sorted(cameras, key=lambda row: row["camera_id"])
    slots = {row["camera_id"]: index for index, row in enumerate(cameras)}
    tie_order = sorted(frames, key=lambda row: (row["tie_break"], row["frame_index"]))
    tie_rank = {row["frame_index"]: rank for rank, row in enumerate(tie_order)}
    frames = sorted(frames, key=lambda row: (row["camera_id"], tie_rank[row["frame_index"]]))
    rotations, centers, frame_camera, frame_index, frame_rank = [], [], [], [], []
    for row in frames:
        values = row["c2w"]
        rotations.append([[values[4 * i + j] for j in range(3)] for i in range(3)])
        centers.append([values[3], values[7], values[11]])
        frame_camera.append(slots[row["camera_id"]])
        frame_index.append(row["frame_index"])
        frame_rank.append(tie_rank[row["frame_index"]])

    maximum_height = max(row["height"] for row in cameras)
    models, widths, heights, params, caps, run_lo, run_hi = [], [], [], [], [], [], []
    for row in cameras:
        runs = row["rows"] + [[0, 0] for _ in range(maximum_height - row["height"])]
        models.append(MODEL_IDS[row["model"]])
        widths.append(row["width"])
        heights.append(row["height"])
        params.append(list(row["params"]) + [0.0] * (PACKED_PARAMETER_COUNT - len(row["params"])))
        caps.append(-1.0 if row["max_angle"] is None else float(row["max_angle"]))
        run_lo.append([item[0] for item in runs])
        run_hi.append([item[1] for item in runs])

    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device).contiguous()

    return CameraTable(
        tensor(rotations, torch.float64), tensor(centers, torch.float64),
        tensor(frame_camera, torch.int32), tensor(frame_index, torch.int32), tensor(frame_rank, torch.int64),
        tensor(models, torch.int32), tensor(widths, torch.int32), tensor(heights, torch.int32),
        tensor(params, torch.float64), tensor(caps, torch.float64),
        tensor(run_lo, torch.int32), tensor(run_hi, torch.int32),
    )


@torch.no_grad()
def compute_filter(positions: torch.Tensor, table: CameraTable, variance: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(filter_3D [N,1], filter_dir [N,3])`` in the dtype of ``positions``.

    ``filter_3D`` is ``sqrt(variance) / r`` with r the finest pixel sampling rate of any frame whose
    support contains the point (the largest observed filter for unobserved points); ``filter_dir`` is
    the unit direction from that frame's camera to the point.  ``variance`` is
    ``render.gofish.mip_filter_variance`` (px^2).
    """
    points64 = positions.detach().to(dtype=torch.float64).contiguous()
    best, _, observed_u8, direction64 = load_extension().best_rate(points64, *table.arguments(), True)
    observed = observed_u8.to(torch.bool)
    if not bool(observed.any()):
        raise RuntimeError("no Gaussian is observed by a training camera")
    scalar64 = torch.empty_like(best)
    scalar64[observed] = math.sqrt(variance) / best[observed]
    scalar64[~observed] = torch.max(scalar64[observed])
    return (
        scalar64.to(positions.dtype)[:, None].contiguous(),
        direction64.to(positions.dtype).contiguous(),
    )


def refresh_filter(model, train_dataset) -> None:
    """Recompute ``model.filter_3D`` / ``model.filter_dir`` from the current centres."""
    table = getattr(train_dataset, "_mip_camera_table", None)
    if table is None:
        table = build_camera_table(train_dataset, model.get_positions().device)
        train_dataset._mip_camera_table = table
    model.filter_3D, model.filter_dir = compute_filter(model.get_positions(), table, filter_variance(model.conf))
