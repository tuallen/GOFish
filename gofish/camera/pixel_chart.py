# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""COLMAP corner-origin pixel chart shared by every camera consumer.

The centre of the upper-left pixel is ``(0.5, 0.5)``. When an image is
resized from native to active dimensions, the per-axis ratios are
``a_x = W_active / W_native`` and ``a_y = H_active / H_native``, so::

    f_active = a * f_native
    c_active = a * c_native
    u_native(index) = (index + 0.5) / a
"""

from __future__ import annotations

import math
from typing import Any


_SIMPLE_MODELS = frozenset(("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"))
_XY_FOCAL_MODELS = frozenset(("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"))
_MODEL_ARITY = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "FULL_OPENCV": 12,
    "OPENCV_FISHEYE": 8,
}


def _ratio(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("active/native ratio must be finite and positive")
    return result


def resize_ratios(native_width, native_height, active_width, active_height) -> tuple[float, float]:
    """Per-axis resize ratios from the actual raster dimensions."""
    native_w, native_h, active_w, active_h = (
        int(native_width), int(native_height), int(active_width), int(active_height))
    if min(native_w, native_h, active_w, active_h) <= 0:
        raise ValueError("raster dimensions must be positive")
    return active_w / native_w, active_h / native_h


def raster_center(index: Any):
    return index + 0.5


def native_center(index: Any, active_ratio: Any):
    """Lift an active raster index centre to the native chart."""
    return raster_center(index) / _ratio(active_ratio)


def native_sample(index: int, active_ratio: float) -> float:
    return float(native_center(float(index), active_ratio))


def native_to_raster(value: Any, active_ratio: Any):
    return value * _ratio(active_ratio)


def raster_to_index_coordinate(value: Any):
    return value - 0.5


def scale_camera_parameters(model: str, parameters: Any, active_ratio_x: Any,
                            active_ratio_y: Any | None = None) -> list[float]:
    """Active projection ``[fx, fy, cx, cy, *distortion]`` for a COLMAP model.

    Distortion coefficients are dimensionless and copied unchanged.
    """
    ax = _ratio(active_ratio_x)
    ay = ax if active_ratio_y is None else _ratio(active_ratio_y)
    values = [float(item) for item in parameters]
    model = str(model)
    expected = _MODEL_ARITY.get(model)
    if expected is None:
        raise ValueError(f"unsupported camera model {model!r}")
    if len(values) != expected:
        raise ValueError(f"{model} requires {expected} parameters; got {len(values)}")
    if model in _SIMPLE_MODELS:
        focal, cx, cy = values[:3]
        distortion = values[3:]
        fx = fy = focal
    else:
        fx, fy, cx, cy = values[:4]
        distortion = values[4:]
    return [fx * ax, fy * ay, native_to_raster(cx, ax), native_to_raster(cy, ay), *distortion]


def torch_pixel_indices(u, v, width: int, height: int):
    """Integer pixel of projected chart coordinates and an in-raster flag."""
    import torch

    ix = torch.floor(u).to(torch.int64)
    iy = torch.floor(v).to(torch.int64)
    inside = (
        torch.isfinite(u) & torch.isfinite(v)
        & (ix >= 0) & (ix < int(width))
        & (iy >= 0) & (iy < int(height))
    )
    return ix, iy, inside


def torch_raster_centers(width: int, height: int, *, device=None, dtype=None):
    """``[1, H, W, 2]`` grid of pixel centres."""
    import torch

    actual_dtype = torch.float32 if dtype is None else dtype
    y, x = torch.meshgrid(
        torch.arange(int(height), dtype=actual_dtype, device=device),
        torch.arange(int(width), dtype=actual_dtype, device=device),
        indexing="ij",
    )
    return torch.stack((raster_center(x), raster_center(y)), dim=-1).unsqueeze(0)
