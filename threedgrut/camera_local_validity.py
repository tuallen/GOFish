# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Near-plane test in the camera-local coordinate of each camera model.

The inherited 3DGS/3DGUT near distance (0.2) is applied to view-space ``z``
for pinhole and distorted-pinhole models and to the radial camera distance
for fisheye models. Works on Python scalars and tensors.
"""

from __future__ import annotations

from typing import Any


MINIMUM_CAMERA_LOCAL_DISTANCE = 0.2
IDEAL_PINHOLE_MODELS = frozenset(("SIMPLE_PINHOLE", "PINHOLE"))
DISTORTED_PINHOLE_MODELS = frozenset(("SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"))
PINHOLE_MODELS = IDEAL_PINHOLE_MODELS | DISTORTED_PINHOLE_MODELS
FISHEYE_MODELS = frozenset(("OPENCV_FISHEYE",))


def camera_local_coordinate(model: str, view_z: Any, radial_distance: Any) -> Any:
    model = str(model)
    if model in PINHOLE_MODELS:
        return view_z
    if model in FISHEYE_MODELS:
        return radial_distance
    raise ValueError(f"unsupported camera model for local validity: {model!r}")


def camera_local_near_valid(model: str, view_z: Any, radial_distance: Any) -> Any:
    return camera_local_coordinate(model, view_z, radial_distance) > MINIMUM_CAMERA_LOCAL_DISTANCE
