# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""ScanNet++ Common region: the official undistorted DSLR raster at half resolution.

Common is the footprint of ScanNet++'s official undistorted DSLR pinhole
raster (``nerfstudio/transforms_undistorted.json``), evaluated at 876 x 584:

* ground truth: the official undistorted image, reduced 2x with a Pillow BOX
  filter;
* a native fisheye prediction (876 x 584, OPENCV_FISHEYE at half resolution)
  is sampled with the official full-resolution map F (native fisheye ->
  undistorted pinhole, ``cv2.fisheye.initUndistortRectifyMap``, CV_32FC1)
  expressed in half-resolution pixels as ``(F - 0.5) / 2``, INTER_LINEAR with
  BORDER_REFLECT_101, then reduced 2x with the same BOX filter.

Requires OpenCV 4.9.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gofish.eval.common.colmap import read_cameras

FULL_WH = (1752, 1168)
HALF_WH = (876, 584)


def half_resolution_map(cameras_txt: str | Path, transforms_undistorted: str | Path):
    """(map_x, map_y) sampling a half-resolution native prediction onto the full undistorted grid."""
    (camera,) = read_cameras(cameras_txt).values()
    assert camera.model == "OPENCV_FISHEYE" and (camera.width, camera.height) == FULL_WH
    fx, fy, cx, cy, *distortion = camera.params
    source = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    official = json.loads(Path(transforms_undistorted).read_text())
    target = np.array([[official["fl_x"], 0.0, official["cx"]],
                       [0.0, official["fl_y"], official["cy"]], [0.0, 0.0, 1.0]])
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        source, np.asarray(distortion), np.eye(3), target, FULL_WH, cv2.CV_32FC1)
    return (map_x - np.float32(0.5)) / np.float32(2.0), (map_y - np.float32(0.5)) / np.float32(2.0)


def box_half(full: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(full, mode="RGB").resize(HALF_WH, resample=Image.Resampling.BOX),
                      dtype=np.uint8).copy()


def transport(prediction: np.ndarray, maps) -> np.ndarray:
    """Native half-resolution RGB uint8 prediction -> Common raster (876 x 584)."""
    full = cv2.remap(prediction, maps[0], maps[1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)
    return box_half(full)


def common_ground_truth(undistorted_image: str | Path) -> np.ndarray:
    with Image.open(undistorted_image) as image:
        return box_half(np.asarray(image.convert("RGB"), dtype=np.uint8))
