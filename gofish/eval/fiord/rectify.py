# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Canonical rectified pinhole of FIORD (Common region).

The Common region is the 90 x 90 degree field of view of an 800 x 800 pinhole
(f = 400 / tan(45 deg), principal point at the raster centre) sharing each
fisheye camera's pose.  Both sides of a Common comparison live on this raster:

* ``gt``: the native ``images_4`` (816 x 816) images are rectified once with
  OpenCV and stored under ``<out>/images/<basename>``, together with a PINHOLE
  COLMAP model in the same world frame.  Its images are the Common ground
  truth.
* ``transport``: a native 816 x 816 prediction is carried to the same raster
  with the same map, so prediction and ground truth are scored on identical pixels.

Map: ``cv2.fisheye.initUndistortRectifyMap`` with the fisheye intrinsics
scaled by 1/4, CV_16SC2, INTER_LINEAR, BORDER_CONSTANT 0 (OpenCV 4.9).
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np

from gofish.eval.common.colmap import read_cameras, read_images

RECT_SIZE = 800
RECT_HFOV_DEG = 90.0
NATIVE_FULL_WIDTH = 3264.0


def rectified_focal() -> float:
    return (RECT_SIZE / 2.0) / np.tan(np.deg2rad(RECT_HFOV_DEG) / 2.0)


def rectify_maps(cameras_path: str | Path, source_width: int = 816) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-camera (map1, map2) from the native OPENCV_FISHEYE model to the canonical pinhole."""
    scale = source_width / NATIVE_FULL_WIDTH
    focal = rectified_focal()
    target = np.array([[focal, 0.0, RECT_SIZE / 2.0], [0.0, focal, RECT_SIZE / 2.0], [0.0, 0.0, 1.0]])
    maps = {}
    for camera_id, camera in read_cameras(cameras_path).items():
        assert camera.model == "OPENCV_FISHEYE", camera.model
        fx, fy, cx, cy, *distortion = camera.params
        intrinsic = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        maps[camera_id] = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, np.asarray(distortion, dtype=np.float64), np.eye(3), target,
            (RECT_SIZE, RECT_SIZE), cv2.CV_16SC2)
    return maps


def transport(image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Native 816 x 816 RGB uint8 image -> canonical 800 x 800 raster."""
    return cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _write_pinhole_cameras(path: Path, camera_ids) -> None:
    focal = rectified_focal()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(camera_ids)))
        for camera_id in sorted(camera_ids):
            f.write(struct.pack("<iiQQ", camera_id, 1, RECT_SIZE, RECT_SIZE))
            f.write(struct.pack("<dddd", focal, focal, RECT_SIZE / 2.0, RECT_SIZE / 2.0))


def _write_points_ply(points_bin: Path, ply: Path) -> None:
    xyz, rgb = [], []
    with points_bin.open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            f.read(8)
            xyz.append(struct.unpack("<ddd", f.read(24)))
            rgb.append(struct.unpack("<BBB", f.read(3)))
            f.read(8)
            f.seek(8 * struct.unpack("<Q", f.read(8))[0], 1)
    xyz, rgb = np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8)
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.zeros(len(xyz), dtype=dtype)
    rows["x"], rows["y"], rows["z"] = xyz.T
    rows["red"], rows["green"], rows["blue"] = rgb.T
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with ply.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def rectify_scene(scene_root: Path, out_root: Path, image_dir: str = "images_4") -> None:
    """Rectify every posed image of a FIORD scene; poses and points are copied unchanged."""
    sparse = scene_root / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.bin")
    images = read_images(sparse / "images.bin")
    sample = next(cv2.imread(str(scene_root / image_dir / im.name)) for im in images
                  if (scene_root / image_dir / im.name).exists())
    height, width = sample.shape[:2]
    maps = rectify_maps(sparse / "cameras.bin", width)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for im in images:
        src = scene_root / image_dir / im.name
        image = cv2.imread(str(src)) if src.exists() else None
        if image is None:
            continue
        if (image.shape[1], image.shape[0]) != (width, height):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        dst = out_root / "images" / Path(im.name).name
        assert not dst.exists(), f"basename collision: {dst.name}"
        cv2.imwrite(str(dst), transport(image, maps[im.camera_id]))
    _write_pinhole_cameras(out_root / "sparse" / "0" / "cameras.bin", cameras.keys())
    shutil.copy(sparse / "images.bin", out_root / "sparse" / "0" / "images.bin")
    shutil.copy(sparse / "points3D.bin", out_root / "sparse" / "0" / "points3D.bin")
    _write_points_ply(sparse / "points3D.bin", out_root / "sparse" / "0" / "points3D.ply")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="FIORD scene root (sparse/0, images_4)")
    parser.add_argument("--out", type=Path, required=True, help="rectified scene root to write")
    args = parser.parse_args()
    rectify_scene(args.scene, args.out)


if __name__ == "__main__":
    main()
