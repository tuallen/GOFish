# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Minimal COLMAP model reader and the held-out split used by every evaluator."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8)}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class Image:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str

    def camera_to_world(self) -> np.ndarray:
        w, x, y, z = np.asarray(self.qvec, dtype=np.float64) / np.linalg.norm(self.qvec)
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ np.asarray(self.tvec, dtype=np.float64)
        return pose


def read_cameras(path: str | Path) -> dict[int, Camera]:
    path = Path(path)
    cameras = {}
    if path.suffix == ".bin":
        with path.open("rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                camera_id, model_id = struct.unpack("<ii", f.read(8))
                width, height = struct.unpack("<QQ", f.read(16))
                name, count = CAMERA_MODELS[model_id]
                params = struct.unpack(f"<{count}d", f.read(8 * count))
                cameras[camera_id] = Camera(camera_id, name, int(width), int(height), params)
        return cameras
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        t = line.split()
        cameras[int(t[0])] = Camera(int(t[0]), t[1], int(t[2]), int(t[3]), tuple(float(v) for v in t[4:]))
    return cameras


def read_images(path: str | Path) -> list[Image]:
    path = Path(path)
    images = []
    if path.suffix == ".bin":
        with path.open("rb") as f:
            for _ in range(struct.unpack("<Q", f.read(8))[0]):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = struct.unpack("<dddd", f.read(32))
                tvec = struct.unpack("<ddd", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                name = b""
                while (ch := f.read(1)) not in (b"\x00", b""):
                    name += ch
                f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
                images.append(Image(image_id, qvec, tvec, camera_id, name.decode()))
    else:
        lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
        for line in lines[0::2]:
            t = line.split()
            images.append(Image(int(t[0]), tuple(map(float, t[1:5])), tuple(map(float, t[5:8])),
                                int(t[8]), t[9]))
    return images


def heldout_images(images: list[Image], interval: int = 8) -> list[Image]:
    """Every ``interval``-th image after sorting by COLMAP image name (index 0 first).

    This is the split of the GOFish dataset loader.
    """
    ordered = sorted(images, key=lambda image: image.name)
    return [image for index, image in enumerate(ordered) if index % interval == 0]
