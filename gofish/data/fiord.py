# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare a FIORD scene for GOFish, and convert its LiDAR scan.

    python -m gofish.data.fiord scene <FIORD>/<folder> <out>/<label>
    python -m gofish.data.fiord lidar <FIORD>/<folder>/<scan>.e57 <lidar>/<label>.npy
    python -m gofish.data.fiord folder <label>

Scenes are named by their paper labels (``SCENES``); the release archives unpack
to the folder names that ``SCENES`` maps them to, which are used only to find
the source files.

``scene`` takes one unzipped scene of the FIORD release (Zenodo community
``fiord``), copies its OPENCV_FISHEYE COLMAP model to ``sparse/0`` and writes
``images_4/<name>``: every registered image reduced 4x with Pillow's bicubic
filter and saved as JPEG with Pillow's defaults, as the paper's inputs
(byte-identical with Pillow 12.3).  The release keeps the model in ``model/``,
``colmap/model/`` or ``colmap/sparse/0/`` and the images in ``images/`` or
``fisheyeimages/``, depending on the scene; the first that exists is used.

``lidar`` reads every scan of the E57 file with its pose (pye57), keeps the
finite points and, above 12M points, a seeded random subset (numpy
default_rng(0), without replacement), saved as float32 ``[N, 3]``, as the
paper's ground truth.  It needs ``pip install pye57``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from gofish.eval.common.colmap import read_images

# Paper label -> folder of the unzipped FIORD release, in the order of the dataset script.
SCENES = {
    "Kitchen_In": "Kitchen_In",
    "Corridor_Out": "festia_out_corridor",
    "MeetingRoom_In": "meetingroom",
    "Road_Out": "main_campus",
    "Bridge_Out": "outcorridor",
    "Building_In": "parakennus",
    "Hall_In": "sportunifront",
}
# Scenes scored for geometry; Corridor_Out is scored for rendering only, as in the paper.
GEOMETRY_SCENES = [label for label in SCENES if label != "Corridor_Out"]

MODEL_FILES = ("cameras", "images", "points3D")
MODEL_DIRS = ("model", "colmap/model", "colmap/sparse/0")
IMAGE_DIRS = ("images", "fisheyeimages")


def prepare(source: Path, target: Path, factor: int = 4) -> None:
    model = next((source / d for d in MODEL_DIRS if (source / d / "images.bin").is_file()), None)
    if model is None:
        raise SystemExit(f"{source}: no fisheye COLMAP model in {', '.join(MODEL_DIRS)}")
    sparse = target / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    for stem in MODEL_FILES:
        for suffix in (".bin", ".txt"):
            if (model / f"{stem}{suffix}").is_file():
                shutil.copyfile(model / f"{stem}{suffix}", sparse / f"{stem}{suffix}")
    names = sorted(image.name for image in read_images(model / "images.bin"))
    images = next((source / d for d in IMAGE_DIRS if (source / d / names[0]).is_file()), None)
    if images is None:
        raise SystemExit(f"{source}: no image folder ({', '.join(IMAGE_DIRS)}) holds {names[0]}")
    for name in names:
        dst = target / f"images_{factor}" / name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(images / name) as image:
            image.resize((image.width // factor, image.height // factor), Image.Resampling.BICUBIC).save(dst)
    print(f"{target}: {len(names)} registered images at 1/{factor}")


def lidar(e57: Path, out: Path, max_points: int = 12_000_000, seed: int = 0) -> None:
    import numpy as np
    import pye57

    scans = pye57.E57(str(e57))
    chunks = []
    for index in range(scans.scan_count):
        data = scans.read_scan(index, ignore_missing_fields=True)
        if not all(key in data for key in ("cartesianX", "cartesianY", "cartesianZ")):
            continue
        xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=-1).astype(np.float32)
        chunks.append(xyz[np.isfinite(xyz).all(1)])
    if not chunks:
        raise SystemExit(f"{e57}: no cartesian points")
    points = np.concatenate(chunks, 0)
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, points.astype(np.float32))
    print(f"{out}: {len(points)} points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    scene = sub.add_parser("scene", help="sparse/0 and images_4 of one scene")
    scene.add_argument("source", type=Path)
    scene.add_argument("target", type=Path)
    scan = sub.add_parser("lidar", help="E57 scan to the ground-truth .npy")
    scan.add_argument("e57", type=Path)
    scan.add_argument("out", type=Path)
    folder = sub.add_parser("folder", help="release folder of a paper scene label")
    folder.add_argument("label", choices=list(SCENES))
    args = parser.parse_args()
    if args.command == "scene":
        prepare(args.source, args.target)
    elif args.command == "lidar":
        lidar(args.e57, args.out)
    else:
        print(SCENES[args.label])


if __name__ == "__main__":
    main()
