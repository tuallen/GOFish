# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""FullCircle held-out rendering metrics (Full and Common regions).

Predictions are the ``render.py`` outputs of the tripod test frames,
``<camera>__<frame>.png`` at 720 x 720.  Ground truth is the released
``images_4`` image of the same frame.

Full
    the calibrated first-branch disc of the frame's camera minus the tripod:
    pixels whose centre lies inside ``rho(theta_open)`` in normalized
    coordinates, without the tripod of FullCircle's ``mask_test.png``, i.e. its
    masked pixels within 1300 native pixels of the image centre, resized to
    720 x 720 by nearest neighbour.  Formula: ``aperture_metrics``.

Common
    the canonical 800 x 800, 90-degree pinhole of FIORD's Common region: both
    the prediction and the ground truth are rectified with the fisheye
    intrinsics scaled by 1/4.  Formula: ``raster_metrics``.

    python -m gofish.eval.fullcircle.render_metrics --scene <prepared scene> \\
        --pred runs/fullcircle/<scene>/heldout/pred --mask-test <FullCircle>/mask_test.png \\
        --region full --out eval_full.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gofish.eval.common import metrics as M
from gofish.eval.common.colmap import read_cameras

RASTER = 720
NATIVE = 2880
TRIPOD_RADIUS = 1300.0
RECT_SIZE = 800


def open_limit(k) -> float:
    """First angle where the radial map rho(theta) stops increasing, just below it."""
    k1, k2, k3, k4 = k
    roots = np.roots([9 * k4, 7 * k3, 5 * k2, 3 * k1, 1.0])
    ys = sorted(float(r.real) for r in roots
                if abs(float(r.imag)) <= 1e-10 * max(1.0, abs(float(r.real))) and float(r.real) > 0)
    thetas = [math.sqrt(y) for y in ys if math.sqrt(y) < math.pi]
    limit = float(np.nextafter(thetas[0] if thetas else math.pi, 0.0))

    def derivative(theta: float) -> float:
        y = theta * theta
        return 1.0 + y * (3.0 * k1 + y * (5.0 * k2 + y * (7.0 * k3 + y * 9.0 * k4)))

    while derivative(limit) <= 0.0:
        limit = float(np.nextafter(limit, 0.0))
    return limit


def radial(theta: float, k) -> float:
    y = theta * theta
    return theta * (1.0 + y * (k[0] + y * (k[1] + y * (k[2] + y * k[3]))))


def tripod_mask(mask_test: Path) -> np.ndarray:
    keep = np.asarray(Image.open(mask_test).convert("L")) > 127
    yy, xx = np.mgrid[:NATIVE, :NATIVE]
    tripod = (~keep) & (np.hypot(xx - (NATIVE - 1) / 2.0, yy - (NATIVE - 1) / 2.0) < TRIPOD_RADIUS)
    small = Image.fromarray(tripod.astype(np.uint8) * 255).resize((RASTER, RASTER), Image.NEAREST)
    return np.asarray(small) > 127


def full_masks(cameras, tripod: np.ndarray) -> dict[int, np.ndarray]:
    masks = {}
    j, i = np.mgrid[:RASTER, :RASTER]
    for camera_id, camera in cameras.items():
        fx, fy, cx, cy = (value * RASTER / NATIVE for value in camera.params[:4])
        k = camera.params[4:8]
        disc = np.hypot((i + 0.5 - cx) / fx, (j + 0.5 - cy) / fy) < radial(open_limit(k), k)
        masks[camera_id] = disc & ~tripod
    return masks


def rectify_maps(cameras) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    focal = (RECT_SIZE / 2.0) / np.tan(np.deg2rad(90.0) / 2.0)
    target = np.array([[focal, 0.0, RECT_SIZE / 2.0], [0.0, focal, RECT_SIZE / 2.0], [0.0, 0.0, 1.0]])
    maps = {}
    for camera_id, camera in cameras.items():
        fx, fy, cx, cy = (value * RASTER / NATIVE for value in camera.params[:4])
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        maps[camera_id] = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, np.asarray(camera.params[4:8], dtype=np.float64), np.eye(3), target,
            (RECT_SIZE, RECT_SIZE), cv2.CV_16SC2)
    return maps


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def evaluate(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    cameras = read_cameras(args.scene / "sparse" / "0" / "cameras.bin")
    camera_of = {"camera1": 1, "camera2": 2}
    names = sorted(args.scene.joinpath("test_list.txt").read_text().split())
    lpips = M.make_lpips(args.device)
    masks = full_masks(cameras, tripod_mask(args.mask_test)) if args.region == "full" else None
    maps = rectify_maps(cameras) if args.region == "common" else None
    frames = []
    with torch.inference_mode():
        for name in names:
            folder, stem = Path(name).parent.name, Path(name).stem
            camera_id = camera_of[folder]
            pred = _rgb(args.pred / f"{folder}__{stem}.png")
            gt = _rgb(args.scene / "images_4" / name)
            assert pred.shape == gt.shape == (RASTER, RASTER, 3), name
            if args.region == "full":
                values = M.aperture_metrics(M.to_tensor(pred.astype(np.float32) / 255.0, args.device),
                                            M.to_tensor(gt.astype(np.float32) / 255.0, args.device),
                                            torch.from_numpy(masks[camera_id]).to(args.device), lpips)
            else:
                remap = lambda image: cv2.remap(image, *maps[camera_id], interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                values = M.raster_metrics(M.to_tensor(remap(pred).astype(np.float32) / 255.0, args.device),
                                          M.to_tensor(remap(gt).astype(np.float32) / 255.0, args.device), lpips)
            frames.append({"name": name, **values})
    return {"region": args.region, "frames": len(frames), "mean": M.mean_metrics(frames), "per_frame": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", "--data", dest="scene", type=Path, required=True, help="prepared scene (sparse/0, images_4, test_list.txt)")
    parser.add_argument("--pred", type=Path, required=True, help="render.py heldout/pred directory")
    parser.add_argument("--mask-test", type=Path, required=True, help="FullCircle's masking/mask_test.png")
    parser.add_argument("--region", choices=("full", "common"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean"]))


if __name__ == "__main__":
    main()
