# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""ScanNet++ held-out rendering metrics (Full and Common regions).

Held-out frames are every 8th DSLR image after sorting by COLMAP name.
Predictions are PNGs named by image stem, as render.py writes them.

Full
    native OPENCV_FISHEYE half-resolution raster (876 x 584) against
    ``dslr/resized_images_2/<name>``; the calibrated domain is the whole
    raster.  Saved predictions are scored with ``raster_metrics``.

Common
    official undistorted DSLR raster at half resolution (``common_region``).
    Native predictions are transported to that raster.  ``raster_metrics``.

Scene scores are the mean over held-out frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gofish.eval.common import metrics as M
from gofish.eval.common.colmap import heldout_images, read_images
from gofish.eval.scannetpp.common_region import common_ground_truth, half_resolution_map, transport


def _load_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def evaluate(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    device = args.device
    dslr = args.scene / "dslr"
    images = heldout_images(read_images(dslr / "colmap" / "images.txt"), args.interval)
    lpips = M.make_lpips(device)
    maps = None
    if args.region == "common":
        maps = half_resolution_map(dslr / "colmap" / "cameras.txt",
                                   dslr / "nerfstudio" / "transforms_undistorted.json")
    frames = []
    with torch.inference_mode():
        for index, image in enumerate(images):
            stem = Path(image.name).stem
            pred = _load_uint8(args.pred / f"{stem}.png")
            assert pred.shape == (584, 876, 3), image.name
            if args.region == "full":
                gt = _load_uint8(dslr / "resized_images_2" / image.name)
            else:
                pred = transport(pred, maps)
                gt = common_ground_truth(dslr / "undistorted_images" / image.name)
            pred_t = M.to_tensor(pred.astype(np.float32) / 255.0, device)
            gt_t = M.to_tensor(gt.astype(np.float32) / 255.0, device)
            values = M.raster_metrics(pred_t, gt_t, lpips)
            frames.append({"index": index, "name": image.name, **values})
    return {"region": args.region, "frames": len(frames), "mean": M.mean_metrics(frames), "per_frame": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", "--data", dest="scene", type=Path, required=True, help="ScanNet++ scene root (contains dslr/)")
    parser.add_argument("--pred", type=Path, required=True,
                        help="directory of <image stem>.png predictions, as written by render.py")
    parser.add_argument("--region", choices=("full", "common"), required=True)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"region": result["region"], "frames": result["frames"], **result["mean"]}))


if __name__ == "__main__":
    main()
