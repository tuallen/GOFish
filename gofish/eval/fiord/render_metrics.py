# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""FIORD held-out rendering metrics (Full and Common regions).

Predictions are saved PNGs, one per held-out frame (every 8th image after
sorting by COLMAP name), named by image stem as render.py writes them.

Full
    native 816 x 816 prediction against the native ``images_4`` image,
    restricted to the calibrated aperture of the frame's camera
    (``--masks``: one Boolean PNG per COLMAP camera id, ``<id>.png``, as
    written by ``gofish.eval.common.dataset_cameras``).
    Formula: ``aperture_metrics``.

Common
    canonical 800 x 800, 90-degree pinhole (``gofish.eval.fiord.rectify``).  A native
    816 x 816 prediction is transported with the rectification map.  Ground
    truth is the rectified image of the same frame.  Formula: ``raster_metrics``.

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
from gofish.eval.fiord.rectify import rectify_maps, transport


def _load_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _load_mask(path: Path):
    import torch

    return torch.from_numpy(np.asarray(Image.open(path).convert("L")) > 0)


def evaluate(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    device = args.device
    images = heldout_images(read_images(args.scene / "sparse" / "0" / "images.bin"), args.interval)
    lpips = M.make_lpips(device)
    maps = None
    if args.region == "common":
        maps = rectify_maps(args.scene / "sparse" / "0" / "cameras.bin")
    masks = {}
    frames = []
    with torch.inference_mode():
        for index, image in enumerate(images):
            pred_path = args.pred / f"{Path(image.name).stem}.png"
            if args.region == "full":
                pred = _load_uint8(pred_path)
                gt = _load_uint8(args.scene / "images_4" / image.name)
                assert pred.shape == gt.shape == (816, 816, 3), pred_path
                if image.camera_id not in masks:
                    masks[image.camera_id] = _load_mask(args.masks / f"{image.camera_id}.png").to(device)
                values = M.aperture_metrics(M.to_tensor(pred.astype(np.float32) / 255.0, device),
                                            M.to_tensor(gt.astype(np.float32) / 255.0, device),
                                            masks[image.camera_id], lpips)
            else:
                pred = _load_uint8(pred_path)
                assert pred.shape == (816, 816, 3), pred_path
                pred = transport(pred, maps[image.camera_id])
                gt = _load_uint8(args.common_gt / "images" / Path(image.name).name)
                assert pred.shape == gt.shape == (800, 800, 3), pred_path
                values = M.raster_metrics(M.to_tensor(pred.astype(np.float32) / 255.0, device),
                                          M.to_tensor(gt.astype(np.float32) / 255.0, device), lpips)
            frames.append({"index": index, "name": image.name, **values})
    return {"region": args.region, "frames": len(frames), "mean": M.mean_metrics(frames), "per_frame": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", "--data", dest="scene", type=Path, required=True, help="FIORD scene root (sparse/0, images_4)")
    parser.add_argument("--pred", type=Path, required=True, help="directory of <image stem>.png predictions, as written by render.py")
    parser.add_argument("--region", choices=("full", "common"), required=True)
    parser.add_argument("--masks", type=Path, help="full: per-camera aperture masks <camera_id>.png")
    parser.add_argument("--common-gt", type=Path, help="common: rectified scene root from gofish.eval.fiord.rectify")
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.region == "full" and args.masks is None:
        parser.error("--region full needs --masks")
    if args.region == "common" and args.common_gt is None:
        parser.error(f"--region {args.region} needs --common-gt")
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"region": result["region"], "frames": result["frames"], **result["mean"]}))


if __name__ == "__main__":
    main()
