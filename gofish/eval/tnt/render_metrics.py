# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Tanks & Temples held-out rendering metrics.

Saved 8-bit predictions are paired with the ground-truth PNGs by held-out name; when the raster sizes
differ, the *prediction* is resized to the ground-truth size with Pillow
LANCZOS (ground truth is never resized).  Metrics per frame on uint8 / 255
float32 images: PSNR from the float64 MSE, SSIM (torchmetrics, 11 x 11
Gaussian window, sigma 1.5, data range 1) and LPIPS-VGG (torchmetrics,
normalize=True).  Scene scores are the mean over frames.

``--pred`` and ``--gt`` hold one PNG per held-out frame with matching file
names (e.g. ``{index:05d}.png`` from render.py).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def evaluate(pred_dir: Path, gt_dir: Path, device: str) -> dict:
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    gt_files = sorted(gt_dir.glob("*.png"))
    assert gt_files and [p.name for p in sorted(pred_dir.glob("*.png"))] == [p.name for p in gt_files], \
        "prediction and ground-truth frame sets differ"
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()
    frames = []
    with torch.inference_mode():
        for gt_path in gt_files:
            gt_image = Image.open(gt_path).convert("RGB")
            pred_image = Image.open(pred_dir / gt_path.name).convert("RGB")
            if pred_image.size != gt_image.size:
                pred_image = pred_image.resize(gt_image.size, Image.Resampling.LANCZOS)
            pred = np.asarray(pred_image, dtype=np.float32) / 255.0
            target = np.asarray(gt_image, dtype=np.float32) / 255.0
            mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
            pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
            gt_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device)
            ssim_metric.reset()
            lpips_metric.reset()
            frames.append({"frame": gt_path.name, "psnr": -10.0 * math.log10(mse),
                           "ssim": float(ssim_metric(pred_t, gt_t).item()),
                           "lpips": float(lpips_metric(pred_t, gt_t).item())})
    mean = {key: float(np.mean([f[key] for f in frames])) for key in ("psnr", "ssim", "lpips")}
    return {"frames": len(frames), "mean": mean, "per_frame": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.pred, args.gt, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["mean"]))


if __name__ == "__main__":
    main()
