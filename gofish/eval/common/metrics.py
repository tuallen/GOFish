# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Image metrics shared by the FIORD, FullCircle and ScanNet++ evaluators.

Two formulas are used in the paper:

* ``raster_metrics``: full-raster PSNR, SSIM over every complete 11x11 window
  (Gaussian window, sigma 1.5) and LPIPS-VGG.  Used for the Common region and
  for unmasked native rasters.
* ``aperture_metrics``: the same statistics restricted to a Boolean region:
  masked-MSE PSNR in float64, SSIM averaged over windows lying wholly inside
  the region, and LPIPS-VGG with both exteriors set to zero.  Used for the
  Full (calibrated aperture) region.

Images are float tensors in [0, 1] with shape [1, 3, H, W]; PNG inputs are
read as uint8 / 255 in float32.  Scene scores are the mean over frames.
"""

from __future__ import annotations

import math
from math import exp

import numpy as np

WINDOW_SIZE = 11
WINDOW_SIGMA = 1.5


def to_tensor(image: np.ndarray, device="cpu"):
    import torch

    return torch.from_numpy(image).permute(2, 0, 1)[None].to(device)


def make_lpips(device="cpu"):
    """LPIPS with the VGG backbone on [0, 1] inputs (torchmetrics, normalize=True)."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device).eval()


# ---------------------------------------------------------------------------
# Full-raster formula


def gaussian_window(reference):
    import torch

    values = torch.tensor([exp(-(x - 5) ** 2 / float(2 * WINDOW_SIGMA**2)) for x in range(WINDOW_SIZE)])
    values = (values / values.sum()).unsqueeze(1)
    return (
        values.mm(values.t()).float().unsqueeze(0).unsqueeze(0)
        .expand(reference.shape[1], 1, WINDOW_SIZE, WINDOW_SIZE)
        .contiguous().to(device=reference.device, dtype=reference.dtype)
    )


def ssim_map(image_a, image_b, window):
    import torch
    import torch.nn.functional as F

    channels = image_a.shape[1]
    mu_a = F.conv2d(image_a, window, padding=5, groups=channels)
    mu_b = F.conv2d(image_b, window, padding=5, groups=channels)
    mu_a_sq, mu_b_sq, mu_ab = mu_a.pow(2), mu_b.pow(2), mu_a * mu_b
    sigma_a_sq = F.conv2d(image_a * image_a, window, padding=5, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(image_b * image_b, window, padding=5, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(image_a * image_b, window, padding=5, groups=channels) - mu_ab
    sigma_a_sq = torch.clamp(sigma_a_sq, min=1e-12)
    sigma_b_sq = torch.clamp(sigma_b_sq, min=1e-12)
    luminance = (2 * mu_ab + 0.01**2) / (mu_a_sq + mu_b_sq + 0.01**2)
    contrast = (2 * sigma_ab + 0.03**2) / (sigma_a_sq + sigma_b_sq + 0.03**2)
    return luminance * contrast


def raster_metrics(pred, gt, lpips) -> dict:
    """PSNR / SSIM / LPIPS over the whole raster (SSIM on complete windows only)."""
    import torch
    import torch.nn.functional as F

    dev = gt.device
    psnr = float(-10.0 * torch.log10((pred - gt).square().mean().clamp_min(1e-12)))
    smap = ssim_map(gt, pred, gaussian_window(gt))
    support = F.conv2d(torch.ones((1, 1, gt.shape[2], gt.shape[3]), device=dev),
                       torch.ones((1, 1, 11, 11), device=dev), padding=5) == 121.0
    ssim = float(smap[support.expand_as(smap)].mean())
    return {"PSNR": psnr, "SSIM": ssim, "LPIPS": float(lpips(pred, gt))}


# ---------------------------------------------------------------------------
# Calibrated-aperture formula


def gaussian_kernel_1d(size: int = WINDOW_SIZE, sigma: float = WINDOW_SIGMA) -> np.ndarray:
    coordinate = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(coordinate * coordinate) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def masked_psnr(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    support = np.asarray(mask, dtype=bool)
    if not bool(support.any()):
        return None
    mse = float(np.mean(np.square(pred[support] - gt[support]), dtype=np.float64))
    return float(-10.0 * math.log10(max(mse, 1.0e-12)))


def masked_ssim(prediction, target, mask) -> tuple[float | None, int]:
    """11x11, sigma 1.5 SSIM averaged over windows wholly inside ``mask``."""
    import torch
    import torch.nn.functional as F

    if not bool(mask.any().item()):
        return None, 0
    one_d = torch.as_tensor(gaussian_kernel_1d(), device=prediction.device, dtype=prediction.dtype)
    window = torch.outer(one_d, one_d)[None, None].expand(3, 1, -1, -1).contiguous()
    padding = WINDOW_SIZE // 2
    mu_pred = F.conv2d(prediction, window, padding=padding, groups=3)
    mu_gt = F.conv2d(target, window, padding=padding, groups=3)
    pred_sq = F.conv2d(prediction.square(), window, padding=padding, groups=3) - mu_pred.square()
    gt_sq = F.conv2d(target.square(), window, padding=padding, groups=3) - mu_gt.square()
    covariance = F.conv2d(prediction * target, window, padding=padding, groups=3) - mu_pred * mu_gt
    numerator = (2.0 * mu_pred * mu_gt + 0.01**2) * (2.0 * covariance + 0.03**2)
    denominator = (mu_pred.square() + mu_gt.square() + 0.01**2) * (
        pred_sq.clamp_min(1.0e-12) + gt_sq.clamp_min(1.0e-12) + 0.03**2
    )
    smap = numerator / denominator
    ones = torch.ones((1, 1, WINDOW_SIZE, WINDOW_SIZE), device=mask.device, dtype=torch.float32)
    complete = F.conv2d(mask[None, None].float(), ones, padding=padding)[0, 0]
    complete = complete == float(WINDOW_SIZE * WINDOW_SIZE)
    count = int(complete.sum().item())
    if count == 0:
        return None, 0
    return float(smap.permute(0, 2, 3, 1)[0][complete].mean().item()), count


def masked_lpips(lpips, prediction, target, mask) -> float | None:
    if not bool(mask.any().item()):
        return None
    gate = mask[None, None].to(device=prediction.device, dtype=prediction.dtype)
    return float(lpips(prediction * gate, target * gate).item())


def aperture_metrics(prediction, target, mask, lpips) -> dict | None:
    """Metrics restricted to the Boolean ``mask`` [H, W]; ``None`` if the mask is empty."""
    pred = prediction.detach().clamp(0.0, 1.0)
    gt = target.detach().clamp(0.0, 1.0)
    pixels = int(mask.sum().item())
    if pixels == 0:
        return None
    pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
    gt_np = gt[0].permute(1, 2, 0).cpu().numpy()
    ssim, windows = masked_ssim(pred, gt, mask)
    return {
        "PSNR": masked_psnr(pred_np, gt_np, mask.cpu().numpy()),
        "SSIM": ssim,
        "LPIPS": masked_lpips(lpips, pred, gt, mask),
        "pixels": pixels,
        "complete_ssim_windows": windows,
    }


def mean_metrics(frames: list[dict], keys=("PSNR", "SSIM", "LPIPS")) -> dict:
    """Frame mean, skipping frames whose value is undefined."""
    out = {}
    for key in keys:
        values = [float(f[key]) for f in frames if f.get(key) is not None]
        out[key] = float(sum(values) / len(values)) if values else None
    return out
