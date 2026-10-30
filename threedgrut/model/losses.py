# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import exp

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


# ---------------------------------------------------------------------------
# Losses restricted to the calibrated camera domain
# ---------------------------------------------------------------------------

WINDOW_SIZE = 11
WINDOW_SIGMA = 1.5


class MaskedLossSupport:
    """Boolean pixel support of one camera (or frame) and its complete SSIM windows."""

    __slots__ = ("mask", "ssim_support", "_flat_index")

    def __init__(self, mask_bhwc: torch.Tensor, ssim_support_b1hw: torch.Tensor):
        self.mask = mask_bhwc
        self.ssim_support = ssim_support_b1hw
        self._flat_index: dict[tuple, torch.Tensor] = {}

    def gather(self, value: torch.Tensor, mask_view: torch.Tensor) -> torch.Tensor:
        """``value[mask_view]`` without a host synchronization.

        The supports are immutable, so the flat indices of each mask view are
        computed once.  The selected elements and their row-major order are
        those of Boolean indexing, so every reduction is bit-identical.
        """
        key = (tuple(mask_view.shape), tuple(mask_view.stride()), mask_view.data_ptr())
        index = self._flat_index.get(key)
        if index is None:
            index = mask_view.reshape(-1).nonzero().squeeze(1)
            self._flat_index[key] = index
        return value.reshape(-1).index_select(0, index)

    def mask_like(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            return self.mask.expand_as(value)
        if value.ndim == 3:
            return self.mask[0].expand_as(value)
        if value.ndim == 2:
            return self.mask[0, ..., 0]
        raise ValueError(f"unsupported reduction shape {tuple(value.shape)}")

    def ssim_support_like(self, value_bchw: torch.Tensor) -> torch.Tensor:
        return self.ssim_support.expand(1, int(value_bchw.shape[1]), *value_bchw.shape[2:])


class MaskedLossCache:
    """Per-camera (and per-frame, when a frame mask applies) support cache.

    The supports and the SSIM window are computed once; a cache hit only
    creates tensor views.
    """

    def __init__(self) -> None:
        self._support: dict[tuple, MaskedLossSupport] = {}
        self._windows: dict[tuple, torch.Tensor] = {}

    def bind_batch(self, batch) -> MaskedLossSupport:
        mask = batch.training_valid_mask
        frame_idx = int(batch.frame_idx) if batch.semantic_validity_applied else -1
        key = (int(batch.camera_intrinsic_id), frame_idx, mask.device.type, mask.device.index, tuple(mask.shape))
        existing = self._support.get(key)
        if existing is not None:
            return existing
        owned_mask = mask.detach().clone(memory_format=torch.contiguous_format)
        if int(owned_mask.sum().item()) == 0:
            raise RuntimeError("camera domain has no valid pixel")
        mask_b1hw = owned_mask.permute(0, 3, 1, 2).float()
        kernel = torch.ones((1, 1, WINDOW_SIZE, WINDOW_SIZE), device=mask.device, dtype=mask_b1hw.dtype)
        counts = F.conv2d(mask_b1hw, kernel, padding=WINDOW_SIZE // 2)
        support = (counts == float(WINDOW_SIZE * WINDOW_SIZE)).detach()
        entry = MaskedLossSupport(owned_mask, support)
        self._support[key] = entry
        return entry

    def gaussian_window(self, reference_bchw: torch.Tensor) -> torch.Tensor:
        """11x11 Gaussian SSIM window (sigma 1.5), one per device/dtype/channel count."""
        channels = int(reference_bchw.shape[1])
        key = (reference_bchw.device.type, reference_bchw.device.index, reference_bchw.dtype, channels)
        existing = self._windows.get(key)
        if existing is not None:
            return existing
        one_d = torch.tensor(
            [exp(-(x - WINDOW_SIZE // 2) ** 2 / float(2 * WINDOW_SIGMA**2)) for x in range(WINDOW_SIZE)]
        )
        one_d = (one_d / one_d.sum()).unsqueeze(1)
        window = (
            one_d.mm(one_d.t())
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(channels, 1, WINDOW_SIZE, WINDOW_SIZE)
            .contiguous()
            .to(device=reference_bchw.device, dtype=reference_bchw.dtype)
        )
        self._windows[key] = window
        return window


def _ssim_maps(image_a: torch.Tensor, image_b: torch.Tensor, window: torch.Tensor):
    channels = int(image_a.shape[1])
    mu_a = F.conv2d(image_a, window, padding=WINDOW_SIZE // 2, groups=channels)
    mu_b = F.conv2d(image_b, window, padding=WINDOW_SIZE // 2, groups=channels)
    mu_a_sq = mu_a.pow(2)
    mu_b_sq = mu_b.pow(2)
    mu_ab = mu_a * mu_b
    sigma_a_sq = F.conv2d(image_a * image_a, window, padding=WINDOW_SIZE // 2, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(image_b * image_b, window, padding=WINDOW_SIZE // 2, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(image_a * image_b, window, padding=WINDOW_SIZE // 2, groups=channels) - mu_ab
    sigma_a_sq = torch.clamp(sigma_a_sq, min=1e-12)
    sigma_b_sq = torch.clamp(sigma_b_sq, min=1e-12)
    luminance = (2 * mu_ab + 0.01**2) / (mu_a_sq + mu_b_sq + 0.01**2)
    contrast_structure = (2 * sigma_ab + 0.03**2) / (sigma_a_sq + sigma_b_sq + 0.03**2)
    return luminance, contrast_structure


def masked_mean(value: torch.Tensor, support: MaskedLossSupport) -> torch.Tensor:
    return support.gather(value, support.mask_like(value)).mean()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, support: MaskedLossSupport) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), support)


def masked_l2(prediction: torch.Tensor, target: torch.Tensor, support: MaskedLossSupport) -> torch.Tensor:
    return masked_mean((prediction - target).square(), support)


def masked_dssim(
    ground_truth: torch.Tensor, prediction: torch.Tensor, support: MaskedLossSupport, cache: MaskedLossCache
) -> torch.Tensor:
    """1 - SSIM averaged over the windows that lie entirely inside the support."""
    window = cache.gaussian_window(ground_truth)
    luminance, contrast_structure = _ssim_maps(ground_truth, prediction, window)
    gate = support.ssim_support_like(luminance)
    return 1.0 - support.gather(luminance * contrast_structure, gate).mean()
