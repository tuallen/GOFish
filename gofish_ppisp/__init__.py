# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
PPISP: Physically-Plausible Image Signal Processing

A learned post-processing module for radiance field rendering that models:
- Exposure compensation (per-frame)
- Vignetting (per-camera)
- Color correction (per-frame)
- Camera Response Function / tone mapping (per-camera)

Includes a controller for predicting exposure and color
parameters from rendered radiance images.
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from gofish_ppisp.wide_angle import _radial_decrements, flatfield_attenuation, pixel_centres, radial_attenuation
from gofish_ppisp._cuda import load as _load_cuda


class _LazyExtension:
    """PPISP's CUDA extension, compiled and loaded on first use."""

    _module = None

    def __getattr__(self, name):
        if _LazyExtension._module is None:
            _LazyExtension._module = _load_cuda()
        return getattr(_LazyExtension._module, name)


_C = _LazyExtension()


# =============================================================================
# Version
# =============================================================================

from gofish_ppisp._version import __version__

# =============================================================================
# Constants
# =============================================================================

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
# Stored as 8x8 block-diagonal matrix for efficient single-matmul application
# Used by PPISP.get_regularization_loss() for color mean regularization
_COLOR_PINV_BLOCK_DIAG = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),  # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),  # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),  # Green
    torch.tensor([[0.0128369, -0.0034654],
                 [-0.0034654, 0.0128158]]),  # Neutral
).to(torch.float32)

# Constants (match config.h)
NUM_VIGNETTING_ALPHA_TERMS = 3


def set_validate_inputs(enabled: bool) -> None:
    """Enable or disable input validation in the CUDA extension wrappers.

    Off by default. When enabled, every extension call checks device, dtype,
    shape, contiguity, and camera/frame indices and raises on malformed inputs,
    which otherwise reach the kernels unchecked. Useful when debugging.
    """
    _C.set_validate_inputs(enabled)


def validate_inputs_enabled() -> bool:
    """Return whether the CUDA extension wrappers validate their inputs."""
    return _C.validate_inputs_enabled()


def _normalize_index(idx: torch.Tensor | int | None, name: str) -> int:
    """Normalize camera/frame index to int.

    Args:
        idx: Index as tensor (numel==1), int, or None
        name: Parameter name for error messages

    Returns:
        -1 if None, otherwise the integer value
    """
    if idx is None:
        return -1
    if isinstance(idx, torch.Tensor):
        assert idx.numel() == 1, (
            f"{name} tensor must have exactly 1 element (batch_size==1), "
            f"got {idx.numel()}"
        )
        return int(idx.item())
    return int(idx)


COLOR_PARAMS_PER_FRAME = 8
CRF_PARAMS_PER_CHANNEL = 4
VIGNETTING_PARAMS_PER_CHANNEL = 2 + NUM_VIGNETTING_ALPHA_TERMS


# =============================================================================
# PPISP Configuration
# =============================================================================

@dataclass
class PPISPConfig:
    """Configuration for PPISP module.

    Includes regularization weights, optimizer settings, scheduler settings,
    and controller activation settings.
    All defaults are tuned for typical radiance field training scenarios.
    """

    # Controller settings
    use_controller: bool = True
    """Enable the controller for predicting per-frame corrections.
    When False, zero corrections are used for novel views.
    """

    controller_distillation: bool = True
    """Use distillation to train the controller.
    When True, the controller is trained with freezing the other PPISP parameters and detaching
    the input so that gradients only flow through the controller.
    """

    controller_activation_ratio: float = 0.8
    """Relative training step at which to activate the controller.
    Controller activates when step >= controller_activation_ratio * max_optimization_iters.
    Default 0.8 means the controller activates at 80% of training.
    """

    # Regularization weights
    exposure_mean: float = 1.0
    """Encourage exposure mean ~ 0 to resolve SH <-> exposure ambiguity."""

    vig_center: float = 0.02
    """Encourage vignetting optical center near image center."""

    vig_channel: float = 0.1
    """Encourage similar vignetting across RGB channels."""

    vig_non_pos: float = 0.01
    """Penalize positive vignetting alpha coefficients (should be <= 0)."""

    color_mean: float = 1.0
    """Encourage color correction mean ~ 0 across frames."""

    crf_channel: float = 0.1
    """Encourage similar CRF parameters across RGB channels."""

    radial_attenuation_knots: int = 0
    """GOFish wide-angle extension: knots of the per-camera nonincreasing radial profile w(rho) (0 = released PPISP)."""

    radial_attenuation_smoothness: float = 0.0001
    """Second-difference prior on the radial log-attenuation knots."""

    flatfield_grid: int = 0
    """GOFish wide-angle extension: per-camera flat-field log-attenuation grid a(u) (cells per max(W,H); 0 = off)."""

    # Optimizer settings for PPISP main params
    ppisp_lr: float = 0.002
    """Learning rate for PPISP main parameters."""

    ppisp_eps: float = 1e-15
    """Adam epsilon for PPISP main parameters."""

    ppisp_betas: tuple[float, float] = (0.9, 0.999)
    """Adam betas for PPISP main parameters."""

    # Optimizer settings for controller params
    controller_lr: float = 0.001
    """Learning rate for controller parameters (fixed, no scheduler)."""

    controller_eps: float = 1e-15
    """Adam epsilon for controller parameters."""

    controller_betas: tuple[float, float] = (0.9, 0.999)
    """Adam betas for controller parameters."""

    # Scheduler settings (PPISP main params only)
    scheduler_type: str = "linear_exp"
    """Scheduler type: 'linear_exp' for linear warmup + exponential decay."""

    scheduler_base_lr: float = 0.002
    """Base learning rate for scheduler (should match ppisp_lr)."""

    scheduler_warmup_steps: int = 500
    """Number of warmup steps for linear warmup phase."""

    scheduler_start_factor: float = 0.01
    """Starting factor for warmup (lr starts at start_factor * base_lr)."""

    scheduler_decay_max_steps: int = 30000
    """Decay horizon of the PPISP main parameters (the joint scene steps)."""

    scheduler_final_factor: float = 0.01
    """Final factor for decay (lr decays toward final_factor * base_lr)."""


# Default configuration
DEFAULT_PPISP_CONFIG = PPISPConfig()


# =============================================================================
# Autograd Functions
# =============================================================================

class _PPISPFunction(torch.autograd.Function):
    """Custom autograd function for PPISP forward/backward."""

    @staticmethod
    def forward(
        ctx,
        exposure_params: torch.Tensor,
        vignetting_params: torch.Tensor,
        color_params: torch.Tensor,
        crf_params: torch.Tensor,
        rgb_in: torch.Tensor,
        pixel_coords: torch.Tensor | None,
        resolution_w: int,
        resolution_h: int,
        camera_idx: int,
        frame_idx: int,
    ) -> torch.Tensor:
        rgb_out = _C.ppisp_forward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            pixel_coords,
            resolution_w,
            resolution_h,
            camera_idx,
            frame_idx,
        )

        # The backward recomputes the pipeline from rgb_in, so rgb_out is not saved.
        ctx.save_for_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, pixel_coords
        )
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h
        ctx.camera_idx = camera_idx
        ctx.frame_idx = frame_idx

        return rgb_out

    @staticmethod
    def backward(ctx, v_rgb_out: torch.Tensor):
        (exposure_params, vignetting_params,
         color_params, crf_params, rgb_in, pixel_coords) = ctx.saved_tensors

        (v_exposure_params, v_vignetting_params,
         v_color_params, v_crf_params, v_rgb_in) = _C.ppisp_backward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            pixel_coords,
            v_rgb_out.contiguous(),
            ctx.resolution_w,
            ctx.resolution_h,
            ctx.camera_idx,
            ctx.frame_idx,
        )

        return (
            v_exposure_params,
            v_vignetting_params,
            v_color_params,
            v_crf_params,
            v_rgb_in,
            None,  # pixel_coords
            None,  # resolution_w
            None,  # resolution_h
            None,  # camera_idx
            None,  # frame_idx
        )


def _as_float_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` unchanged when it is already float32 and contiguous.

    The dtype and layout checks are plain attribute reads; ``.float()`` and
    ``.contiguous()`` each go through the dispatcher even when they return the
    input, and the PPISP parameters are float32 and contiguous by construction.
    """
    if tensor.dtype is torch.float32 and tensor.is_contiguous():
        return tensor
    return tensor.float().contiguous()


def _as_float2_aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``_as_float_contiguous`` plus the 8-byte alignment of the kernels' float2 loads.

    The extension also copies a misaligned view, but only for its own call;
    copying here once lets the forward, the backward and the saved tensor share
    the aligned copy.
    """
    tensor = _as_float_contiguous(tensor)
    if tensor.data_ptr() % 8 != 0:
        return tensor.clone()
    return tensor


class _PPISPRegularizationFunction(torch.autograd.Function):
    """Custom autograd function for the PPISP regularization loss.

    ``weights`` is one tuple of six floats in the order
    ``(exposure_mean, vig_center, vig_channel, vig_non_pos, color_mean,
    crf_channel)`` so the autograd machinery handles five arguments instead
    of ten every step.
    """

    @staticmethod
    def forward(
        ctx,
        exposure_params: torch.Tensor,
        vignetting_params: torch.Tensor,
        color_params: torch.Tensor,
        crf_params: torch.Tensor,
        weights: tuple[float, float, float, float, float, float],
    ) -> torch.Tensor:
        exposure_params = _as_float_contiguous(exposure_params)
        vignetting_params = _as_float_contiguous(vignetting_params)
        color_params = _as_float_contiguous(color_params)
        crf_params = _as_float_contiguous(crf_params)

        loss, frame_mean_sums = _C.ppisp_regularization_forward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            *weights,
        )

        ctx.save_for_backward(
            exposure_params, vignetting_params, color_params, crf_params, frame_mean_sums
        )
        ctx.weights = weights
        return loss

    @staticmethod
    def backward(ctx, v_loss: torch.Tensor):
        exposure_params, vignetting_params, color_params, crf_params, frame_mean_sums = (
            ctx.saved_tensors
        )

        # The wrapper makes v_loss contiguous; a 0-d tensor already is.
        grads = _C.ppisp_regularization_backward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            v_loss,
            frame_mean_sums,
            *ctx.weights,
        )

        return grads + (None,)


# =============================================================================
# Public Functions
# =============================================================================

def ppisp_apply(
    exposure_params: torch.Tensor,
    vignetting_params: torch.Tensor,
    color_params: torch.Tensor,
    crf_params: torch.Tensor,
    rgb_in: torch.Tensor,
    pixel_coords: torch.Tensor | None,
    resolution_w: int,
    resolution_h: int,
    camera_idx: torch.Tensor | int | None = None,
    frame_idx: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Apply PPISP processing pipeline to input RGB.

    Tensor shapes are flattened on input and restored in the output. The last dimension
    of rgb_in must be 3 (RGB channels), and the last dimension of pixel_coords
    must be 2 (x, y).

    Args:
        exposure_params: Per-frame exposure [num_frames] or [1] from controller
        vignetting_params: Per-camera vignetting [num_cameras, 3, 5]
        color_params: Per-frame color correction [num_frames, 8]
        crf_params: Per-camera CRF [num_cameras, 3, 4]
        rgb_in: Input RGB [H, W, 3] or [N, 3]
        pixel_coords: Pixel coordinates [H, W, 2], [N, 2], or None. Ignored when
            camera_idx is None. When None with a camera, pixel centers are derived
            from the resolution.
        resolution_w: Image width
        resolution_h: Image height
        camera_idx: Camera index (Tensor, int, or None). None disables per-camera effects.
        frame_idx: Frame index (Tensor, int, or None). None disables per-frame effects.

    Returns:
        Processed RGB [H, W, 3] or [N, 3] - same shape as rgb_in
    """
    # Normalize indices: Tensor/int/None -> int (-1 for None)
    camera_idx = _normalize_index(camera_idx, "camera_idx")
    frame_idx = _normalize_index(frame_idx, "frame_idx")

    # Store original shape for restoring output
    original_shape = rgb_in.shape

    # Flatten tensors for processing and assert correct dimensions
    rgb_flat = rgb_in.view(-1, rgb_in.shape[-1])
    assert rgb_flat.shape[-1] == 3, f"Expected 3 RGB channels, got {rgb_flat.shape[-1]}"
    if camera_idx == -1:
        # Only the per-camera effects read coordinates; drop them so they are
        # neither converted nor saved for the backward.
        coords_flat = None
    elif pixel_coords is None:
        coords_flat = None
        assert rgb_flat.shape[0] == resolution_w * resolution_h, (
            f"resolution must be consistent with num_pixels in rgb, "
            f"{resolution_w * resolution_h} vs {rgb_flat.shape[0]}"
        )
    else:
        coords_flat = pixel_coords.view(-1, pixel_coords.shape[-1])
        assert coords_flat.shape[-1] == 2, (
            f"Expected 2D coords, got {coords_flat.shape[-1]}"
        )
        assert rgb_flat.shape[0] == coords_flat.shape[0], (
            f"rgb and pixel_coords must have same num_pixels after flattening, "
            f"got {rgb_flat.shape[0]} vs {coords_flat.shape[0]}"
        )

    # Convert to float32 and ensure contiguous memory layout
    exposure_params = _as_float_contiguous(exposure_params)
    vignetting_params = _as_float_contiguous(vignetting_params)
    color_params = _as_float2_aligned(color_params)
    crf_params = _as_float_contiguous(crf_params)
    rgb_flat = _as_float_contiguous(rgb_flat)
    if coords_flat is not None:
        coords_flat = _as_float2_aligned(coords_flat)

    rgb_out = _PPISPFunction.apply(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_flat,
        coords_flat,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    )

    # Restore output to original shape
    return rgb_out.view(original_shape)


# =============================================================================
# Controller Network
# =============================================================================

def _masked_max_pool2d(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    pool: nn.MaxPool2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-pool over valid samples only; empty windows are exact finite zeros."""
    valid = valid_mask.to(device=values.device, dtype=torch.bool)
    if valid.ndim != 4 or valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
        raise RuntimeError("PPISP max-pool mask shape differs")
    kwargs = {
        "kernel_size": pool.kernel_size,
        "stride": pool.stride,
        "padding": pool.padding,
        "dilation": pool.dilation,
        "ceil_mode": pool.ceil_mode,
    }
    if pool.dilation not in (1, (1, 1)):
        raise RuntimeError("PPISP masked support requires unit-dilation MaxPool")
    masked = torch.where(
        valid.expand_as(values), values, torch.full_like(values, -torch.inf)
    )
    pooled = F.max_pool2d(masked, **kwargs)
    pooled_support = F.avg_pool2d(
        valid.to(values.dtype), kernel_size=pool.kernel_size,
        stride=pool.stride, padding=pool.padding, ceil_mode=pool.ceil_mode,
        count_include_pad=False,
    )
    pooled_valid = pooled_support > 0
    pooled = torch.where(
        pooled_valid.expand_as(pooled), pooled, torch.zeros_like(pooled)
    )
    return pooled, pooled_support


def _valid_support_adaptive_average(
    values: torch.Tensor,
    support: torch.Tensor,
    output_size: tuple[int, int],
    denominator: torch.Tensor | None = None,
    contiguous: bool = False,
) -> torch.Tensor:
    """Area-weighted valid-support average, zero where support is empty."""
    weighted = values * support
    if contiguous:
        # Real-time renderer: NCHW-contiguous inputs select PyTorch's faster NCHW adaptive-average kernel
        # (cuDNN's 1x1 convolutions can return channels-last tensors). Training and the exact renderer keep
        # the reference layout.
        weighted, support = weighted.contiguous(), support.contiguous()
    numerator = F.adaptive_avg_pool2d(weighted, output_size)
    if denominator is None:  # depends only on the mask; the real-time renderer passes it precomputed per mask
        denominator = F.adaptive_avg_pool2d(support, output_size)
    positive = denominator > 0
    safe = torch.where(positive, denominator, torch.ones_like(denominator))
    return torch.where(
        positive.expand_as(numerator), numerator / safe, torch.zeros_like(numerator)
    )


class _PPISPController(nn.Module):
    """CNN-based controller that predicts exposure and color params from rendered radiance images.

    Fixed architecture:
    - CNN feature extraction with 1x1 convolutions
    - Adaptive average pooling to fixed spatial grid
    - MLP head for exposure + color parameter prediction
    - Uses prior exposure (from EXIF metadata) as additional input

    Input: Rendered radiance image + prior exposure
    Output: Exposure scalar + 8 color correction parameters
    """

    def __init__(
        self,
        input_downsampling: int = 3,
        cnn_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        pool_grid_size: tuple = (5, 5),
        device: str = "cuda",
    ):
        super().__init__()
        self.pool_grid_size = tuple(pool_grid_size)

        # CNN encoder: 1x1 convolutions for per-pixel feature extraction
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, device=device),
            nn.MaxPool2d(kernel_size=input_downsampling,
                         stride=input_downsampling),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cnn_feature_dim, kernel_size=1, device=device),
            nn.AdaptiveAvgPool2d(pool_grid_size),
            nn.Flatten(),
        )

        # Input dimension: CNN features + prior exposure
        cnn_output_dim = cnn_feature_dim * \
            pool_grid_size[0] * pool_grid_size[1]
        input_dim = cnn_output_dim + 1  # +1 for prior_exposure

        # MLP trunk (shared hidden layers)
        trunk_layers = [nn.Linear(input_dim, hidden_dim, device=device),
                        nn.ReLU(inplace=True)]
        for _ in range(num_mlp_layers - 1):
            trunk_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim, device=device), nn.ReLU(inplace=True)])
        self.mlp_trunk = nn.Sequential(*trunk_layers)

        # Separate output heads (avoids misaligned views from slicing)
        self.exposure_head = nn.Linear(hidden_dim, 1, device=device)
        self.color_head = nn.Linear(
            hidden_dim, COLOR_PARAMS_PER_FRAME, device=device)

    def forward(
        self,
        rgb: torch.Tensor,
        prior_exposure: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple:
        """Predict exposure and color parameters from rendered radiance image.

        Args:
            rgb: Rendered radiance image [H, W, 3]
            prior_exposure: Prior exposure from EXIF [1] (defaults to zero if not provided)

        Returns:
            exposure: Predicted exposure scalar
            color_params: Predicted color parameters [8]
        """
        # Default prior exposure to zero if not provided
        if prior_exposure is None:
            prior_exposure = torch.zeros(1, device=rgb.device)

        # Extract CNN features
        image = rgb.permute(2, 0, 1).unsqueeze(0).detach()
        if valid_mask is None:
            features = self.cnn_encoder(image)  # released PPISP controller
        else:
            # Real-time renderer only (no autograd): the mask-derived tensors depend only on the camera's
            # domain mask, so they are prepared once per mask tensor; the per-frame operations are unchanged.
            realtime = bool(getattr(self, "realtime", False)) and not torch.is_grad_enabled()
            ck = None
            if realtime:
                ck = (valid_mask.data_ptr(), tuple(valid_mask.shape), str(valid_mask.device), str(rgb.device))
                cache = self.__dict__.setdefault("_rt_mask_cache", {})
            prep = cache.get(ck) if ck is not None else None
            if prep is None:
                mask = valid_mask.to(device=rgb.device)
                if mask.ndim == 4:
                    mask = mask.permute(0, 3, 1, 2)
                elif mask.ndim == 2:
                    mask = mask[None, None]
                if mask.shape != (1, 1, rgb.shape[0], rgb.shape[1]):
                    raise RuntimeError("PPISP valid mask shape differs from rendered raster")
                valid = mask.to(torch.bool)
                if not bool(valid.any()):
                    raise RuntimeError("PPISP controller mask is empty")
                # The mask itself is kept in the entry so that its storage (the key) cannot be reused.
                prep = (valid, valid.expand_as(image), valid_mask)
                if ck is not None:
                    cache[ck] = prep
            valid, valid3, _ = prep
            # Conv1 is pointwise.  Replace invalid RGB before it, then exclude
            # invalid Conv1 activations (including bias) from the 3x3 max.
            masked_image = torch.where(valid3, image, torch.zeros_like(image))
            conv1 = self.cnn_encoder[0](masked_image)
            pooled, support = _masked_max_pool2d(conv1, valid, self.cnn_encoder[1])
            # The remaining convolutions are 1x1, so an empty pooled cell cannot
            # leak spatially.  Its eventual support is exactly zero.
            encoded = self.cnn_encoder[2:6](pooled)
            den = None
            if ck is not None:
                den = cache.get((ck, "den"))
                if den is None:
                    den = F.adaptive_avg_pool2d(support.to(encoded.dtype).contiguous(), self.pool_grid_size)
                    cache[(ck, "den")] = den
            features = _valid_support_adaptive_average(
                encoded, support.to(encoded.dtype), self.pool_grid_size, den, contiguous=realtime
            ).flatten(1)
        features = torch.cat([features.squeeze(0), prior_exposure], dim=0)
        hidden = self.mlp_trunk(features)
        return self.exposure_head(hidden).squeeze(-1), self.color_head(hidden)

# =============================================================================
# Main PPISP Module
# =============================================================================


class PPISP(nn.Module):
    """Physically-Plausible Image Signal Processing module with integrated controller.

    Combines learned ISP parameters with a CNN-based controller for
    predicting exposure and color correction from rendered radiance images.

    The controller automatically activates at a specified fraction of training
    (controller_activation_ratio * max_optimization_iters). When activated,
    per-camera vignetting and CRF parameters are frozen.

    Note: Call create_schedulers() before forward() to set up the scheduler
    reference used for controller activation timing.

    Args:
        num_cameras: Number of cameras in the scene
        num_frames: Total number of frames across all cameras
        config: PPISP configuration (regularization, optimizer, scheduler settings)
    """

    wa_order = "sequential"  # float association of the wide-angle attenuation, see wide_angle_factors

    def __init__(
        self,
        num_cameras: int,
        num_frames: int,
        config: PPISPConfig = DEFAULT_PPISP_CONFIG,
        radial_attenuation_geometry: torch.Tensor | None = None,
    ):
        """``radial_attenuation_geometry``: ``[num_cameras, 3]`` rows ``(pp_x, pp_y, rho_ap)`` of the
        Wide-Angle Appearance Modeling (``gofish_ppisp.wide_angle.geometry_from``); required when
        ``config.radial_attenuation_knots > 0`` and unused otherwise."""
        super().__init__()
        if int(config.flatfield_grid) > 0 and int(config.radial_attenuation_knots) <= 0:
            raise ValueError("flatfield_grid > 0 needs radial_attenuation_knots > 0 (the flat field shares its gauge)")

        self.config = config

        # Warn if controller is enabled but will never train (ratio >= 1.0)
        if config.use_controller and config.controller_activation_ratio >= 1.0:
            print(
                f"[PPISP] Warning: controller_activation_ratio="
                f"{config.controller_activation_ratio} >= 1.0. "
                "Controller will not be trained. "
                "Zero per-frame corrections will be used for novel views (frame_idx=-1)."
            )

        # Scheduler reference (set by create_schedulers, not serialized)
        self._ppisp_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # Controller activation step (set by create_schedulers)
        # -1 means not yet set
        self._controller_activation_step: int = -1

        # Per-frame exposure offset (in log space)
        self.exposure_params = nn.Parameter(
            torch.zeros(num_frames, device="cuda")
        )

        # Per-camera vignetting: [center_x, center_y, alpha_0, alpha_1, alpha_2] per channel
        # Per-channel vignetting [num_cameras, 3, 5]
        self.vignetting_params = nn.Parameter(
            torch.zeros(num_cameras, 3,
                        VIGNETTING_PARAMS_PER_CHANNEL, device="cuda")
        )

        # Per-frame color params [num_frames, 8]
        # [db_r, db_g, dr_r, dr_g, dg_r, dg_g, dgray_r, dgray_g]
        self.color_params = nn.Parameter(
            torch.zeros(num_frames, COLOR_PARAMS_PER_FRAME, device="cuda")
        )

        # Per-camera CRF: [toe_raw, shoulder_raw, gamma_raw, center_raw] per channel
        # Initialize to identity-like response
        def softplus_inverse(x: float, min_value: float = 0.0, epsilon: float = 1e-5) -> float:
            clamped_value = max(epsilon, x - min_value)
            return float(torch.log(torch.expm1(torch.tensor(clamped_value))))

        crf_raw = torch.zeros(CRF_PARAMS_PER_CHANNEL, device="cuda")
        crf_raw[0] = softplus_inverse(1.0, min_value=0.3)  # toe
        crf_raw[1] = softplus_inverse(1.0, min_value=0.3)  # shoulder
        crf_raw[2] = softplus_inverse(1.0, min_value=0.1)  # gamma
        crf_raw[3] = 0.0  # center_raw -> sigmoid(0) = 0.5

        self.crf_params = nn.Parameter(
            crf_raw.view(1, 1, CRF_PARAMS_PER_CHANNEL)
            .repeat(num_cameras, 3, 1)
            .contiguous()
        )

        # GOFish wide-angle extension: per-camera nonincreasing radial profile (identity at init).
        if int(config.radial_attenuation_knots) > 0:
            self.radial_attenuation_params = nn.Parameter(
                torch.zeros(num_cameras, int(config.radial_attenuation_knots), device="cuda"))
            if radial_attenuation_geometry is None:
                raise ValueError("radial_attenuation_knots > 0 needs radial_attenuation_geometry "
                                 "(threedgrut.datasets.wide_angle.wide_angle_geometry)")
            geometry = torch.as_tensor(radial_attenuation_geometry, dtype=torch.float32).reshape(num_cameras, 3)
            self.register_buffer("radial_attenuation_geometry", geometry.to(device="cuda", dtype=torch.float32))
        if int(config.radial_attenuation_knots) > 0 and int(config.flatfield_grid) > 0:
            gn = int(config.flatfield_grid)
            self.flatfield_params = nn.Parameter(torch.zeros(num_cameras, gn, gn, device="cuda"))
            self.register_buffer("flatfield_free", torch.ones(num_cameras, gn, gn, device="cuda"))

        # ZCA pinv block-diagonal matrix for color loss computation (constant, non-persistent)
        # 8x8 block-diagonal: each 2x2 block transforms latent (dr, dg) to real chromaticity offsets
        # Order: [Blue, Red, Green, Neutral]
        self.register_buffer(
            "color_pinv_block_diag",
            _COLOR_PINV_BLOCK_DIAG.to(device="cuda"),
            persistent=False,
        )

        # Controllers for predicting per-frame corrections
        if config.use_controller:
            self.controllers = nn.ModuleList([
                _PPISPController() for _ in range(num_cameras)
            ])
        else:
            self.controllers = nn.ModuleList()

    @property
    def num_cameras(self) -> int:
        """Number of cameras, inferred from crf_params shape."""
        return self.crf_params.shape[0]

    @property
    def num_frames(self) -> int:
        """Number of frames, inferred from exposure_params shape."""
        return self.exposure_params.shape[0]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        config: PPISPConfig = DEFAULT_PPISP_CONFIG,
    ) -> "PPISP":
        """Create PPISP instance from a state dict.

        Infers num_cameras and num_frames from parameter shapes in the state dict.
        This enables checkpoint restoration without needing to store dimensions explicitly.

        Args:
            state_dict: State dict from a saved PPISP module (via state_dict())
            config: PPISP configuration (default: DEFAULT_PPISP_CONFIG)

        Returns:
            New PPISP instance with loaded state
        """
        # Infer dimensions from parameter shapes
        # Use crf_params for num_cameras
        num_cameras = state_dict["crf_params"].shape[0]
        num_frames = state_dict["exposure_params"].shape[0]

        # Create instance with inferred dimensions
        if "radial_attenuation_params" in state_dict:
            import dataclasses
            config = dataclasses.replace(
                config, radial_attenuation_knots=int(state_dict["radial_attenuation_params"].shape[1]),
                flatfield_grid=int(state_dict["flatfield_params"].shape[1]) if "flatfield_params" in state_dict else 0)
        instance = cls(
            num_cameras=num_cameras,
            num_frames=num_frames,
            config=config,
            radial_attenuation_geometry=state_dict.get("radial_attenuation_geometry"),
        )
        instance.load_state_dict(state_dict)

        return instance

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor | None = None,
        resolution: tuple[int, int] | None = None,
        camera_idx: torch.Tensor | int | None = None,
        frame_idx: torch.Tensor | int | None = None,
        exposure_prior: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply PPISP processing to input RGB.

        Tensor shapes are flattened on input and restored in the output.
        The last dimension of rgb must be 3 (RGB channels), and the last
        dimension of pixel_coords must be 2 (x, y).

        Args:
            rgb: Input RGB [H, W, 3] or [N, 3]
            pixel_coords: Pixel coordinates [H, W, 2] or [N, 2]. Uses pixel centers when None.
            resolution: Image resolution as (width, height). Inferred from rgb when None.
            camera_idx: Camera index (Tensor, int, or None). None disables per-camera effects.
            frame_idx: Frame index (Tensor, int, or None). None disables per-frame effects.
            exposure_prior: Prior exposure value [1] (defaults to zero if not provided)
            valid_mask: Optional boolean mask of the camera's calibrated domain on the rgb raster;
                the controller pools only over these pixels.  None pools over the whole raster.

        Returns:
            Processed RGB [H, W, 3] or [N, 3] - same shape as rgb
        """
        if resolution is None or pixel_coords is None:
            assert rgb.dim() == 3, (
                f"resolution and pixel_coords can only be inferred from [H, W, 3] input (got shape {rgb.shape})"
            )

        if resolution is None:
            resolution_w, resolution_h = rgb.shape[1], rgb.shape[0]
        else:
            resolution_w, resolution_h = resolution
            if pixel_coords is None:
                assert resolution_w == rgb.shape[1] and resolution_h == rgb.shape[0], (
                    f"Explicit resolution ({resolution_w}x{resolution_h}) must match "
                    f"rgb shape ({rgb.shape[1]}x{rgb.shape[0]}) when pixel_coords is None"
                )

        # Normalize indices: Tensor/int/None -> int (-1 for None)
        camera_idx_int = _normalize_index(camera_idx, "camera_idx")
        frame_idx_int = _normalize_index(frame_idx, "frame_idx")

        # Validate indices against parameter dimensions
        if camera_idx_int != -1:
            assert 0 <= camera_idx_int < self.num_cameras, f"Invalid camera_idx: {camera_idx_int}"
        if frame_idx_int != -1:
            assert 0 <= frame_idx_int < self.num_frames, f"Invalid frame_idx: {frame_idx_int}"

        # Check if controller is trained (enabled AND step threshold reached)
        # Controller is not trained if: disabled, ratio >= 1, or training hasn't reached
        # activation step yet. During inference (no scheduler), assume controller is ready.
        if self._ppisp_scheduler is None:
            # Inference mode: controller is ready if enabled and exists
            controller_trained = (
                self.config.use_controller
                and self.config.controller_activation_ratio < 1.0
                and len(self.controllers) > 0
            )
        else:
            # Training mode: check activation step threshold
            controller_trained = (
                self.config.use_controller
                and self.config.controller_activation_ratio < 1.0
                and self._controller_activation_step >= 0
                and self._ppisp_scheduler.last_epoch >= self._controller_activation_step
            )

        if controller_trained and self.config.controller_distillation:
            # Distillation: freeze the other PPISP parameters
            self.exposure_params.requires_grad = False
            self.vignetting_params.requires_grad = False
            self.color_params.requires_grad = False
            self.crf_params.requires_grad = False
            if hasattr(self, "radial_attenuation_params"):
                self.radial_attenuation_params.requires_grad = False
            if hasattr(self, "flatfield_params"):
                self.flatfield_params.requires_grad = False
            # Distillation: detach the input
            rgb = rgb.detach()

        # Determine if we should apply correction overrides
        # Overrides are used for:
        # 1. Novel views (frame_idx=-1): either controller predictions or zeros
        # 2. Training after controller activation: controller predictions
        is_novel_view = (camera_idx_int != -1 and frame_idx_int == -1)
        apply_correction_override = is_novel_view or (
            controller_trained and camera_idx_int != -1)

        if apply_correction_override:
            if is_novel_view and not controller_trained:
                # Novel view with untrained controller: apply identity corrections
                exposure = torch.zeros(1, device=rgb.device, dtype=rgb.dtype)
                # Homography offsets, identity is zeros
                color = torch.zeros(
                    1, COLOR_PARAMS_PER_FRAME, device=rgb.device, dtype=rgb.dtype
                )
            else:
                # Use controller predictions
                # View rgb as [H, W, 3] image for controller
                rgb_image = rgb.view(resolution_h, resolution_w, 3)

                # Predict exposure and color params using controller
                controller = self.controllers[camera_idx_int]
                exposure_pred, color_pred = controller(
                    rgb_image, exposure_prior, valid_mask=valid_mask)

                # Use 1-element tensors for predictions to save memory bandwidth
                exposure = exposure_pred.unsqueeze(0)  # [1]
                color = color_pred.unsqueeze(0)  # [1, 8]

            # When override is applied, kernel always reads from index 0
            frame_idx_for_kernel = 0
        else:
            exposure = self.exposure_params
            color = self.color_params
            frame_idx_for_kernel = frame_idx_int

        rgb = self.wide_angle_attenuate(rgb, pixel_coords, camera_idx_int, resolution_w, resolution_h)

        return ppisp_apply(
            exposure,
            self.vignetting_params,
            color,
            self.crf_params,
            rgb,
            pixel_coords,
            resolution_w,
            resolution_h,
            camera_idx_int,
            frame_idx_for_kernel,
        )

    def wide_angle_factors(self, pixel_coords, camera_idx_int, resolution_w, resolution_h):
        """Per-pixel factors of the GOFish wide-angle PPISP attenuation W of one camera (None for released checkpoints).
        Float association (``self.wa_order``, set from ``render.wa_order`` for test renders): "sequential" (default) =
        (rgb * W_r) * W_ff, as in training; "product" = rgb * (W_r * W_ff), as used for the paper's FIORD test renders.
        W depends only on the camera and the pixel coordinates, so callers may cache the factors per camera."""
        if not (hasattr(self, "radial_attenuation_params") and camera_idx_int != -1):
            return None
        if pixel_coords is None:
            pixel_coords = pixel_centres(resolution_w, resolution_h, device=self.radial_attenuation_params.device)
        order = self.wa_order
        pix = pixel_coords.reshape(-1, 2)
        radial = radial_attenuation(
            self.radial_attenuation_params[camera_idx_int], self.radial_attenuation_geometry[camera_idx_int],
            pix, max(resolution_w, resolution_h))
        flat = (flatfield_attenuation(self.flatfield_params[camera_idx_int], self.flatfield_free[camera_idx_int],
                                      pix, resolution_w, resolution_h) if hasattr(self, "flatfield_params") else None)
        if order == "sequential":
            return ("sequential", radial, flat)
        if order != "product":
            raise ValueError(f"unknown wa_order {order!r}")
        wm = torch.ones(pix.shape[0], 1, device=pix.device) * radial
        if flat is not None:
            wm = wm * flat
        return ("product", wm, None)

    @staticmethod
    def apply_wide_angle_factors(rgb, factors):
        """rgb * W with the factors of wide_angle_factors (identity for None)."""
        if factors is None:
            return rgb
        order, a, b = factors
        if order == "sequential":
            rgb = rgb * a.reshape(rgb.shape[:-1] + (1,)).to(rgb.dtype)
            if b is not None:
                rgb = rgb * b.reshape(rgb.shape[:-1] + (1,)).to(rgb.dtype)
            return rgb
        xin = rgb.contiguous().view(-1, 3)
        return (xin * a.to(xin.dtype)).view(rgb.shape)

    def wide_angle_attenuate(self, rgb, pixel_coords, camera_idx_int, resolution_w, resolution_h):
        """GOFish wide-angle PPISP: rgb * W(p) (radial [x flat-field]) entering ppisp_apply, after the
        controller saw the un-attenuated radiance; identity for released checkpoints (no extension keys)."""
        return self.apply_wide_angle_factors(
            rgb, self.wide_angle_factors(pixel_coords, camera_idx_int, resolution_w, resolution_h))

    def get_regularization_loss(self) -> torch.Tensor:
        """Compute weighted regularization loss for PPISP parameters.

        Regularization terms:
            - Exposure mean regularization
            - Vignetting center/channel/non-positivity regularization
            - Color mean regularization
            - CRF channel variance regularization

        Returns:
            Single scalar tensor with the total weighted regularization loss.
        """
        cfg = self.config
        reg = _PPISPRegularizationFunction.apply(
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
            (
                cfg.exposure_mean,
                cfg.vig_center,
                cfg.vig_channel,
                cfg.vig_non_pos,
                cfg.color_mean,
                cfg.crf_channel,
            ),
        )
        if hasattr(self, "radial_attenuation_params") and float(cfg.radial_attenuation_smoothness) > 0.0:
            dec = _radial_decrements(self.radial_attenuation_params)
            dec = torch.cat([dec.new_zeros(dec.shape[0], 1), dec], 1)
            reg = reg + float(cfg.radial_attenuation_smoothness) * (dec[:, 1:] - dec[:, :-1]).square().sum()
        if hasattr(self, "flatfield_params") and float(cfg.radial_attenuation_smoothness) > 0.0:
            a = _radial_decrements(self.flatfield_params) * self.flatfield_free
            reg = reg + float(cfg.radial_attenuation_smoothness) * (
                (a[:, 1:, :] - a[:, :-1, :]).square().sum() + (a[:, :, 1:] - a[:, :, :-1]).square().sum())
        return reg

    def create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create optimizers for PPISP parameters.

        Returns Adam optimizers:
        1. PPISP main params optimizer (exposure, vignetting, color, CRF)
        2. Controller params optimizer (neural network weights) - only if use_controller=True

        Returns:
            List of optimizers: [ppisp_optimizer] or [ppisp_optimizer, controller_optimizer]
        """
        cfg = self.config

        # PPISP main parameters
        ppisp_params = [
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
        ]
        if hasattr(self, "radial_attenuation_params"):
            ppisp_params.append(self.radial_attenuation_params)
        if hasattr(self, "flatfield_params"):
            ppisp_params.append(self.flatfield_params)

        ppisp_optimizer = torch.optim.Adam(
            ppisp_params,
            lr=cfg.ppisp_lr,
            eps=cfg.ppisp_eps,
            betas=cfg.ppisp_betas,
        )

        optimizers = [ppisp_optimizer]

        # Controller parameters (only if controller is enabled)
        if cfg.use_controller and len(self.controllers) > 0:
            controller_optimizer = torch.optim.Adam(
                self.controllers.parameters(),
                lr=cfg.controller_lr,
                eps=cfg.controller_eps,
                betas=cfg.controller_betas,
            )
            optimizers.append(controller_optimizer)

        return optimizers

    def create_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        max_optimization_iters: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate schedulers for the optimizers.

        Only the PPISP main params optimizer gets a scheduler (linear warmup + exp decay).
        The controller optimizer uses a fixed learning rate and has no scheduler.

        This method also stores the scheduler reference internally so that forward()
        can read the current step for controller activation timing.

        Args:
            optimizers: List of optimizers from create_optimizers()
                        [ppisp_optimizer] or [ppisp_optimizer, controller_optimizer]
            max_optimization_iters: Total number of optimization steps for training.
                        Used to compute controller activation step.

        Returns:
            List containing the PPISP scheduler.
        """
        cfg = self.config
        ppisp_optimizer = optimizers[0]

        # Compute controller activation step
        self._controller_activation_step = int(
            cfg.controller_activation_ratio * max_optimization_iters)

        # Compute decay gamma for exponential decay phase
        # After decay_max_steps, lr should be final_factor * base_lr
        gamma = cfg.scheduler_final_factor ** (1.0 /
                                               cfg.scheduler_decay_max_steps)

        # Linear warmup + exponential decay using built-in schedulers
        ppisp_scheduler = torch.optim.lr_scheduler.SequentialLR(
            ppisp_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    ppisp_optimizer,
                    start_factor=cfg.scheduler_start_factor,
                    total_iters=cfg.scheduler_warmup_steps,
                ),
                torch.optim.lr_scheduler.ExponentialLR(
                    ppisp_optimizer,
                    gamma=gamma,
                ),
            ],
            milestones=[cfg.scheduler_warmup_steps],
        )

        # Store reference so forward() can read current step
        self._ppisp_scheduler = ppisp_scheduler

        return [ppisp_scheduler]
