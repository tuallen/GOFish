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

import torch

# Real spherical-harmonics constants (as in 3D Gaussian Splatting and PlenOctrees).
C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def apply_background(background, outputs: dict, gpu_batch, training: bool = False) -> dict:
    """Composite the background behind the rendered RGB."""
    if background is None or outputs["pred_features"].shape[-1] != 3:
        return outputs
    pred_features, pred_opacity = background(
        gpu_batch.T_to_world.contiguous(),
        gpu_batch.rays_dir.contiguous(),
        outputs["pred_features"],
        outputs["pred_opacity"],
        training,
    )
    outputs["pred_features"] = pred_features
    return outputs


def install_physical_fov_gradient_gate(outputs: dict, gpu_batch) -> list:
    """Zero the gradients of invalid pixels of every raw render output before
    the renderer backward (and its absolute-gradient accumulation)."""
    mask = getattr(gpu_batch, "training_valid_mask", None)
    if mask is None:
        mask = getattr(gpu_batch, "physical_fov_mask", None)
    if mask is None:
        return []
    handles = []
    for value in outputs.values():
        if not torch.is_tensor(value) or not value.requires_grad or value.ndim != 4:
            continue
        if value.shape[:3] != mask.shape[:3]:
            continue
        gate = mask.to(device=value.device, dtype=value.dtype).expand_as(value)
        handles.append(value.register_hook(lambda gradient, gate=gate: gradient * gate))
    gpu_batch.physical_fov_gradient_gate_installed = True
    return handles


def apply_post_processing(
    post_processing,
    outputs: dict,
    gpu_batch,
    training: bool = False,
) -> dict:
    """Apply post-processing to rendered output.

    Args:
        post_processing: Post-processing module
        outputs: Model outputs including pred_features
        gpu_batch: Batch containing camera_idx, frame_idx, pixel_coords, exposure
        training: If True, use actual frame_idx; if False, use -1 for novel view mode

    Returns:
        Updated outputs dict with post-processed pred_features
    """
    assert outputs["pred_features"].shape[0] == 1, "Post-processing requires batch_size=1"

    pred_features = outputs["pred_features"]
    physical_fov_mask = getattr(gpu_batch, "physical_fov_mask", None)
    learning_valid_mask = (
        getattr(gpu_batch, "training_valid_mask", None)
        if training else physical_fov_mask
    )
    if learning_valid_mask is None:
        learning_valid_mask = physical_fov_mask
    pred_features_for_ppisp = pred_features
    if learning_valid_mask is not None:
        # Pixels outside the camera domain enter PPISP as the exterior value
        # and are excluded from its pooling.
        gate = learning_valid_mask.to(
            device=pred_features.device, dtype=torch.bool).expand_as(pred_features)
        exterior = torch.full_like(pred_features, float(gpu_batch.physical_fov_exterior_value))
        pred_features_for_ppisp = torch.where(gate, pred_features, exterior)
    camera_idx = gpu_batch.camera_idx
    frame_idx = gpu_batch.frame_idx if training else -1
    H, W = pred_features.shape[1], pred_features.shape[2]

    # Flatten: [1, H, W, 3] -> [H*W, 3]
    # Ensure contiguous memory for CUDA kernels
    pred_features_flat = pred_features_for_ppisp.contiguous().view(-1, 3)
    pixel_coords_flat = gpu_batch.pixel_coords.contiguous().view(-1, 2)

    # Apply post-processing
    if learning_valid_mask is not None:
        pred_features_pp = post_processing(
            pred_features_flat, pixel_coords_flat, resolution=(W, H),
            camera_idx=camera_idx, frame_idx=frame_idx,
            exposure_prior=gpu_batch.exposure,
            valid_mask=learning_valid_mask,
        )
    else:
        pred_features_pp = post_processing(
            pred_features_flat,
            pixel_coords_flat,
            resolution=(W, H),
            camera_idx=camera_idx,
            frame_idx=frame_idx,
            exposure_prior=gpu_batch.exposure,
        )

    # Reshape back: [H*W, 3] -> [1, H, W, 3]
    pred_features_pp = pred_features_pp.view(pred_features.shape)
    if learning_valid_mask is not None:
        pred_features_pp = torch.where(gate, pred_features_pp, exterior)
        if "pred_opacity" not in outputs:
            raise RuntimeError("post-processing requires pred_opacity")
        opacity = outputs["pred_opacity"]
        opacity_gate = learning_valid_mask.to(
            device=opacity.device, dtype=torch.bool).expand_as(opacity)
        outputs["pred_opacity"] = torch.where(opacity_gate, opacity, torch.zeros_like(opacity))
    outputs["pred_features"] = pred_features_pp
    return outputs
