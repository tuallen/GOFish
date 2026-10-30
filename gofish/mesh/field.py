# Portions derived from GOF (https://github.com/autonomousvision/gaussian-opacity-fields),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Matched camera-min opacity field O(x) at query points.

O(x) = min over the training cameras that observe x (inside the calibrated
camera domain) of the opacity accumulated on the segment from the camera
center to x, evaluated with the renderer's own response and alpha rules on
the intrinsic (Mip-filtered) Gaussians. Points no training camera observes
are treated as opaque.
"""

from __future__ import annotations

import numpy as np
import torch

from .cameras import TrainingCamera
from .shape import EffectiveShape


def camera_inputs(camera: TrainingCamera, device: torch.device):
    """Rays, physical mask and pose arguments of one camera for the native field."""
    direction = camera.ray_direction.to(device).float().contiguous()
    origin = camera.ray_origin.to(device).view(1, 1, 1, 3).expand_as(direction).contiguous()
    mask = (torch.empty(0, dtype=torch.uint8, device=device)
            if camera.physical_mask is None else camera.physical_mask.to(device).contiguous())
    return (
        int(camera.frame_number), origin, direction, mask, camera.sensor,
        int(camera.poses.timestamps_us[0]), int(camera.poses.timestamps_us[1]),
        camera.poses.T_world_sensors[0], camera.poses.T_world_sensors[1],
    )


def evaluate_field(raster, shape: EffectiveShape, cameras: list[TrainingCamera],
                   points: np.ndarray, device: torch.device):
    """Return (minimum opacity [Q] float32, observed [Q] uint8) at the points."""
    packed = torch.from_numpy(shape.packed_density()).to(device).contiguous()
    sigma = torch.from_numpy(np.array(shape.sigma_max, dtype=np.float32, order="C", copy=True)).to(device).contiguous()
    query = np.array(points, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(query).all():
        raise ValueError("query points are not finite")
    query = torch.from_numpy(query).to(device).contiguous()

    token = raster.begin_opacity_field(packed, sigma, query)
    finished = False
    try:
        for camera in cameras:
            raster.integrate_opacity_field_camera(token, *camera_inputs(camera, device))
        minimum, observed, _above = raster.finish_opacity_field(token)
        finished = True
    finally:
        if finished:
            raster.release_opacity_field(token)
    torch.cuda.synchronize(device)
    minimum = np.array(minimum.detach().cpu().numpy(), dtype="<f4", order="C", copy=True)
    observed = np.array(observed.detach().cpu().numpy(), dtype=np.uint8, order="C", copy=True)
    return minimum, observed


def signed_field(minimum: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """1/2 - O(x), with unobserved points opaque (O = 1)."""
    return np.ascontiguousarray(
        np.float32(0.5) - np.where(observed != 0, minimum, np.float32(1.0)), dtype="<f4")
