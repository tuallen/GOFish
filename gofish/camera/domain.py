# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Calibrated camera domain of an OPENCV_FISHEYE (Kannala-Brandt) camera.

The domain of a fisheye camera is the set of raster pixels whose calibrated
ray lies on the first monotone branch of the distortion polynomial

    rho(theta) = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8),

optionally intersected with a measured lens aperture (a circle in the native
image). Rays are obtained by inverting rho on that branch at the exact native
pixel centres, so a downsampled raster samples the same calibrated chart as
the native one.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from gofish.camera.pixel_chart import (
    native_center,
    native_sample,
    native_to_raster,
    raster_to_index_coordinate,
    resize_ratios,
)


DATA_DIR = Path(__file__).resolve().parent / "data"


def radial_value(theta: float, coefficients: Iterable[float]) -> float:
    k1, k2, k3, k4 = (float(value) for value in coefficients)
    y = theta * theta
    return theta * (1.0 + y * (k1 + y * (k2 + y * (k3 + y * k4))))


def radial_derivative(theta: float, coefficients: Iterable[float]) -> float:
    k1, k2, k3, k4 = (float(value) for value in coefficients)
    y = theta * theta
    return 1.0 + y * (3.0 * k1 + y * (5.0 * k2 + y * (7.0 * k3 + y * 9.0 * k4)))


def monotone_branch_limit(coefficients: Iterable[float]) -> float:
    """Open upper limit of the first strictly increasing branch of rho on [0, pi]."""
    k1, k2, k3, k4 = (float(value) for value in coefficients)
    polynomial = np.asarray([9.0 * k4, 7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0], dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(polynomial) > 0.0)
    roots = np.roots(polynomial[int(nonzero[0]):])
    roots_y = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= 1.0e-10 * max(1.0, abs(float(root.real))) and float(root.real) > 0.0
    )
    folds = [math.sqrt(value) for value in roots_y if math.sqrt(value) < math.pi]
    limit = math.nextafter(folds[0] if folds else math.pi, 0.0)
    while radial_derivative(limit, (k1, k2, k3, k4)) <= 0.0:
        limit = math.nextafter(limit, 0.0)
    return limit


@dataclasses.dataclass(frozen=True)
class FisheyeDomain:
    """First-branch angular domain of one fisheye intrinsic at one raster size."""

    native_params: tuple[float, ...]  # fx, fy, cx, cy, k1..k4 in the native chart
    max_angle_radians: float
    active_ratio_x: float
    active_ratio_y: float

    def first_branch_rays(self, image_points, *, iterations: int = 48):
        """Invert rho on the first branch by bisection (float64).

        ``image_points`` are native-chart pixel coordinates ``[N, 2]``. Points
        outside the branch receive a finite placeholder ray and ``valid=False``.
        """
        import torch

        work = image_points.to(dtype=torch.float64)
        focal = torch.as_tensor(self.native_params[:2], dtype=torch.float64, device=work.device)
        principal = torch.as_tensor(self.native_params[2:4], dtype=torch.float64, device=work.device)
        normalized = (work - principal) / focal
        distorted_radius = torch.linalg.vector_norm(normalized, dim=-1)
        coefficients = self.native_params[4:8]
        cap = float(self.max_angle_radians)
        cap_radius = radial_value(cap, coefficients)
        finite = torch.isfinite(normalized).all(dim=-1) & torch.isfinite(distorted_radius)
        valid = finite & (distorted_radius < cap_radius)
        lo = torch.zeros_like(distorted_radius)
        hi = torch.full_like(distorted_radius, cap)
        k1, k2, k3, k4 = (float(value) for value in coefficients)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            y = mid * mid
            value = mid * (1.0 + y * (k1 + y * (k2 + y * (k3 + y * k4))))
            move_lo = value < distorted_radius
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(move_lo, hi, mid)
        theta = 0.5 * (lo + hi)
        safe_radius = torch.where(distorted_radius > 0.0, distorted_radius, torch.ones_like(distorted_radius))
        azimuth = normalized / safe_radius.unsqueeze(-1)
        rays = torch.cat([torch.sin(theta).unsqueeze(-1) * azimuth, torch.cos(theta).unsqueeze(-1)], dim=-1)
        forward = torch.zeros_like(rays)
        forward[..., 2] = 1.0
        rays = torch.where((distorted_radius == 0.0).unsqueeze(-1), forward, rays)
        rays = torch.where(valid.unsqueeze(-1), rays, forward)
        return rays.to(dtype=torch.float32), valid.unsqueeze(-1)

    def unproject_raster(self, raster_indices, support_mask):
        """Camera rays of integer raster pixels ``[N, 2]``; every pixel in
        ``support_mask`` must lie on the first branch."""
        import torch

        values = raster_indices.to(dtype=torch.float64)
        native_points = torch.stack(
            (native_center(values[:, 0], self.active_ratio_x), native_center(values[:, 1], self.active_ratio_y)),
            dim=1,
        )
        rays, valid = self.first_branch_rays(native_points)
        support = torch.as_tensor(support_mask, device=rays.device, dtype=torch.bool).reshape(-1)
        if bool((support & ~valid.reshape(-1)).any()):
            raise RuntimeError("camera support contains pixels outside the calibrated branch")
        return rays, valid


def fisheye_domain(native_params, native_width, native_height, raster_width, raster_height,
                   max_angle_cap: Optional[float] = None) -> FisheyeDomain:
    """First-branch domain, optionally capped by a calibrated angular limit."""
    params = tuple(float(value) for value in native_params)
    active_ratio_x, active_ratio_y = resize_ratios(native_width, native_height, raster_width, raster_height)
    max_angle = monotone_branch_limit(params[4:8])
    if max_angle_cap is not None:
        max_angle = min(max_angle, float(max_angle_cap))
    return FisheyeDomain(params, max_angle, active_ratio_x, active_ratio_y)


# ---------------------------------------------------------------------------
# Per-camera calibrated support
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CameraSupport:
    """Pixel support and angular limit of one camera at one raster size."""

    max_angle_radians: float
    rows: tuple[tuple[int, int], ...]  # [x0, x1) per raster row
    width: int
    height: int

    def torch_mask(self, device=None):
        import torch

        mask = np.zeros((1, self.height, self.width, 1), dtype=np.bool_)
        for y, (x0, x1) in enumerate(self.rows):
            mask[0, y, x0:x1, 0] = True
        return torch.from_numpy(mask).to(device=device)


def full_raster_support(width: int, height: int, max_angle_radians: float) -> CameraSupport:
    return CameraSupport(float(max_angle_radians), tuple((0, int(width)) for _ in range(int(height))),
                         int(width), int(height))


def branch_support(domain: FisheyeDomain, raster_width: int, raster_height: int) -> CameraSupport:
    """Raster pixels whose calibrated ray lies on the first branch of ``domain``
    (the support of a fisheye camera without a measured aperture)."""
    import torch

    width, height = int(raster_width), int(raster_height)
    u = torch.arange(width, dtype=torch.float64).repeat(height)
    v = torch.arange(height, dtype=torch.float64).repeat_interleave(width)
    native_points = torch.stack(
        (native_center(u, domain.active_ratio_x), native_center(v, domain.active_ratio_y)), dim=1)
    _, valid = domain.first_branch_rays(native_points)
    valid = valid.reshape(height, width).numpy()
    runs: list[tuple[int, int]] = []
    for row in valid:
        columns = np.flatnonzero(row)
        if columns.size == 0:
            runs.append((0, 0))
            continue
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        if columns.size != x1 - x0:
            raise RuntimeError("the first-branch support of a raster row is not one interval")
        runs.append((x0, x1))
    return CameraSupport(float(domain.max_angle_radians), tuple(runs), width, height)


def aperture_support(record: dict, raster_width: int, raster_height: int) -> CameraSupport:
    """Measured lens aperture (a native-chart circle) intersected with the
    calibrated branch ``|normalized radius| < rho_limit``, as row runs of the
    raster pixels whose centres lie inside both."""
    width, height = int(record["width"]), int(record["height"])
    active_ratio_x, active_ratio_y = resize_ratios(width, height, raster_width, raster_height)
    fx, fy, cx, cy = (float(value) for value in record["params"][:4])
    rho_limit = float(record["rho_limit"])
    aperture_cx, aperture_cy = (float(value) for value in record["aperture_center"])
    aperture_radius = float(record["aperture_radius"])
    runs: list[tuple[int, int]] = []
    for y in range(raster_height):
        source_y = native_sample(y, active_ratio_y)
        circle_residual = aperture_radius ** 2 - (source_y - aperture_cy) ** 2
        calibrated_residual = rho_limit ** 2 - ((source_y - cy) / fy) ** 2
        if circle_residual < 0.0 or calibrated_residual < 0.0:
            runs.append((0, 0))
            continue
        aperture_dx = math.sqrt(max(0.0, circle_residual))
        calibrated_dx = fx * math.sqrt(max(0.0, calibrated_residual))
        lo_native = max(aperture_cx - aperture_dx, cx - calibrated_dx)
        hi_native = min(aperture_cx + aperture_dx, cx + calibrated_dx)
        if hi_native < lo_native:
            runs.append((0, 0))
            continue
        lo_real = raster_to_index_coordinate(native_to_raster(lo_native, active_ratio_x))
        hi_real = raster_to_index_coordinate(native_to_raster(hi_native, active_ratio_x))
        x0 = math.ceil(math.nextafter(lo_real, -math.inf))
        x1 = math.floor(math.nextafter(hi_real, math.inf)) + 1
        x0 = max(0, min(raster_width, x0))
        x1 = max(x0, min(raster_width, x1))
        runs.append((x0, x1))
    return CameraSupport(float(record["theta_limit"]), tuple(runs), int(raster_width), int(raster_height))


def _same_calibration(record: dict, model: str, width: int, height: int, params) -> bool:
    return (record["model"] == model and int(record["width"]) == int(width)
            and int(record["height"]) == int(height)
            and [float(value) for value in record["params"]] == [float(value) for value in params])


class CameraDomainTable:
    """Calibrated camera data keyed by the native calibration.

    Two record kinds are supported (see ``gofish/camera/data/README.md``):
      * ``aperture``: a measured lens aperture plus the calibrated branch
        limits (``rho_limit``, ``theta_limit``);
      * ``max_angle``: a full-raster camera with an angular cap.
    A fisheye camera without a record uses the raster pixels on its own
    first monotone branch.
    """

    def __init__(self, paths: Iterable[str | Path] = ()):
        self.records: list[dict] = []
        for path in paths:
            path = Path(path)
            if not path.is_absolute() and not path.exists():
                path = DATA_DIR / path
            self.records.extend(json.loads(Path(path).read_text())["cameras"])

    def find(self, model: str, width: int, height: int, params) -> Optional[dict]:
        matches = [r for r in self.records if _same_calibration(r, model, width, height, params)]
        if len(matches) > 1:
            raise RuntimeError("camera calibration matches more than one domain record")
        return matches[0] if matches else None
