# Portions derived from GOF (https://github.com/autonomousvision/gaussian-opacity-fields),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Tetrahedralization sites: each Gaussian's center and its eight support-box corners.

For B = R Sbar and the renderer support radius r, the support ellipsoid is
mu + B u with ||u|| <= r. Its enclosing oriented box has corners
mu + B (r z) for z in {-1, +1}^3. Sites are placed only for Gaussians with
opacity above 1/255.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .shape import DEAD_OPACITY, EffectiveShape

SIGNS = np.asarray(
    [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)
# Renderer kernel response threshold as float32, and the radius it implies.
KERNEL_MIN_RESPONSE = float(np.float32("0.0113"))
RADIUS_CAP = math.sqrt(-2.0 * math.log(KERNEL_MIN_RESPONSE))
SITES_PER_GAUSSIAN = 9


@dataclass
class Sites:
    points: np.ndarray  # [M, 3] float32
    scale: np.ndarray  # [M, 1] float32, owner support extent sigma_max * r
    owner: np.ndarray  # [M] int64, owning Gaussian index

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def support_radius(opacity: np.ndarray) -> np.ndarray:
    """Opacity-adaptive renderer support radius, capped by the kernel threshold."""
    value = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all() or not bool(np.all(value > DEAD_OPACITY)):
        raise ValueError("site opacities must be finite and above 1/255")
    radius = np.minimum(RADIUS_CAP, np.sqrt(2.0 * np.log(255.0 * value)))
    if not np.isfinite(radius).all() or not bool(np.all(radius > 0.0)):
        raise ValueError("support radius is not finite and positive")
    return radius


def support_box_sites(positions: np.ndarray, factor: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """[N, 9, 3] float64: the center, then the eight corners in sign order."""
    centers = np.asarray(positions, dtype=np.float64)
    factors = np.asarray(factor, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    local_corners = radius[:, None, None] * SIGNS[None, :, :]
    corners = centers[:, None, :] + np.einsum("nij,nkj->nki", factors, local_corners, optimize=False)
    return np.concatenate((centers[:, None, :], corners), axis=1)


def generate_sites(shape: EffectiveShape) -> Sites:
    opacity = np.asarray(shape.density[:, 0], dtype=np.float64)
    if not np.array_equal(shape.retained, opacity > DEAD_OPACITY):
        raise ValueError("retained Gaussians differ from opacity > 1/255")
    indices = np.flatnonzero(shape.retained)
    if len(indices) == 0:
        raise ValueError("no Gaussian has opacity above 1/255")

    radius = support_radius(opacity[indices])
    points = support_box_sites(
        np.asarray(shape.positions[indices], dtype="<f4"),
        np.asarray(shape.factor[indices], dtype="<f4"),
        radius,
    ).astype("<f4")
    scale = np.asarray(shape.sigma_max[indices, 0], dtype=np.float64) * radius
    return Sites(
        points=np.ascontiguousarray(points.reshape(-1, 3)),
        scale=np.repeat(scale.astype("<f4"), SITES_PER_GAUSSIAN).reshape(-1, 1),
        owner=np.repeat(indices.astype("<i8"), SITES_PER_GAUSSIAN),
    )
