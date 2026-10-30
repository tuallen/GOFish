# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Calibrated Mip Filtering, per-view part: the pixel footprint.

The intrinsic part, the Mip 3-D filter of every Gaussian, is in
``threedgrut.physical_mip``; ``render.gofish.scene_sensor_factorization``
switches both.

Example, the per-view shape inputs of one Gaussian::

    q = footprint_covariance(J)                 # [1, 3, 3]; J [1, 2, 3] pixel Jacobian
    m = mass_preserving_factor(Sigma, J)        # [1]; the renderer draws Sigma + Q_v with opacity m * o

The checkpoint stores only the intrinsic scene: positions, Mip-filtered
covariances and opacities.  For one render, the pixel variance gamma is pulled
back through the active camera as the minimum-norm 3-D covariance
Q_v = gamma J^+ J^+^T, so that J Q_v J^T = gamma I, and ``Sigma + Q_v`` is
passed to the renderer together with the mass-preserving factor m.  Q_v is
detached, transient, and never saved or used by mesh extraction.
"""

from __future__ import annotations

from typing import Any

import torch

from threedgrut.camera_local_validity import camera_local_near_valid


# 3DGS's inherited projected pixel variance gamma (px^2), the default of
# render.gofish.pixel_footprint_variance; read it through pixel_footprint_variance.
FILTER_VARIANCE_PIXELS = 0.3


def pixel_footprint_variance(conf) -> float:
    """Effective per-view pixel variance gamma of ``conf`` (0 without the factorization)."""
    import math

    from omegaconf import OmegaConf

    if not bool(OmegaConf.select(conf, "render.gofish.scene_sensor_factorization", default=True)):
        return 0.0
    gamma = float(OmegaConf.select(conf, "render.gofish.pixel_footprint_variance", default=FILTER_VARIANCE_PIXELS))
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("render.gofish.pixel_footprint_variance must be finite and >= 0")
    return gamma


def footprint_covariance(jacobian: torch.Tensor, pixel_variance: float = FILTER_VARIANCE_PIXELS) -> torch.Tensor:
    """Reference Q_v = gamma J^+ J^+^T, float64 ``[N, 3, 3]``, for full-rank pixel Jacobians ``[N, 2, 3]``.

    It is the minimum-norm 3-D covariance whose projection is the pixel footprint, J Q_v J^T = gamma I.
    The renderer builds the same Q_v with a pivoted Gram--Schmidt factorization (``current_view_q6``).
    """
    pseudo_inverse = torch.linalg.pinv(jacobian.detach().to(torch.float64))
    return float(pixel_variance) * pseudo_inverse @ pseudo_inverse.transpose(1, 2)


def mass_preserving_factor(covariance: torch.Tensor, jacobian: torch.Tensor,
                           pixel_variance: float = FILTER_VARIANCE_PIXELS) -> torch.Tensor:
    """Reference m = sqrt(det S0 / det(S0 + gamma I)), float64 ``[N]``, with S0 = J Sigma J^T the
    projected covariance of the intrinsic Gaussian ``covariance`` ``[N, 3, 3]``: the opacity factor that
    keeps the ray-marginal mass of Sigma + Q_v equal to that of Sigma."""
    j = jacobian.detach().to(torch.float64)
    projected = j @ covariance.detach().to(torch.float64) @ j.transpose(1, 2)
    eye = torch.eye(2, dtype=torch.float64, device=projected.device)
    return torch.sqrt(torch.linalg.det(projected) / torch.linalg.det(projected + float(pixel_variance) * eye))


def _as_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _pack_q6(q: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            q[:, 0, 0],
            q[:, 0, 1],
            q[:, 0, 2],
            q[:, 1, 1],
            q[:, 1, 2],
            q[:, 2, 2],
        ),
        dim=1,
    )


def _minimum_norm_q6_from_float32_jacobian(
    jacobian_world: torch.Tensor,
    chart_valid: torch.Tensor,
    pixel_variance: float = FILTER_VARIANCE_PIXELS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q = gamma J+ J+^T from the float32 J that the factor operator receives.

    The 3x2 transpose is factorized by vectorized column-pivoted modified
    Gram--Schmidt with one reorthogonalization pass.  Validity uses only the
    camera-domain predicate and finite, strictly-positive QR pivots; there is
    no determinant threshold or condition-number tuning.
    """
    if jacobian_world.dtype != torch.float32:
        raise RuntimeError("current-view Jacobian must be float32")
    if jacobian_world.ndim != 3 or jacobian_world.shape[1:] != (2, 3):
        raise RuntimeError("current-view Jacobian must be [N,2,3]")
    if chart_valid.shape != (jacobian_world.shape[0],):
        raise RuntimeError("camera-chart validity must be [N]")

    # Promote the exact bytes that footprint receives; never recompute J in double.
    transpose64 = jacobian_world.detach().to(torch.float64).transpose(1, 2)
    first = transpose64[:, :, 0]
    second = transpose64[:, :, 1]
    first_norm2 = torch.sum(first * first, dim=1)
    second_norm2 = torch.sum(second * second, dim=1)
    swap = second_norm2 > first_norm2
    leading = torch.where(swap[:, None], second, first)
    trailing = torch.where(swap[:, None], first, second)

    pivot0 = torch.linalg.vector_norm(leading, dim=1)
    pivot0_valid = torch.isfinite(pivot0) & (pivot0 > 0.0)
    safe_pivot0 = torch.where(pivot0_valid, pivot0, torch.ones_like(pivot0))
    basis0 = leading / safe_pivot0[:, None]

    coefficient = torch.sum(basis0 * trailing, dim=1)
    residual = trailing - coefficient[:, None] * basis0
    correction = torch.sum(basis0 * residual, dim=1)
    residual = residual - correction[:, None] * basis0
    coefficient = coefficient + correction
    pivot1 = torch.linalg.vector_norm(residual, dim=1)
    pivot1_valid = torch.isfinite(pivot1) & (pivot1 > 0.0)
    safe_pivot1 = torch.where(pivot1_valid, pivot1, torch.ones_like(pivot1))
    basis1 = residual / safe_pivot1[:, None]

    valid = (
        chart_valid
        & torch.isfinite(jacobian_world).all(dim=2).all(dim=1)
        & pivot0_valid
        & pivot1_valid
        & torch.isfinite(coefficient)
        & torch.isfinite(basis0).all(dim=1)
        & torch.isfinite(basis1).all(dim=1)
    )

    # For A=J^T with A Pi=U R, J+=U R^-T Pi^T.  Pi cancels in J+J+^T.
    inverse00 = 1.0 / safe_pivot0
    inverse10 = -coefficient / (safe_pivot0 * safe_pivot1)
    inverse11 = 1.0 / safe_pivot1
    pseudo_column0 = (
        basis0 * inverse00[:, None] + basis1 * inverse10[:, None]
    )
    pseudo_column1 = basis1 * inverse11[:, None]
    q64 = pixel_variance * (
        pseudo_column0[:, :, None] * pseudo_column0[:, None, :]
        + pseudo_column1[:, :, None] * pseudo_column1[:, None, :]
    )
    valid = valid & torch.isfinite(q64).all(dim=2).all(dim=1)
    q64 = torch.where(valid[:, None, None], q64, torch.zeros_like(q64))
    jacobian_world = torch.where(
        valid[:, None, None], jacobian_world, torch.zeros_like(jacobian_world)
    )
    return (
        _pack_q6(q64).detach().contiguous(),
        jacobian_world.detach().contiguous(),
        valid.detach().contiguous(),
    )


def _ideal_pinhole_focals(
    intrinsics: Any, reference: torch.Tensor, host_sync_free: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ideal-pinhole focal lengths without ignoring lens distortion."""
    focal = _as_tensor(intrinsics["focal_length"], reference).reshape(-1)
    if focal.numel() == 1:  # SIMPLE_PINHOLE
        fx = fy = focal[0]
    elif focal.numel() == 2:  # PINHOLE
        fx, fy = focal.unbind()
    else:
        raise RuntimeError("ideal pinhole focal_length must contain one or two values")

    # SIMPLE_PINHOLE and PINHOLE arrive through NCore's OpenCV-pinhole ABI.
    # Refuse unknown or nonzero distortion rather than silently applying the
    # ideal-pinhole Jacobian to OPENCV/FULL_OPENCV calibration.
    for name in ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"):
        if name not in intrinsics:
            raise RuntimeError(
                f"ideal pinhole Q requires explicit zero {name}; "
                "distorted OPENCV is unsupported"
            )
        if host_sync_free:
            # Same predicate on the host copy of the calibration (no device sync).
            values = torch.as_tensor(intrinsics[name]).reshape(-1)
        else:
            values = _as_tensor(intrinsics[name], reference).reshape(-1)
        if not bool(torch.isfinite(values).all()) or bool(torch.count_nonzero(values)):
            raise RuntimeError(
                "view observation filter does not approximate distorted OPENCV "
                "with an ideal-pinhole Jacobian"
            )
    return fx, fy


def _current_view_q6_ideal_pinhole(
    positions: torch.Tensor,
    camera_points: torch.Tensor,
    rotation: torch.Tensor,
    intrinsics: Any,
    pixel_variance: float,
    host_sync_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build transient Q for SIMPLE_PINHOLE/PINHOLE through analytic J."""
    fx, fy = _ideal_pinhole_focals(intrinsics, positions, host_sync_free)
    x, y, z = camera_points.unbind(dim=1)
    # Pinhole validity is view-depth based; do not compute an unused radial norm.
    in_local_chart = camera_local_near_valid("PINHOLE", z, None)
    safe_z = torch.where(in_local_chart, z, torch.ones_like(z))

    jacobian_camera = positions.new_zeros((positions.shape[0], 2, 3))
    jacobian_camera[:, 0, 0] = fx / safe_z
    jacobian_camera[:, 0, 2] = -fx * x / safe_z.square()
    jacobian_camera[:, 1, 1] = fy / safe_z
    jacobian_camera[:, 1, 2] = -fy * y / safe_z.square()

    # Row-vector camera coordinates imply J_world = J_camera R_c2w^T.
    jacobian_world = torch.einsum(
        "nij,kj->nik", jacobian_camera, rotation
    ).to(torch.float32)
    return _minimum_norm_q6_from_float32_jacobian(
        jacobian_world, in_local_chart, pixel_variance)


@torch.no_grad()
def current_view_q6(
    positions: torch.Tensor, gpu_batch: Any, pixel_variance: float = FILTER_VARIANCE_PIXELS,
    *, host_sync_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build detached minimum-norm Q for one active supported camera view.

    ``host_sync_free`` (inference renderers) writes the on-axis Jacobian rows
    with ``torch.where`` instead of a data-dependent branch; the values are
    the same operations on the same operands, so Q, J and validity are
    bit-identical, without a device-to-host synchronization.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise RuntimeError("positions must be [N,3]")
    if getattr(gpu_batch, "rays_in_world_space", False):
        raise RuntimeError("view observation filter requires camera-space rays")
    poses = getattr(gpu_batch, "T_to_world", None)
    if not torch.is_tensor(poses) or tuple(poses.shape) != (1, 4, 4):
        raise RuntimeError("view observation filter requires exactly one active pose")
    end = getattr(gpu_batch, "T_to_world_end", None)
    if end is not None and not torch.equal(poses, end):
        raise RuntimeError("view observation filter is scoped to global shutter")

    intrinsics = getattr(
        gpu_batch, "intrinsics_OpenCVFisheyeCameraModelParameters", None
    )
    if intrinsics is None:
        pinhole_intrinsics = getattr(
            gpu_batch, "intrinsics_OpenCVPinholeCameraModelParameters", None
        )
        if pinhole_intrinsics is None:
            raise RuntimeError(
                "view observation filter supports OPENCV_FISHEYE and ideal pinhole"
            )
    else:
        pinhole_intrinsics = None

    pose = poses[0].to(device=positions.device, dtype=positions.dtype)
    rotation = pose[:3, :3]  # camera to world
    center = pose[:3, 3]
    camera_points = (positions.detach() - center) @ rotation
    if pinhole_intrinsics is not None:
        return _current_view_q6_ideal_pinhole(
            positions, camera_points, rotation, pinhole_intrinsics, pixel_variance,
            host_sync_free,
        )
    x, y, z = camera_points.unbind(dim=1)
    distance = torch.linalg.vector_norm(camera_points, dim=1)
    rho = torch.hypot(x, y)
    theta = torch.atan2(rho, z)

    focal = _as_tensor(intrinsics["focal_length"], positions)
    radial = _as_tensor(intrinsics["radial_coeffs"], positions)
    fx, fy = focal[0], focal[1]
    k1, k2, k3, k4 = radial.unbind()
    theta2 = theta.square()
    radial_theta = theta * (
        1.0
        + theta2
        * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
    )
    radial_derivative = 1.0 + theta2 * (
        3.0 * k1
        + theta2 * (5.0 * k2 + theta2 * (7.0 * k3 + theta2 * 9.0 * k4))
    )

    tiny = torch.finfo(positions.dtype).tiny
    safe_rho = rho.clamp_min(torch.sqrt(positions.new_tensor(tiny)))
    safe_distance2 = distance.square().clamp_min(tiny)
    cos_phi, sin_phi = x / safe_rho, y / safe_rho
    grad_theta = torch.stack(
        (
            z * x / (safe_rho * safe_distance2),
            z * y / (safe_rho * safe_distance2),
            -rho / safe_distance2,
        ),
        dim=1,
    )
    grad_phi = torch.stack(
        (
            -y / safe_rho.square(),
            x / safe_rho.square(),
            torch.zeros_like(x),
        ),
        dim=1,
    )

    jacobian_camera = positions.new_zeros((positions.shape[0], 2, 3))
    jacobian_camera[:, 0] = fx * (
        radial_derivative[:, None] * cos_phi[:, None] * grad_theta
        - radial_theta[:, None] * sin_phi[:, None] * grad_phi
    )
    jacobian_camera[:, 1] = fy * (
        radial_derivative[:, None] * sin_phi[:, None] * grad_theta
        + radial_theta[:, None] * cos_phi[:, None] * grad_phi
    )
    at_axis = (rho <= 1.0e-12) & (z > 0.0)
    if host_sync_free:
        axis = jacobian_camera.new_zeros(jacobian_camera.shape)
        axis[:, 0, 0] = fx / distance
        axis[:, 1, 1] = fy / distance
        jacobian_camera = torch.where(at_axis[:, None, None], axis, jacobian_camera)
    elif bool(at_axis.any()):
        jacobian_camera[at_axis] = 0.0
        jacobian_camera[at_axis, 0, 0] = fx / distance[at_axis]
        jacobian_camera[at_axis, 1, 1] = fy / distance[at_axis]

    # Row-vector camera coordinates imply J_world = J_camera R_c2w^T.
    jacobian_world = torch.einsum(
        "nij,kj->nik", jacobian_camera, rotation
    ).to(torch.float32)
    max_angle = float(intrinsics["max_angle"])
    chart_valid = (
        camera_local_near_valid("OPENCV_FISHEYE", z, distance)
        & ((rho > 1.0e-12) | (z > 0.0))
        & (theta < max_angle)
        & (radial_derivative > 0.0)
    )
    return _minimum_norm_q6_from_float32_jacobian(jacobian_world, chart_valid, pixel_variance)


def build_intrinsic_view_shape(
    scales: torch.Tensor,
    quaternion: torch.Tensor,
    density: torch.Tensor,
    positions: torch.Tensor,
    gpu_batch: Any,
    *,
    pixel_footprint_variance: float = FILTER_VARIANCE_PIXELS,
    frame_id: int | None = None,
):
    """Return C = Sigma + Q_v, its factors, and the m-compensated density.

    The renderer owns the min(0.99, response * rho') VJP; the factor operator
    consumes its already-masked density adjoint.
    """
    from threedgut_tracer.footprint_shape_operator import build_full_shape

    del frame_id
    q6, jacobian, observed = current_view_q6(positions, gpu_batch, pixel_footprint_variance)
    return build_full_shape(scales, quaternion, density, q6, jacobian, observed, mass_preserving=True)


__all__ = [
    "FILTER_VARIANCE_PIXELS",
    "footprint_covariance",
    "mass_preserving_factor",
    "build_intrinsic_view_shape",
    "current_view_q6",
    "pixel_footprint_variance",
]
