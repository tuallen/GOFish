# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Export the FIORD training cameras used for geometry visibility.

For every training image of a scene (GOFish loader, training split, no ray
jitter) this writes its camera-to-world pose and, once per distinct camera,
a ray asset: the central origin, the unit ray of every pixel centre inside
the calibrated aperture (float32) and the cosine of the cone through that
pixel's four corners (float64, padded outward by one float32 ulp).  Corner
rays come from the calibration (OpenCV fisheye by 64-step bisection of the
Kannala-Brandt radius, capped at the first critical angle and the physical
maximum angle).  No image is read.

Output: ``<out>/training_cameras.json`` and ``<out>/rays/camera_<id>.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1)[..., None]


def _first_fisheye_cap(coefficients: np.ndarray) -> float:
    k1, k2, k3, k4 = (float(v) for v in coefficients)
    polynomial = np.asarray([9.0 * k4, 7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0], dtype=np.float64)
    nonzero = np.flatnonzero(polynomial != 0.0)
    roots = np.roots(polynomial[int(nonzero[0]):])
    positive = sorted(
        math.sqrt(float(root.real)) for root in roots
        if abs(float(root.imag)) <= 1.0e-10 * max(1.0, abs(float(root.real)))
        and float(root.real) > 0.0 and math.sqrt(float(root.real)) < math.pi)
    return math.nextafter(positive[0] if positive else math.pi, 0.0)


def fisheye_rays(coordinates: np.ndarray, parameters: Mapping[str, Any], physical_cap: float | None) -> np.ndarray:
    focal = np.asarray(parameters["focal_length"], dtype=np.float64).reshape(-1)
    principal = np.asarray(parameters["principal_point"], dtype=np.float64).reshape(-1)
    coefficients = np.asarray(parameters["radial_coeffs"], dtype=np.float64).reshape(-1)
    cap = _first_fisheye_cap(coefficients)
    if physical_cap is not None:
        cap = min(cap, math.nextafter(float(physical_cap), 0.0))
    normalised = (np.asarray(coordinates, dtype=np.float64) - principal) / focal
    radius = np.linalg.norm(normalised, axis=-1)
    k1, k2, k3, k4 = (float(v) for v in coefficients)

    def radial(theta):
        theta = np.asarray(theta, dtype=np.float64)
        square = theta * theta
        return theta * (1.0 + square * (k1 + square * (k2 + square * (k3 + square * k4))))

    clipped = np.minimum(radius, float(radial(cap)))
    lo, hi = np.zeros_like(clipped), np.full_like(clipped, cap)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        below = radial(mid) < clipped
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    theta = 0.5 * (lo + hi)
    azimuth = normalised / np.where(radius > 0.0, radius, 1.0)[..., None]
    rays = np.concatenate((np.sin(theta)[..., None] * azimuth, np.cos(theta)[..., None]), axis=-1)
    rays[radius == 0.0] = (0.0, 0.0, 1.0)
    return _normalise(rays)


def pinhole_rays(coordinates: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """OpenCV fixed-point undistortion (exact in one step for zero distortion)."""
    focal = np.asarray(parameters["focal_length"], dtype=np.float64).reshape(-1)
    principal = np.asarray(parameters["principal_point"], dtype=np.float64).reshape(-1)
    radial = np.asarray(parameters.get("radial_coeffs", np.zeros(6)), dtype=np.float64).reshape(-1)
    tangential = np.asarray(parameters.get("tangential_coeffs", np.zeros(2)), dtype=np.float64).reshape(-1)
    prism = np.asarray(parameters.get("thin_prism_coeffs", np.zeros(4)), dtype=np.float64).reshape(-1)
    radial = radial if radial.size else np.zeros(6)
    tangential = tangential if tangential.size else np.zeros(2)
    prism = prism if prism.size else np.zeros(4)
    distorted = (np.asarray(coordinates, dtype=np.float64) - principal) / focal
    x, y = distorted[..., 0].copy(), distorted[..., 1].copy()
    k1, k2, k3, k4, k5, k6 = radial
    p1, p2 = tangential
    s1, s2, s3, s4 = prism
    for _ in range(24):
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        inverse_radial = (1.0 + k4 * r2 + k5 * r4 + k6 * r6) / (1.0 + k1 * r2 + k2 * r4 + k3 * r6)
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4
        x = (distorted[..., 0] - dx) * inverse_radial
        y = (distorted[..., 1] - dy) * inverse_radial
    return _normalise(np.stack((x, y, np.ones_like(x)), axis=-1))


def calibrated_rays(model: str, parameters, coordinates, physical_cap=None) -> np.ndarray:
    if model == "fisheye":
        return fisheye_rays(coordinates, parameters, physical_cap)
    return pinhole_rays(coordinates, parameters)


def ray_asset(origins: np.ndarray, directions: np.ndarray, mask: np.ndarray, model: str, parameters,
              physical_cap=None, chunk_size: int = 131072):
    """Central origin, physical pixel-centre rays (float32) and per-pixel cone cosines (float64)."""
    yy, xx = np.nonzero(mask)
    origin = np.asarray(origins[mask], dtype=np.float64)[0]
    centre_rays = _normalise(directions[mask])
    cone_cos = np.empty(len(centre_rays), dtype=np.float64)
    offsets = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    base = np.stack((xx, yy), axis=-1).astype(np.float64)
    for begin in range(0, len(base), chunk_size):
        end = min(begin + chunk_size, len(base))
        corners = (base[begin:end, None, :] + offsets[None, :, :]).reshape(-1, 2)
        corner_rays = calibrated_rays(model, parameters, corners, physical_cap).reshape(end - begin, 4, 3)
        cone_cos[begin:end] = np.min(np.einsum("ij,ikj->ik", centre_rays[begin:end], corner_rays), axis=1)
    cone_cos = np.nextafter(np.clip(cone_cos - np.finfo(np.float32).eps, -1.0, 1.0), -np.inf)
    return origin, np.ascontiguousarray(centre_rays, dtype=np.float32), np.ascontiguousarray(cone_cos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="GOFish checkpoint (configuration only)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="overrides of the checkpoint's config, e.g. dataset.test_split_interval=8")
    args = parser.parse_args()

    import torch

    from gofish.eval.common.dataset_cameras import iterate_batches, plain_config

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = plain_config(checkpoint, args.data, args.overrides)
    del checkpoint
    (args.out / "rays").mkdir(parents=True, exist_ok=True)
    rows, written = [], set()
    with torch.inference_mode():
        for index, dataset, gpu in iterate_batches(conf, split="train"):
            camera_id = int(gpu.camera_intrinsic_id)
            asset = args.out / "rays" / f"camera_{camera_id}.npz"
            if camera_id not in written:
                fisheye = getattr(gpu, "intrinsics_OpenCVFisheyeCameraModelParameters", None)
                model, parameters = (("fisheye", fisheye) if fisheye is not None else
                                     ("pinhole", gpu.intrinsics_OpenCVPinholeCameraModelParameters))
                parameters = {k: np.asarray(torch.as_tensor(v).cpu()) if torch.is_tensor(v) else v
                              for k, v in parameters.items()}
                height, width = int(gpu.rays_dir.shape[1]), int(gpu.rays_dir.shape[2])
                mask = getattr(gpu, "physical_fov_mask", None)
                mask = (np.ones((height, width), bool) if mask is None
                        else mask[0, ..., 0].bool().cpu().numpy())
                cap = float(parameters["max_angle"]) if model == "fisheye" else None
                origin, directions, cone_cos = ray_asset(gpu.rays_ori[0].cpu().numpy(),
                                                         gpu.rays_dir[0].cpu().numpy(), mask, model,
                                                         parameters, physical_cap=cap)
                np.savez(asset, origins_camera=origin, directions_camera=directions, cone_cos=cone_cos)
                written.add(camera_id)
            rows.append({"image_id": dataset.cam_extrinsics[index].name,
                         "camera_to_world": np.asarray(dataset.poses[index], dtype=np.float64).tolist(),
                         "camera_id": camera_id, "rays": f"rays/camera_{camera_id}.npz"})
    (args.out / "training_cameras.json").write_text(json.dumps({"training_images": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
