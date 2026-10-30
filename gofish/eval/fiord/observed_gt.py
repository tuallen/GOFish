# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Training-camera visibility for FIORD geometry evaluation.

A LiDAR ground-truth point is *observed* when Open3D hidden-point removal
(radius = 100 x GT bounding-box diameter) finds it visible from a training
camera and its direction lies inside that camera's calibrated domain, given as
the union of per-pixel ray cones (``gofish.eval.fiord.training_cameras``).  Points
observed from at least ``nmin = 2`` training cameras form the ground truth for
recall.  The same visibility test, applied to predicted surface samples with
``>= 1`` camera, defines the precision crop ``C``.

Only the supplied ``transform_mesh_to_gt`` alignment is applied; nothing is
fitted here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THRESHOLDS_M = (0.05, 0.10, 0.25, 0.30)
NUM_MESH_SAMPLES = 400_000
GT_DOWNSAMPLE = 1_000_000
NMIN = 2
SEED = 0


@dataclass(frozen=True)
class TrainingFrame:
    image_id: str
    camera_to_world: np.ndarray
    rays_path: Path


@dataclass(frozen=True)
class RayAsset:
    origin_camera: np.ndarray
    directions_camera: np.ndarray
    cone_cos: np.ndarray


def load_alignment(path: str | Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text())["transform_mesh_to_gt"], dtype=np.float64)


def load_training_cameras(path: str | Path) -> list[TrainingFrame]:
    """Training-split inventory written by ``gofish.eval.fiord.training_cameras``."""
    path = Path(path)
    value = json.loads(path.read_text())
    return [TrainingFrame(row["image_id"], np.asarray(row["camera_to_world"], dtype=np.float64),
                          (path.parent / row["rays"]).resolve())
            for row in value["training_images"]]


def load_ray_asset(path: Path) -> RayAsset:
    with np.load(path, allow_pickle=False) as payload:
        directions = np.asarray(payload["directions_camera"], dtype=np.float64)
        origins = np.asarray(payload["origins_camera"], dtype=np.float64)
        cone_cos = np.asarray(payload["cone_cos"], dtype=np.float64)
    origin = origins if origins.shape == (3,) else origins[0]
    norms = np.linalg.norm(directions, axis=1)
    return RayAsset(np.asarray(origin, dtype=np.float64),
                    np.ascontiguousarray(directions / norms[:, None]),
                    np.ascontiguousarray(cone_cos))


def domain_membership(directions: np.ndarray, rays: np.ndarray, cone_cos: np.ndarray) -> np.ndarray:
    """Exact membership of unit directions in the union of pixel-ray cones."""
    from scipy.spatial import cKDTree

    directions = np.asarray(directions, dtype=np.float64)
    if len(directions) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(rays)
    maximum_chord = math.sqrt(max(0.0, 2.0 - 2.0 * float(np.min(cone_cos))))
    result = np.zeros(len(directions), dtype=bool)
    for begin in range(0, len(directions), 65536):
        end = min(begin + 65536, len(directions))
        candidate_lists = tree.query_ball_point(directions[begin:end], maximum_chord)
        for local_index, candidates in enumerate(candidate_lists):
            if not candidates:
                continue
            candidate_array = np.asarray(candidates, dtype=np.int64)
            dots = rays[candidate_array] @ directions[begin + local_index]
            result[begin + local_index] = bool(np.any(dots >= cone_cos[candidate_array]))
    return result


def apply_transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T + transform[:3, 3:4]).T


def hpr_visible_indices(points: np.ndarray, camera_origin: np.ndarray, radius: float) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, np.float64))
    _, indices = cloud.hidden_point_removal(camera_origin.tolist(), radius)
    return np.asarray(indices, dtype=np.int64)


def visible_in_domain(points: np.ndarray, frame: TrainingFrame, asset: RayAsset,
                      transform: np.ndarray, radius: float) -> np.ndarray:
    """Indices of ``points`` (GT frame) visible from ``frame`` inside its calibrated domain."""
    camera_to_gt = transform @ frame.camera_to_world
    origin = apply_transform(camera_to_gt, asset.origin_camera.reshape(1, 3))[0]
    indices = np.unique(hpr_visible_indices(points, origin, radius))
    if len(indices) == 0:
        return indices
    vectors = (np.linalg.inv(camera_to_gt[:3, :3]) @ (points[indices] - origin).T).T
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 0.0
    directions = np.zeros_like(vectors)
    directions[nonzero] = vectors[nonzero] / norms[nonzero, None]
    inside = np.zeros(len(indices), dtype=bool)
    inside[nonzero] = domain_membership(directions[nonzero], asset.directions_camera, asset.cone_cos)
    return indices[inside]


def load_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            array = payload["points"] if "points" in payload else payload["xyz"]
    else:
        import open3d as o3d

        array = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64)[:, :3])


def build_observed_gt(gt_points: np.ndarray, frames: list[TrainingFrame], transform: np.ndarray,
                      nmin: int = NMIN, gt_downsample: int = GT_DOWNSAMPLE,
                      seed: int = SEED, workers: int = 8) -> tuple[np.ndarray, float]:
    """Observed ground truth and the HPR radius used to build it."""
    points = np.asarray(gt_points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    rng = np.random.default_rng(seed)
    if len(points) > gt_downsample:
        points = points[np.sort(rng.choice(len(points), gt_downsample, replace=False))]
    radius = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * 100.0
    count = observation_counts(points, frames, transform, radius, workers=workers)
    return np.ascontiguousarray(points[count >= nmin], dtype=np.float64), radius


def observation_counts(points: np.ndarray, frames: list[TrainingFrame], transform: np.ndarray,
                       radius: float, workers: int = 8) -> np.ndarray:
    """Number of training cameras observing each point (fork pool over cameras)."""
    from multiprocessing import get_context

    global _SHARED
    _SHARED = (points, frames, transform, radius)
    count = np.zeros(len(points), np.uint32)
    with get_context("fork").Pool(workers) as pool:
        for admitted in pool.imap_unordered(_count_worker, range(len(frames)), chunksize=4):
            count[admitted] += 1
    return count


_SHARED = None


def _count_worker(index: int) -> np.ndarray:
    points, frames, transform, radius = _SHARED
    frame = frames[index]
    return visible_in_domain(points, frame, load_ray_asset(frame.rays_path), transform, radius)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(np.asarray(target, np.float64)).query(np.asarray(source, np.float64), k=1, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def sample_mesh(mesh_path: str | Path, transform: np.ndarray, count: int = NUM_MESH_SAMPLES,
                seed: int = SEED) -> np.ndarray:
    """Uniform area samples of the mesh after alignment to the GT frame."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    transformed = o3d.geometry.TriangleMesh(mesh)
    transformed.transform(transform)
    try:
        cloud = transformed.sample_points_uniformly(number_of_points=count, seed=seed)
    except TypeError:
        # Open3D 0.19 has no ``seed`` argument here; seed its global generator instead.
        o3d.utility.random.seed(seed)
        cloud = transformed.sample_points_uniformly(number_of_points=count)
    return np.asarray(cloud.points, dtype=np.float64)


def prf(d_pred_to_gt: np.ndarray, recall: dict[int, float]) -> dict:
    out = {}
    for threshold in THRESHOLDS_M:
        cm = int(round(threshold * 100))
        precision = float(np.mean(d_pred_to_gt < threshold)) if len(d_pred_to_gt) else 0.0
        out[f"P{cm}"], out[f"R{cm}"] = precision, recall[cm]
        out[f"F{cm}"] = 2 * precision * recall[cm] / (precision + recall[cm]) if precision + recall[cm] else 0.0
    return out
