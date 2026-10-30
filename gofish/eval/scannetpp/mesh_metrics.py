# Portions derived from CoMe (https://github.com/r4dl/CoMe),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""ScanNet++ mesh metrics with CoMe's released per-mesh protocol (F@5 cm).

Reproduces ``CoMe/mesh_utils/eval_SCN.py``:

* reconstruction points are every mesh vertex plus every triangle centroid
  (mesh read with trimesh);
* ground truth is ``scans/mesh_aligned_0.05.ply`` read as a point cloud;
* starting from the ``--frame`` transform, three scaled-ICP stages
  (Open3D point-to-point with scaling, the released convergence criteria) are fitted for this
  mesh: 5 cm voxels / 4 m, 2.5 cm voxels / 1 m, then the uniformly limited
  (<= 4M points) clouds / 0.10 m; every stage crops both clouds to the GT
  axis-aligned bounding box;
* final scoring crops to the GT box, voxelises both clouds at 2.5 cm and
  counts point-to-point distances below 5 cm.

Frames: ``colmap`` (identity; COLMAP/laser-aligned meshes, e.g. GOFish) or
``nerfstudio`` (CoMe's OpenGL convention).  Two sanity checks only fail a run,
they never choose a transform: at least half of the initial points must lie
in the GT box, and recall / precision must be at least 0.25.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

TAU_M = 0.05
FINAL_VOXEL_M = 0.025
ICP_MAX_ITER = 20
MAX_UNIFORM_POINTS = 4_000_000
FRAMES = {
    "colmap": np.eye(4),
    "nerfstudio": np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]),
}
MIN_INITIAL_INSIDE_FRACTION = 0.5
MIN_RECALL_TO_PRECISION = 0.25


class CoMeProtocol:
    """Open3D operations in the order of the released CoMe script."""

    def __init__(self, o3d):
        self.o3d = o3d

    def crop(self, cloud, volume, transform):
        result = copy.deepcopy(cloud)
        result.transform(transform)
        return result.crop(volume)

    def scaled_icp(self, source, target, initial, threshold):
        reg = self.o3d.pipelines.registration
        # Positional criteria arguments exactly as in the released script.
        result = reg.registration_icp(source, target, threshold, np.eye(4),
                                      reg.TransformationEstimationPointToPoint(True),
                                      reg.ICPConvergenceCriteria(1e-6, ICP_MAX_ITER))
        return np.matmul(np.asarray(result.transformation), initial)

    def stage_voxel(self, source, target, initial, volume, voxel, threshold):
        s = self.crop(source, volume, initial).voxel_down_sample(voxel)
        t = self.crop(target, volume, np.eye(4)).voxel_down_sample(voxel)
        return self.scaled_icp(s, t, initial, threshold)

    @staticmethod
    def uniform_limit(cloud):
        count = len(cloud.points)
        if count > MAX_UNIFORM_POINTS:
            return cloud.uniform_down_sample(int(round(count / float(MAX_UNIFORM_POINTS))))
        return cloud

    def stage_uniform(self, source, target, initial, volume, threshold):
        s = self.uniform_limit(self.crop(source, volume, initial))
        t = self.uniform_limit(self.crop(target, volume, np.eye(4)))
        return self.scaled_icp(s, t, initial, threshold)

    def distances(self, source, target, transform, volume):
        s = self.crop(source, volume, transform).voxel_down_sample(FINAL_VOXEL_M)
        t = self.crop(target, volume, np.eye(4)).voxel_down_sample(FINAL_VOXEL_M)
        return (np.asarray(s.compute_point_cloud_distance(t), dtype=np.float64),
                np.asarray(t.compute_point_cloud_distance(s), dtype=np.float64))


def evaluate(mesh_path: Path, gt_path: Path, frame: str) -> dict:
    import open3d as o3d
    import trimesh

    mesh = trimesh.load_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces).astype(np.int64)
    points = np.concatenate((vertices, vertices[faces].mean(axis=1)), axis=0)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)
    target = o3d.io.read_point_cloud(str(gt_path))
    initial = FRAMES[frame].copy()

    volume = target.get_axis_aligned_bounding_box()
    probe = copy.deepcopy(source)
    probe.transform(initial)
    inside = len(probe.crop(volume).points) / float(len(points))

    protocol = CoMeProtocol(o3d)
    t1 = protocol.stage_voxel(source, target, initial.copy(), volume, TAU_M, TAU_M * 80.0)
    t2 = protocol.stage_voxel(source, target, t1.copy(), volume, TAU_M / 2.0, TAU_M * 20.0)
    t3 = protocol.stage_uniform(source, target, t2.copy(), volume, 2.0 * TAU_M)
    d_source, d_target = protocol.distances(source, target, t3.copy(), volume)
    precision = float(np.mean(d_source < TAU_M))
    recall = float(np.mean(d_target < TAU_M))
    f_score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    if inside < MIN_INITIAL_INSIDE_FRACTION:
        raise RuntimeError(f"frame '{frame}' leaves only {inside:.3f} of the mesh inside the GT volume")
    if precision > 0.0 and recall / precision < MIN_RECALL_TO_PRECISION:
        raise RuntimeError("registration produced a recall collapse; check the frame")
    return {"precision": precision, "recall": recall, "F": f_score,
            "accuracy": float(d_source.mean()), "completeness": float(d_target.mean()),
            "transform": t3.tolist(), "initial_inside_fraction": inside}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="<scene>/scans/mesh_aligned_0.05.ply")
    parser.add_argument("--frame", choices=sorted(FRAMES), default="colmap")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.mesh, args.gt, args.frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"P {result['precision']:.4f} R {result['recall']:.4f} F@5 {result['F']:.4f}")


if __name__ == "__main__":
    main()
