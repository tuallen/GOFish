# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""FIORD mesh metrics: precision / recall / F-score at 5, 10, 25 and 30 cm.

The mesh is aligned with the fixed per-scene SfM-to-LiDAR transform
(``alignments/<scene>.json``) and 400k points are sampled uniformly over its
area (Open3D, seed 0).  Recall is measured against the HPR-observed training
ground truth (``observed_gt``, >= 2 training cameras, GT downsampled to 1M
points with seed 0).  Two precision populations are reported:

* ``nocrop``: every sample;
* ``C`` (main tables): samples inside the axis-aligned bounding box of the
  observed GT that are observed from >= 1 training camera inside its
  calibrated domain (the same HPR rule applied to the prediction).

Both rules depend only on the ground truth and the training cameras.  The
observed GT is cached in ``--cache`` and reused across runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gofish.eval.fiord import observed_gt as O


def observed_ground_truth(args, frames, transform) -> tuple[np.ndarray, float]:
    points_path = args.cache / "observed_gt.npy"
    meta_path = args.cache / "observed_gt.json"
    if points_path.exists():
        return np.load(points_path), float(json.loads(meta_path.read_text())["hpr_radius"])
    points, radius = O.build_observed_gt(O.load_points(args.gt), frames, transform, workers=args.workers)
    args.cache.mkdir(parents=True, exist_ok=True)
    np.save(points_path, points)
    meta_path.write_text(json.dumps({"hpr_radius": radius, "observed_gt_count": int(len(points)),
                                     "nmin": O.NMIN, "gt_downsample": O.GT_DOWNSAMPLE, "seed": O.SEED}) + "\n")
    return points, radius


def evaluate(args) -> dict:
    transform = O.load_alignment(args.alignment)
    frames = O.load_training_cameras(args.cameras)
    gt, radius = observed_ground_truth(args, frames, transform)
    pred = O.sample_mesh(args.mesh, transform)
    d_pred_to_gt = O.nearest_distances(pred, gt)
    d_gt_to_pred = O.nearest_distances(gt, pred)
    recall = {int(round(t * 100)): float(np.mean(d_gt_to_pred < t)) for t in O.THRESHOLDS_M}
    counts = O.observation_counts(pred, frames, transform, radius, workers=args.workers)
    lo, hi = gt.min(0), gt.max(0)
    in_box = np.all((pred >= lo) & (pred <= hi), axis=1)
    crop = in_box & (counts >= 1)
    return {
        "nocrop": O.prf(d_pred_to_gt, recall),
        "C": {"kept_fraction": float(crop.mean()), **O.prf(d_pred_to_gt[crop], recall)},
        "accuracy_mean": float(np.mean(d_pred_to_gt)),
        "completeness_mean": float(np.mean(d_gt_to_pred)),
        "observed_gt_points": int(len(gt)),
        "training_cameras": len(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="LiDAR ground-truth point cloud (.npy/.ply)")
    parser.add_argument("--alignment", type=Path, required=True, help="alignments/<scene>.json")
    parser.add_argument("--cameras", type=Path, required=True,
                        help="training_cameras.json from gofish.eval.fiord.training_cameras")
    parser.add_argument("--cache", type=Path, required=True, help="observed-GT cache directory (per scene)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"F@10 C {result['C']['F10']:.4f}  no-crop {result['nocrop']['F10']:.4f}")


if __name__ == "__main__":
    main()
