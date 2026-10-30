# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Tanks & Temples mesh metrics with the official evaluator.

Runs ``official/eval_TNT.py`` (the Tanks and Temples toolbox as packaged by
GOF: mesh vertices plus triangle centroids, official crop, trajectory
alignment and per-scene thresholds) with Open3D's random generator seeded to
0 so the RANSAC stage is deterministic.

``--gt-dir`` is the official evaluation folder of the scene (``<Scene>.ply``,
``<Scene>.json`` crop, ``<Scene>_trans.txt``, ``<Scene>_COLMAP_SfM.log``);
``--traj`` is the COLMAP SfM log of the training reconstruction used by GOF's
preprocessing.  Results (precision, recall, F-score at the official tau) are
written to ``<out>/results.json``.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parent / "official"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import open3d as o3d

    # The official script runs from its own directory, so every path is made absolute first.
    argv = [str(OFFICIAL / "eval_TNT.py"), "--dataset-dir", str(args.gt_dir.resolve()),
            "--traj-path", str(args.traj.resolve()), "--ply-path", str(args.mesh.resolve()),
            "--out-dir", str(args.out.resolve())]
    o3d.utility.random.seed(args.seed)
    sys.path.insert(0, str(OFFICIAL))
    os.chdir(OFFICIAL)
    sys.argv = argv
    runpy.run_path(str(OFFICIAL / "eval_TNT.py"), run_name="__main__")


if __name__ == "__main__":
    main()
