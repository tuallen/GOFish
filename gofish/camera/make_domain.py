# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Write camera-domain records for the fisheye cameras of a COLMAP model.

    python -m gofish.camera.make_domain <sparse/0> out.json \
        [--aperture CAMERA_ID CX CY RADIUS ...]

Cameras with an ``--aperture`` (native-pixel circle) get an ``aperture``
record; the others a full-raster ``max_angle`` record at the first monotone
branch limit of their distortion polynomial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gofish.camera.domain import monotone_branch_limit, radial_value


def _read_cameras(model_dir: Path):
    from threedgrut.datasets.utils import read_colmap_intrinsics_binary, read_colmap_intrinsics_text

    if (model_dir / "cameras.bin").is_file():
        return read_colmap_intrinsics_binary(str(model_dir / "cameras.bin"))
    return read_colmap_intrinsics_text(str(model_dir / "cameras.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="COLMAP model directory with cameras.bin/.txt")
    parser.add_argument("output", type=Path)
    parser.add_argument("--aperture", nargs=4, action="append", default=[], metavar=("ID", "CX", "CY", "R"))
    args = parser.parse_args()

    apertures = {int(row[0]): (float(row[1]), float(row[2]), float(row[3])) for row in args.aperture}
    records = []
    for camera_id, intrinsic in sorted(_read_cameras(args.model_dir).items()):
        if intrinsic.model != "OPENCV_FISHEYE":
            continue
        params = [float(value) for value in intrinsic.params]
        theta_limit = monotone_branch_limit(params[4:8])
        record = {"camera_id": int(camera_id), "model": intrinsic.model, "width": int(intrinsic.width),
                  "height": int(intrinsic.height), "params": params}
        if camera_id in apertures:
            cx, cy, radius = apertures[camera_id]
            record |= {"kind": "aperture", "aperture_center": [cx, cy], "aperture_radius": radius,
                       "theta_limit": theta_limit, "rho_limit": radial_value(theta_limit, params[4:8])}
        else:
            record |= {"kind": "max_angle", "max_angle": theta_limit}
        records.append(record)
    args.output.write_text(json.dumps({"cameras": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
