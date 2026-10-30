#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Add a new camera calibration to the renderer's complete-admission table.

Complete admission needs a per-calibration curvature bound compiled into the
renderer.  ``train.py`` and ``render.py`` stop before compiling when a camera
has no row and print the ``--camera`` command for it; run that command once per
new camera or training resolution.  The renderer recompiles on the next run,
because its extension name hashes the table.  Existing rows are never
modified.  Only OPENCV_FISHEYE and distortion-free PINHOLE cameras have rows;
undistort other COLMAP models first (``colmap image_undistorter``).

Examples::

    # COLMAP calibration, images downsampled by 4
    python -m gofish.tools.build_calibration_table \
        --colmap data/scene/sparse/0/cameras.bin --downsample 4

    # exact runtime parameters, as printed by train.py, render.py or the renderer on a table miss
    python -m gofish.tools.build_calibration_table \
        --camera=OPENCV_FISHEYE,816,816,fx,fy,cx,cy,k1,k2,k3,k4,max_angle

``--camera`` takes one comma-separated value, so negative coefficients such as
``-0x1.4p-3`` are never read as options.  Numbers are decimal or float.hex().
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from threedgut_tracer.kb_table.table import (
    HEADER_PATH, PackedTable, RuntimeCamera, TableError, cameras_from_colmap, f32)


def _parse_camera(value: str) -> RuntimeCamera:
    model, width, height, *numbers = [field.strip() for field in value.split(",")]
    numbers = [float.fromhex(v) if v.lower().startswith(("0x", "-0x")) else float(v)
               for v in numbers]
    if model == "OPENCV_FISHEYE":
        if len(numbers) != 9:
            raise TableError("OPENCV_FISHEYE needs fx fy cx cy k1 k2 k3 k4 max_angle")
        parameters, max_angle = numbers[:8], numbers[8]
    elif model == "PINHOLE":
        if len(numbers) != 4:
            raise TableError("PINHOLE needs fx fy cx cy")
        parameters, max_angle = numbers, 0.0
    else:
        raise TableError(f"unsupported model {model}")
    return RuntimeCamera(model, int(width), int(height), tuple(f32(v) for v in parameters), max_angle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--colmap", type=Path, help="COLMAP cameras.txt or cameras.bin")
    parser.add_argument("--downsample", type=int, default=1,
                        help="training image downsample factor (active size = native // factor)")
    parser.add_argument("--width", type=int, help="active raster width (overrides --downsample)")
    parser.add_argument("--height", type=int, help="active raster height (overrides --downsample)")
    parser.add_argument("--max-angle", type=float,
                        help="fisheye maximum ray angle in radians (default: calibrated domain)")
    parser.add_argument("--camera", action="append", default=[], metavar="MODEL,W,H,P1,...",
                        help="one camera as comma-separated exact runtime values; repeatable")
    parser.add_argument("--header", type=Path, default=HEADER_PATH,
                        help="table header to extend in place")
    args = parser.parse_args(argv)

    try:
        cameras = [_parse_camera(values) for values in args.camera]
        if args.colmap is not None:
            cameras += cameras_from_colmap(args.colmap, args.width, args.height,
                                           args.downsample, args.max_angle)
        if not cameras:
            parser.error("give --colmap and/or --camera")
        table = PackedTable.load(args.header)
        added = table.append(cameras)
        if added:
            table.write(args.header)
    except TableError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{added} camera(s) added, {len(table.selector_rows)} in table: {args.header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
