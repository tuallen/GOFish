# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Binary little-endian PLY output (float32 vertices, uchar/int32 triangle lists)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FACE_DTYPE = np.dtype([("n", "u1"), ("v", "<i4", (3,))])
_CHUNK = 1_000_000


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("expected [V, 3] vertices and [F, 3] faces")
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError("face indices are out of range")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    with Path(path).open("wb") as stream:
        stream.write(header)
        stream.write(vertices.tobytes())
        for start in range(0, len(faces), _CHUNK):
            current = faces[start:start + _CHUNK]
            packed = np.empty(len(current), dtype=_FACE_DTYPE)
            packed["n"] = 3
            packed["v"] = current
            stream.write(packed.tobytes())
