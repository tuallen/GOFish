# Portions derived from GOF and SOF (https://github.com/autonomousvision/gaussian-opacity-fields, https://github.com/r4dl/SOF),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Mesh filters: GOF's tetrahedron cell filter and SOF's vertex filter."""

from __future__ import annotations

import numpy as np
import torch

from .marching_tetrahedra import marching_tetrahedra


def sof_vertex_filter(points: torch.Tensor, scale: torch.Tensor, edges: np.ndarray,
                      device: torch.device) -> np.ndarray:
    """SOF's released filter: keep a crossing vertex when its edge is no longer
    than the sum of the two endpoint Gaussians' support extents.

    ``points`` [M, 3] and ``scale`` [M, 1] are the site tensors, ``edges`` [E, 2]
    the crossing site pairs. Returns the per-vertex keep mask [E] (bool).
    """
    with torch.no_grad():
        pts = points.float().to(device)
        scales = scale.float().reshape(-1).to(device)
        edge = torch.from_numpy(np.asarray(edges).copy()).long().to(device)
        distance = torch.linalg.vector_norm(pts[edge[:, 0]] - pts[edge[:, 1]], dim=-1)
        return (distance <= scales[edge[:, 0]] + scales[edge[:, 1]]).cpu().numpy()


def apply_vertex_filter(vertices: np.ndarray, faces: np.ndarray, keep_vertex: np.ndarray):
    """Keep the selected vertices and the faces whose three vertices are all kept."""
    keep_face = keep_vertex[np.asarray(faces)].all(axis=1)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[keep_vertex] = np.arange(int(keep_vertex.sum()), dtype=np.int64)
    kept_vertices = np.ascontiguousarray(np.asarray(vertices)[keep_vertex], dtype="<f4")
    kept_faces = np.ascontiguousarray(remap[np.asarray(faces)[keep_face]], dtype="<i4")
    return kept_vertices, kept_faces


def gof_cell_filter(cells: torch.Tensor, points: torch.Tensor, scale: torch.Tensor,
                    owner: torch.Tensor) -> torch.Tensor:
    """GOF's cell filter: drop every tetrahedron with an edge that joins two
    different Gaussians farther apart than the sum of their support extents.
    Evaluated on the CPU. Returns the kept cells.
    """
    pscale = scale.float().reshape(-1)
    keep = torch.ones(cells.shape[0], dtype=torch.bool)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ci, cj = cells[:, i], cells[:, j]
        d = torch.linalg.vector_norm(points[ci] - points[cj], dim=-1)
        bad = (owner[ci] != owner[cj]) & (d > pscale[ci] + pscale[cj])
        keep &= ~bad
        del ci, cj, d, bad
    return cells[keep]


def cell_filtered_mesh(points: torch.Tensor, kept_cells: torch.Tensor, values: torch.Tensor,
                       scale: torch.Tensor, raw_edges: np.ndarray, raw_vertices: np.ndarray,
                       keep_vertex: np.ndarray, device: torch.device):
    """Marching Tetrahedra on the kept cells, reusing the refined vertex and the
    vertex-filter decision of each crossing edge from the full extraction.

    Every crossing edge of a cell subset is a crossing edge of the full set, so
    no field evaluation or bisection is repeated. Returns (vertices, faces).
    """
    count = int(points.shape[0])
    pscale = scale.float().reshape(-1)
    with torch.no_grad():
        # Only integer topology is used, so evaluating on the GPU is exact.
        vt, _, ft, et = marching_tetrahedra(
            points[None].to(device), kept_cells.to(device), values[None].to(device),
            pscale[None, :, None].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    edges = et[0].cpu().numpy().astype("<i8")
    faces = ft[0].cpu().numpy().astype(np.int64)
    del vt, ft, et

    key0 = raw_edges[:, 0] * count + raw_edges[:, 1]
    order0 = np.argsort(key0, kind="stable")
    key0_sorted = key0[order0]
    key = edges[:, 0] * count + edges[:, 1]
    position = np.searchsorted(key0_sorted, key)
    if not (np.all(position < len(key0_sorted)) and np.all(key0_sorted[position] == key)):
        raise RuntimeError("a cell-filtered crossing edge is not a crossing edge of the full set")
    source = order0[position]

    vertices = raw_vertices[source]
    keep = keep_vertex[source]
    keep_face = keep[faces].all(axis=1)
    remap = np.full(len(edges), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    kept_vertices = np.ascontiguousarray(vertices[keep], dtype="<f4")
    kept_faces = np.ascontiguousarray(remap[faces[keep_face]], dtype="<i4")
    return kept_vertices, kept_faces
