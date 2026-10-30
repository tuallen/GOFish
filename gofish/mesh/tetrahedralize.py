# Portions derived from GOF (https://github.com/autonomousvision/gaussian-opacity-fields),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Delaunay tetrahedralization of the sites with CGAL.

Uses the CGAL extension from Tetra-NeRF (``tetranerf.utils.extension``), as in
GOF. Only site coordinates enter the triangulation.
"""

from __future__ import annotations

import numpy as np
import torch

from .sites import Sites


def tetrahedralize(sites: Sites) -> torch.Tensor:
    """Return the tetrahedra as an int64 [T, 4] CPU tensor of site indices."""
    try:
        from tetranerf.utils.extension import triangulate
    except ImportError as error:
        raise ImportError(
            "mesh extraction needs the Tetra-NeRF CGAL extension; see the README"
        ) from error

    if not np.isfinite(sites.points).all():
        raise ValueError("site coordinates are not finite")
    points = torch.from_numpy(np.ascontiguousarray(sites.points)).contiguous().cpu()
    cells = triangulate(points.clone()).to(torch.int64).cpu().contiguous()
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.numel() == 0 \
            or int(cells.min()) < 0 or int(cells.max()) >= len(points):
        raise RuntimeError("CGAL returned invalid tetrahedra")
    return cells
