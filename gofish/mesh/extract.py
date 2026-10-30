# Portions derived from GOF (https://github.com/autonomousvision/gaussian-opacity-fields),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""GOFish mesh extraction from a geometry checkpoint.

Pipeline:
  1. effective shape: Mip-filtered Gaussians of the geometry checkpoint;
  2. sites: center plus eight support-box corners per Gaussian (opacity > 1/255);
  3. fresh CGAL Delaunay tetrahedralization of the sites;
  4. matched camera-min opacity field O at the sites over the training cameras;
  5. Marching Tetrahedra of O = 1/2, then eight bisection rounds per crossing;
  6. SOF's vertex filter, and GOF's cell filter before Marching Tetrahedra.

Outputs ``mesh_raw.ply`` (5), ``mesh_sof.ply`` (5 + SOF filter) and
``mesh.ply`` (cell filter + SOF filter, the reported GOFish mesh).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from threedgrut.utils.misc import apply_config_overrides

from .bisection import refine_crossings
from .build import load_field_plugin
from .cameras import all_frames_config, load_training_cameras
from .field import evaluate_field, signed_field
from .filters import apply_vertex_filter, cell_filtered_mesh, gof_cell_filter, sof_vertex_filter
from .marching_tetrahedra import marching_tetrahedra
from .ply import write_ply
from .shape import effective_shape
from .sites import generate_sites
from .tetrahedralize import tetrahedralize


def _new_raster(plugin, conf):
    torch.zeros(1, device="cuda")  # initialize the CUDA context
    return plugin.SplatRaster(OmegaConf.to_container(conf))


def extract_mesh(checkpoint_path: Path, output: Path, dataset_path: str | None = None,
                 save_intermediates: bool = False, overrides=None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conf = apply_config_overrides(checkpoint["config"], overrides)

    tick = time.perf_counter()
    shape = effective_shape(checkpoint, device="cuda")
    del checkpoint
    sites = generate_sites(shape)
    timings["sites"] = time.perf_counter() - tick

    tick = time.perf_counter()
    cells = tetrahedralize(sites)
    timings["tetrahedralization"] = time.perf_counter() - tick

    tick = time.perf_counter()
    plugin = load_field_plugin(conf)
    cameras = load_training_cameras(conf, plugin, dataset_path)
    raster_conf = all_frames_config(conf, dataset_path)
    timings["setup"] = time.perf_counter() - tick

    tick = time.perf_counter()
    raster = _new_raster(plugin, raster_conf)
    minimum, observed = evaluate_field(raster, shape, cameras, sites.points, device)
    del raster
    torch.cuda.empty_cache()
    timings["field"] = time.perf_counter() - tick

    tick = time.perf_counter()
    points = torch.from_numpy(sites.points)
    scale = torch.from_numpy(sites.scale)
    values = torch.from_numpy(signed_field(minimum, observed))
    with torch.no_grad():
        vertices_tuple, _, faces_tuple, edge_tuple = marching_tetrahedra(
            points.float().cuda()[None], cells.long().cuda(), values.cuda()[None],
            scale.float().cuda()[None])
        endpoints, endpoint_values = vertices_tuple[0]
        endpoints = endpoints.cpu().numpy().astype("<f4", copy=False)
        endpoint_fields = endpoint_values[..., 0].cpu().numpy().astype("<f4", copy=False)
        raw_faces = faces_tuple[0].cpu().numpy().astype("<i4", copy=False)
        raw_edges = edge_tuple[0].cpu().numpy().astype("<i8", copy=False)
    del vertices_tuple, faces_tuple, edge_tuple, endpoint_values
    torch.cuda.empty_cache()
    timings["marching_tetrahedra"] = time.perf_counter() - tick

    tick = time.perf_counter()
    raster = _new_raster(plugin, raster_conf)
    raw_vertices = refine_crossings(raster, shape, cameras, endpoints, endpoint_fields, device)
    del raster
    torch.cuda.empty_cache()
    timings["bisection"] = time.perf_counter() - tick

    tick = time.perf_counter()
    write_ply(output / "mesh_raw.ply", raw_vertices, raw_faces)
    keep_vertex = sof_vertex_filter(points, scale, raw_edges, device)
    sof_vertices, sof_faces = apply_vertex_filter(raw_vertices, raw_faces, keep_vertex)
    write_ply(output / "mesh_sof.ply", sof_vertices, sof_faces)

    owner = torch.from_numpy(sites.owner)
    kept_cells = gof_cell_filter(cells, points.float(), scale, owner)
    mesh_vertices, mesh_faces = cell_filtered_mesh(
        points.float(), kept_cells, values, scale, raw_edges, raw_vertices, keep_vertex, device)
    write_ply(output / "mesh.ply", mesh_vertices, mesh_faces)
    timings["filters"] = time.perf_counter() - tick
    timings["total"] = time.perf_counter() - started

    if save_intermediates:
        np.save(output / "sites.npy", sites.points)
        np.save(output / "field_min_opacity.npy", minimum)
        np.save(output / "field_observed.npy", observed)

    summary = {
        "checkpoint": str(checkpoint_path),
        "gaussians": shape.count,
        "sites": sites.count,
        "tetrahedra": int(cells.shape[0]),
        "cells_after_cell_filter": int(kept_cells.shape[0]),
        "training_cameras": len(cameras),
        "crossing_edges": int(raw_edges.shape[0]),
        "faces": {"raw": int(len(raw_faces)), "sof": int(len(sof_faces)), "mesh": int(len(mesh_faces))},
        "vertices": {"raw": int(len(raw_vertices)), "sof": int(len(sof_vertices)), "mesh": int(len(mesh_vertices))},
        "seconds": timings,
    }
    (output / "mesh_info.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
