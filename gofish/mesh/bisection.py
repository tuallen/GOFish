# Portions derived from GOF and SOF (https://github.com/autonomousvision/gaussian-opacity-fields, https://github.com/r4dl/SOF),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Eight sign-preserving bisection rounds on each Marching-Tetrahedra crossing edge.

Each crossing edge is bracketed by a solid (O >= 1/2) and a free endpoint.
Every round evaluates the same opacity field at the bracket midpoint and
replaces the endpoint of the same sign; the vertex is the final midpoint.
Unobserved midpoints are opaque. Every edge is refined independently, so the
batch partition does not change any output.
"""

from __future__ import annotations

import numpy as np
import torch

from .cameras import TrainingCamera
from .field import camera_inputs
from .shape import EffectiveShape

ROUNDS = 8
DEVICE_HEADROOM_BYTES = 1536 << 20
MAX_BATCH_EDGES = 1 << 24
BATCH_EDGE_QUANTUM = 1 << 16
BYTES_PER_EDGE = 1024  # conservative device memory per edge (about 4x measured)


def batch_size(edge_count: int, free_bytes: int, camera_bytes: int) -> int:
    """All edges in one batch when they fit, else the largest multiple of 2^16 that fits."""
    available = free_bytes - camera_bytes - DEVICE_HEADROOM_BYTES
    if available < BYTES_PER_EDGE:
        raise MemoryError("not enough device memory for the bisection camera table")
    fit = available // BYTES_PER_EDGE
    if fit < edge_count:
        fit = max(BATCH_EDGE_QUANTUM, fit // BATCH_EDGE_QUANTUM * BATCH_EDGE_QUANTUM)
    return int(min(edge_count, MAX_BATCH_EDGES, fit))


def _padded_batch(endpoints: np.ndarray, fields: np.ndarray, start: int, end: int, capacity: int):
    """Fixed-capacity copy of edges [start, end); padding repeats the last edge."""
    count = end - start
    endpoint_values = np.empty((capacity, 2, 3), dtype="<f4")
    field_values = np.empty((capacity, 2), dtype="<f4")
    endpoint_values[:count] = endpoints[start:end]
    field_values[:count] = fields[start:end]
    if count < capacity:
        endpoint_values[count:] = endpoint_values[count - 1]
        field_values[count:] = field_values[count - 1]
    return endpoint_values, field_values


def refine_crossings(raster, shape: EffectiveShape, cameras: list[TrainingCamera],
                     endpoints: np.ndarray, endpoint_fields: np.ndarray,
                     device: torch.device) -> np.ndarray:
    """Return the refined vertex [E, 3] float32 of every crossing edge.

    ``endpoints`` is [E, 2, 3] and ``endpoint_fields`` is [E, 2] (values of 1/2 - O).
    """
    edge_count = int(endpoints.shape[0])
    packed = torch.from_numpy(shape.packed_density()).to(device).contiguous()
    sigma = torch.from_numpy(np.array(shape.sigma_max, dtype=np.float32, order="C", copy=True)).to(device).contiguous()

    free_bytes, _total = torch.cuda.mem_get_info(device)
    camera_bytes = 0
    for camera in cameras:
        camera_bytes += int(camera.ray_direction.numel()) * 4
        if camera.physical_mask is not None:
            camera_bytes += int(camera.physical_mask.numel()) * int(camera.physical_mask.element_size())
    capacity = batch_size(edge_count, int(free_bytes), camera_bytes)

    vertices = np.empty((edge_count, 3), dtype="<f4")
    token = None
    released = False
    try:
        for start in range(0, edge_count, capacity):
            end = min(edge_count, start + capacity)
            count = end - start
            endpoint_values, field_values = _padded_batch(endpoints, endpoint_fields, start, end, capacity)
            ends = torch.from_numpy(endpoint_values).to(device).contiguous()
            fields = torch.from_numpy(field_values).to(device).contiguous()
            f0, f1 = fields[:, 0], fields[:, 1]
            crossing = ((f0 <= 0.0) & (f1 >= 0.0)) | ((f1 <= 0.0) & (f0 >= 0.0))
            if not (bool(torch.isfinite(fields).all().item()) and bool(crossing.all().item())):
                raise ValueError("bisection input edges are not finite sign crossings")
            swap = f0 > 0.0
            solid = torch.where(swap[:, None], ends[:, 1], ends[:, 0]).contiguous()
            free = torch.where(swap[:, None], ends[:, 0], ends[:, 1]).contiguous()
            del endpoint_values, field_values, ends, fields, f0, f1, crossing, swap

            if token is None:
                # The camera table is staged once and reused by every batch.
                probe = torch.add(solid, free).mul_(0.5).contiguous()
                token = raster.begin_bisection_table(packed, sigma, probe)
                for camera in cameras:
                    raster.stage_bisection_camera(token, *camera_inputs(camera, device))
                raster.seal_bisection_table(token)
                del probe

            for round_index in range(ROUNDS):
                midpoint = torch.add(solid, free).mul_(0.5).contiguous()
                above_u8, observed_u8, exact_u8 = raster.evaluate_bisection_round(token, midpoint, round_index)
                observed = observed_u8.bool()
                above = above_u8.bool() | ~observed
                exact = exact_u8.bool() & observed
                solid = torch.where(above[:, None], midpoint, solid).contiguous()
                free = torch.where(((~above) | exact)[:, None], midpoint, free).contiguous()
                del midpoint, above_u8, observed_u8, exact_u8, observed, above, exact

            refined = torch.add(solid, free).mul_(0.5).contiguous()
            if not bool(torch.isfinite(refined).all().item()):
                raise ValueError("refined vertices are not finite")
            vertices[start:end] = np.asarray(refined[:count].detach().cpu().numpy(), dtype="<f4")
            del solid, free, refined
            if end < edge_count:
                raster.reset_bisection_batch(token)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        raster.release_bisection_table(token)
        released = True
    finally:
        if token is not None and not released:
            try:
                raster.release_bisection_table(token)
            except RuntimeError:
                pass
    return vertices
