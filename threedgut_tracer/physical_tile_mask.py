# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""16x16 tile mask of the calibrated camera domain.

Tiles without a single valid pixel are skipped by the renderer.  The mask is
built once per camera domain and registered with the native rasterizer under
a 64-hex token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


TILE = 16


@dataclass(frozen=True)
class PhysicalTileBinding:
    mask: object
    token: str


def _token(*, domain: str, height: int, width: int, device: str, tile: int = TILE) -> str:
    payload = {"domain": domain, "device": device, "height": height, "tile": tile, "width": width}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_physical_tile_binding(batch, rays, domain: str = "full-raster", tile: int = TILE):
    """Build the tile mask of ``batch.physical_fov_mask`` (all tiles if absent); ``tile`` = tile edge."""
    TILE_ = int(tile)
    import torch
    import torch.nn.functional as functional

    if rays.ndim != 4 or rays.shape[0] != 1 or rays.shape[-1] != 3 or rays.device.type != "cuda":
        raise RuntimeError("physical tile pruning requires CUDA [1,H,W,3] rays")
    height, width = int(rays.shape[1]), int(rays.shape[2])
    physical = getattr(batch, "physical_fov_mask", None)
    if physical is None:
        tiles = torch.ones(
            ((height + TILE_ - 1) // TILE_, (width + TILE_ - 1) // TILE_),
            dtype=torch.uint8, device=rays.device,
        )
    else:
        if (not torch.is_tensor(physical) or physical.dtype != torch.bool
                or physical.ndim != 4 or tuple(physical.shape) != (1, height, width, 1)
                or physical.device != rays.device):
            raise RuntimeError("physical_fov_mask must be a bool [1,H,W,1] tensor on the ray device")
        mask = physical[0, ..., 0]
        padded_height = ((height + TILE_ - 1) // TILE_) * TILE_
        padded_width = ((width + TILE_ - 1) // TILE_) * TILE_
        padded = functional.pad(
            mask, (0, padded_width - width, 0, padded_height - height),
            mode="constant", value=False,
        )
        tiles = padded.reshape(
            padded_height // TILE_, TILE_, padded_width // TILE_, TILE_
        ).any(dim=3).any(dim=1).to(dtype=torch.uint8).contiguous()
    return PhysicalTileBinding(
        mask=tiles,
        token=_token(domain=domain, height=height, width=width, device=str(rays.device), tile=TILE_),
    )
