# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Wide-Angle Appearance Modeling inputs of a GOFish dataset (``gofish_ppisp.wide_angle`` on its cameras)."""

from __future__ import annotations

import torch

from gofish_ppisp.wide_angle import flatfield_gauge, geometry_from


def wide_angle_geometry(dataset, num_cameras: int) -> list:
    """Per camera (pp_x, pp_y, rho_ap): principal point and the calibrated-domain radius
    |p - pp| / max(W, H) of the farthest valid pixel (the image corner without a domain)."""
    geometry = [None] * num_cameras
    for cam_id, cam_idx in dataset._camera_id_to_idx.items():
        pd, _ro, _rd, _nm, pix = dataset.intrinsics[cam_id]
        pp = pd["principal_point"] if isinstance(pd, dict) else pd.principal_point
        res = pd["resolution"] if isinstance(pd, dict) else pd.resolution
        pix = torch.as_tensor(pix).reshape(-1, 2)
        support = getattr(dataset, "camera_supports", {}).get(int(cam_id))
        if support is not None:
            m = torch.as_tensor(support.torch_mask(device="cpu")).reshape(-1).bool()
            if m.numel() != pix.shape[0] or not bool(m.any()):
                raise RuntimeError("wide-angle PPISP: calibrated-domain mask/raster mismatch")
            pix = pix[m]
        geometry[cam_idx] = geometry_from(pp, res, pix)
    if any(g is None for g in geometry):
        raise RuntimeError("wide-angle PPISP: camera without geometry")
    return geometry


def init_flatfield_gauge(ppisp, dataset, geometry: list, gn: int) -> None:
    """Static flat-field gauge of every camera: cells whose centre lies within rho_ap / 2 stay at identity."""
    with torch.no_grad():
        for cam_idx, camera_geometry in enumerate(geometry):
            pd = dataset.intrinsics[[c for c, i in dataset._camera_id_to_idx.items() if i == cam_idx][0]][0]
            res = pd["resolution"] if isinstance(pd, dict) else pd.resolution
            ppisp.flatfield_free[cam_idx] = flatfield_gauge(camera_geometry, res, gn).to(ppisp.flatfield_free.device)
