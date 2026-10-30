# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Wide-Angle Appearance Modeling factors of GOFish's PPISP extension, in torch only.

``radial_attenuation`` and ``flatfield_attenuation`` are the per-pixel factors
W_r = w(rho) and W_ff = exp(-a(u)), whose product W is applied with PPISP's vignetting; ``geometry_from`` and
``flatfield_gauge`` build their per-camera inputs from tensors.  See
``gofish.appearance.wide_angle`` for the model and the configuration keys.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_centres(width: int, height: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``[H * W, 2]`` (x, y) pixel centres, ``(i + 0.5, j + 0.5)``, row-major as PPISP's pixel_coords."""
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack([x + 0.5, y + 0.5], -1).reshape(-1, 2)


def _radial_decrements(params: torch.Tensor) -> torch.Tensor:
    # straight-through clamp: forward max(d, 0) (monotone), identity gradient (can leave the identity d = 0)
    return params + (params.clamp(min=0.0) - params).detach()


class _KnotLerp(torch.autograd.Function):
    """Linear interpolation between log-knots, logw[i0] * (1 - f) + logw[i0 + 1] * f.

    The forward is the indexing expression itself.  The backward replaces the
    sort-based gradient of advanced indexing by two dense one-hot products,
    which compute the same gradient, up to floating-point summation order,
    several times faster.
    """

    @staticmethod
    def forward(ctx, logw, i0, f):
        ctx.save_for_backward(i0, f)
        ctx.num_knots = int(logw.shape[0])
        return logw[i0] * (1.0 - f) + logw[i0 + 1] * f

    @staticmethod
    def backward(ctx, grad):
        i0, f = ctx.saved_tensors
        n = ctx.num_knots
        g = grad.to(f.dtype)
        lo = g.new_zeros((i0.shape[0], n)).scatter_(1, i0.unsqueeze(1), 1.0)
        hi = g.new_zeros((i0.shape[0], n)).scatter_(1, (i0 + 1).unsqueeze(1), 1.0)
        grad_logw = lo.t().mv(g * (1.0 - f)) + hi.t().mv(g * f)
        return grad_logw, None, None


def radial_attenuation(params: torch.Tensor, geometry: torch.Tensor, pixel_coords: torch.Tensor,
                       max_res: float) -> torch.Tensor:
    """W(rho) in (0, 1], monotone non-increasing in rho = |p - pp| / max(W,H).

    params [K] raw knot decrements; geometry [3] = (pp_x, pp_y, rho_ap); pixel_coords [N,2] pixel centres.
    K+1 uniform knots over [rho_ap/2, rho_ap]; W = 1 inside rho_ap/2.  Returns [N,1]."""
    k = params.shape[-1]
    hi = geometry[2].to(params.dtype)
    lo = 0.5 * hi
    dec = _radial_decrements(params)
    logw = torch.cat([dec.new_zeros(1), -torch.cumsum(dec, 0)])
    rho = torch.linalg.vector_norm(pixel_coords.to(params.dtype) - geometry[:2].to(params.dtype), dim=-1) / float(max_res)
    t = ((rho - lo) / (hi - lo) * k).clamp(0.0, float(k))
    i0 = t.floor().long().clamp(max=k - 1)
    f = t - i0.to(t.dtype)
    return torch.exp(_KnotLerp.apply(logw, i0, f)).unsqueeze(-1)


def flatfield_attenuation(params: torch.Tensor, free: torch.Tensor, pixel_coords: torch.Tensor,
                          resolution_w: int, resolution_h: int) -> torch.Tensor:
    """Static per-camera flat-field W_ff(p) = exp(-a(p)) in (0, 1]; a = bilinear upsampling of a G x G grid of
    non-negative log-attenuations (straight-through clamp; exact identity at 0) laid over a max(W,H) square centred on the
    raster; cells inside rho_ap/2 are fixed at 0 (free = 0) so the map cannot trade a global scale with exposure.
    Returns [N,1]."""
    a = _radial_decrements(params) * free
    s = float(max(resolution_w, resolution_h))
    gx = (pixel_coords[:, 0].to(a.dtype) - 0.5 * resolution_w) / (0.5 * s)
    gy = (pixel_coords[:, 1].to(a.dtype) - 0.5 * resolution_h) / (0.5 * s)
    g = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
    v = F.grid_sample(a[None, None], g, mode="bilinear", padding_mode="border", align_corners=False)
    return torch.exp(-v.view(-1, 1))


def geometry_from(principal_point, resolution, valid_pixel_coords) -> list:
    """``[pp_x, pp_y, rho_ap]`` of one camera: the principal point and the calibrated-domain radius
    ``max |p - pp| / max(W, H)`` over the valid pixel centres ``[N, 2]`` (all pixels without a domain)."""
    pp = [float(v) for v in torch.as_tensor(principal_point).reshape(-1)[:2]]
    max_res = float(max(int(v) for v in torch.as_tensor(resolution).reshape(-1)[:2]))
    pix = torch.as_tensor(valid_pixel_coords).reshape(-1, 2).double()
    if pix.shape[0] == 0:
        raise ValueError("geometry_from needs at least one valid pixel")
    rad = torch.linalg.vector_norm(pix - torch.tensor(pp, dtype=torch.float64), dim=-1) / max_res
    return pp + [float(rad.max())]


def flatfield_gauge(geometry, resolution, grid: int) -> torch.Tensor:
    """``[G, G]`` float mask of the flat-field cells that may train: 0 for the cells whose centre lies
    within ``rho_ap / 2`` of the principal point, which stay at identity, 1 elsewhere."""
    ppx, ppy, rho_ap = (float(v) for v in geometry)
    rw, rh = (int(v) for v in torch.as_tensor(resolution).reshape(-1)[:2])
    sz = float(max(rw, rh))
    cc = (torch.arange(grid, dtype=torch.float64) + 0.5) / grid * sz
    cy, cx = torch.meshgrid(cc - 0.5 * sz + 0.5 * rh, cc - 0.5 * sz + 0.5 * rw, indexing="ij")
    inner = torch.hypot(cx - ppx, cy - ppy) / sz <= 0.5 * rho_ap
    return (~inner).float()
