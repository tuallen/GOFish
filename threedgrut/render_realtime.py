# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Real-time renderer: the trained scene rendered by a faster rasterizer.

Built on the exact forward-only tracer (threedgut_tracer.inference). Only image
formation changes; the stored scene and its mesh do not. Options are a JSON object
(``render.py --renderer realtime --realtime-options '{...}'``); the default,
``DEFAULT_OPTIONS``, is the renderer of the paper:

``admission``: ``{"mode": "ut", "scale": s, "fallback": f, "partial_ut": p,
    "fallback_min_angle_deg": a}`` admits Gaussians by 3DGUT's UT-projected
    footprint enlarged by s >= 1. With ``partial_ut`` a sigma point that leaves
    the camera domain is mirrored about the projected mean instead of dropping the
    Gaussian; ``fallback`` "cone_big" admits UT-invalid Gaussians whose half-angle
    exceeds a degrees by a conservative bounding-sphere / tile-cone test ("none"
    drops them as 3DGUT does, "cone" tests all of them).
``sort``: ``{"mode": "tile", "k": K, "key": "tile"|"global"}`` replaces SOF's
    per-pixel hierarchical sort by the per-tile depth order refined by a
    per-pixel K-entry buffer; key "global" sorts by camera distance as 3DGUT.
``split``: ``{"ways": S, "min": L}`` processes tiles with at least L entries by S
    blocks over contiguous chunks of the depth-sorted list, composited front to
    back.
``prefetch``: ``"slim"`` keeps only what the forward hit test reads (exact).
``shape``: ``"fused32"`` forms the per-view full-shape factors in one float32
    kernel with a float64 retry for invalid or ill-conditioned rows (``"exact"``:
    the reference operators, ``"fused"``: one float64 kernel).
``fastpose``: numpy rotation-to-quaternion for the native camera pose.
``ppisp``: ``{"cache": "camera"}`` runs PPISP's controller and colour transform
    every frame and caches only the per-camera wide-angle factors W.
"""

from __future__ import annotations

import contextlib
import json

import torch

from threedgrut.render_inference import InferenceRenderer


class CameraCachedPPISP(torch.nn.Module):
    """PPISP with the wide-angle factors W cached per camera; the controller runs every frame."""

    def __init__(self, ppisp):
        super().__init__()
        self.ppisp = ppisp
        for controller in getattr(ppisp, "controllers", ()):
            controller.realtime = True  # per-mask controller preparation and the NCHW pooling kernel
        self.wcache = {}

    def forward(self, rgb, pixel_coords=None, resolution=None, camera_idx=None, frame_idx=None,
                exposure_prior=None, valid_mask=None):
        from gofish_ppisp import ppisp_apply

        camera = int(camera_idx)
        width, height = resolution
        config = self.ppisp.config
        if config.use_controller and config.controller_activation_ratio < 1.0 and len(self.ppisp.controllers) > 0:
            exposure, color = self.ppisp.controllers[camera](rgb.view(height, width, 3), exposure_prior,
                                                             valid_mask=valid_mask)
            params = (exposure.unsqueeze(0), color.unsqueeze(0))
        else:  # no trained controller: identity corrections, as PPISP.forward for novel views
            params = (torch.zeros(1, device=rgb.device, dtype=rgb.dtype),
                      torch.zeros(1, 8, device=rgb.device, dtype=rgb.dtype))
        # Wide-angle PPISP (W) after the controller, exactly as PPISP.forward; W depends only on the camera
        # (and its pixel grid), so its factors are cached per camera.
        wkey = (camera, width, height, tuple(pixel_coords.shape))
        if wkey not in self.wcache:
            self.wcache[wkey] = self.ppisp.wide_angle_factors(pixel_coords, camera, width, height)
        rgb = self.ppisp.apply_wide_angle_factors(rgb, self.wcache[wkey])
        return ppisp_apply(params[0], self.ppisp.vignetting_params, params[1], self.ppisp.crf_params,
                           rgb, pixel_coords, width, height, camera, 0)


class _PoseProxy:
    """Batch view whose camera poses are host tensors (the native camera parameters are built from host data)."""

    def __init__(self, batch, host, host_end):
        self._batch, self.T_to_world, self.T_to_world_end = batch, host, host_end

    def __getattr__(self, name):
        return getattr(self._batch, name)


DEFAULT_OPTIONS = {
    "ppisp": {"cache": "camera"},
    "shape": "fused32",
    "admission": {"mode": "ut", "scale": 1.0, "fallback": "cone_big", "partial_ut": 1.0, "fallback_min_angle_deg": 5.0},
    "sort": {"mode": "tile", "k": 12, "key": "global"},
    "prefetch": "slim",
    "fastpose": True,
    "split": {"ways": 4, "min": 2048},
}


def parse_realtime_options(text=None):
    """JSON options (``@FILE`` for a JSON file); default: ``DEFAULT_OPTIONS``."""
    if not text:
        return json.loads(json.dumps(DEFAULT_OPTIONS))
    if text.startswith("@"):
        with open(text[1:]) as handle:
            text = handle.read()
    options = json.loads(text)
    unknown = set(options) - {"ppisp", "sort", "admission", "shape", "prefetch", "split", "fastpose"}
    if unknown:
        raise ValueError(f"unknown real-time renderer options {sorted(unknown)}")
    return options


def realtime_overrides(options, overrides=None):
    """Config overrides (and validation) implied by the real-time renderer options."""
    overrides = list(overrides or [])
    if "sort" in options:
        if options["sort"].get("mode") != "tile":
            raise ValueError("real-time sort mode must be 'tile'")
        overrides += ["render.splat.hierarchical_sort=false", "render.splat.sof_mid8_hierarchy=false",
                      f"render.splat.k_buffer_size={int(options['sort'].get('k', 0))}"]
        if options["sort"].get("key", "tile") == "global":
            if "admission" not in options:
                raise ValueError("the global sort key requires the UT admission")
            overrides.append("render.gofish.fast_global_depth_key=true")
        elif options["sort"].get("key", "tile") != "tile":
            raise ValueError("real-time sort key must be 'tile' or 'global'")
    if "admission" in options:
        if options["admission"].get("mode") != "ut":
            raise ValueError("real-time admission mode must be 'ut'")
        overrides += [f"render.gofish.fast_admission_scale={float(options['admission'].get('scale', 1.0))}"]
        if "partial_ut" in options["admission"]:
            overrides.append(f"render.gofish.fast_partial_ut_scale={float(options['admission']['partial_ut'])}")
        fallback = options["admission"].get("fallback", "cone")
        if fallback not in ("none", "cone", "cone_big"):
            raise ValueError("real-time admission fallback must be none, cone or cone_big")
        if "fallback_min_angle_deg" in options["admission"]:
            overrides.append("render.gofish.fast_admission_fallback_min_angle_deg="
                             f"{float(options['admission']['fallback_min_angle_deg'])}")
        overrides.append(f"render.gofish.fast_admission_fallback={fallback}")
    if "prefetch" in options:
        if options["prefetch"] != "slim":
            raise ValueError("real-time prefetch option must be 'slim'")
        overrides.append("render.gofish.fast_slim_prefetch=true")
    if "split" in options:  # {"ways": S, "min": L}
        overrides += [f"render.gofish.fast_tile_split={int(options['split']['ways'])}",
                      f"render.gofish.fast_tile_split_min={int(options['split'].get('min', 2048))}"]
    if options.get("shape", "exact") not in ("exact", "fused", "fused32"):
        raise ValueError("real-time shape option must be exact, fused or fused32")
    return overrides


def _so3_to_quat_numpy(R, unbatch=True):
    """float32 numpy copy of SensorPose3DModel.__so3_matrix_to_quat (same branch rule and formulas) for one or a
    batch of 3x3 rotations, avoiding ~25 small CPU torch ops per call."""
    import numpy as np
    Rn = (R.numpy() if isinstance(R, torch.Tensor) else np.asarray(R)).reshape(-1, 3, 3)
    out = np.empty((Rn.shape[0], 4), dtype=Rn.dtype)
    for b in range(Rn.shape[0]):
        M = Rn[b]; d = np.array([M[0, 0], M[1, 1], M[2, 2], 0], dtype=Rn.dtype); d[3] = d[:3].sum()
        c = int(np.argmax(d)); q = np.empty(4, dtype=Rn.dtype)
        if c != 3:
            i = c; j = (i + 1) % 3; k = (j + 1) % 3
            q[i] = 1 - d[3] + 2 * M[i, i]; q[j] = M[j, i] + M[i, j]; q[k] = M[k, i] + M[i, k]; q[3] = M[k, j] - M[j, k]
        else:
            q[0] = M[2, 1] - M[1, 2]; q[1] = M[0, 2] - M[2, 0]; q[2] = M[1, 0] - M[0, 1]; q[3] = 1 + d[3]
        out[b] = q / np.sqrt((q * q).sum())
    t = torch.from_numpy(out)
    return t.squeeze() if unbatch else t


@contextlib.contextmanager
def _fast_pose(enabled: bool):
    """Within a real-time frame, the camera-parameter builder converts poses with ``_so3_to_quat_numpy``."""
    if not enabled:
        yield
        return
    from threedgut_tracer import tracer

    model = tracer.SensorPose3DModel
    original = model.__dict__["_SensorPose3DModel__so3_matrix_to_quat"]
    model._SensorPose3DModel__so3_matrix_to_quat = staticmethod(_so3_to_quat_numpy)
    try:
        yield
    finally:
        model._SensorPose3DModel__so3_matrix_to_quat = original


class RealtimeRenderer(InferenceRenderer):
    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, path="", save_gt=True, metrics=False, overrides=None,
                        options=None):
        options = parse_realtime_options(options)
        overrides = realtime_overrides(options, overrides)
        model, tracer, conf, step, post_processing = cls.load(
            checkpoint_path, overrides, path, tracer_kwargs={"shape": options.get("shape", "exact")})
        if "ppisp" in options:
            if post_processing is None:
                raise RuntimeError("checkpoint has no PPISP")
            if options["ppisp"].get("cache") != "camera":
                raise ValueError("real-time ppisp option must be {'cache': 'camera'}")
            post_processing = CameraCachedPPISP(post_processing)
        renderer = cls(model, tracer, conf, step, out_dir, post_processing, save_gt=save_gt, metrics=metrics)
        renderer.realtime_options = options
        return renderer

    def render_frame(self, gpu_batch):
        # The pose is read back to the host once per frame and shared by the fused shape and the
        # native camera parameters.
        host = gpu_batch.T_to_world.detach().cpu()
        end = getattr(gpu_batch, "T_to_world_end", None)
        host_end = None if end is None else (host if end is gpu_batch.T_to_world else end.detach().cpu())
        self.tracer.frame_host_pose = (gpu_batch, host, host_end)
        try:
            with _fast_pose(bool(self.realtime_options.get("fastpose"))):
                return super().render_frame(gpu_batch)
        finally:
            self.tracer.frame_host_pose = None


__all__ = ["DEFAULT_OPTIONS", "RealtimeRenderer", "CameraCachedPPISP", "parse_realtime_options", "realtime_overrides"]
