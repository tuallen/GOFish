# SPDX-License-Identifier: Apache-2.0
"""Exact forward-only renderer of a frozen scene.

Produces the same pixels as the reference renderer (``Tracer.render`` with
``train=False``) while doing less work per frame:

* training-only state is not built: no autograd context or saved tensors, no
  geometric training channels (normals, distortion moments, extent, quantile
  and exact-crossing depth, opacity-field replay), no hit counts, no kernel
  timers;
* per-scene fixed data are computed once: the activated rotation and opacity,
  the intrinsic scale with the frozen 3-D (Mip) filter added in quadrature, the
  SH feature buffer and the static slots of the packed particle buffer;
* without a pixel footprint (gamma = 0) the full-shape factors do not depend
  on the view and are computed once per scene;
* per-camera fixed data are cached: the 16x16 domain tile mask and its
  registration (as in the reference), and the calibration predicates of the
  pinhole Jacobian;
* the per-view footprint Q_v is built without device-to-host synchronization,
  and the full-shape operator keeps one cuSOLVER handle across frames.

Only the view-dependent quantities are recomputed per frame: Q_v and the
full-shape factor C = Sigma + Q_v with its mass-preserving opacity, followed by
complete admission, the SOF hierarchical sort and blending, all through the
same code as the reference.
"""

from __future__ import annotations

import atexit

import torch
from omegaconf import OmegaConf

from .physical_tile_mask import build_physical_tile_binding
from .setup_3dgut import pixel_footprint_variance, setup_3dgut
from .tracer import create_camera_parameters


# Packed per-particle density layout consumed by the native renderer
# (Tracer._Autograd.forward): position, density, rotation, scale,
# intrinsic density, factor C^(1/2) (3x3), inverse factor (3x3), padding.
_SLOTS = {"position": (0, 3), "density": (3, 4), "rotation": (4, 8), "scale": (8, 11),
          "intrinsic_density": (11, 12), "factor": (12, 21), "inverse_factor": (21, 30),
          "padding": (30, 32)}
_PACKED_WIDTH = 32


_fast_shape = None


def load_fast_shape():
    """JIT-build the fused approximate full-shape kernel (real-time fast path)."""
    global _fast_shape
    if _fast_shape is None:
        import os

        from threedgrut.utils import jit

        prefix = os.path.dirname(__file__)
        name = "gofish_fast_shape"
        _fast_shape = jit.load(
            name=name, sources=[os.path.join(prefix, "src", "fast_shape.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
            build_directory=torch.utils.cpp_extension._get_build_directory(name, verbose=True), with_cuda=True)
    return _fast_shape


def load_inference_plugin(conf):
    """The forward-only extension of ``conf`` (setup_3dgut caches it by extension name)."""
    return setup_3dgut(conf, inference=True)


class InferenceTracer:
    """Forward-only GOFish renderer bound to one frozen scene."""

    def __init__(self, conf, model, cache_static_shape=True, shape="exact"):
        if conf.render.primitive_type == "trisurfel":
            raise RuntimeError("GOFish renders volumetric Gaussians only")
        self.conf = conf
        self.device = "cuda"
        torch.zeros(1, device=self.device)  # CUDA context
        self.plugin = load_inference_plugin(conf)
        self.wrapper = self.plugin.SplatRaster(OmegaConf.to_container(conf))
        self.gamma = pixel_footprint_variance(conf)
        # One cuSOLVER handle for all frames (no numerical state; bit-identical factors).
        from . import footprint_shape_operator

        native = footprint_shape_operator._native()
        native.set_persistent_solver(True)
        atexit.register(native.set_persistent_solver, False)
        self._tile_tokens = {}
        self._registered = set()
        # gamma = 0: the full-shape factors are view-independent (see view_shape).
        self.cache_static_shape = bool(cache_static_shape)
        self.static_shape = None
        # "exact": reference operators; "fused": one approximate float64 kernel per
        # frame (threedgut_tracer/src/fast_shape.cu), Cholesky factor B.
        if shape not in ("exact", "fused", "fused32", "fusedmix"):
            raise ValueError(f"unknown shape mode {shape!r}")
        self.shape_mode = shape
        self._fused = load_fast_shape() if shape in ("fused", "fused32", "fusedmix") else None
        self._retries = None
        self.freeze(model)

    @torch.no_grad()
    def freeze(self, model):
        """Precompute the per-scene fixed data of ``model`` (same ops as the reference)."""
        self.positions = model.positions.detach().contiguous()
        self.rotation = model.get_rotation().detach().contiguous()
        self.raw_density = model.get_density().detach().contiguous()
        scale = (model.get_scale_with_3D_filter()
                 if bool(getattr(model, "_use_physical_mip3d_filter", False)) else model.get_scale())
        self.physical_scale = scale.detach().contiguous()
        self.features = model.get_features().detach().contiguous()
        self.n_active_features = int(model.n_active_features)
        self.ray_feature_dim = int(model.ray_feature_dim)
        n = int(self.positions.shape[0])
        for name, value in (("positions", self.positions), ("rotation", self.rotation),
                            ("density", self.raw_density), ("scale", self.physical_scale)):
            if value.dtype != torch.float32:
                raise RuntimeError(f"inference renderer expects float32 {name}")
        packed = torch.zeros((n, _PACKED_WIDTH), dtype=torch.float32, device=self.positions.device)
        packed[:, slice(*_SLOTS["position"])] = self.positions
        packed[:, slice(*_SLOTS["rotation"])] = self.rotation
        packed[:, slice(*_SLOTS["scale"])] = self.physical_scale
        packed[:, slice(*_SLOTS["intrinsic_density"])] = self.raw_density
        self.packed = packed
        self.num_gaussians = n
        self.static_shape = None
        self._packed_shape = None

    def _tile_token(self, gpu_batch, rays):
        mask = getattr(gpu_batch, "physical_fov_mask", None)
        domain = getattr(gpu_batch, "physical_fov_key", None)
        if domain is None:
            domain = "full-raster" if mask is None else f"mask:{id(mask)}"
        key = (str(domain), str(rays.device), int(rays.shape[1]), int(rays.shape[2]))
        binding = self._tile_tokens.get(key)
        if binding is None:
            tile = int(OmegaConf.select(self.conf, "render.gofish.fast_tile_size", default=16) or 16)
            binding = build_physical_tile_binding(gpu_batch, rays, str(domain), tile=tile)
            self._tile_tokens[key] = binding
        if binding.token not in self._registered:
            self.wrapper.register_physical_tile_mask(binding.mask, binding.token)
            self._registered.add(binding.token)
        return binding.token

    def view_shape(self, gpu_batch):
        """C = Sigma + Q_v and the mass-preserving density (reference operators).

        With a pixel footprint (gamma > 0) both depend on the view through the
        camera Jacobian and are recomputed per frame.  Without one (gamma = 0)
        Q_v = 0 and m = 1, so the factors are those of the stored scene for every
        view: they are computed once and reused (``static_shape``).
        """
        from .footprint_shape_operator import build_full_shape
        from .view_observation_filter import current_view_q6

        if self.static_shape is not None:
            return self.static_shape
        q6, jacobian, observed = current_view_q6(
            self.positions, gpu_batch, self.gamma, host_sync_free=True)
        shape = build_full_shape(self.physical_scale, self.rotation, self.raw_density,
                                 q6, jacobian, observed, mass_preserving=True)
        if self.gamma == 0.0 and self.cache_static_shape:
            self.static_shape = shape
        return shape

    def _host_pose(self, gpu_batch):
        frame = getattr(self, "frame_host_pose", None)
        return frame[1] if frame is not None and frame[0] is gpu_batch else None

    def _fused_shape(self, gpu_batch):
        fisheye = getattr(gpu_batch, "intrinsics_OpenCVFisheyeCameraModelParameters", None)
        if fisheye is not None:
            fx, fy = (float(v) for v in fisheye["focal_length"])
            intrinsics = [fx, fy, *(float(v) for v in fisheye["radial_coeffs"]), float(fisheye["max_angle"])]
        else:
            pinhole = gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters
            focal = [float(v) for v in torch.as_tensor(pinhole["focal_length"]).reshape(-1)]
            fx, fy = (focal[0], focal[0]) if len(focal) == 1 else focal
            intrinsics = [fx, fy, 0.0, 0.0, 0.0, 0.0, 0.0]
        if self.shape_mode in ("fused32", "fusedmix"):
            # Real-time renderer: float32 factors, float64 retry for invalid / ill-conditioned rows (retry count kept on GPU)
            if self._retries is None:  # [count, deferred rows...] (count reset on device every frame)
                self._retries = torch.zeros(self.num_gaussians + 1, dtype=torch.int32, device=self.positions.device)
            pose = self._host_pose(gpu_batch)
            self._fused.fast_shape32(self.positions, self.rotation, self.physical_scale, self.raw_density,
                                     (pose if pose is not None else gpu_batch.T_to_world)[0], intrinsics, fisheye is not None, float(self.gamma),
                                     self.packed, self._retries, 1 if self.shape_mode == "fusedmix" else 0)
            return
        self._fused.fast_shape(self.positions, self.rotation, self.physical_scale, self.raw_density,
                               gpu_batch.T_to_world[0], intrinsics, fisheye is not None, float(self.gamma),
                               self.packed)

    @torch.no_grad()
    def render(self, gpu_batch):
        rays_o = gpu_batch.rays_ori.contiguous()
        rays_d = gpu_batch.rays_dir.contiguous()
        token = self._tile_token(gpu_batch, rays_d)
        frame = getattr(self, "frame_host_pose", None)
        if frame is not None and frame[0] is gpu_batch:
            from threedgrut.render_realtime import _PoseProxy
            sensor, poses = create_camera_parameters(_PoseProxy(gpu_batch, frame[1], frame[2]), self.plugin)
        else:
            sensor, poses = create_camera_parameters(gpu_batch, self.plugin)

        if self.shape_mode in ("fused", "fused32", "fusedmix"):
            self._fused_shape(gpu_batch)
            shape = self._packed_shape = None
        else:
            shape = self.view_shape(gpu_batch)
        if shape is not None and shape is not self._packed_shape:  # a cached static shape is packed once
            n = self.num_gaussians
            for name, value in (("density", shape.density), ("factor", shape.factor),
                                ("inverse_factor", shape.inverse_factor)):
                if value.dtype != torch.float32:
                    raise RuntimeError(f"full-shape {name} must be float32")
            self.packed[:, slice(*_SLOTS["density"])] = shape.density
            self.packed[:, slice(*_SLOTS["factor"])] = shape.factor.reshape(n, 9)
            self.packed[:, slice(*_SLOTS["inverse_factor"])] = shape.inverse_factor.reshape(n, 9)
            self._packed_shape = shape

        features_alpha, hit_distance = self.wrapper.trace_inference(
            self.n_active_features, self.packed, self.features, rays_o, rays_d, sensor,
            poses.timestamps_us[0], poses.timestamps_us[1],
            poses.T_world_sensors[0], poses.T_world_sensors[1], token)
        features_alpha = features_alpha.float()
        f = self.ray_feature_dim
        return {
            "pred_features": features_alpha[..., :f].unsqueeze(0).contiguous(),
            "pred_opacity": features_alpha[..., f:].unsqueeze(0).contiguous(),
            "pred_dist": hit_distance.unsqueeze(0).contiguous(),
        }


__all__ = ["InferenceTracer", "load_inference_plugin"]
