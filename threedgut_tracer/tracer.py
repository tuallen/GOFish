# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from ncore.data import FThetaCameraModelParameters, ShutterType
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch
from .physical_tile_mask import build_physical_tile_binding
from .view_observation_filter import pixel_footprint_variance

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_3dgut_plugin = None  # extension of the most recent Tracer, for the camera-parameter helpers


def load_3dgut_plugin(conf):
    """The training renderer extension of ``conf`` (setup_3dgut caches it by extension name)."""
    global _3dgut_plugin
    from .setup_3dgut import setup_3dgut

    _3dgut_plugin = setup_3dgut(conf)
    return _3dgut_plugin


# ----------------------------------------------------------------------------
#


@dataclass
class SensorPose3D:
    T_world_sensors: list  # represents two tquat [t,q] poses
    timestamps_us: list


class SensorPose3DModel:
    def __init__(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(SensorPose3DModel, self).__init__()
        self.R = R
        self.T = T
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = (
            torch.tensor(SensorPose3DModel.__getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
        )

    @staticmethod
    def __getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    @staticmethod
    def __so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
        """
        Converts a singe / batch of SO3 rotation matrices (3x3) to unit quaternion representation.

        Args:
            R: single / batch of SO3 rotation matrices [bs, 3, 3] or [3,3]
            unbatch: if the single example should be unbatched (first dimension removed) or not

        Returns:
            single / batch of unit quaternions (XYZW convention)  [bs, 4] or [4]
        """

        # Convert numpy array to torch tensor
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        R = R.reshape((-1, 3, 3))  # batch dimensions unconditionally
        num_rotations, D1, D2 = R.shape
        assert (D1, D2) == (3, 3), "so3_matrix_to_quat: Input has to be a Bx3x3 tensor."

        decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
        quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)

        decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
        choices = decision_matrix.argmax(dim=1)

        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
        quat[ind, j] = R[ind, j, i] + R[ind, i, j]
        quat[ind, k] = R[ind, k, i] + R[ind, i, k]
        quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
        quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
        quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]

        quat = quat / torch.norm(quat, dim=1)[:, None]

        if unbatch:  # unbatch dimensions conditionally
            quat = quat.squeeze()

        return quat  # (N,4) or (4,)

# ----------------------------------------------------------------------------
#


class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            physical_tile_token,
            frame_id,
            sof_exact_opacity_active,
            n_active_features,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_intrinsic_dns,
            mog_factor,
            mog_inverse_factor,
            mog_sph,
            sensor_params,
            sensor_poses,
            mog_confidence,
        ):
            has_confidence = mog_confidence.numel() != 0
            if has_confidence and mog_confidence.shape != (mog_pos.shape[0], 1):
                raise ValueError("CoMe confidence must have shape [N,1]")
            if mog_factor.shape != (mog_pos.shape[0], 3, 3) or mog_inverse_factor.shape != mog_factor.shape:
                raise ValueError("full-shape factor tensors must have shape [N,3,3]")
            # Rotation/scale/density slots carry the intrinsic (Mip-filtered) shape used by the
            # world-unit extent term; factor and inverse factor carry C = Sigma + Q_v for rendering.
            # The extent VJP returns only through the intrinsic slots, so they are not detached.
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, mog_intrinsic_dns,
                 mog_factor.reshape(-1, 9), mog_inverse_factor.reshape(-1, 9),
                 # ABI padding word 0 carries the CoMe confidence conf_i = exp(gamma_i) when enabled.
                 (torch.concat([mog_confidence.to(mog_pos.dtype),
                                torch.zeros((mog_pos.shape[0], 1), device=mog_pos.device, dtype=mog_pos.dtype)], dim=1)
                  if has_confidence else
                  torch.zeros((mog_pos.shape[0], 2), device=mog_pos.device, dtype=mog_pos.dtype))], dim=1
            ).contiguous()
            particle_features = mog_sph.contiguous()

            ray_time = (
                torch.ones(
                    (ray_ori.shape[0], ray_ori.shape[1], ray_ori.shape[2], 1), device=ray_ori.device, dtype=torch.long
                )
                * sensor_poses.timestamps_us[0]
            )

            ray_features_density, ray_hit_distance, ray_hit_count, mog_visibility, ray_normal, ray_depth_var, ray_quant_depth, ray_extent, ray_opacity_field, ray_invd, ray_invd2 = tracer_wrapper.trace(
                frame_id,
                sof_exact_opacity_active,
                n_active_features,
                particle_density,
                particle_features,
                ray_ori.contiguous(),
                ray_dir.contiguous(),
                ray_time.contiguous(),
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
                physical_tile_token,
            )

            ctx.save_for_backward(
                ray_ori,
                ray_dir,
                ray_time,
                ray_features_density,
                ray_hit_distance,
                ray_depth_var,
                ray_extent,
                ray_opacity_field,
                ray_invd,
                ray_invd2,
                ray_quant_depth,
                ray_normal,
                particle_density,
                particle_features,
            )

            ctx.frame_id = frame_id
            ctx.sof_exact_opacity_active = bool(sof_exact_opacity_active)
            ctx.n_active_features = n_active_features
            ctx.sensor_params = sensor_params
            ctx.sensor_poses = sensor_poses
            ctx.tracer_wrapper = tracer_wrapper
            ctx.physical_tile_token = physical_tile_token
            ctx.has_confidence = has_confidence

            return (
                ray_features_density.float(),  # always fp32 to caller; fp16 saved in ctx for trace_bwd
                ray_hit_distance,
                ray_hit_count,
                mog_visibility,
                ray_normal,
                ray_depth_var,
                ray_quant_depth,
                ray_extent,
                ray_opacity_field,
                ray_invd,
                ray_invd2,
            )

        @staticmethod
        def backward(
            ctx,
            ray_features_density_grd,  # always fp32 (gradient buffer is never fp16)
            ray_hit_distance_grd,
            ray_hit_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
            ray_normal_grd,
            ray_depth_var_grd,
            ray_quant_depth_grd,
            ray_extent_grd,
            ray_opacity_field_grd,
            ray_invd_grd,
            ray_invd2_grd,
        ):
            (
                ray_ori,
                ray_dir,
                ray_time,
                ray_features_density,
                ray_hit_distance,
                ray_depth_var,
                ray_extent,
                ray_opacity_field,
                ray_invd,
                ray_invd2,
                ray_quant_depth,
                ray_normal,
                particle_density,
                particle_features,
            ) = ctx.saved_variables

            frame_id = ctx.frame_id
            sof_exact_opacity_active = ctx.sof_exact_opacity_active
            n_active_features = ctx.n_active_features
            sensor_params = ctx.sensor_params
            sensor_poses = ctx.sensor_poses

            particle_density_grd, particle_features_grd = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                sof_exact_opacity_active,
                n_active_features,
                particle_density,
                particle_features,
                ray_ori,
                ray_dir,
                ray_time,
                sensor_params,
                sensor_poses.timestamps_us[0],
                sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0],
                sensor_poses.T_world_sensors[1],
                ray_features_density,
                ray_features_density_grd,
                ray_hit_distance,
                ray_hit_distance_grd,
                ray_depth_var,
                (ray_depth_var_grd if ray_depth_var_grd is not None else torch.zeros_like(ray_depth_var)),
                ray_extent,
                (ray_extent_grd if ray_extent_grd is not None else torch.zeros_like(ray_extent)),
                ray_opacity_field,
                (ray_opacity_field_grd if ray_opacity_field_grd is not None else torch.zeros_like(ray_opacity_field)),
                ray_quant_depth,
                (ray_quant_depth_grd if ray_quant_depth_grd is not None else torch.zeros_like(ray_quant_depth)),
                ray_normal,
                (ray_normal_grd if ray_normal_grd is not None else torch.zeros_like(ray_normal)),
                ray_invd,
                (ray_invd_grd if ray_invd_grd is not None else torch.zeros_like(ray_invd)),
                ray_invd2,
                (ray_invd2_grd if ray_invd2_grd is not None else torch.zeros_like(ray_invd2)),
                ctx.physical_tile_token,
            )

            (
                mog_pos_grd,
                mog_dns_grd,
                mog_rot_grd,
                mog_scl_grd,
                mog_intrinsic_dns_grd,
                mog_factor_grd,
                mog_inverse_factor_grd,
                mog_padding_grd,
            ) = torch.split(particle_density_grd, [3, 1, 4, 3, 1, 9, 9, 2], dim=1)
            mog_confidence_grd = mog_padding_grd[:, 0:1].contiguous() if ctx.has_confidence else None
            mog_factor_grd = mog_factor_grd.reshape(-1, 3, 3)
            mog_inverse_factor_grd = mog_inverse_factor_grd.reshape(-1, 3, 3)
            mog_sph_grd = particle_features_grd

            return (
                None,  # tracer_wrapper
                None,  # physical_tile_token
                None,  # frame_id
                None,  # sof_exact_opacity_active
                None,  # n_active_features
                None,  # ray_ori
                None,  # ray_dir
                mog_pos_grd.contiguous(),
                mog_rot_grd.contiguous(),
                mog_scl_grd.contiguous(),
                mog_dns_grd.contiguous(),
                mog_intrinsic_dns_grd.contiguous(),
                mog_factor_grd,
                mog_inverse_factor_grd,
                mog_sph_grd.contiguous(),
                None,  # sensor_params
                None,  # sensor_poses
                mog_confidence_grd,
            )

    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf
        # The renderer always consumes the full-shape factor C = Sigma + Q_v
        # built per view by Calibrated Mip Filtering.
        self._view_observation_filter = True
        # Without the factorization the per-view footprint vanishes (Q = 0, m = 1).
        self._pixel_footprint_variance = pixel_footprint_variance(conf)

        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        self.tracer_wrapper = load_3dgut_plugin(conf).SplatRaster(OmegaConf.to_container(conf))

    def _physical_tile_token(self, wrapper, gpu_batch, rays):
        """Register the 16x16 tile mask of the calibrated domain once per camera."""
        mask = getattr(gpu_batch, "physical_fov_mask", None)
        domain = getattr(gpu_batch, "physical_fov_key", None)
        if domain is None:
            domain = "full-raster" if mask is None else f"mask:{id(mask)}"
        cache_key = (str(domain), str(rays.device), int(rays.shape[1]), int(rays.shape[2]))
        cache = getattr(self, "_physical_tile_binding_cache", None)
        if cache is None:
            cache = {}
            self._physical_tile_binding_cache = cache
        if cache_key not in cache:
            cache[cache_key] = build_physical_tile_binding(gpu_batch, rays, str(domain))
        binding = cache[cache_key]

        registered = getattr(self, "_physical_tile_registration_cache", None)
        if registered is None:
            registered = set()
            self._physical_tile_registration_cache = registered
        registration_key = (id(wrapper), binding.token)
        if registration_key not in registered:
            wrapper.register_physical_tile_mask(binding.mask, binding.token)
            registered.add(registration_key)
        return binding.token

    def _density_shape_inputs(self, gaussians, gpu_batch=None, detach=False, frame_id=None):
        """Intrinsic (Mip-filtered) scene shape plus this view's pixel footprint."""
        if gpu_batch is None:
            raise RuntimeError("the per-view footprint requires the active camera batch")
        if self.conf.render.primitive_type == "trisurfel":
            raise RuntimeError("GOFish renders volumetric Gaussians only")
        from .view_observation_filter import build_intrinsic_view_shape

        rotation = gaussians.get_rotation()
        raw_density = gaussians.get_density()
        physical_scale = (
            gaussians.get_scale_with_3D_filter()
            if bool(getattr(gaussians, "_use_physical_mip3d_filter", False))
            else gaussians.get_scale()
        )
        if detach:
            rotation, raw_density, physical_scale = rotation.detach(), raw_density.detach(), physical_scale.detach()
        shape = build_intrinsic_view_shape(
            physical_scale,
            rotation,
            raw_density,
            gaussians.positions.detach(),
            gpu_batch,
            pixel_footprint_variance=self._pixel_footprint_variance,
        )
        return rotation, physical_scale, shape.density, raw_density, shape.factor, shape.inverse_factor

    @property
    def timings(self):
        return self.tracer_wrapper.collect_times()

    def build_acc(self, gaussians, rebuild=True):
        pass  # no-op for 3DGUT

    def render(self, gaussians, gpu_batch: Batch, train=False, frame_id=0):
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        physical_tile_token = self._physical_tile_token(
            self.tracer_wrapper, gpu_batch, rays_d
        )

        sensor, poses = Tracer.__create_camera_parameters(gpu_batch)
        lam_opa = float(OmegaConf.select(
            self.conf, "loss.lambda_opa", default=0.0) or 0.0)
        opa_warmup = int(OmegaConf.select(
            self.conf, "loss.opa_warmup",
            default=int(OmegaConf.select(
                self.conf, "loss.geom_warmup", default=15_000) or 0)) or 0)
        total_steps = int(OmegaConf.select(
            self.conf, "n_iterations", default=0) or 0)
        controller_steps = int(OmegaConf.select(
            self.conf, "post_processing.n_distillation_steps", default=0)
            or 0)
        scene_steps = total_steps - controller_steps
        sof_exact_opacity_active = bool(
            train and lam_opa > 0.0
            and opa_warmup <= int(frame_id) < scene_steps)

        rot, scale, dens, intrinsic_dens, factor, inverse_factor = self._density_shape_inputs(
            gaussians, gpu_batch, detach=False, frame_id=frame_id
        )

        num_gaussians = gaussians.num_gaussians
        # CoMe confidence (alpha-detached composite): conf_i = exp(gamma_i); absent/frozen -> empty (channel = 0).
        _come_gamma = getattr(gaussians, "confidence", None)
        if (bool(OmegaConf.select(self.conf, "loss.color_confidence", default=False))
                and isinstance(_come_gamma, torch.nn.Parameter) and _come_gamma.numel() > 0
                and _come_gamma.requires_grad):
            mog_confidence = torch.exp(_come_gamma).contiguous()
        else:
            mog_confidence = gaussians.positions.new_empty((0, 1))
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            (
                pred_features_alpha,
                pred_dist,
                hits_count,
                mog_visibility,
                pred_normal_buf,
                pred_depthvar_buf,
                pred_quant_buf,
                pred_extent_buf,
                pred_opacity_field_buf,
                pred_invd_buf,
                pred_invd2_buf,
            ) = Tracer._Autograd.apply(
                self.tracer_wrapper,
                physical_tile_token,
                frame_id,
                sof_exact_opacity_active,
                gaussians.n_active_features,
                rays_o.contiguous(),
                rays_d.contiguous(),
                gaussians.positions.contiguous(),
                rot.contiguous(),
                scale.contiguous(),
                dens.contiguous(),
                intrinsic_dens.contiguous(),
                factor.contiguous(),
                inverse_factor.contiguous(),
                gaussians.get_features().contiguous(),
                sensor,
                poses,
                mog_confidence,
            )

            # pred_features_alpha is [..., RAY_FEATURE_DIM + 1]: features (fp32) + density
            ray_feature_dim = gaussians.ray_feature_dim
            pred_features = pred_features_alpha[..., :ray_feature_dim].unsqueeze(0).contiguous()
            pred_opacity = pred_features_alpha[..., ray_feature_dim:].unsqueeze(0).contiguous()
            pred_dist = pred_dist.unsqueeze(0).contiguous()
            hits_count = hits_count.unsqueeze(0).contiguous()

            timings = self.tracer_wrapper.collect_times()

        _come_out = ({"pred_confidence": pred_quant_buf[..., 10:11].unsqueeze(0).contiguous()}
                     if pred_quant_buf.shape[-1] > 10 else {})
        return {
            **_come_out,
            "pred_features": pred_features,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": pred_normal_buf.unsqueeze(0).contiguous(),
            "pred_depth_var": pred_depthvar_buf.unsqueeze(0).contiguous(),
            "pred_quant": pred_quant_buf.unsqueeze(0).contiguous(),
            "pred_crossing": pred_quant_buf[..., 8:9].unsqueeze(0).contiguous(),  # ch FRONT_NQ+1 = exact 0.5-crossing t* (differentiable via IFT)
            "pred_crossing_rate": pred_quant_buf[..., 9:10].unsqueeze(0).contiguous(),  # ch FRONT_NQ+2 = Odot*=dO/dt|t* (crossing sharpness; detach for confidence)
            "pred_extent": pred_extent_buf.unsqueeze(0).contiguous(),
            "pred_opacity_field": pred_opacity_field_buf.unsqueeze(0).contiguous(),
            "pred_invd": pred_invd_buf.unsqueeze(0).contiguous(),
            "pred_invd2": pred_invd2_buf.unsqueeze(0).contiguous(),
            "hits_count": hits_count,
            "frame_time_ms": timings["forward_render"] if "forward_render" in timings else 0.0,
            "mog_visibility": mog_visibility,
        }

    @staticmethod
    def __fov2focal(fov_radians: float, pixels: int):
        return pixels / (2 * math.tan(fov_radians / 2))

    @staticmethod
    def __focal2fov(focal: float, pixels: int):
        return 2 * math.atan(pixels / (2 * focal))

    @staticmethod
    def __create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end):
        """
        Create SensorPose3D from start and end R, T matrices.
        Supports both global shutter (R_start == R_end) and rolling shutter.
        """
        # Convert START pose to quaternion
        T_world_sensor_t_start = torch.from_numpy(T_start).float()
        T_world_sensor_R_start = torch.from_numpy(R_start.transpose()).float()
        T_world_sensor_quat_start = SensorPose3DModel._SensorPose3DModel__so3_matrix_to_quat(T_world_sensor_R_start)
        T_world_sensor_tquat_start = torch.hstack([T_world_sensor_t_start.cpu(), T_world_sensor_quat_start.cpu()])

        # Convert END pose to quaternion (the same arrays: the START result, identical values)
        if R_end is R_start and T_end is T_start:
            T_world_sensor_tquat_end = T_world_sensor_tquat_start
        else:
            T_world_sensor_t_end = torch.from_numpy(T_end).float()
            T_world_sensor_R_end = torch.from_numpy(R_end.transpose()).float()
            T_world_sensor_quat_end = SensorPose3DModel._SensorPose3DModel__so3_matrix_to_quat(T_world_sensor_R_end)
            T_world_sensor_tquat_end = torch.hstack([T_world_sensor_t_end.cpu(), T_world_sensor_quat_end.cpu()])

        return SensorPose3D(
            T_world_sensors=[T_world_sensor_tquat_start, T_world_sensor_tquat_end],
            timestamps_us=[0, 1],  # Relative timestamps (start=0, end=1)
        )

    @staticmethod
    def __create_camera_parameters(gpu_batch, plugin=None):
        # ``plugin``: the native extension whose camera types are built (the
        # training renderer by default, or the inference-only extension).
        plugin = _3dgut_plugin if plugin is None else plugin
        SHUTTER_TYPE_MAP = {
            ShutterType.ROLLING_TOP_TO_BOTTOM.name: plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT.name: plugin.ShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP.name: plugin.ShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT.name: plugin.ShutterType.ROLLING_RIGHT_TO_LEFT,
            ShutterType.GLOBAL.name: plugin.ShutterType.GLOBAL,
        }

        # Check if rays are already in world space (rolling shutter with per-pixel poses)
        rays_in_world_space = gpu_batch.rays_in_world_space if hasattr(gpu_batch, "rays_in_world_space") else False

        if rays_in_world_space:
            # Rays are already in world space with correct per-pixel poses baked in
            # Use identity transform (no-op) so rays stay in world space
            R_start = R_end = np.eye(3, dtype=np.float32)
            T_start = T_end = np.zeros(3, dtype=np.float32)
        else:
            # Process the camera extrinsics (START and END poses for rolling shutter)
            pose_start = gpu_batch.T_to_world.squeeze()
            assert pose_start.ndim == 2

            # Extract END pose for rolling shutter (if available)
            if gpu_batch.T_to_world_end is not None:
                pose_end = gpu_batch.T_to_world_end.squeeze()
                assert pose_end.ndim == 2
            else:
                # Global shutter: END pose = START pose
                pose_end = pose_start

            # Helper function to convert pose to world-to-camera parameters
            def pose_to_world_to_camera(pose):
                C2W = np.concatenate((pose[:3, :4].cpu().detach().numpy(), np.zeros((1, 4))))
                C2W[3, 3] = 1.0
                W2C = np.linalg.inv(C2W)
                R = np.transpose(W2C[:3, :3])
                T = W2C[:3, 3]
                return R, T

            R_start, T_start = pose_to_world_to_camera(pose_start)
            # Global shutter: the END pose is the START tensor; reuse its conversion (identical values).
            R_end, T_end = (R_start, T_start) if pose_end is pose_start else pose_to_world_to_camera(pose_end)

        # Process the camera intrinsics
        if (K := gpu_batch.intrinsics) is not None:
            focalx, focaly, cx, cy = K[0], K[1], K[2], K[3]
            orig_w = int(2 * cx)
            orig_h = int(2 * cy)
            FovX = Tracer.__focal2fov(focalx, orig_w)
            FovY = Tracer.__focal2fov(focaly, orig_h)
            # Compute the camera model parameters
            camera_model_parameters = plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=np.array([orig_w, orig_h], dtype=np.uint32),
                shutter_type=plugin.ShutterType.GLOBAL,
                principal_point=np.array([orig_w, orig_h], dtype=np.float32) / 2,
                focal_length=np.array(
                    [orig_w / (2.0 * math.tan(FovX * 0.5)), orig_h / (2.0 * math.tan(FovY * 0.5))], dtype=np.float32
                ),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters) is not None:
            camera_model_parameters = plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                tangential_coeffs=K["tangential_coeffs"],
                thin_prism_coeffs=K.get("thin_prism_coeffs", np.zeros((4,), dtype=np.float32)),
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters) is not None:
            camera_model_parameters = plugin.fromOpenCVFisheyeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                max_angle=K["max_angle"],
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        elif (K := gpu_batch.intrinsics_FThetaCameraModelParameters) is not None:
            POLYNOMIAL_TYPE_MAP = {
                FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE.name: plugin.PolynomialType.PIXELDIST_TO_ANGLE,
                FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST.name: plugin.PolynomialType.ANGLE_TO_PIXELDIST,
            }
            camera_model_parameters = plugin.fromFThetaCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                reference_poly=POLYNOMIAL_TYPE_MAP[K["reference_poly"]],
                pixeldist_to_angle_poly=K["pixeldist_to_angle_poly"],
                angle_to_pixeldist_poly=K["angle_to_pixeldist_poly"],
                max_angle=K["max_angle"],
                linear_cde=K["linear_cde"],
            )
            return camera_model_parameters, Tracer.__create_sensor_pose_from_R_T(R_start, T_start, R_end, T_end)

        raise ValueError(
            f"Camera intrinsics unavailable or unsupported, input keys are [{', '.join(gpu_batch.keys())}]"
        )


def create_camera_parameters(gpu_batch, plugin):
    """Native camera model and sensor poses of ``gpu_batch`` for ``plugin``."""
    return Tracer._Tracer__create_camera_parameters(gpu_batch, plugin)
