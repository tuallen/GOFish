# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import torch

from gofish.camera.pixel_chart import torch_pixel_indices
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.physical_mip import refresh_filter
from threedgrut.strategy.base import BaseStrategy
from threedgrut.strategy.extent import WORLD_SIZE_EXTENTS, sof_max_radius_extent
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, quaternion_to_so3


class GSStrategy(BaseStrategy):
    """3DGS densification and pruning with SOF's densifier.

    With ``strategy.densify.generic_camera_absgrad`` the clone/split statistic
    is the image-space gradient of the projected mean in NDC, obtained through
    the pseudo-inverse of the calibrated camera Jacobian (so it remains valid
    beyond 90 degrees), plus AbsGS absolute gradients gated by the quantile
    matched to the signed-gradient fraction. Otherwise the stock 3DGUT
    world-gradient statistic is used.
    """

    # Densification statistics (all [N, 1]).
    _DENSIFICATION_BUFFERS = (
        ("densify_grad_norm_accum", torch.float),
        ("densify_grad_norm_denom", torch.int),
        ("densify_grad_norm_ndc_accum", torch.float),
        ("densify_grad_norm_ndc_abs_accum", torch.float),
        ("densify_grad_norm_ndc_denom", torch.float),
    )

    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        # Parameters related to densification, pruning and reset
        self.split_n_gaussians = self.conf.strategy.densify.split.n_gaussians
        self.relative_size_threshold = self.conf.strategy.densify.relative_size_threshold
        self.prune_density_threshold = self.conf.strategy.prune.density_threshold
        self.clone_grad_threshold = self.conf.strategy.densify.clone_grad_threshold
        self.split_grad_threshold = self.conf.strategy.densify.split_grad_threshold
        self.new_max_density = self.conf.strategy.reset_density.new_max_density
        self.generic_camera_absgrad = bool(self.conf.strategy.densify.get("generic_camera_absgrad", True))
        self.abs_grad = bool(self.conf.strategy.densify.get("abs_grad", False)) and self.generic_camera_absgrad
        self.world_size_threshold = float(self.conf.strategy.prune_big.get("scale_threshold", 0.1))
        self.world_size_extent = str(self.conf.strategy.prune_big.get("extent", "median"))
        if self.world_size_extent not in WORLD_SIZE_EXTENTS:
            raise ValueError(f"strategy.prune_big.extent must be one of {WORLD_SIZE_EXTENTS}, "
                             f"got {self.world_size_extent!r}")

        # The Mip filter is refreshed every 100 steps on the scene clock and
        # after every topology change.
        scene_steps = int(self.conf.n_iterations) - int(self.conf.post_processing.n_distillation_steps)
        self.mip_refresh_frequency = 100
        self.mip_last_periodic_refresh = scene_steps - self.mip_refresh_frequency

        for name, dtype in self._DENSIFICATION_BUFFERS:
            setattr(self, name, torch.empty([0, 1], dtype=dtype))

    def get_strategy_parameters(self) -> dict:
        return {name: (getattr(self, name),) for name, _ in self._DENSIFICATION_BUFFERS}

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        for name, dtype in self._DENSIFICATION_BUFFERS:
            if checkpoint is not None:
                setattr(self, name, checkpoint[name][0].detach())
            else:
                setattr(self, name, torch.zeros((self.model.num_gaussians, 1), dtype=dtype, device=self.model.device))

    def _pre_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        # The renderer accumulates per-hit absolute image-space gradients
        # through the camera-native map set here.
        if not self.abs_grad:
            return False
        if not check_step_condition(step, 0, self.conf.strategy.densify.end_iteration, 1):
            return False
        mapping = self._build_ndc_gradient_map(batch, train_dataset)
        self.model.renderer.tracer_wrapper.set_exact_screen_absgrad_map(mapping)
        return False

    def _post_backward(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        if check_step_condition(step, 0, self.conf.strategy.densify.end_iteration, 1):
            with torch.cuda.nvtx.range(f"train_{step}_grad_buffer"):
                self.update_gradient_buffer(
                    sensor_position=batch.T_to_world[0, :3, 3],
                    T_to_world=batch.T_to_world[0],
                    train_dataset=train_dataset,
                    batch=batch,
                )
        return False

    def _post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        scene_updated = False
        densify = self.conf.strategy.densify

        if check_step_condition(step, densify.start_iteration, densify.end_iteration, densify.frequency):
            self.densify_gaussians(scene_extent=scene_extent)
            scene_updated = True

        # SOF world-size prune, on the densification cadence after the first opacity reset.
        reset_interval = int(self.conf.strategy.reset_density.frequency)
        if (
            self.conf.strategy.prune_big.enabled
            and reset_interval > 0
            and step > reset_interval
            and check_step_condition(step, densify.start_iteration, densify.end_iteration, densify.frequency)
        ):
            self.prune_gaussians_big(self.world_size_prune_extent(scene_extent, train_dataset))
            scene_updated = True

        prune = self.conf.strategy.prune
        if check_step_condition(step, prune.start_iteration, prune.end_iteration, prune.frequency):
            self.prune_gaussians_opacity()
            scene_updated = True

        # Refresh the Mip filter after topology changes and periodically on the scene clock.
        if self.model._use_physical_mip3d_filter:
            periodic = 0 < step <= self.mip_last_periodic_refresh and step % self.mip_refresh_frequency == 0
            if scene_updated or periodic:
                refresh_filter(self.model, train_dataset)

        reset = self.conf.strategy.reset_density
        if check_step_condition(step, reset.start_iteration, reset.end_iteration, reset.frequency):
            self.reset_density()

        return scene_updated

    @staticmethod
    def _densification_intrinsics(train_dataset):
        """``(fx, fy, cx, cy, W, H)`` of the first camera at full COLMAP resolution."""
        intrinsic = list(train_dataset.cam_intrinsics.values())[0]
        p = [float(x) for x in list(intrinsic.params)]
        W = float(intrinsic.width)
        H = float(intrinsic.height)
        model = str(getattr(intrinsic, "model", "PINHOLE")).upper()
        if model.startswith("SIMPLE") or len(p) == 3:  # SIMPLE_PINHOLE: f, cx, cy
            fx = fy = p[0]
            cx = p[1]
            cy = p[2]
        else:  # PINHOLE/OPENCV: fx, fy, cx, cy
            fx = p[0]
            fy = p[1]
            cx = p[2]
            cy = p[3]
        return fx, fy, cx, cy, W, H

    @staticmethod
    def _fisheye_projection_jacobian(xyz_cam, intr):
        """Kannala-Brandt projection of camera-space centres and its pixel Jacobian rows."""
        dev, dtype = xyz_cam.device, xyz_cam.dtype
        fl = torch.as_tensor(intr["focal_length"], device=dev, dtype=dtype).reshape(-1)
        pp = torch.as_tensor(intr["principal_point"], device=dev, dtype=dtype).reshape(-1)
        kk = torch.as_tensor(intr["radial_coeffs"], device=dev, dtype=dtype).reshape(-1)[:4]
        max_angle = torch.as_tensor(intr["max_angle"], device=dev, dtype=dtype).reshape(-1)[0]
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        q2 = x * x + y * y
        q = torch.sqrt(q2.clamp_min(1e-16))
        r2 = (q2 + z * z).clamp_min(1e-16)
        theta = torch.atan2(q, z)
        t2 = theta * theta
        rho = theta * (1.0 + kk[0] * t2 + kk[1] * t2**2 + kk[2] * t2**3 + kk[3] * t2**4)
        drho = 1.0 + 3.0 * kk[0] * t2 + 5.0 * kk[1] * t2**2 + 7.0 * kk[2] * t2**3 + 9.0 * kk[3] * t2**4
        dthx = z * x / (q * r2)
        dthy = z * y / (q * r2)
        dthz = -q / r2
        dqdx, dqdy = x / q, y / q
        qsq = q * q
        dsdx = (drho * dthx * q - rho * dqdx) / qsq
        dsdy = (drho * dthy * q - rho * dqdy) / qsq
        dsdz = drho * dthz / q
        s = rho / q
        du = torch.stack((fl[0] * (s + x * dsdx), fl[0] * x * dsdy, fl[0] * x * dsdz), dim=1)
        dv = torch.stack((fl[1] * y * dsdx, fl[1] * (s + y * dsdy), fl[1] * y * dsdz), dim=1)
        # Optical-axis limit of the Jacobian.
        axis = q2 < 1e-12
        zsafe = z.abs().clamp_min(1e-6) * torch.where(z >= 0, 1.0, -1.0)
        du_axis = torch.stack((fl[0] / zsafe, torch.zeros_like(z), torch.zeros_like(z)), dim=1)
        dv_axis = torch.stack((torch.zeros_like(z), fl[1] / zsafe, torch.zeros_like(z)), dim=1)
        du = torch.where(axis[:, None], du_axis, du)
        dv = torch.where(axis[:, None], dv_axis, dv)
        u = fl[0] * x * s + pp[0]
        v = fl[1] * y * s + pp[1]
        u = torch.where(axis, pp[0].expand_as(u), u)
        v = torch.where(axis, pp[1].expand_as(v), v)
        return du, dv, u, v, r2, theta, drho, max_angle

    @staticmethod
    def _fisheye_visibility(physical, u, v, r2, theta, drho, max_angle, det, W, H):
        ui, vi, inside = torch_pixel_indices(u, v, W, H)
        pixel_valid = torch.zeros_like(inside)
        if bool(inside.any()):
            if physical is None:
                pixel_valid[inside] = True
            else:
                pixel_valid[inside] = physical[0, vi[inside], ui[inside], 0]
        return (
            (torch.sqrt(r2) > 0.2) & (theta <= max_angle) & (drho > 0.0) & (det > 1e-20) & inside & pixel_valid
        )

    @staticmethod
    def _raster_size(batch, physical):
        if physical is not None:
            return int(physical.shape[1]), int(physical.shape[2])
        return int(batch.rays_dir.shape[1]), int(batch.rays_dir.shape[2])

    @torch.no_grad()
    def _build_ndc_gradient_map(self, batch, train_dataset):
        """Per-Gaussian ``[N, 2, 3]`` map from a world-space hit gradient to the NDC
        gradient of the projected mean (zero for centres outside the camera domain)."""
        T_to_world = batch.T_to_world[0]
        Rc = T_to_world[:3, :3].to(self.model.device).float()
        tc = T_to_world[:3, 3].to(self.model.device).float()
        xyz = self.model.positions.detach()
        xyz_cam = (xyz - tc[None, :]) @ Rc

        intr = batch.intrinsics_OpenCVFisheyeCameraModelParameters
        if intr is not None:
            physical = batch.training_valid_mask
            du, dv, u, v, r2, theta, drho, max_angle = self._fisheye_projection_jacobian(xyz_cam, intr)
            a00 = (du * du).sum(1)
            a01 = (du * dv).sum(1)
            a11 = (dv * dv).sum(1)
            det = a00 * a11 - a01 * a01
            detsafe = det.clamp_min(1e-20)
            B0 = (a11[:, None] * du - a01[:, None] * dv) / detsafe[:, None]
            B1 = (-a01[:, None] * du + a00[:, None] * dv) / detsafe[:, None]
            H, W = self._raster_size(batch, physical)
            Bcam = torch.stack((B0 * (W * 0.5), B1 * (H * 0.5)), dim=1)
            visible = self._fisheye_visibility(physical, u, v, r2, theta, drho, max_angle, det, W, H)
        else:
            fx, fy, cx, cy, W, H = self._densification_intrinsics(train_dataset)
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            zc = z.clamp(min=1e-6)
            u, v = fx * (x / zc) + cx, fy * (y / zc) + cy
            _, _, inside = torch_pixel_indices(u, v, W, H)
            visible = (z > 0.2) & inside
            Bcam = torch.zeros((xyz.shape[0], 2, 3), device=xyz.device, dtype=xyz.dtype)
            Bcam[:, 0, 0] = z * (W / (2.0 * fx))
            Bcam[:, 1, 1] = z * (H / (2.0 * fy))

        # Row-vector convention: g_cam = g_world @ Rc, hence B_world = B_cam @ Rc^T.
        Bworld = torch.matmul(Bcam, Rc.transpose(0, 1))
        Bworld = torch.nan_to_num(Bworld, nan=0.0, posinf=0.0, neginf=0.0)
        Bworld[~visible] = 0.0
        return Bworld.detach().to(dtype=torch.float32).contiguous()

    def _fisheye_ndc_statistics(self, xyz_cam, g_cam, abs_grad, batch):
        """Camera-native NDC gradient of the projected mean and its visibility.

        Solves ``g_cam = J^T g_pixel`` in the least-squares sense, then scales
        the pixel gradient to NDC.
        """
        intr = batch.intrinsics_OpenCVFisheyeCameraModelParameters
        physical = batch.training_valid_mask
        du, dv, u, v, r2, theta, drho, max_angle = self._fisheye_projection_jacobian(xyz_cam, intr)
        a00 = (du * du).sum(1)
        a01 = (du * dv).sum(1)
        a11 = (dv * dv).sum(1)
        det = a00 * a11 - a01 * a01
        detsafe = det.clamp_min(1e-20)
        b0 = (du * g_cam).sum(1)
        b1 = (dv * g_cam).sum(1)
        gu = (a11 * b0 - a01 * b1) / detsafe
        gv = (-a01 * b0 + a00 * b1) / detsafe
        H, W = self._raster_size(batch, physical)
        ndc_sig = torch.sqrt((gu * (W * 0.5)) ** 2 + (gv * (H * 0.5)) ** 2)
        vis = self._fisheye_visibility(physical, u, v, r2, theta, drho, max_angle, det, W, H)
        ndc_sig = torch.nan_to_num(ndc_sig, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)
        ndc_abs = None
        if abs_grad is not None:
            ndc_abs = torch.linalg.vector_norm(abs_grad, dim=1, keepdim=True)
        return ndc_sig, ndc_abs, vis

    @torch.no_grad()
    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, sensor_position: torch.Tensor, T_to_world=None, train_dataset=None,
                               batch=None) -> None:
        params_grad = self.model.positions.grad
        assert params_grad is not None
        mask = (params_grad != 0).max(dim=1)[0]
        distance_to_camera = (self.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)

        self.densify_grad_norm_accum[mask] += (
            torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2
        )
        self.densify_grad_norm_denom[mask] += 1

        if not self.generic_camera_absgrad:
            return

        # Per-view absolute NDC gradient [N, 2] accumulated by the renderer backward.
        abs_grad = None
        if self.abs_grad:
            abs_grad = self.model.renderer.tracer_wrapper.collect_abs_grad()
            if tuple(abs_grad.shape) != (self.model.num_gaussians, 2):
                raise RuntimeError("per-view absolute gradient must have shape [N, 2]")

        Rc = T_to_world[:3, :3].to(self.model.device).float()
        tc = T_to_world[:3, 3].to(self.model.device).float()
        xyz = self.model.positions
        xyz_cam = (xyz - tc[None, :]) @ Rc
        z = xyz_cam[:, 2]
        zc = z.clamp(min=1e-6)
        g_cam = params_grad @ Rc
        if batch.intrinsics_OpenCVFisheyeCameraModelParameters is not None:
            ndc_sig, ndc_abs, vis = self._fisheye_ndc_statistics(xyz_cam, g_cam, abs_grad, batch)
        else:
            # Pinhole: dL/dNDC_x = dL/dx_cam * (W z) / (2 f_x).
            fx, fy, cx, cy, Wpx, Hpx = self._densification_intrinsics(train_dataset)
            cxc = Wpx / (2.0 * fx)
            cyc = Hpx / (2.0 * fy)
            ndc_gx = g_cam[:, 0] * (cxc * z)
            ndc_gy = g_cam[:, 1] * (cyc * z)
            ndc_sig = torch.nan_to_num(
                torch.sqrt(ndc_gx * ndc_gx + ndc_gy * ndc_gy), nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(1)
            u = fx * (xyz_cam[:, 0] / zc) + cx
            v = fy * (xyz_cam[:, 1] / zc) + cy
            _, _, inside = torch_pixel_indices(u, v, Wpx, Hpx)
            vis = (z > 0.2) & inside
            ndc_abs = None if abs_grad is None else torch.linalg.vector_norm(abs_grad, dim=1, keepdim=True)
        self.densify_grad_norm_ndc_accum[vis] += ndc_sig[vis]
        self.densify_grad_norm_ndc_denom[vis] += 1
        if ndc_abs is not None:
            self.densify_grad_norm_ndc_abs_accum[vis] += ndc_abs[vis]

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        assert (
            self.model.optimizer is not None
        ), "Optimizer need to be initialized before splitting and cloning the Gaussians"
        densify_grad_norm = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        grad_abs = None
        Q = None
        if self.generic_camera_absgrad:
            denom = self.densify_grad_norm_ndc_denom.clamp(min=1)
            densify_grad_norm = torch.nan_to_num(
                self.densify_grad_norm_ndc_accum / denom, nan=0.0, posinf=0.0, neginf=0.0
            )
            if self.abs_grad:
                # AbsGS gate: select by absolute gradient the same fraction of
                # Gaussians that exceed the signed clone threshold.
                grad_abs = torch.nan_to_num(
                    self.densify_grad_norm_ndc_abs_accum / denom, nan=0.0, posinf=0.0, neginf=0.0
                )
                grad_abs[grad_abs.isnan()] = 0.0
                grad_abs = grad_abs.squeeze()
                ratio = (densify_grad_norm.squeeze() >= self.clone_grad_threshold).float().mean()
                Q = (
                    torch.quantile(grad_abs.reshape(-1), (1.0 - ratio).clamp(0.0, 1.0))
                    if grad_abs.numel() > 0
                    else torch.tensor(1e4, device=grad_abs.device)
                )
                # A non-positive quantile would select every zero-gradient Gaussian.
                if (not bool(torch.isfinite(Q))) or float(Q) <= 0.0:
                    Q = None

        self.clone_gaussians(densify_grad_norm.squeeze(), scene_extent, grad_abs, Q)
        self.split_gaussians(densify_grad_norm.squeeze(), scene_extent, grad_abs, Q)

        torch.cuda.empty_cache()

    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float,
                        grad_abs: torch.Tensor = None, Q: torch.Tensor = None):
        n_init_points = self.model.num_gaussians

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        mask = torch.where(padded_grad >= self.split_grad_threshold, True, False)
        if grad_abs is not None and Q is not None:
            padded_abs = torch.zeros((n_init_points), device="cuda")
            padded_abs[: grad_abs.shape[0]] = grad_abs.squeeze()
            mask = torch.logical_or(mask, (padded_abs >= Q) & (padded_abs > 0.0))
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values > self.relative_size_threshold * scene_extent
        )

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Split {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [self.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scale":
                p_split = self.model.scale_activation_inv(
                    self.model.scale_activation(param[mask].repeat(repeats)) / (0.8 * self.split_n_gaussians)
                )
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.nn.Parameter(torch.cat([param[~mask], p_split]), requires_grad=param.requires_grad)

            return p_new

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros((self.split_n_gaussians * int(mask.sum()), *v.shape[1:]), device=v.device)
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float,
                        grad_abs: torch.Tensor = None, Q: torch.Tensor = None):
        assert densify_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(densify_grad_norm >= self.clone_grad_threshold, True, False)
        if grad_abs is not None and Q is not None:
            mask = torch.logical_or(mask, (grad_abs >= Q) & (grad_abs > 0.0))

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent
        )

        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        # SOF clone: the child centre is sampled from the parent Gaussian,
        # parent_xyz + R_parent @ N(0, parent_scale).
        clone_offsets = None
        if bool(mask.any()):
            stds = self.model.get_scale()[mask]
            means = torch.zeros((stds.size(0), 3), device=stds.device)
            samples = torch.normal(mean=means, std=stds)
            rots = quaternion_to_so3(self.model.rotation[mask])
            clone_offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            child = param[mask]
            if clone_offsets is not None and name == "positions":
                child = child + clone_offsets
            return torch.nn.Parameter(torch.cat([param, child]), requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat([v, torch.zeros((int(mask.sum()), *v.shape[1:]), device=v.device)])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    def world_size_prune_extent(self, scene_extent: float, train_dataset) -> float:
        """Extent multiplying ``scale_threshold`` in the world-size prune (``strategy.prune_big.extent``)."""
        if self.world_size_extent == "median":
            return scene_extent
        return sof_max_radius_extent(train_dataset.get_observer_points())

    @torch.no_grad()
    def prune_gaussians_big(self, scene_extent: float):
        """SOF world-size prune: remove Gaussians larger than ``scale_threshold x scene_extent``."""
        mask = ~(self.model.get_scale().max(dim=1).values > self.world_size_threshold * scene_extent)
        if self.conf.strategy.print_stats:
            logger.info(f"World-size prune ({self.world_size_extent}): scene_extent={scene_extent:.9g} "
                        f"world_size_threshold={self.world_size_threshold * scene_extent:.9g}")
            n_before = mask.shape[0]
            n_prune = n_before - int(mask.sum())
            logger.info(f"Big-points-pruned {n_prune} / {n_before} ({n_prune/max(n_before,1)*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def prune_gaussians_opacity(self):
        mask = self.model.get_density().squeeze() >= self.prune_density_threshold

        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Density-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def reset_densification_buffers(self) -> None:
        num_gaussians = self.model.get_positions().shape[0]
        for name, dtype in self._DENSIFICATION_BUFFERS:
            setattr(self, name, torch.zeros((num_gaussians, 1), dtype=dtype, device=self.model.device))

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        for name, _ in self._DENSIFICATION_BUFFERS:
            setattr(self, name, getattr(self, name)[valid_mask])

    def reset_density(self):
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "density", "wrong parameter passed to update_param_fn"
            densities = torch.clamp(
                param,
                max=self.model.density_activation_inv(torch.tensor(self.new_max_density)).item(),
            )
            return torch.nn.Parameter(densities)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn, names=["density"])
