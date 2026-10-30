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
#
# Portions derived from SOF and CoMe (https://github.com/r4dl/SOF, https://github.com/r4dl/CoMe):
# the SOF geometry terms and CoMe's confidence and normal-variance terms in get_losses, distributed under the Gaussian-Splatting License (Inria / MPII;
# non-commercial research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import (
    DEFAULT_DEVICE,
    MultiEpochsDataLoader,
    configure_camera_sampler_seed,
    configure_dataloader_for_platform,
)
from threedgrut.model.losses import MaskedLossCache, _ssim_maps, masked_dssim, masked_l1, masked_l2, masked_mean, ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.physical_mip import refresh_filter
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer
from threedgrut.utils.render import apply_background, apply_post_processing, install_physical_fov_gradient_gate
from threedgrut.utils.timer import CudaTimer
from threedgut_tracer.kb_table.table import check_dataset as check_calibration_table


_PPISP_MAIN_PARAMETER_NAMES = (
    "exposure_params",
    "vignetting_params",
    "color_params",
    "crf_params",
    "radial_attenuation_params",
    "flatfield_params",
)


def _cross5(mask):
    """Centre/left/right/up/down closure of an [H, W] boolean mask (interior pixels)."""
    return mask[1:-1, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:] & mask[:-2, 1:-1] & mask[2:, 1:-1]


class Trainer3DGRUT:
    """GOFish trainer (3DGUT renderer, SOF geometry objective, PPISP).

    The schedule has two clocks: ``n_iterations - n_distillation_steps`` joint
    scene + PPISP steps, followed by ``n_distillation_steps`` steps that train
    only PPISP's novel-view controller with the scene frozen. Checkpoint
    ``ours_<scene_steps>`` is the geometry checkpoint used for meshing; the
    final checkpoint carries the trained controller and is used for rendering.
    """

    model: MixtureOfGaussians
    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    scene_extent: float = 1.0
    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # (min, max)
    strategy: BaseStrategy
    criterions: Dict
    tracking: Dict
    post_processing: Optional[nn.Module] = None
    post_processing_optimizers: Optional[list] = None
    post_processing_schedulers: Optional[list] = None
    _distillation_start_step: int = -1

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        self.conf = conf
        self.device = device if device is not None else DEFAULT_DEVICE
        self.global_step = 0
        self.n_iterations = conf.n_iterations
        self.scene_steps = int(conf.n_iterations) - int(conf.post_processing.n_distillation_steps)
        self.n_epochs = 0
        self.val_frequency = conf.val_frequency

        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self._masked_loss_cache = MaskedLossCache()
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.init_post_processing(conf)
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)

        # Controller-only phase: the scene is frozen, so the pre-PPISP render
        # of each training frame is computed once and reused.
        self._controller_render_cache: dict[tuple[int, int], dict] = {}

    def init_dataloaders(self, conf: DictConfig):
        configure_camera_sampler_seed(int(conf.dataset.camera_sampler_seed))

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )
        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        # Fail before any compilation when a camera has no complete-admission table row.
        check_calibration_table(train_dataset)
        self.train_dataset = train_dataset
        self.train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        self.val_dataset = val_dataset
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        self.scene_extent = train_dataset.get_scene_extent()
        self.scene_bbox = train_dataset.get_scene_bbox()

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        if conf.strategy.method != "GSStrategy":
            raise ValueError(f"unrecognized strategy {conf.strategy.method}")
        from threedgrut.strategy.gs import GSStrategy

        self.strategy = GSStrategy(conf, self.model)

    def setup_training(self, conf: DictConfig, model: MixtureOfGaussians, train_dataset: BoundedMultiViewDataset):
        """Initialize the Gaussians from the COLMAP points, the optimizer, the
        densification buffers and the Mip filter."""
        if conf.initialization.method != "colmap":
            raise ValueError(f"unrecognized initialization.method {conf.initialization.method}")
        logger.info("Initiating new training...")
        observer_points = torch.tensor(train_dataset.get_observer_points(), dtype=torch.float32, device=self.device)
        model.init_from_colmap(conf.path, observer_points)
        self.strategy.init_densification_buffer()
        model.build_acc()
        model.setup_optimizer()

        # The Mip filter is established from the training cameras before the first render.
        if model._use_physical_mip3d_filter:
            refresh_filter(model, train_dataset)
            model.build_acc(rebuild=True)

        self.global_step = 0
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(conf, object_name, conf.out_dir, conf.experiment_name)
        logger.info(f"Training logs will be saved to: {out_dir}")

        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        self.tracking = Dict(writer=writer, run_name=run_name, object_name=object_name, output_dir=out_dir)

    def init_post_processing(self, conf: DictConfig):
        """PPISP trained jointly with the scene, then its controller alone."""
        method = conf.post_processing.method
        if method is None:
            return
        if method != "ppisp":
            raise ValueError(f"Unknown post-processing method: {method}")
        from gofish.appearance.wide_angle import ppisp_config_from
        from threedgrut.datasets.wide_angle import init_flatfield_gauge, wide_angle_geometry
        from gofish_ppisp import PPISP

        frames_per_camera = self.train_dataset.get_frames_per_camera()
        num_cameras = len(frames_per_camera)
        num_frames = sum(frames_per_camera)

        ppisp_config, self._distillation_start_step = ppisp_config_from(conf, training=True, scene_steps=self.scene_steps)
        if self._distillation_start_step >= 0:
            logger.info(f"PPISP controller activates at step {self.scene_steps}")
        radial_geometry = None
        geometry = None
        if ppisp_config.radial_attenuation_knots > 0:
            geometry = wide_angle_geometry(self.train_dataset, num_cameras)
            radial_geometry = torch.tensor(geometry, dtype=torch.float32)
            logger.info(f"Wide-angle PPISP geometry (pp_x, pp_y, rho_ap): {geometry}")
        self.post_processing = PPISP(
            radial_attenuation_geometry=radial_geometry,
            num_cameras=num_cameras,
            num_frames=num_frames,
            config=ppisp_config,
        ).to(self.device)
        if geometry is not None and hasattr(self.post_processing, "flatfield_free"):
            init_flatfield_gauge(self.post_processing, self.train_dataset, geometry, ppisp_config.flatfield_grid)
        self.post_processing_optimizers = self.post_processing.create_optimizers()
        self.post_processing_schedulers = self.post_processing.create_schedulers(
            self.post_processing_optimizers,
            max_optimization_iters=conf.n_iterations,
        )
        logger.info(f"{method.upper()} initialized: {num_cameras} cameras, {num_frames} frames")

    def _freeze_ppisp_main(self) -> None:
        """Freeze PPISP's per-frame/per-camera parameters; only the controller trains on."""
        for name in _PPISP_MAIN_PARAMETER_NAMES:
            if hasattr(self.post_processing, name):  # wide-angle parameters exist only when enabled
                getattr(self.post_processing, name).requires_grad_(False)

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(self, gpu_batch, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Photometric loss plus SOF's geometry objective.

        With a calibrated camera domain every term is averaged over the valid
        pixels only.
        """
        conf = self.conf.loss
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_features"]
        mask = gpu_batch.mask
        physical_fov_enabled = gpu_batch.physical_fov_enabled
        training_valid_mask = gpu_batch.training_valid_mask
        support = self._masked_loss_cache.bind_batch(gpu_batch) if physical_fov_enabled else None
        if mask is not None and not physical_fov_enabled:
            rgb_gt = rgb_gt * mask
            rgb_pred = rgb_pred * mask

        zero = torch.zeros(1, device=self.device)
        step = self.global_step

        # CoMe confidence (ScanNet++ recipe): C = sum_i w_i conf_i / max(1-T, 1e-3)
        # with detached weights, clamped to [min, max]; the photometric residuals are weighted
        # by C and -beta*log C is added. C == 1 reproduces the unweighted loss exactly.
        conf_map = None
        if (bool(conf.get("color_confidence", False))
                and int(conf.get("color_confidence_from_iter", 500)) <= step < self.scene_steps):
            if "pred_confidence" not in outputs:
                raise RuntimeError("CoMe confidence is enabled but the renderer emitted no pred_confidence")
            if not physical_fov_enabled:
                raise RuntimeError("CoMe confidence is defined on the calibrated-domain loss supports")
            coverage = outputs["pred_opacity"].detach().clamp(min=1e-3)
            conf_map = (outputs["pred_confidence"] / coverage).clamp(
                min=float(conf.get("color_confidence_min", 1e-3)),
                max=float(conf.get("color_confidence_max", 5.0)))  # [1,H,W,1]

        # L1 / L2 / D-SSIM
        loss_l1, lambda_l1 = zero, 0.0
        if conf.use_l1:
            with torch.cuda.nvtx.range("loss-l1"):
                loss_l1 = (masked_l1(rgb_pred, rgb_gt, support) if physical_fov_enabled
                           else torch.abs(rgb_pred - rgb_gt).mean())
                if conf_map is not None:
                    loss_l1 = masked_mean((rgb_pred - rgb_gt).abs() * conf_map, support)
                lambda_l1 = conf.lambda_l1

        loss_l2, lambda_l2 = zero, 0.0
        if conf.use_l2:
            with torch.cuda.nvtx.range("loss-l2"):
                loss_l2 = (masked_l2(outputs["pred_features"], rgb_gt, support) if physical_fov_enabled
                           else torch.nn.functional.mse_loss(outputs["pred_features"], rgb_gt))
                lambda_l2 = conf.lambda_l2

        loss_ssim, lambda_ssim = zero, 0.0
        if conf.use_ssim:
            with torch.cuda.nvtx.range("loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_features_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = (masked_dssim(rgb_gt_full, pred_features_full, support, self._masked_loss_cache)
                             if physical_fov_enabled else 1.0 - ssim(pred_features_full, rgb_gt_full))
                if conf_map is not None:
                    luminance, contrast_structure = _ssim_maps(
                        rgb_gt_full, pred_features_full, self._masked_loss_cache.gaussian_window(rgb_gt_full))
                    gate = support.ssim_support_like(luminance)
                    weight = conf_map.permute(0, 3, 1, 2).expand_as(luminance)
                    loss_ssim = support.gather((1.0 - luminance * contrast_structure) * weight, gate).mean()
                lambda_ssim = conf.lambda_ssim

        loss_conf = zero
        if conf_map is not None:
            loss_conf = -float(conf.get("color_confidence_beta", 0.0)) * masked_mean(torch.log(conf_map), support)

        # SOF distortion on the inverse-depth-mapped coordinate
        # m(t) = far/(far-near) * (1 - near/t); with the renderer sums
        # A = sum w, S1 = sum w/t, S2 = sum w/t^2 the per-ray loss is
        # C^2 (A S2 - S1^2), C = far*near/(far-near). A is detached.
        loss_distort = zero
        lam_distort = float(conf.lambda_distortion)
        if lam_distort > 0.0 and step >= int(conf.geom_warmup):
            opac = outputs["pred_opacity"][0, ..., 0]
            C2 = float(conf.dist_far) * float(conf.dist_near) / (float(conf.dist_far) - float(conf.dist_near))
            Ow = opac.detach().clamp(min=1e-3)
            S1 = outputs["pred_invd"][0, ..., 0]
            S2 = outputs["pred_invd2"][0, ..., 0]
            dmap = (C2 * C2) * (Ow * S2 - S1 * S1).clamp(min=0.0)
            loss_distort = lam_distort * (masked_mean(dmap, support) if physical_fov_enabled else dmap.mean())

        # SOF depth-normal consistency and edge-aware normal smoothness. The
        # depth normal is built from the (detached) 0.5-crossing depth along the
        # calibrated rays, so it is camera-model agnostic; gradients flow
        # through the rendered normal.
        loss_esmooth = zero
        loss_dnc = zero
        lam_dnc = float(conf.lambda_dnc)
        lam_esmooth = float(conf.lambda_esmooth)
        if (lam_dnc > 0.0 or lam_esmooth > 0.0) and step >= int(conf.dnc_warmup):
            crossing = outputs["pred_crossing"][0, ..., 0]
            fallback = outputs["pred_dist"][0, ..., 0].detach()
            crossing_valid = crossing > 1e-6
            D = torch.where(crossing_valid, crossing.detach(), fallback)
            D = torch.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
            rd = gpu_batch.rays_dir[0]  # [H,W,3] sensor-frame ray dirs
            rd = rd / torch.sqrt((rd * rd).sum(-1, keepdim=True) + 1e-12)
            Rws = gpu_batch.T_to_world[0][:3, :3]
            Cw = gpu_batch.T_to_world[0][:3, 3]
            P = (D.unsqueeze(-1) * rd) @ Rws.T + Cw  # [H,W,3] world points
            dPu = P[1:-1, 2:, :] - P[1:-1, :-2, :]
            dPv = P[2:, 1:-1, :] - P[:-2, 1:-1, :]
            Nd_raw = torch.cross(dPu, dPv, dim=-1)
            Nd_n2 = (Nd_raw * Nd_raw).sum(-1, keepdim=True)
            Nd = Nd_raw / torch.sqrt(Nd_n2 + 1e-6)
            view_raw = Cw - P[1:-1, 1:-1, :]
            view = view_raw / torch.sqrt((view_raw * view_raw).sum(-1, keepdim=True) + 1e-6)
            sgn = torch.sign((Nd * view).sum(-1, keepdim=True))
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            Nd = Nd * sgn  # orient toward the camera
            Nd = torch.nan_to_num(Nd, nan=0.0, posinf=0.0, neginf=0.0)
            opac = outputs["pred_opacity"][0, ..., 0]
            vmask = opac > 0.5
            if mask is not None:
                vmask = vmask & (mask[0, ..., 0] > 0.5)
            deg_ok = (Nd_n2[..., 0] > 1e-8) & torch.isfinite(Nd).all(-1)
            vm = vmask[1:-1, 1:-1] & deg_ok
            if physical_fov_enabled:
                vm = vm & _cross5(training_valid_mask[0, ..., 0])
            vm = vm & crossing_valid[1:-1, 1:-1]
            vm = vm.float()
            nvm = vm.sum().clamp(min=1.0)
            if lam_dnc > 0.0:
                Nrw = outputs["pred_normals"][0]  # world-frame rendered normal
                Nrw = Nrw / Nrw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                Nrw = Nrw[1:-1, 1:-1, :]
                sgnr = torch.sign((Nrw * view).sum(-1, keepdim=True))
                sgnr = torch.where(sgnr == 0, torch.ones_like(sgnr), sgnr)
                Nrw = Nrw * sgnr
                Nrw = torch.nan_to_num(Nrw, nan=0.0, posinf=0.0, neginf=0.0)
                cons = 1.0 - (Nd * Nrw).sum(-1)
                loss_dnc = lam_dnc * (cons * vm).sum() / nvm
            if lam_esmooth > 0.0:
                # SOF Eq. 22 on the rendered normal: |grad N| exp(-|grad I|).
                Nrw_e = outputs["pred_normals"][0]
                Nrw_e = Nrw_e / Nrw_e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                Nrw_e = torch.nan_to_num(Nrw_e, nan=0.0, posinf=0.0, neginf=0.0)
                Igt = gpu_batch.rgb_gt[0]
                dNx = (Nrw_e[:, 1:, :] - Nrw_e[:, :-1, :]).abs().sum(-1)
                dNy = (Nrw_e[1:, :, :] - Nrw_e[:-1, :, :]).abs().sum(-1)
                dIx = (Igt[:, 1:, :] - Igt[:, :-1, :]).abs().sum(-1)
                dIy = (Igt[1:, :, :] - Igt[:-1, :, :]).abs().sum(-1)
                gx = dNx * torch.exp(-dIx)
                gy = dNy * torch.exp(-dIy)
                if physical_fov_enabled:
                    pm = mask[0, ..., 0] > 0.5
                    mx = pm[:, 1:] & pm[:, :-1]
                    my = pm[1:, :] & pm[:-1, :]
                    gx_mean = gx[mx].mean() if bool(mx.any()) else gx.new_zeros(())
                    gy_mean = gy[my].mean() if bool(my.any()) else gy.new_zeros(())
                    loss_esmooth = lam_esmooth * 0.5 * (gx_mean + gy_mean)
                else:
                    loss_esmooth = lam_esmooth * 0.5 * (gx.mean() + gy.mean())

        # SOF opacity-field supervision: the opacity accumulated up to the
        # exact 0.5 crossing is pushed to 0.5 along observed rays.
        loss_opa = zero
        lam_opa = float(conf.lambda_opa)
        if lam_opa > 0.0 and step >= int(conf.geom_warmup):
            O = outputs["pred_opacity_field"][0, ..., 0]
            vm = (mask[0, ..., 0] > 0.5) if mask is not None else torch.ones_like(O, dtype=torch.bool)
            loss_opa = lam_opa * (((O - 0.5) ** 2) * vm).sum() / vm.sum().clamp(min=1).float()

        # World-unit extent: alpha-weighted along-ray standard deviation (alphas detached).
        loss_extent = zero
        lam_extent = float(conf.lambda_extent)
        if lam_extent > 0.0 and step >= int(conf.geom_warmup):
            E = outputs["pred_extent"][0, ..., 0]
            vm = (mask[0, ..., 0] > 0.5) if mask is not None else torch.ones_like(E, dtype=torch.bool)
            loss_extent = lam_extent * (E * vm).sum() / vm.sum().clamp(min=1).float()

        # CoMe normal variance sum_i w_i ||n_i - N||^2 with N = sum_i w_i n_i held constant in the
        # gradient. With unit per-hit normals and A = sum_i w_i the value is A - 2 N.sg(N) + A |sg(N)|^2.
        loss_nvar = zero
        lam_nvar = float(conf.get("lambda_normal_variance", 0.0))
        if lam_nvar > 0.0 and int(conf.get("normal_variance_warmup", 15000)) <= step < self.scene_steps:
            A = outputs["pred_opacity"][0, ..., 0]
            N = outputs["pred_normals"][0]
            Nd = N.detach()
            nvar = A - 2.0 * (N * Nd).sum(-1) + A * (Nd * Nd).sum(-1)
            loss_nvar = lam_nvar * (masked_mean(nvar, support) if physical_fov_enabled else nvar.mean())

        loss = (lambda_l1 * loss_l1 + lambda_ssim * loss_ssim
                + loss_distort + loss_esmooth + loss_opa + loss_dnc + loss_extent
                + loss_conf + loss_nvar)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("non-finite total loss")
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            distortion_loss=loss_distort,
            esmooth_loss=loss_esmooth,
            opa_loss=loss_opa,
            dnc_loss=loss_dnc,
            extent_loss=loss_extent,
            confidence_loss=loss_conf,
            normal_variance_loss=loss_nvar,
        )

    @torch.cuda.nvtx.range("log_training_iter")
    def log_training_iter(self, batch_losses: dict[str, torch.Tensor]) -> None:
        writer = self.tracking.writer
        global_step = self.global_step
        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            losses = {k: v.detach().item() for k, v in batch_losses.items()}
            for name, value in losses.items():
                writer.add_scalar(f"loss/{name}/train", value, global_step)
            writer.add_scalar("num_particles/train", self.model.num_gaussians, global_step)
        logger.log_progress(task_name="Training", advance=1, step=f"{global_step}", loss=batch_losses["total_loss"].detach().item())

    @torch.cuda.nvtx.range("on_training_end")
    def on_training_end(self):
        if self._controller_render_cache:
            size = sum(v.numel() * v.element_size() for entry in self._controller_render_cache.values()
                       for v in entry.values() if torch.is_tensor(v))
            logger.info(f"controller-phase render cache: {len(self._controller_render_cache)} frames, "
                        f"{size / 1e9:.2f} GB of host RAM")
        self.teardown_dataloaders()
        self.save_checkpoint(last_checkpoint=True)

    @torch.cuda.nvtx.range("save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Save ``ours_<step>/ckpt_<step>.pt`` (or ``ckpt_last.pt``)."""
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        # Before the scene freezes, a checkpoint carries a filter that is
        # current with the saved centres.
        if self.model._use_physical_mip3d_filter and global_step < self.scene_steps and not last_checkpoint:
            refresh_filter(self.model, self.train_dataset)
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}
        parameters = {**parameters, **self.strategy.get_strategy_parameters()}

        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.post_processing_optimizers],
                "schedulers": [sched.state_dict() for sched in self.post_processing_schedulers],
            }

        if not last_checkpoint:
            os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def _controller_cache_active(self, global_step: int) -> bool:
        return 0 <= self._distillation_start_step <= global_step < self.n_iterations

    @torch.cuda.nvtx.range("run_train_iter")
    def run_train_iter(self, global_step: int, batch: dict, profilers: dict, conf: DictConfig):
        # Controller-only phase: Gaussians frozen, densification suspended.
        if self._distillation_start_step >= 0 and global_step >= self._distillation_start_step:
            self.model.freeze_gaussians()
            self.strategy.suspend()

        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

        profilers["step_total"].start()

        if self.val_frequency > 0 and global_step > 0 and global_step % self.val_frequency == 0:
            self.run_validation_pass(conf)

        cache_active = self._controller_cache_active(global_step)
        cache_key = (int(gpu_batch.camera_idx), int(gpu_batch.frame_idx))
        cached = self._controller_render_cache.get(cache_key) if cache_active else None

        with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
            profilers["inference"].start()
            if cached is None:
                outputs = self.model(gpu_batch, train=True, frame_id=global_step)
            else:
                outputs = {
                    name: value.to(device=self.model.positions.device, copy=True) if torch.is_tensor(value) else value
                    for name, value in cached.items()
                }
                outputs["frame_time_ms"] = 0.0
            install_physical_fov_gradient_gate(outputs, gpu_batch)
            profilers["inference"].end()

        if cached is None:
            outputs = apply_background(self.model.background, outputs, gpu_batch, training=True)
            if cache_active:
                self._controller_render_cache[cache_key] = {
                    name: (None if name == "frame_time_ms"
                           else value.detach().cpu().contiguous() if torch.is_tensor(value) else value)
                    for name, value in outputs.items()
                }

        if self.post_processing is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing"):
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=True)

        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            batch_losses = self.get_losses(gpu_batch, outputs)
            if self.post_processing is not None:
                post_processing_reg_loss = self.post_processing.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + post_processing_reg_loss
                batch_losses["post_processing_reg_loss"] = post_processing_reg_loss
            if not bool(torch.isfinite(batch_losses["total_loss"]).all()):
                raise FloatingPointError("non-finite total loss")

        with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
            self.strategy.pre_backward(
                step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset,
                batch=gpu_batch, writer=self.tracking.writer,
            )

        with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
            profilers["backward"].start()
            batch_losses["total_loss"].backward()
            profilers["backward"].end()

        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            self.strategy.post_backward(
                step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset,
                batch=gpu_batch, writer=self.tracking.writer,
            )

        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            if not self.model._gaussians_frozen:
                self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            if global_step < self.scene_steps:
                self.model.scheduler_step(global_step)

        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing_opt"):
                if global_step >= self.scene_steps:
                    # Controller only: optimizers[0] owns the frozen PPISP parameters.
                    active_optimizers, active_schedulers = self.post_processing_optimizers[1:], ()
                else:
                    active_optimizers, active_schedulers = self.post_processing_optimizers, self.post_processing_schedulers
                for opt in active_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in active_schedulers:
                    sched.step()

        with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
            scene_updated = self.strategy.post_optimizer_step(
                step=global_step, scene_extent=self.scene_extent, train_dataset=self.train_dataset,
                batch=gpu_batch, writer=self.tracking.writer,
            )

        # Last scene update: refresh the Mip filter from the final centres,
        # then freeze the scene and the PPISP per-frame parameters.
        if self._distillation_start_step >= 0 and global_step + 1 == self._distillation_start_step:
            if self.model._use_physical_mip3d_filter:
                refresh_filter(self.model, self.train_dataset)
            self.model.freeze_gaussians()
            self._freeze_ppisp_main()
            self.strategy.suspend()

        if self.model.progressive_training and check_step_condition(
            global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        if scene_updated or (conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0):
            with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                self.model.build_acc(rebuild=True)

        profilers["step_total"].end()

        global_step += 1
        self.global_step = global_step

        with torch.cuda.nvtx.range(f"train_{global_step - 1}_log_iter"):
            self.log_training_iter(batch_losses)
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_save_ckpt"):
            # The geometry checkpoint at the end of the scene steps is always saved, for meshing.
            if global_step in conf.checkpoint.iterations or global_step == self.scene_steps:
                self.save_checkpoint()

    @torch.cuda.nvtx.range("run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "step_total": CudaTimer(enabled=self.conf.enable_frame_timings),
        }
        for batch in self.train_dataloader:
            if self.global_step >= conf.n_iterations:
                return
            self.run_train_iter(self.global_step, batch, profilers, conf)

    @torch.cuda.nvtx.range("run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """PSNR on the held-out split (novel-view PPISP)."""
        logger.info(f"Step {self.global_step} -- Running validation..")
        psnrs = []
        for batch_idx in self.val_dataloader:
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)
            outputs = self.model(gpu_batch, train=False)
            outputs = apply_background(self.model.background, outputs, gpu_batch, training=False)
            if self.post_processing is not None:
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
            psnrs.append(self.criterions["psnr"](outputs["pred_features"], gpu_batch.rgb_gt).item())
        mean_psnr = float(np.mean(psnrs))
        self.tracking.writer.add_scalar("psnr/val", mean_psnr, self.global_step)
        logger.info(f"Validation PSNR at step {self.global_step}: {mean_psnr:.3f}")
        return {"psnr": mean_psnr}

    def run_training(self):
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")
        for _ in range(self.n_epochs):
            self.run_train_pass(conf)
        logger.end_progress(task_name="Training")

        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table("Training Statistics", record=table)

        self.on_training_end()
        logger.info("Training Complete.")
