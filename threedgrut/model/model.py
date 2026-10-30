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

import os

import numpy as np
import torch

import threedgrut.model.background as background
import threedgut_tracer
from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import read_colmap_points3D_text, read_next_bytes
from threedgrut.model.features import Features
from threedgrut.model.geometry import k_nearest_neighbors, nearest_neighbor_dist_cpuKD
from threedgrut.physical_mip import configured as physical_mip_configured
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    get_activation_function,
    get_scheduler,
    sh_degree_to_specular_dim,
    to_np,
    to_torch,
)
from threedgrut.utils.render import RGB2SH


class MixtureOfGaussians(torch.nn.Module):
    """Mixture of 3D Gaussians with spherical-harmonics radiance."""

    @property
    def num_gaussians(self):
        return self.positions.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_max_n_features(self) -> int:
        return self.max_n_features

    def get_n_active_features(self) -> int:
        return self.n_active_features

    def get_features_albedo(self) -> torch.Tensor:
        return self.features_albedo

    def get_features_specular(self) -> torch.Tensor:
        return self.features_specular

    def get_features(self):
        return torch.cat((self.features_albedo, self.features_specular), dim=1)

    def get_scale(self, preactivation=False):
        if preactivation:
            return self.scale
        return self.scale_activation(self.scale)

    def get_rotation(self, preactivation=False):
        if preactivation:
            return self.rotation
        return self.rotation_activation(self.rotation)

    def get_density(self, preactivation=False):
        if preactivation:
            return self.density
        return self.density_activation(self.density)

    def get_scale_with_3D_filter(self):
        """Scale with the 3D Mip filter added in quadrature, ``sqrt(s^2 + f^2)``.

        The filter is a detached function of the Gaussian centres; opacity is
        not compensated here (the renderer applies the ray-maximum correction).
        """
        s = self.get_scale()
        if not self._use_physical_mip3d_filter:
            return s
        if self.filter_3D is None or tuple(self.filter_3D.shape) != (self.num_gaussians, 1):
            raise RuntimeError("Mip filter is not current with the Gaussian topology")
        scalar = self.filter_3D.detach()
        return torch.sqrt(s * s + scalar * scalar)



    def get_model_parameters(self) -> dict:
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"

        model_params = {
            "positions": self.positions,
            "rotation": self.rotation,
            "scale": self.scale,
            "density": self.density,
            "features_albedo": self.features_albedo,
            "features_specular": self.features_specular,
            "background": self.background.state_dict(),
            "n_active_features": self.n_active_features,
            "max_n_features": self.max_n_features,
            "progressive_training": self.progressive_training,
            "scene_extent": self.scene_extent,
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf,
            "feature_type": "sh",
            "particle_feature_dim": self.particle_feature_dim,
            "ray_feature_dim": self.ray_feature_dim,
            # Intrinsic Mip filter (None when the factorization is disabled).
            "filter_3D": None if self.filter_3D is None else self.filter_3D.detach(),
            "filter_dir": None if self.filter_dir is None else self.filter_dir.detach(),
        }

        if self.progressive_training:
            model_params["feature_dim_increase_interval"] = self.feature_dim_increase_interval
            model_params["feature_dim_increase_step"] = self.feature_dim_increase_step

        if isinstance(getattr(self, "confidence", None), torch.nn.Parameter):
            model_params["confidence"] = self.confidence  # CoMe training-only state
        return model_params

    def __init__(self, conf, scene_extent=None, build_renderer=True):
        super().__init__()

        self.conf = conf
        self.scene_extent = scene_extent

        sh_degree = conf.model.progressive_training.max_n_features
        render_sph_degree = conf.render.particle_radiance_sph_degree
        if sh_degree != render_sph_degree:
            # The particle feature layout (3 (degree + 1)^2) and the kernels' coefficient count must agree.
            raise ValueError(f"model.progressive_training.max_n_features ({sh_degree}) must equal "
                             f"render.particle_radiance_sph_degree ({render_sph_degree})")
        specular_dim = sh_degree_to_specular_dim(sh_degree)
        self.positions = torch.nn.Parameter(torch.empty([0, 3]))  # [n_gaussians, 3]
        self.rotation = torch.nn.Parameter(torch.empty([0, 4]))  # unit quaternion [n_gaussians, 4]
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))  # anisotropic scale [n_gaussians, 3]
        self.density = torch.nn.Parameter(torch.empty([0, 1]))  # [n_gaussians, 1]

        Features(self.conf)  # only spherical-harmonics features are supported
        self.features_albedo = torch.nn.Parameter(torch.empty([0, 3]))  # 0th-order SH [n_gaussians, 3]
        self.features_specular = torch.nn.Parameter(torch.empty([0, specular_dim]))  # higher-order SH
        self.particle_feature_dim = 3 + specular_dim
        self.ray_feature_dim = 3

        self.max_sh_degree = sh_degree

        # 3D filter of Calibrated Mip Filtering (threedgrut.physical_mip).
        self._use_physical_mip3d_filter = physical_mip_configured(conf)
        self.filter_3D = None
        self.filter_dir = None

        self.device = "cuda"
        self.optimizer = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.density_activation_inv = get_activation_function(self.conf.model.density_activation, inverse=True)
        self.scale_activation = get_activation_function(self.conf.model.scale_activation)
        self.scale_activation_inv = get_activation_function(self.conf.model.scale_activation, inverse=True)
        self.rotation_activation = get_activation_function("normalize")  # The default value of the dim parameter is 1

        self.background = background.make(self.conf.model.background.name, self.conf.model.background)

        self.n_active_features = min(self.conf.model.progressive_training.init_n_features, sh_degree)
        self.max_n_features = sh_degree
        self.progressive_training = False
        if self.n_active_features < self.max_n_features:
            self.feature_dim_increase_interval = self.conf.model.progressive_training.increase_frequency
            self.feature_dim_increase_step = self.conf.model.progressive_training.increase_step
            self.progressive_training = True

        if conf.render.method != "3dgut":
            raise ValueError(f"Unknown rendering method: {conf.render.method}")
        # The forward-only renderers (render.py --renderer exact-forward|realtime) bring their own
        # forward-only extension and skip building the training renderer.
        self.renderer = threedgut_tracer.Tracer(conf) if build_renderer else None

        self._gaussians_frozen = False

    @torch.no_grad()
    def build_acc(self, rebuild=True):
        self.renderer.build_acc(self, rebuild)

    def freeze_gaussians(self) -> None:
        """Freeze all Gaussian parameters for the controller-only PPISP phase."""
        if self._gaussians_frozen:
            return

        self.positions.requires_grad = False
        self.rotation.requires_grad = False
        self.scale.requires_grad = False
        self.density.requires_grad = False
        self.features_albedo.requires_grad = False
        self.features_specular.requires_grad = False
        if isinstance(getattr(self, "confidence", None), torch.nn.Parameter):
            self.confidence.requires_grad = False  # CoMe confidence is scene state: frozen with the scene

        self._gaussians_frozen = True
        logger.info("Gaussian parameters frozen")

    def validate_fields(self):
        num_gaussians = self.num_gaussians
        assert self.positions.shape == (num_gaussians, 3)
        assert self.density.shape == (num_gaussians, 1)
        assert self.rotation.shape == (num_gaussians, 4)
        assert self.scale.shape == (num_gaussians, 3)
        assert self.features_albedo.shape == (num_gaussians, 3)
        assert self.features_specular.shape == (num_gaussians, sh_degree_to_specular_dim(self.max_n_features))

    def init_from_colmap(self, root_path: str, observer_pts):
        if self.conf.dataset.type == "scannetpp_fisheye":
            points_file = os.path.join(root_path, "colmap", "points3D.txt")
            pts, rgb, _ = read_colmap_points3D_text(points_file)
            file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)

        else:
            points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
            # also handle nonbinary points files
            if not os.path.isfile(points_file):
                points_file = os.path.join(root_path, "sparse/0", "points3D.txt")
                pts, rgb, _ = read_colmap_points3D_text(points_file)
                file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)
            else:
                with open(points_file, "rb") as file:
                    n_pts = read_next_bytes(file, 8, "Q")[0]
                    logger.info(f"Found {n_pts} colmap points")

                    file_pts = np.zeros((n_pts, 3), dtype=np.float32)
                    file_rgb = np.zeros((n_pts, 3), dtype=np.float32)

                    for i_pt in range(n_pts):
                        pt_data = read_next_bytes(file, 43, "QdddBBBd")
                        file_pts[i_pt, :] = np.array(pt_data[1:4])
                        file_rgb[i_pt, :] = np.array(pt_data[4:7])

                        # skip the track data
                        t_len = read_next_bytes(file, num_bytes=8, format_char_sequence="Q")[0]
                        read_next_bytes(file, num_bytes=8 * t_len, format_char_sequence="ii" * t_len)

                file_pts = torch.tensor(file_pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(file_rgb, dtype=torch.uint8, device=self.device)

        assert file_rgb.dtype == torch.uint8, "Expecting RGB values to be in [0, 255] range"
        self.default_initialize_from_points(
            file_pts,
            observer_pts,
            file_rgb,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features_albedo = checkpoint["features_albedo"]
        self.features_specular = checkpoint["features_specular"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]
        self.particle_feature_dim = checkpoint.get("particle_feature_dim", self.particle_feature_dim)
        self.ray_feature_dim = checkpoint.get("ray_feature_dim", self.ray_feature_dim)

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint["feature_dim_increase_interval"]
            self.feature_dim_increase_step = checkpoint["feature_dim_increase_step"]

        if "confidence" in checkpoint:
            self.confidence = checkpoint["confidence"]  # CoMe training-only state
        self.background.load_state_dict(checkpoint["background"])
        filter_3d, filter_dir = load_checkpoint_filter(checkpoint)
        if self._use_physical_mip3d_filter:
            if filter_3d is None:
                raise RuntimeError("checkpoint lacks the Mip filter required by its configuration")
            dtype = self.get_positions().dtype
            self.filter_3D = filter_3d.detach().to(device=self.device, dtype=dtype).contiguous()
            self.filter_dir = filter_dir.detach().to(device=self.device, dtype=dtype).contiguous()
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint["optimizer"])
        self.validate_fields()

    def default_initialize_from_points(self, pts, observer_pts, colors=None, use_observer_pts=True):
        """Initialize Gaussians at ``pts`` (Nx3) with optional uint8 colors (Nx3)."""

        dtype = torch.float32

        # Local generator for deterministic random initialization (does not affect global RNG)
        rng = torch.Generator(device=self.device).manual_seed(self.conf.seed_initialization)

        dist_to_observers = None
        if use_observer_pts:
            # Scales from the distance to the nearest camera; points that coincide
            # with a camera centre are dropped.
            dist_to_observers = nearest_neighbor_dist_cpuKD(pts, observer_pts)
            camera_coincident_floor = 1e-7
            keep = dist_to_observers > camera_coincident_floor
            removed = int((~keep).sum().item())
            if removed:
                logger.info(f"Rejected {removed}/{int(keep.numel())} points coincident with a camera centre")
            if not bool(keep.any().item()):
                raise RuntimeError("all initial points coincide with camera centres")
            if removed:
                pts = pts[keep]
                dist_to_observers = dist_to_observers[keep]
                if colors is not None:
                    colors = colors[keep]
            dist_to_observers = torch.clamp_min(dist_to_observers, camera_coincident_floor)

        N = pts.shape[0]
        positions = pts

        # Random rotations
        rots = torch.rand((N, 4), dtype=dtype, device=self.device, generator=rng)

        if use_observer_pts:
            observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        else:
            # Initialize the GS size to be the average dist of the 3 nearest neighbors
            dist2_avg = (k_nearest_neighbors(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
            observation_scale = torch.sqrt(dist2_avg)

        observation_scale = observation_scale * self.conf.model.default_scale_factor

        scales = self.scale_activation_inv(observation_scale)[:, None].repeat(1, 3)

        # set density as a constant
        opacities = self.density_activation_inv(
            torch.full((N, 1), fill_value=self.conf.model.default_density, dtype=dtype, device=self.device)
        )

        # set colors, random if they weren't given
        if colors is None:
            colors = torch.randint(0, 256, (N, 3), dtype=torch.uint8, device=self.device, generator=rng)

        features_albedo = to_torch(RGB2SH(to_np(colors.float() / 255.0)), device=self.device)
        num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims))

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()

    def setup_optimizer(self, state_dict=None):
        # CoMe per-Gaussian confidence gamma (init 0 -> conf 1, as in CoMe's gaussian_model.py).
        from omegaconf import OmegaConf as _OCc
        if bool(_OCc.select(self.conf, "loss.color_confidence", default=False)):
            _g = getattr(self, "confidence", None)
            if not isinstance(_g, torch.nn.Parameter) or tuple(_g.shape) != tuple(self.density.shape):
                self.confidence = torch.nn.Parameter(torch.zeros_like(self.density.detach()), requires_grad=True)
        params = []
        for name, args in self.conf.optimizer.params.items():
            if not hasattr(self, name):
                continue
            module = getattr(self, name)
            if isinstance(module, torch.nn.Parameter) and module.requires_grad:
                params.append({"params": [module], "name": name, **args})

        if self.conf.optimizer.type != "adam":
            raise ValueError(f"Unknown optimizer type: {self.conf.optimizer.type}")
        self.optimizer = torch.optim.Adam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps, fused=True)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "positions":
                param_group["lr"] *= self.scene_extent  # Multiply the position lr by the scene scale

        self.setup_scheduler()

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)

    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if not hasattr(self, name):
                continue
            attr = getattr(self, name)
            if not (hasattr(attr, "requires_grad") and attr.requires_grad):
                continue
            if args.type is None:
                continue
            if name == "positions":
                self.schedulers[name] = get_scheduler(args.type)(
                    lr_init=args.lr_init * self.scene_extent,
                    lr_final=args.lr_final * self.scene_extent,
                    max_steps=args.max_steps,
                )
            else:
                self.schedulers[name] = get_scheduler(args.type)(**args)

    def scheduler_step(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.schedulers:
                lr = self.schedulers[param_group["name"]](step)
                if lr is not None:
                    param_group["lr"] = lr

    def set_optimizable_parameters(self):
        if not self.conf.model.optimize_density:
            self.density.requires_grad = False
        if not self.conf.model.optimize_rotation:
            self.rotation.requires_grad = False
        if not self.conf.model.optimize_scale:
            self.scale.requires_grad = False
        if not self.conf.model.optimize_position:
            self.positions.requires_grad = False
        if not self.conf.model.optimize_features_albedo:
            self.features_albedo.requires_grad = False
        if not self.conf.model.optimize_features_specular:
            self.features_specular.requires_grad = False


    def increase_num_active_features(self) -> None:
        self.n_active_features = min(self.max_n_features, self.n_active_features + self.feature_dim_increase_step)


    def forward(self, batch: Batch, train=False, frame_id=0) -> dict[str, torch.Tensor]:
        """Render ``batch``; returns the renderer output dictionary."""
        return self.renderer.render(self, batch, train, frame_id)


def load_checkpoint_filter(checkpoint: dict):
    """``(filter_3D, filter_dir)`` of a checkpoint (top level or under ``filter_ownership``), or ``(None, None)``."""
    if checkpoint.get("filter_3D") is not None:
        return checkpoint["filter_3D"], checkpoint["filter_dir"]
    owned = checkpoint.get("filter_ownership")
    if isinstance(owned, dict) and owned.get("filter_3D") is not None:
        return owned["filter_3D"], owned["filter_dir"]
    return None, None
