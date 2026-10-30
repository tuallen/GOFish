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

import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import threedgrut.datasets as datasets
from threedgrut.datasets.utils import configure_dataloader_for_platform
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import apply_config_overrides
from threedgrut.utils.render import apply_background, apply_post_processing
from threedgut_tracer.kb_table.table import check_dataset as check_calibration_table


PNG_ROUNDING = ("half_up", "half_even")


def _to_uint8(image: np.ndarray, rounding: str = "half_up") -> np.ndarray:
    """8-bit PNG quantization of a [0, 1] float32 image.

    ``half_up`` is torchvision's ``save_image`` (float32 ``x * 255 + 0.5``,
    truncated), the upstream 3DGRUT convention; ``half_even`` is
    ``numpy.rint(x * 255)``. They differ only where ``x * 255`` is exactly a
    half-integer.
    """
    value = np.clip(image, 0.0, 1.0).astype(np.float32) * np.float32(255.0)
    if rounding == "half_up":
        return np.clip(value + np.float32(0.5), 0.0, 255.0).astype(np.uint8)
    if rounding == "half_even":
        return np.rint(value).astype(np.uint8)
    raise ValueError(f"unknown PNG rounding {rounding!r}; choose from {PNG_ROUNDING}")


def load_post_processing(checkpoint, conf):
    """PPISP of a checkpoint in novel-view mode (with its controller), or None."""
    post_processing = None
    if "post_processing" in checkpoint and conf.post_processing.method == "ppisp":
        from gofish.appearance.wide_angle import ppisp_config_from
        from gofish_ppisp import PPISP

        ppisp_config, _ = ppisp_config_from(conf)
        post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
        post_processing = post_processing.to("cuda").eval()
        post_processing.wa_order = str(OmegaConf.select(conf, "render.wa_order", default="sequential"))

    return post_processing


class Renderer:
    """Render the held-out split of a checkpoint over the calibrated camera domain.

    For every held-out image ``<stem>`` it writes ``pred/<stem>.png`` (pixels
    outside the camera domain set to zero), ``gt/<stem>.png`` (same masking)
    and ``mask/<stem>.png`` (255 inside the domain).  When two cameras share
    image stems, as on FullCircle, the files are named ``<camera>__<stem>``.
    PPISP runs in novel-view mode with its controller.
    """

    def __init__(self, model, conf, global_step, out_dir, post_processing=None, save_gt=True,
                 metrics=False) -> None:
        png_rounding = str(OmegaConf.select(conf, "render.png_rounding", default="half_up"))
        if png_rounding not in PNG_ROUNDING:
            raise ValueError(f"unknown PNG rounding {png_rounding!r}; choose from {PNG_ROUNDING}")
        self.png_rounding = png_rounding
        self.model = model
        self.metrics = metrics
        self.conf = conf
        self.global_step = global_step
        self.out_dir = Path(out_dir)
        self.post_processing = post_processing
        self.save_gt = save_gt
        self.dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        check_calibration_table(self.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            **configure_dataloader_for_platform({"num_workers": 8, "batch_size": 1, "shuffle": False, "collate_fn": None}),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, path="", save_gt=True, metrics=False, overrides=None):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        conf = apply_config_overrides(checkpoint["config"], overrides)
        if path:
            conf.path = path

        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.build_acc()

        post_processing = load_post_processing(checkpoint, conf)
        model.eval()
        return Renderer(model, conf, checkpoint["global_step"], out_dir, post_processing, save_gt, metrics)

    def render_frame(self, gpu_batch):
        """One held-out frame: render, composite the background, apply PPISP."""
        outputs = self.model(gpu_batch, train=False)
        outputs = apply_background(self.model.background, outputs, gpu_batch, training=False)
        if self.post_processing is not None:
            outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
        return outputs

    renderer_name = "exact"  # set by make_renderer; recorded with the renders

    @torch.inference_mode()
    def render_all(self):
        for name in ("pred", "gt", "mask"):
            os.makedirs(self.out_dir / name, exist_ok=True)
        (self.out_dir / "render.json").write_text(json.dumps({
            "renderer": self.renderer_name, "png_rounding": self.png_rounding,
            "wa_order": str(OmegaConf.select(self.conf, "render.wa_order", default="sequential"))}, indent=1) + "\n")
        if self.metrics:
            # Full-aperture metrics on the float renders.
            from gofish.eval.common.metrics import aperture_metrics, make_lpips, mean_metrics

            lpips = make_lpips(device="cuda")
            frames = []
        paths = [Path(str(path)) for path in self.dataset.image_paths]
        stems = [path.stem for path in paths]
        if len(set(stems)) != len(stems):
            stems = [f"{path.parent.name}__{path.stem}" for path in paths]
        for iteration, batch in enumerate(self.dataloader):
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = self.render_frame(gpu_batch)

            prediction = outputs["pred_features"].clamp(0.0, 1.0)[0].cpu().numpy()
            target = gpu_batch.rgb_gt.clamp(0.0, 1.0)[0].cpu().numpy()
            if gpu_batch.physical_fov_mask is not None:
                domain = gpu_batch.physical_fov_mask[0, ..., 0].cpu().numpy()
            else:
                domain = np.ones(prediction.shape[:2], dtype=bool)
            stem = stems[iteration]
            if self.metrics:
                region = torch.from_numpy(domain).to("cuda")
                frame = aperture_metrics(
                    outputs["pred_features"].clamp(0.0, 1.0).permute(0, 3, 1, 2),
                    gpu_batch.rgb_gt.permute(0, 3, 1, 2), region, lpips)
                if frame is not None:
                    frames.append({"image": stem, **frame})
            Image.fromarray(_to_uint8(np.where(domain[..., None], prediction, 0.0), self.png_rounding), mode="RGB").save(
                self.out_dir / "pred" / f"{stem}.png")
            if self.save_gt:
                Image.fromarray(_to_uint8(np.where(domain[..., None], target, 0.0), self.png_rounding), mode="RGB").save(
                    self.out_dir / "gt" / f"{stem}.png")
            Image.fromarray((domain * 255).astype(np.uint8), mode="L").save(self.out_dir / "mask" / f"{stem}.png")
        if self.metrics:
            with open(self.out_dir / "metrics.json", "w") as f:
                json.dump({"renderer": self.renderer_name, "mean": mean_metrics(frames), "frames": frames}, f, indent=2)
        logger.info(f"Rendered {len(self.dataset)} held-out views to {self.out_dir}")
