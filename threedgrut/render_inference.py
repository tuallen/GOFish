# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Renderers selected by ``render.py --renderer``.

``exact``: the training renderer (threedgrut.render.Renderer); the paper's
    evaluation renders.
``exact-forward``: the exact forward-only path
    (threedgut_tracer.inference.InferenceTracer); pixels, opacity and depth are
    bit-identical to ``exact`` without the training-only channels.
``realtime``: the real-time renderer (threedgrut.render_realtime),
    built on the forward-only tracer.

All reuse the reference writer (PNG quantization, domain masking, metrics).
"""

from __future__ import annotations

import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.render import Renderer, load_post_processing
from threedgrut.utils.misc import apply_config_overrides
from threedgrut.utils.render import apply_background, apply_post_processing


class InferenceRenderer(Renderer):
    """Renderer whose per-frame forward is the exact inference-only tracer."""

    def __init__(self, model, tracer, conf, global_step, out_dir, post_processing=None, **kwargs):
        super().__init__(model, conf, global_step, out_dir, post_processing, **kwargs)
        self.tracer = tracer

    @classmethod
    def load(cls, checkpoint_path, overrides=None, path="", tracer_kwargs=None):
        """``(model, tracer, conf, global_step, post_processing)`` of a checkpoint."""
        from threedgut_tracer.inference import InferenceTracer

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        conf = apply_config_overrides(checkpoint["config"], overrides)
        if path:
            conf.path = path
        model = MixtureOfGaussians(conf, build_renderer=False)
        model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.eval()
        post_processing = load_post_processing(checkpoint, conf)
        tracer = InferenceTracer(conf, model, **(tracer_kwargs or {}))
        return model, tracer, conf, checkpoint["global_step"], post_processing

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, path="", save_gt=True, metrics=False, overrides=None):
        model, tracer, conf, step, post_processing = cls.load(checkpoint_path, overrides, path)
        return cls(model, tracer, conf, step, out_dir, post_processing, save_gt=save_gt, metrics=metrics)

    def render_frame(self, gpu_batch):
        outputs = self.tracer.render(gpu_batch)
        outputs = apply_background(self.model.background, outputs, gpu_batch, training=False)
        if self.post_processing is not None:
            outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
        return outputs


def make_renderer(renderer, checkpoint_path, out_dir, realtime_options=None, **kwargs):
    """Renderer for ``render.py --renderer {exact,exact-forward,realtime}``."""
    if realtime_options and renderer != "realtime":
        raise ValueError("--realtime-options require --renderer realtime")
    if renderer == "exact":
        result = Renderer.from_checkpoint(checkpoint_path, out_dir, **kwargs)
    elif renderer == "exact-forward":
        result = InferenceRenderer.from_checkpoint(checkpoint_path, out_dir, **kwargs)
    elif renderer == "realtime":
        from threedgrut.render_realtime import RealtimeRenderer

        result = RealtimeRenderer.from_checkpoint(checkpoint_path, out_dir, options=realtime_options, **kwargs)
    else:
        raise ValueError(f"unknown renderer {renderer!r}")
    result.renderer_name = renderer
    return result
