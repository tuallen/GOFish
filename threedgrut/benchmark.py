# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Render-only frame rate of a checkpoint, the protocol of the paper's efficiency table.

The held-out batches are loaded once; a frame is render + background + PPISP,
clamped to [0, 1] and masked to the camera domain; ``warmup`` frames, then
``frames`` timed frames cycling the held-out cameras, with
``torch.cuda.synchronize()`` after every frame and no image writes.

    python -m threedgrut.benchmark --checkpoint ckpt_35000.pt --out fps.json

The real-time renderer runs with the paper's options, ``DEFAULT_OPTIONS`` of
``threedgrut.render_realtime``.
"""

from __future__ import annotations

import statistics
import tempfile
import time

import torch

from threedgrut.render_inference import make_renderer


def benchmark_fps(checkpoint, renderer="realtime", realtime_options=None, warmup=20, frames=200, overrides=()) -> dict:
    """Mean and median frame time of ``renderer`` on the held-out views of ``checkpoint``."""
    with tempfile.TemporaryDirectory() as scratch:
        renderer_ = make_renderer(renderer, checkpoint, scratch, realtime_options=realtime_options,
                                  save_gt=False, metrics=False, overrides=list(overrides))
        with torch.inference_mode():
            batches = [renderer_.dataset.get_gpu_batch_with_intrinsics(b) for b in renderer_.dataloader]

            def frame(index):
                batch = batches[index % len(batches)]
                image = renderer_.render_frame(batch)["pred_features"].clamp(0.0, 1.0)
                mask = getattr(batch, "physical_fov_mask", None)
                return image if mask is None else torch.where(mask, image, torch.zeros_like(image))

            for index in range(warmup):
                frame(index)
            torch.cuda.synchronize()
            times = []
            for k in range(frames):
                start = time.perf_counter()
                frame(warmup + k)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
    mean = statistics.fmean(times)
    return {"checkpoint": str(checkpoint), "renderer": renderer, "overrides": list(overrides),
            "gpu": torch.cuda.get_device_name(0), "heldout_cameras": len(batches),
            "mean_ms": mean, "median_ms": statistics.median(times), "mean_fps": 1000.0 / mean,
            "per_frame_ms": times}


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--renderer", choices=("exact", "exact-forward", "realtime"), default="realtime")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args()
    record = benchmark_fps(args.checkpoint, renderer=args.renderer, overrides=args.overrides)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{args.renderer}: {record['mean_fps']:.1f} FPS")


if __name__ == "__main__":
    main()
