# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Read evaluation cameras from the GOFish dataset loader.

The calibrated aperture, per-pixel rays and poses used by every evaluator are
the ones the GOFish loader produces for a trained scene; nothing is
re-derived.  The scene configuration is taken from a GOFish checkpoint.  Only
camera tensors of each batch are read; images are never decoded here.

Writes, for one split:

* ``masks/<camera_id>.png``: calibrated aperture of each COLMAP camera
  (all-true when the camera has no explicit physical domain);
* ``frames/{index:05d}.npz``: ``R`` (camera-to-world rotation), ``C``,
  ``W``, ``H``, camera model and intrinsics, ``rays`` (camera-space
  per-pixel rays, float32 [H, W, 3]) and ``mask`` (float32 [H, W]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def plain_config(checkpoint: dict, data_root: Path, overrides=None):
    """Scene configuration stored in a checkpoint, pointed at ``data_root``.

    ``overrides`` are ``key=value`` dot-list entries merged into the config
    (as in render.py / mesh.py), e.g. ``dataset.test_split_interval=8``.
    """
    from omegaconf import OmegaConf

    from threedgrut.utils.misc import apply_config_overrides

    conf = OmegaConf.create(OmegaConf.to_container(checkpoint["config"], resolve=False))
    conf = apply_config_overrides(conf, overrides)
    conf.path = str(Path(data_root).resolve())
    return conf


def iterate_batches(conf, split: str = "test"):
    """Yield ``(index, dataset, gpu_batch)`` in loader order (no shuffling, no ray jitter)."""
    import torch
    import threedgrut.datasets as datasets

    if split == "test":
        dataset = datasets.make_test(name=conf.dataset.type, config=conf)
    else:
        dataset, _ = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for index, batch in enumerate(loader):
        yield index, dataset, dataset.get_gpu_batch_with_intrinsics(batch)


def camera_record(gpu) -> dict:
    """Pose, intrinsics, rays and aperture of one batch as numpy arrays."""
    import torch

    pose = gpu.T_to_world[0].double().cpu().numpy()
    height, width = int(gpu.rays_dir.shape[1]), int(gpu.rays_dir.shape[2])
    record = {"R": pose[:3, :3], "C": pose[:3, 3], "W": width, "H": height,
              "rays": gpu.rays_dir[0].float().cpu().numpy()}
    fisheye = getattr(gpu, "intrinsics_OpenCVFisheyeCameraModelParameters", None)
    pinhole = getattr(gpu, "intrinsics_OpenCVPinholeCameraModelParameters", None)
    if fisheye is not None:
        f = torch.as_tensor(fisheye["focal_length"]).double().reshape(-1).cpu().numpy()
        c = torch.as_tensor(fisheye["principal_point"]).double().reshape(-1).cpu().numpy()
        k = torch.as_tensor(fisheye["radial_coeffs"]).double().reshape(-1).cpu().numpy()[:4]
        record.update(model="fisheye", fx=f[0], fy=f[1], cx=c[0], cy=c[1], k=k,
                      max_angle=float(fisheye["max_angle"]))
    else:
        f = torch.as_tensor(pinhole["focal_length"]).double().reshape(-1).cpu().numpy()
        c = torch.as_tensor(pinhole["principal_point"]).double().reshape(-1).cpu().numpy()
        record.update(model="pinhole", fx=f[0], fy=f[-1], cx=c[0], cy=c[1])
    mask = getattr(gpu, "physical_fov_mask", None)
    record["mask"] = (np.ones((height, width), np.float32) if mask is None
                      else mask[0, ..., 0].float().cpu().numpy())
    record["camera_id"] = int(getattr(gpu, "camera_intrinsic_id", -1))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="GOFish checkpoint (configuration only)")
    parser.add_argument("--data", type=Path, required=True, help="scene data root")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-frames", action="store_true", help="write the aperture masks only")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="overrides of the checkpoint's config, e.g. dataset.test_split_interval=8")
    args = parser.parse_args()

    import torch
    from PIL import Image

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    conf = plain_config(checkpoint, args.data, args.overrides)
    del checkpoint
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    if not args.no_frames:
        (args.out / "frames").mkdir(parents=True, exist_ok=True)
    written = set()
    with torch.inference_mode():
        for index, dataset, gpu in iterate_batches(conf, args.split):
            record = camera_record(gpu)
            if record["camera_id"] not in written:
                mask = (record["mask"] > 0).astype(np.uint8) * 255
                Image.fromarray(mask, mode="L").save(args.out / "masks" / f"{record['camera_id']}.png")
                written.add(record["camera_id"])
            if not args.no_frames:
                np.savez(args.out / "frames" / f"{index:05d}.npz",
                         name=str(Path(dataset.image_paths[index]).name), **record)


if __name__ == "__main__":
    main()
