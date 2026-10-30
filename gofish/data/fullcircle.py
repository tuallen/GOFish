# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare a FullCircle scene for GOFish.

    python -m gofish.data.fullcircle <FullCircle>/<scene> <out>/<scene>

The input is a released FullCircle scene: ``sparse/0`` with every registered
frame, ``images_4/camera{1,2}/`` at 720 x 720 and the capturer masks
``masks_4/masks-20_4/<frame>_mask.png`` (255 on the capturer).  The output
keeps the same COLMAP model and images, writes the supervision masks
``images_4/<frame>_mask.png`` (255 = supervised) of the training frames, and
lists the tripod test frames ``*_test.png`` in ``test_list.txt``, which
``configs/apps/fullcircle.yaml`` holds out.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from gofish.eval.common.colmap import read_images


def prepare(source: Path, target: Path) -> None:
    images = read_images(source / "sparse" / "0" / "images.bin")
    names = sorted(image.name for image in images)
    test = [name for name in names if Path(name).stem.endswith("_test")]
    if not test:
        raise SystemExit(f"{source}: no tripod test frames (*_test.png)")
    (target / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copyfile(source / "sparse" / "0" / name, target / "sparse" / "0" / name)
    for name in names:
        src, dst = source / "images_4" / name, target / "images_4" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src.resolve())
        if name in test:
            continue
        capturer = np.asarray(Image.open(source / "masks_4" / "masks-20_4" / f"{Path(name).with_suffix('')}_mask.png")
                              .convert("L")) > 127
        Image.fromarray(np.where(capturer, 0, 255).astype(np.uint8)).save(dst.with_name(f"{dst.stem}_mask.png"))
    (target / "test_list.txt").write_text("\n".join(test) + "\n")
    print(f"{target}: {len(names) - len(test)} training frames with masks, {len(test)} tripod test frames")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
