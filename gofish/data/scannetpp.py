# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""ScanNet++: ``dslr/resized_images_<factor>`` from ``dslr/resized_images``.

Each image is converted to RGB, box-filtered to ``(W // factor, H // factor)``
and stored as JPEG at quality 95 without EXIF, keeping the file name.

    python -m gofish.data.scannetpp <scannetpp>/data/<scene>/dslr [--factor 2]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from PIL import Image


def prepare(dslr: Path, factor: int = 2) -> None:
    source = dslr / "resized_images"
    target = dslr / f"resized_images_{factor}"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    # Written to a temporary directory and renamed, so an interrupted run leaves no partial target.
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=dslr))
    try:
        for path in sorted(source.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                size = (image.width // factor, image.height // factor)
                image.resize(size, Image.Resampling.BOX).save(staging / path.name, "JPEG", quality=95)
        staging.chmod(0o755)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dslr", type=Path, help="ScanNet++ scene dslr/ directory")
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    prepare(args.dslr, args.factor)


if __name__ == "__main__":
    main()
