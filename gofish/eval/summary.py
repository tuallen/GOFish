# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Dataset means of the per-scene evaluator outputs of one run, as printed by the dataset scripts.

    python -m gofish.eval.summary --fiord runs/fiord

Dataset means are arithmetic means of the per-scene full-precision values:
FIORD over 7 scenes for rendering and 6 for geometry, without Corridor_Out,
FullCircle over 9 scenes, ScanNet++ and Tanks & Temples over 6 scenes. Display: PSNR 2 decimals, the
other metrics 3 decimals (round half up). Missing files are reported, never filled.
"""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from gofish.data.fiord import GEOMETRY_SCENES, SCENES

FIORD = list(SCENES)
FIORD_GEOMETRY = list(GEOMETRY_SCENES)
SCANNETPP = ["0a5c013435", "8d563fc2cc", "bb87c292ad", "d415cc449b", "e8ea9b4da8", "fe1733741f"]
FULLCIRCLE = ["dark", "flat1", "flat2", "lab", "lounge", "persons", "room1", "room2", "room3"]
TNT = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]


def _load(path: Path, *keys):
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    for key in keys:
        value = value[key]
    return float(value)


def _mean(root: Path, scenes, relative, *keys, missing):
    values = []
    for scene in scenes:
        value = _load(root / scene / relative, *keys)
        if value is None:
            missing.append(str(root / scene / relative))
        else:
            values.append(value)
    return sum(values) / len(values) if values and len(values) == len(scenes) else None


def _fmt(value, digits):
    if value is None:
        return "-"
    return str(Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def render_triplet(root, scenes, relative, missing, keys=("PSNR", "SSIM", "LPIPS"), prefix=("mean",)):
    return [_fmt(_mean(root, scenes, relative, *prefix, k, missing=missing), 2 if k.lower() == "psnr" else 3) for k in keys]


def renderers(root: Path, scenes) -> str:
    """The renderer of the scored test renders, from each scene's heldout/render.json; the paper's tables use exact."""
    names = set()
    for scene in scenes:
        path = root / scene / "heldout" / "render.json"
        names.add(json.loads(path.read_text())["renderer"] if path.is_file() else "-")
    return "/".join(sorted(names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fiord", type=Path)
    parser.add_argument("--fullcircle", type=Path)
    parser.add_argument("--scannetpp", type=Path)
    parser.add_argument("--tnt", type=Path)
    parser.add_argument("--scenes", default=None, help="space-separated subset of the paper's scenes")
    args = parser.parse_args()
    global FIORD, FIORD_GEOMETRY, FULLCIRCLE, SCANNETPP, TNT
    if args.scenes:
        keep = set(args.scenes.split())
        FIORD, FIORD_GEOMETRY, FULLCIRCLE, SCANNETPP, TNT = (
            [s for s in group if s in keep] for group in (FIORD, FIORD_GEOMETRY, FULLCIRCLE, SCANNETPP, TNT))
    missing: list[str] = []
    rows = {}
    if args.fiord:
        rows["FIORD real-time FPS"] = [_fmt(_mean(args.fiord, FIORD, "fps.json", "mean_fps", missing=missing), 1)]
        rows["FIORD test renderer"] = [renderers(args.fiord, FIORD)]
        rows["FIORD Full (200 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.fiord, FIORD, "eval_full.json", missing)
        rows["FIORD Common (90 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.fiord, FIORD, "eval_common.json", missing)
        rows["FIORD Geometry P/R/F@10 cm"] = [
            _fmt(_mean(args.fiord, FIORD_GEOMETRY, "eval_mesh.json", "C", k, missing=missing), 3) for k in ("P10", "R10", "F10")]
    if args.fullcircle:
        rows["FullCircle real-time FPS"] = [_fmt(_mean(args.fullcircle, FULLCIRCLE, "fps.json", "mean_fps", missing=missing), 1)]
        rows["FullCircle test renderer"] = [renderers(args.fullcircle, FULLCIRCLE)]
        rows["FullCircle Full (205 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.fullcircle, FULLCIRCLE, "eval_full.json", missing)
        rows["FullCircle Common (90 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.fullcircle, FULLCIRCLE, "eval_common.json", missing)
    if args.scannetpp:
        rows["ScanNet++ real-time FPS"] = [_fmt(_mean(args.scannetpp, SCANNETPP, "fps.json", "mean_fps", missing=missing), 1)]
        rows["ScanNet++ test renderer"] = [renderers(args.scannetpp, SCANNETPP)]
        rows["ScanNet++ Full (127 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.scannetpp, SCANNETPP, "eval_full.json", missing)
        rows["ScanNet++ Common (90 deg) PSNR/SSIM/LPIPS"] = render_triplet(args.scannetpp, SCANNETPP, "eval_common.json", missing)
        rows["ScanNet++ Geometry P/R/F@5 cm"] = [
            _fmt(_mean(args.scannetpp, SCANNETPP, "eval_mesh.json", k, missing=missing), 3) for k in ("precision", "recall", "F")]
    if args.tnt:
        rows["Tanks & Temples real-time FPS"] = [_fmt(_mean(args.tnt, TNT, "fps.json", "mean_fps", missing=missing), 1)]
        rows["Tanks & Temples test renderer"] = [renderers(args.tnt, TNT)]
        rows["Tanks & Temples Pinhole PSNR/SSIM/LPIPS"] = render_triplet(args.tnt, TNT, "eval_render.json", missing, keys=("psnr", "ssim", "lpips"))
        rows["Tanks & Temples Geometry P/R/F"] = [
            _fmt(_mean(args.tnt, TNT, "eval_mesh/results.json", k, missing=missing), 3) for k in ("precision", "recall", "fscore")]
    width = max(len(k) for k in rows) if rows else 0
    for name, values in rows.items():
        print(f"{name:<{width}}  " + "  ".join(values))
    if missing:
        print(f"\n{len(missing)} missing evaluator outputs (their dataset means are '-'):")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
