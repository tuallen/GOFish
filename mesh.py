#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Extract the GOFish mesh from a geometry checkpoint.

Example:
    python mesh.py --checkpoint runs/kitchen/ours_30000/ckpt_30000.pt --out runs/kitchen/mesh

Writes mesh.ply (reported mesh), mesh_sof.ply, mesh_raw.ply and mesh_info.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gofish.mesh import extract_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="geometry checkpoint (end of joint training, step 30000)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--data", "--dataset", dest="dataset", type=str, default=None,
                        help="dataset root, if it moved since training (default: the checkpoint's path)")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="also save the sites and the field values at the sites")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="overrides of the checkpoint's config, e.g. dataset.test_split_interval=8")
    args = parser.parse_args()
    summary = extract_mesh(args.checkpoint, args.out, args.dataset, args.save_intermediates, args.overrides)
    print(json.dumps({key: summary[key] for key in ("faces", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
