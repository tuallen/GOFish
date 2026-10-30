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

import argparse

from threedgrut.render_inference import make_renderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the held-out views of a checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=str, help="rendering checkpoint (final step)")
    parser.add_argument("--data", "--path", dest="path", type=str, default="",
                        help="dataset root, if different from the checkpoint")
    parser.add_argument("--out", "--out-dir", dest="out_dir", required=True, type=str, help="output directory")
    parser.add_argument("--no-gt", action="store_true", help="do not write the masked ground-truth images")
    parser.add_argument("--metrics", action="store_true", help="write Full-aperture PSNR/SSIM/LPIPS to metrics.json")
    parser.add_argument("--renderer", choices=("realtime", "exact", "exact-forward"), default="realtime",
                        help="realtime, the default = the real-time renderer; exact = the training renderer, "
                             "used for the paper's metrics; exact-forward = its forward-only path, same pixels")
    parser.add_argument("--realtime-options", type=str, default=None,
                        help="real-time renderer options as JSON or @FILE (default: the paper's renderer, "
                             "see threedgrut/render_realtime.py)")
    parser.add_argument("--wa-order", choices=("sequential", "product"), default=None,
                        help="render.wa_order of the checkpoint's config, if given: float association of the "
                             "wide-angle attenuation")
    parser.add_argument("--png-rounding", choices=("half_up", "half_even"), default=None,
                        help="render.png_rounding of the checkpoint's config, if given: 8-bit PNG quantization")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="overrides of the checkpoint's config, e.g. dataset.test_split_interval=8")
    args = parser.parse_args()
    overrides = list(args.overrides)
    if args.wa_order:
        overrides.append(f"render.wa_order={args.wa_order}")
    if args.png_rounding:
        overrides.append(f"render.png_rounding={args.png_rounding}")

    renderer = make_renderer(
        args.renderer, args.checkpoint, args.out_dir, realtime_options=args.realtime_options, path=args.path,
        save_gt=not args.no_gt, metrics=args.metrics, overrides=overrides)
    renderer.render_all()
