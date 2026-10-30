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

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import sys

import torch
from omegaconf import OmegaConf

from threedgrut.model.features import Features
from threedgrut.utils import jit
from threedgut_tracer.view_observation_filter import FILTER_VARIANCE_PIXELS, pixel_footprint_variance


def _slang_prelude_include() -> str:
    candidates = [Path(sys.prefix) / "include"]
    try:
        import slangtorch

        package = Path(slangtorch.__file__).resolve().parent
        candidates.extend((package / "include", package.parent / "include"))
    except ImportError:
        pass
    slangc = shutil.which("slangc")
    if slangc:
        candidates.append(Path(slangc).resolve().parent.parent / "include")
    for directory in candidates:
        if (directory / "slang-cuda-prelude.h").is_file():
            return str(directory)
    raise RuntimeError("slang-cuda-prelude.h was not found in the active toolchain")


# Loaded extensions of this process, by extension name.
_LOADED: dict = {}


def build_key(*parts) -> str:
    """Short hash of the build inputs that the torch extension versioner does not see."""
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else str(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:8]


def write_if_changed(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` unless it already holds it, so concurrent
    builds never read a half-written file and an unchanged file keeps its timestamp."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _select(conf, key, default):
    return OmegaConf.select(conf, key, default=default)


def load_full_shape_operator(conf):
    """JIT-build the full-shape factor operator C = Sigma + Q_v and its VJP."""
    from . import footprint_shape_operator

    gamma = pixel_footprint_variance(conf)
    name = "_gofish_footprint_shape_operator"
    if gamma != FILTER_VARIANCE_PIXELS:
        name += "_g" + hashlib.sha1(float(gamma).hex().encode()).hexdigest()[:8]
    prefix = os.path.dirname(__file__)
    define = f"-DGOFISH_PIXEL_FOOTPRINT_VARIANCE={gamma!r}"
    if name in _LOADED:
        footprint_shape_operator.bind_native(_LOADED[name])
        return _LOADED[name]
    module = jit.load(
        name=name,
        sources=[
            os.path.join(prefix, "src", "footprint_shape_operator.cpp"),
            os.path.join(prefix, "src", "footprint_shape_operator_cuda.cu"),
        ],
        extra_cflags=["-O3", define],
        extra_cuda_cflags=["-O3", "-lineinfo", define],
        extra_ldflags=["-lcusolver"],
        build_directory=torch.utils.cpp_extension._get_build_directory(name, verbose=True),
        with_cuda=True,
    )
    footprint_shape_operator.bind_native(module)
    _LOADED[name] = module
    return module


# ----------------------------------------------------------------------------
#
def setup_3dgut(conf, inference=False):
    """JIT-build the GOFish renderer extension.

    ``inference=True`` builds the forward-only extension of
    ``render.py --renderer exact-forward|realtime``: the forward pass omits the
    training-only geometric channels and hit counts; colour, opacity and depth
    code is shared with the training build.  Every build is named by a hash of
    its defines, so all variants coexist in one JIT cache.
    """
    if int(conf.render.particle_kernel_degree) != 2 or float(conf.render.particle_kernel_min_response) != 0.0113:
        # Complete admission certifies the radius-3 support of the degree-2
        # kernel with minimum response 0.0113 (r_ker = 2.994 < 3).
        raise RuntimeError("complete admission requires particle_kernel_degree=2 and min_response=0.0113")
    if not bool(_select(conf, "render.gofish.complete_admission", True)):
        raise NotImplementedError(
            "render.gofish.complete_admission=false: stock 3DGUT tile admission is not part of this renderer"
        )
    if bool(_select(conf, "render.splat.fine_grained_load_balancing", False)):
        raise RuntimeError("physical tile pruning requires ordinary 16x16 raster blocks")
    if not bool(_select(conf, "render.splat.per_tile_depth", True)):
        raise RuntimeError("complete admission emits per-tile depth keys")
    if not (bool(_select(conf, "render.splat.hierarchical_sort", True))
            and bool(_select(conf, "render.splat.sof_mid8_hierarchy", True))) and not inference:
        # The real-time renderer may replace it by the per-tile order (+ a per-pixel k-buffer).
        raise RuntimeError("GOFish renders with SOF's per-pixel hierarchical sort")
    exact_screen_absgrad = bool(_select(conf, "strategy.densify.generic_camera_absgrad", True))
    # Scheduled SOF cascade (sofMid8Schedule.inl), used with the image-space densification
    # statistic; the world-space statistic uses the per-pixel cascade.
    sof_schedule = (not inference and exact_screen_absgrad
                    and bool(_select(conf, "render.splat.sof_schedule", True)))

    load_full_shape_operator(conf)

    # CoMe confidence composite (alpha-detached sum_i w_i conf_i); off by default and
    # built as a separate extension. Training-only: inference builds omit the channel.
    come_confidence = bool(_select(conf, "loss.color_confidence", False)) and not inference
    extension_name = "lib3dgut_gofish" + ("" if exact_screen_absgrad else "_worldabsgrad")
    if exact_screen_absgrad and not sof_schedule and not inference:
        extension_name += "_noschedule"
    if come_confidence:
        extension_name += "_come"
    if inference:
        extension_name += "_inference"

    prefix = os.path.dirname(__file__)
    include_paths = [
        None,  # build directory, for the generated threedgutSlang.cuh
        os.path.join(prefix, "include"),
        os.path.join(prefix, "..", "third_party", "tiny-cuda-nn", "include"),
        os.path.join(prefix, "..", "third_party", "tiny-cuda-nn", "dependencies"),
        os.path.join(prefix, "..", "third_party", "tiny-cuda-nn", "dependencies", "fmt", "include"),
        _slang_prelude_include(),
    ]

    def to_cpp_bool(value):
        return "true" if value else "false"

    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha * ut_alpha * (ut_d + ut_kappa))

    feat = Features(conf)
    half_defines = [
        f"-DPARTICLE_FEATURE_HALF={1 if conf.render.particle_feature_half else 0}",
        f"-DFEATURE_OUTPUT_HALF={1 if conf.render.feature_output_half else 0}",
    ]

    defines = [
        # Generic-camera densification: exact per-hit screen-space AbsGrad
        # (1) or the per-hit world-space |dL/dmu| statistic (0).
        f"-DGOFISH_EXACT_SCREEN_ABSGRAD={1 if exact_screen_absgrad else 0}",
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={to_cpp_bool(conf.render.enable_normals)}",
        f"-DGAUSSIAN_PARTICLE_SURFEL={to_cpp_bool(conf.render.primitive_type=='trisurfel')}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={to_cpp_bool(conf.render.enable_hitcounts and not inference)}",
        # Feature-based radiance dimensions
        *feat.feature_defines,
        # Feature buffer memory layout
        *half_defines,
        # Specific to the 3DGUT renderer
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={conf.render.splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={conf.render.splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={to_cpp_bool(conf.render.splat.global_z_order)}",
        f"-DFINE_GRAINED_LOAD_BALANCING={to_cpp_bool(_select(conf, 'render.splat.fine_grained_load_balancing', False))}",
        # -- Unscented Transform --
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={conf.render.splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={to_cpp_bool(conf.render.splat.ut_require_all_sigma_points_valid)}",
        # -- Culling --
        f"-DGAUSSIAN_RECT_BOUNDING={to_cpp_bool(conf.render.splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={to_cpp_bool(conf.render.splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={to_cpp_bool(conf.render.splat.tile_based_culling)}",
        # -- SOF hierarchical sort: per-ray HEAD, per-tile MID/TAIL buffers keyed by per-tile depth --
        f"-DHIER_PER_TILE_DEPTH={to_cpp_bool(_select(conf, 'render.splat.per_tile_depth', True))}",
        f"-DGAUSSIAN_HIERARCHICAL_SORT={1 if _select(conf, 'render.splat.hierarchical_sort', True) else 0}",
        f"-DGAUSSIAN_HEAD_SIZE={_select(conf, 'render.splat.head_size', 4)}",
        f"-DGAUSSIAN_TAIL_SIZE={_select(conf, 'render.splat.tail_size', 64)}",
        f"-DGAUSSIAN_SOF_MID8_HIERARCHY={1 if _select(conf, 'render.splat.sof_mid8_hierarchy', True) else 0}",
        f"-DGAUSSIAN_MID_SIZE={_select(conf, 'render.splat.mid_size', 8)}",
        f"-DGOFISH_SOF_SCHEDULE={1 if sof_schedule else 0}",
        # CoMe confidence composite channel (alpha-detached sum_i w_i conf_i).
        f"-DGOFISH_COME_CONFIDENCE={1 if come_confidence else 0}",
    ]

    if inference:
        defines.append("-DGOFISH_INFERENCE_ONLY=1")
        fast_scale = _select(conf, "render.gofish.fast_admission_scale", None)
        if fast_scale is not None:
            # Real-time renderer: UT-projected footprint admission enlarged by s (approximate).
            fast_scale = float(fast_scale)
            if not (math.isfinite(fast_scale) and fast_scale >= 1.0):
                raise ValueError("render.gofish.fast_admission_scale must be >= 1")
            defines += ["-DGOFISH_FAST_UT_ADMISSION=1", f"-DGOFISH_FAST_ADMISSION_SCALE={fast_scale!r}f"]
            fallback = {"none": 0, "cone": 1, "cone_big": 5}[str(_select(conf, "render.gofish.fast_admission_fallback", "cone"))]
            defines.append(f"-DGOFISH_FAST_FALLBACK={fallback}")
            min_angle = float(_select(conf, "render.gofish.fast_admission_fallback_min_angle_deg", 0.0))
            if fallback == 5:
                defines.append(f"-DGOFISH_FAST_FALLBACK_MIN_ANGLE={math.radians(min_angle)!r}f")
            max_angle = float(_select(conf, "render.gofish.fast_admission_fallback_max_angle_deg", 90.0))
            defines.append(f"-DGOFISH_FAST_FALLBACK_MAX_ANGLE={math.radians(max_angle)!r}f")
            if bool(_select(conf, "render.gofish.fast_global_depth_key", False)):
                defines.append("-DGOFISH_FAST_GLOBAL_DEPTH_KEY=1")
            partial = _select(conf, "render.gofish.fast_partial_ut_scale", None)
            if partial is not None:
                defines += ["-DGOFISH_FAST_PARTIAL_UT=1", f"-DGOFISH_FAST_PARTIAL_UT_SCALE={float(partial)!r}f"]
        if bool(_select(conf, "render.gofish.fast_slim_prefetch", False)):
            defines.append("-DGOFISH_FAST_SLIM_PREFETCH=1")  # real-time renderer: exact slim tile prefetch
        split = _select(conf, "render.gofish.fast_tile_split", None)
        if split is not None and int(split) > 1:  # real-time renderer heavy-tile split (inference forward)
            defines += [f"-DGOFISH_FAST_TILE_SPLIT={int(split)}",
                        f"-DGOFISH_FAST_TILE_SPLIT_MIN={int(_select(conf, 'render.gofish.fast_tile_split_min', 2048))}"]

    cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        *defines,
    ]

    cuda_cflags = [
        "-DTCNN_MIN_GPU_ARCH=70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-use_fast_math",
        "-O3",
        *defines,
    ]
    # When PARTICLE_FEATURE_HALF=1 the Slang-generated header uses __half types;
    # the Slang prelude only pulls in <cuda_fp16.h> and defines __half when
    # SLANG_CUDA_ENABLE_HALF is set.
    if conf.render.particle_feature_half or conf.render.feature_output_half:
        cuda_cflags.append("-DSLANG_CUDA_ENABLE_HALF=1")

    # One extension per define set, so every training, inference and real-time variant
    # coexists in one JIT cache; the hash also covers the compiled-in calibration table.
    from threedgut_tracer.kb_table.table import HEADER_PATH

    extension_name += "_" + build_key(*cflags, *cuda_cflags, HEADER_PATH.read_bytes())
    if extension_name in _LOADED:
        return _LOADED[extension_name]
    build_dir = torch.utils.cpp_extension._get_build_directory(extension_name, verbose=True)
    include_paths[0] = build_dir

    # List of sources.
    source_files = [
        "src/splatRaster.cpp",
        "src/gutRenderer.cu",
        "src/cudaBuffer.cpp",
        "bindings.cpp",
    ]

    # Compile slang kernels into the private build directory.
    slang_build_inc_dir = os.path.join(prefix, "include", "3dgut")
    raw_slang = Path(build_dir) / f".threedgutSlang.{os.getpid()}.cuh"
    jit.compile_slang_kernel(
        kernel_files=[os.path.join(slang_build_inc_dir, "threedgut.slang")],
        output_file=str(raw_slang),
        include_paths=[os.path.join(prefix, "include")],
        defines=defines,
    )
    generated = raw_slang.read_text(encoding="utf-8")
    raw_slang.unlink()
    generated = re.sub(
        r'^#include\s+"[^"]*/slang-cuda-prelude\.h"$',
        "#include <slang-cuda-prelude.h>", generated, count=1, flags=re.MULTILINE)
    write_if_changed(Path(build_dir) / "threedgutSlang.cuh", generated)

    # Compile and load.
    source_paths = [os.path.join(prefix, fn) for fn in source_files]
    _LOADED[extension_name] = jit.load(
        name=extension_name,
        sources=source_paths,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_include_paths=include_paths,
        build_directory=build_dir,
    )
    return _LOADED[extension_name]
