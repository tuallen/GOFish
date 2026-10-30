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

"""JIT build of the native opacity-field extension (``gofish/mesh/native``)."""

from __future__ import annotations

import math
from pathlib import Path

import torch

NATIVE_ROOT = Path(__file__).resolve().parent / "native"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Loaded field extensions of this process, by extension name.
_LOADED: dict = {}


def _bool(value: object) -> str:
    return "true" if bool(value) else "false"


def load_field_plugin(conf):
    """Compile (or load from the torch extension cache) the field extension."""
    from threedgrut.model.features import Features
    from threedgrut.utils import jit
    from threedgut_tracer.setup_3dgut import _slang_prelude_include, build_key

    if int(conf.render.particle_kernel_degree) != 2:
        raise ValueError("the opacity field requires Gaussian kernel degree 2")

    features = Features(conf)
    ut_d = 3
    ut_alpha = conf.render.splat.ut_alpha
    ut_beta = conf.render.splat.ut_beta
    ut_kappa = conf.render.splat.ut_kappa
    ut_delta = math.sqrt(ut_alpha * ut_alpha * (ut_d + ut_kappa))
    splat = conf.render.splat
    defines = [
        *features.feature_defines,
        f"-DPARTICLE_FEATURE_HALF={1 if conf.render.particle_feature_half else 0}",
        f"-DFEATURE_OUTPUT_HALF={1 if conf.render.feature_output_half else 0}",
        f"-DPARTICLE_RADIANCE_NUM_COEFFS={(conf.render.particle_radiance_sph_degree + 1) ** 2}",
        f"-DGAUSSIAN_PARTICLE_KERNEL_DEGREE={conf.render.particle_kernel_degree}",
        f"-DGAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY={conf.render.particle_kernel_min_response}",
        f"-DGAUSSIAN_PARTICLE_MIN_ALPHA={conf.render.particle_kernel_min_alpha}",
        f"-DGAUSSIAN_PARTICLE_MAX_ALPHA={conf.render.particle_kernel_max_alpha}",
        f"-DGAUSSIAN_PARTICLE_ENABLE_NORMAL={_bool(conf.render.enable_normals)}",
        f"-DGAUSSIAN_PARTICLE_SURFEL={_bool(conf.render.primitive_type == 'trisurfel')}",
        f"-DGAUSSIAN_MIN_TRANSMITTANCE_THRESHOLD={conf.render.min_transmittance}",
        f"-DGAUSSIAN_ENABLE_HIT_COUNT={_bool(conf.render.enable_hitcounts)}",
        f"-DGAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS={splat.n_rolling_shutter_iterations}",
        f"-DGAUSSIAN_K_BUFFER_SIZE={splat.k_buffer_size}",
        f"-DGAUSSIAN_GLOBAL_Z_ORDER={_bool(splat.global_z_order)}",
        f"-DFINE_GRAINED_LOAD_BALANCING={_bool(getattr(splat, 'fine_grained_load_balancing', False))}",
        f"-DGAUSSIAN_UT_ALPHA={ut_alpha}",
        f"-DGAUSSIAN_UT_BETA={ut_beta}",
        f"-DGAUSSIAN_UT_KAPPA={ut_kappa}",
        f"-DGAUSSIAN_UT_DELTA={ut_delta}",
        f"-DGAUSSIAN_UT_IN_IMAGE_MARGIN_FACTOR={splat.ut_in_image_margin_factor}",
        f"-DGAUSSIAN_UT_REQUIRE_ALL_SIGMA_POINTS_VALID={_bool(splat.ut_require_all_sigma_points_valid)}",
        f"-DGAUSSIAN_RECT_BOUNDING={_bool(splat.rect_bounding)}",
        f"-DGAUSSIAN_TIGHT_OPACITY_BOUNDING={_bool(splat.tight_opacity_bounding)}",
        f"-DGAUSSIAN_TILE_BASED_CULLING={_bool(splat.tile_based_culling)}",
        f"-DHIER_PER_TILE_DEPTH={_bool(getattr(splat, 'per_tile_depth', True))}",
        f"-DGAUSSIAN_HIERARCHICAL_SORT={1 if getattr(splat, 'hierarchical_sort', True) else 0}",
        f"-DGAUSSIAN_HEAD_SIZE={getattr(splat, 'head_size', 4)}",
        f"-DGAUSSIAN_TAIL_SIZE={getattr(splat, 'tail_size', 64)}",
        f"-DGAUSSIAN_SOF_MID8_HIERARCHY={1 if getattr(splat, 'sof_mid8_hierarchy', True) else 0}",
        f"-DGAUSSIAN_MID_SIZE={getattr(splat, 'mid_size', 8)}",
    ]
    tcnn = REPOSITORY_ROOT / "third_party" / "tiny-cuda-nn"
    include_paths = [
        str(NATIVE_ROOT / "include"),
        str(tcnn / "include"),
        str(tcnn / "dependencies"),
        str(tcnn / "dependencies" / "fmt" / "include"),
        _slang_prelude_include(),
    ]
    # Host C++ is built with -O2 under ISO C++17 (no fast-math, no FMA
    # contraction), so host float results equal an unoptimized build.
    cflags = ["-O2", "-DTCNN_MIN_GPU_ARCH=70", *defines]
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
    if conf.render.particle_feature_half or conf.render.feature_output_half:
        cuda_cflags.append("-DSLANG_CUDA_ENABLE_HALF=1")
    sources = [
        NATIVE_ROOT / "src" / "splatRaster.cpp",
        NATIVE_ROOT / "src" / "gutRenderer.cu",
        NATIVE_ROOT / "src" / "cudaBuffer.cpp",
        NATIVE_ROOT / "bindings.cpp",
    ]
    # One extension per define set, so differently configured checkpoints coexist in one JIT cache.
    extension_name = "gofish_opacity_field_" + build_key(*cflags, *cuda_cflags)
    if extension_name not in _LOADED:
        _LOADED[extension_name] = jit.load(
            name=extension_name,
            sources=[str(path) for path in sources],
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_include_paths=include_paths,
            build_directory=torch.utils.cpp_extension._get_build_directory(extension_name, verbose=True),
        )
    return _LOADED[extension_name]
