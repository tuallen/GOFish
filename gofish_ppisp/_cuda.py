# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""JIT-build the vendored PPISP CUDA extension.

The extension is always compiled from the sources next to this file, under a
name keyed by their content. A separately installed ``ppisp_cuda`` (e.g. from
the upstream ``ppisp`` package) may expose a different ABI and is never used.
"""

import hashlib
from pathlib import Path

_SOURCES = ("src/ppisp_impl.cu", "ext.cpp", "bindings.h", "src/ppisp_constants.h", "src/ppisp_math.cuh",
            "src/ppisp_math_bwd.cuh")


def load():
    from torch.utils.cpp_extension import load as load_extension

    source = Path(__file__).resolve().parent
    identity = hashlib.sha256(b"".join((source / name).read_bytes() for name in _SOURCES)).hexdigest()[:16]
    return load_extension(
        name=f"ppisp_cuda_{identity}",
        sources=[str(source / "src/ppisp_impl.cu"), str(source / "ext.cpp")],
        extra_include_paths=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "--generate-line-info", "--source-in-ptx"],
        with_cuda=True,
    )
