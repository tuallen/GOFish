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

from enum import IntEnum


class Features:
    """Feature layout of the renderer kernels.

    GOFish stores spherical-harmonic colour (``model.feature_type: sh``).  The
    integer values are passed to the kernels as preprocessor defines; the
    other 3DGRUT feature modes of the vendored kernels are not used.
    """

    class Type(IntEnum):
        SH = 0  # spherical harmonics

    def __init__(self, conf):
        if str(conf.model.feature_type).lower() != "sh":
            raise ValueError(f"unsupported model.feature_type {conf.model.feature_type!r}; GOFish uses 'sh'")
        degree, render_degree = conf.model.progressive_training.max_n_features, conf.render.particle_radiance_sph_degree
        if degree != render_degree:
            raise ValueError(f"model.progressive_training.max_n_features ({degree}) must equal "
                             f"render.particle_radiance_sph_degree ({render_degree})")
        self._conf = conf

    transform_type = Type.SH
    activation_type = 0  # no activation
    activation_num_frequencies = 1
    interpolation_type = 0  # particle centre
    interpolation_support = 0  # particle centre
    num_interpolation_points = 1
    interp_point_feature_dim = 3
    ray_feature_dim = 3  # RGB

    @property
    def particle_feature_dim(self):
        """SH coefficients per particle, 3 (degree + 1)^2."""
        sh_degree = self._conf.model.progressive_training.max_n_features
        return 3 * ((sh_degree + 1) ** 2)

    @property
    def feature_defines(self):
        """C preprocessor defines for all feature-related kernel parameters."""
        return [
            f"-DPARTICLE_FEATURE_DIM={self.particle_feature_dim}",
            f"-DRAY_FEATURE_DIM={self.ray_feature_dim}",
            f"-DFEATURE_TRANSFORM_TYPE={self.transform_type}",
            f"-DFEATURE_INTERPOLATION_TYPE={self.interpolation_type}",
            f"-DFEATURE_INTERPOLATION_SUPPORT={self.interpolation_support}",
            f"-DFEATURE_ACTIVATION_TYPE={self.activation_type}",
            f"-DFEATURE_ACTIVATION_NUM_FREQUENCIES={self.activation_num_frequencies}",
            f"-DINTERP_POINT_FEATURE_DIM={self.interp_point_feature_dim}",
        ]
