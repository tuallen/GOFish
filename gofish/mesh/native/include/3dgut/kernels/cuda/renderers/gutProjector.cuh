// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>
#include <3dgut/kernels/cuda/renderers/cameraNativeRayAdmission.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <tiny-cuda-nn/common_device.h>
#include <3dgut/renderer/renderParameters.h>


template <typename Particles, typename Params, typename UTParams>
struct GUTProjector : Params, UTParams {

    struct BoundingBox2D {
        tcnn::uvec2 min;
        tcnn::uvec2 max;
    };

    static inline __device__ BoundingBox2D computeTileSpaceBBox(const tcnn::uvec2& tileGrid, const tcnn::vec2& position, const tcnn::vec2& extent) {
        return BoundingBox2D{
            {
                min(tileGrid.x, max(0, static_cast<int>(floorf((position.x - 0.5f - extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(floorf((position.y - 0.5f - extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
            {
                min(tileGrid.x, max(0, static_cast<int>(ceilf((position.x - 0.5f + extent.x) / threedgut::GUTParameters::Tiling::BlockX)))),
                min(tileGrid.y, max(0, static_cast<int>(ceilf((position.y - 0.5f + extent.y) / threedgut::GUTParameters::Tiling::BlockY)))),
            },
        };
    }

    static inline __device__ float tileMinParticlePowerResponse(const tcnn::vec2& tileCoords,
                                                                const tcnn::vec4& conicOpacity,
                                                                const tcnn::vec2& meanPosition) {

        const tcnn::vec2 tileSize = tcnn::vec2(threedgut::GUTParameters::Tiling::BlockX, threedgut::GUTParameters::Tiling::BlockY);
        const tcnn::vec2 tileMin  = tileSize * tileCoords;
        const tcnn::vec2 tileMax  = tileSize + tileMin;

        const tcnn::vec2 minOffset  = tileMin - meanPosition;
        const tcnn::vec2 leftAbove  = tcnn::vec2(minOffset.x > 0.0f, minOffset.y > 0.0f);
        const tcnn::vec2 notInRange = tcnn::vec2(leftAbove.x + (meanPosition.x > tileMax.x),
                                                 leftAbove.y + (meanPosition.y > tileMax.y));

        if ((notInRange.x + notInRange.y) > 0.0f) {
            const tcnn::vec2 p    = tcnn::mix(tileMax, tileMin, leftAbove);
            const tcnn::vec2 dxy  = tcnn::copysign(tileSize, minOffset);
            const tcnn::vec2 diff = meanPosition - p;
            const tcnn::vec2 rcp  = tcnn::vec2(__frcp_rn(tileSize.x * tileSize.x * conicOpacity.x),
                                               __frcp_rn(tileSize.y * tileSize.y * conicOpacity.z));

            const float tx = notInRange.y * __saturatef((dxy.x * conicOpacity.x * diff.x + dxy.x * conicOpacity.y * diff.y) * rcp.x);
            const float ty = notInRange.x * __saturatef((dxy.y * conicOpacity.y * diff.x + dxy.y * conicOpacity.z * diff.y) * rcp.y);

            const tcnn::vec2 minPosDiff = meanPosition - tcnn::vec2(p.x + tx * dxy.x, p.y + ty * dxy.y);

            return 0.5f * (conicOpacity.x * minPosDiff.x * minPosDiff.x + conicOpacity.z * minPosDiff.y * minPosDiff.y) + conicOpacity.y * minPosDiff.x * minPosDiff.y;
        }
        // mean position is within the tile
        return 0.f;
    }

    /// Convert a projected particle to its conic/opacity representation
    static inline __device__ bool computeProjectedExtentConicOpacity(tcnn::vec3 covariance,
                                                                     float opacity,
                                                                     tcnn::vec2& extent,
                                                                     tcnn::vec4& conicOpacity,
                                                                     float& maxConicOpacityPower) {

        const tcnn::vec3 dilatedCovariance = tcnn::vec3{covariance.x + Params::CovarianceDilation, covariance.y, covariance.z + Params::CovarianceDilation};
        const float dilatedCovDet          = dilatedCovariance.x * dilatedCovariance.z - dilatedCovariance.y * dilatedCovariance.y;
        if (dilatedCovDet == 0.0f) {
            return false;
        }
        conicOpacity.slice<0, 3>() = tcnn::vec3{dilatedCovariance.z, -dilatedCovariance.y, dilatedCovariance.x} / dilatedCovDet;

        // see Yu et al. in "Mip-Splatting: Alias-free 3D Gaussian Splatting" https://github.com/autonomousvision/mip-splatting
        if constexpr (TGUTProjectorParams::MipSplattingScaling) {
            const float covDet            = covariance.x * covariance.z - covariance.y * covariance.y;
            const float convolutionFactor = sqrtf(fmaxf(0.000025f, covDet / dilatedCovDet));
            // GUT evaluates the final response in 3-D. The 2-D Mip factor is
            // therefore does not delete or shrink the broad-phase support;
            // covariance dilation and conic shape are kept as computed.
            conicOpacity.w                = opacity;
        } else {
            conicOpacity.w = opacity;
        }

        if (conicOpacity.w < Params::AlphaThreshold) {
            return false;
        }

        maxConicOpacityPower     = logf(conicOpacity.w / Params::AlphaThreshold);
        const float extentFactor = Params::TightOpacityBounding ? fminf(3.33f, sqrtf(2.0f * maxConicOpacityPower)) : 3.33f;
        const float minLambda    = 0.01f;
        const float mid          = 0.5f * (dilatedCovariance.x + dilatedCovariance.z);
        const float lambda       = mid + sqrtf(fmaxf(minLambda, mid * mid - dilatedCovDet));
        const float radius       = extentFactor * sqrtf(lambda);
        extent                   = Params::RectBounding ? min(extentFactor * sqrt(tcnn::vec2{dilatedCovariance.x, dilatedCovariance.z}), tcnn::vec2{radius}) : tcnn::vec2{radius};

        return radius > 0.f;
    }

    static inline __device__ bool unscentedParticleProjection(
        const tcnn::ivec2& resolution,
        const threedgut::TSensorModel& sensorModel,
        const tcnn::vec3& sensorWorldPosition,
        const tcnn::mat4x3& sensorMatrix,
        const threedgut::TSensorState& sensorShutterState,
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        tcnn::vec3& particleSensorRay,
        float& particleProjOpacity,
        tcnn::vec2& particleProjCenter,
        tcnn::vec3& particleProjCovariance,
        bool& supportCapEligible) {

        supportCapEligible = false;

        particleProjOpacity = particles.opacity(particleParameters);
        if (particleProjOpacity < Params::AlphaThreshold) {
            return false;
        }

        const tcnn::vec3& particleMean = particles.position(particleParameters);
        const bool cameraNativeNonlinear = threedgut::isNonlinearFisheye(sensorModel);
        if (cameraNativeNonlinear) {
            // A calibrated >90-degree ray can have negative camera z.  The
            // 3DGUT metric near-plane constant therefore bounds radial range.
            const tcnn::vec3 cameraNativeMeanRay = particleMean - sensorWorldPosition;
            if (!threedgut::cameraNativeRadialNearAdmitted(
                    cameraNativeMeanRay, Params::ParticleMinSensorZ)) {
                return false;
            }
        } else
        {
            // Pinhole predicate of the 3DGUT projector.
            if ((particleMean.x * sensorMatrix[0][2] + particleMean.y * sensorMatrix[1][2] +
                 particleMean.z * sensorMatrix[2][2] + sensorMatrix[3][2]) < Params::ParticleMinSensorZ) {
                return false;
            }
        }


        particleSensorRay = particleMean - sensorWorldPosition;

        int numValidPoints = 0;
        tcnn::vec2 projectedSigmaPoints[2 * UTParams::D + 1];

        constexpr float Lambda = UTParams::Alpha * UTParams::Alpha * (UTParams::D + UTParams::Kappa) - UTParams::D;

        const bool cameraNativeCenterProjection =
            threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                particleMean,
                resolution,
                sensorModel,
                sensorShutterState,
                UTParams::ImageMarginFactor,
                projectedSigmaPoints[0], false);
        if (cameraNativeNonlinear && !cameraNativeCenterProjection) {
            // Support-preserving camera-native admission: the mean must be
            // finite and mathematically projectable, but it need not lie in
            // a principal-point circle. UT sigma support and the
            // exact per-pixel physical mask determine raster visibility.
            return false;
        }
        if (cameraNativeCenterProjection) {
            numValidPoints++;
        }
        particleProjCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));

        constexpr float weightI = 1.f / (2.f * (UTParams::D + Lambda));
#pragma unroll
        for (int i = 0; i < UTParams::D; ++i) {
            // Packed B is row-major while tcnn's zero-copy view is B^T.
            // Gather the actual column B e_i used by the finite UT.
            const tcnn::vec3 delta = UTParams::Delta * particles.factorColumn(
                particleParameters, i);

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean + delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1], false)) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1];

            if (threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean - delta,
                    resolution,
                    sensorModel,
                    sensorShutterState,
                    UTParams::ImageMarginFactor,
                    projectedSigmaPoints[i + 1 + UTParams::D], false)) {
                numValidPoints++;
            }
            particleProjCenter += weightI * projectedSigmaPoints[i + 1 + UTParams::D];
        }

        // A fixed-weight partial UT is not a moment of the source Gaussian.
        // Invalid moments use the separate opacity-support angular fallback.
        if (numValidPoints != (2 * UTParams::D + 1)) {
            supportCapEligible = supportCapEligible || numValidPoints > 0;
            return false;
        }

        {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[0] - particleProjCenter;
            constexpr float weight0        = Lambda / (UTParams::D + Lambda) + (1.f - UTParams::Alpha * UTParams::Alpha + UTParams::Beta);
            particleProjCovariance         = weight0 * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                                  centeredPoint.x * centeredPoint.y,
                                                                  centeredPoint.y * centeredPoint.y);
        }
#pragma unroll
        for (int i = 0; i < 2 * UTParams::D; ++i) {
            const tcnn::vec2 centeredPoint = projectedSigmaPoints[i + 1] - particleProjCenter;
            particleProjCovariance += weightI * tcnn::vec3(centeredPoint.x * centeredPoint.x,
                                                           centeredPoint.x * centeredPoint.y,
                                                           centeredPoint.y * centeredPoint.y);
        }

        return true;
    }

    static inline __device__ bool nativeTileAdmissionBase(
        int x, int y, bool validProjection, const BoundingBox2D& tileBBox,
        const tcnn::vec4& conicOpacity, const tcnn::vec2& projectedCenter,
        float maxConicOpacityPower) {
        if (!validProjection || x < static_cast<int>(tileBBox.min.x)
            || x >= static_cast<int>(tileBBox.max.x)
            || y < static_cast<int>(tileBBox.min.y)
            || y >= static_cast<int>(tileBBox.max.y)) return false;
        if constexpr (Params::TileCulling) {
            return tileMinParticlePowerResponse(tcnn::vec2(x, y), conicOpacity, projectedCenter)
                < maxConicOpacityPower;
        }
        return true;
    }
};
