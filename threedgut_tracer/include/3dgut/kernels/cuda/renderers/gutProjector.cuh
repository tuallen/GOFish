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
#include <3dgut/kernels/cuda/renderers/conservativeOpacityBound.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderOneStream.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderProjectorCallbacks.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>
#include <3dgut/kernels/cuda/renderers/flatConeAdapter.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderProvider.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedSupportCap.cuh>
#include <3dgut/kernels/cuda/renderers/admissionPredicates.h>
#include <3dgut/kernels/cuda/renderers/sofDepthFallback.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/renderer/renderParameters.h>

#include <3dgut/kernels/cuda/renderers/sofMaximumResponseDepth.cuh>

// Real-time renderer approximate admission: the UT-projected footprint (3DGUT's
// rectangle / opacity-adaptive conic test, here enlarged by the scale s, with
// the conic power threshold s^2) replaces complete certified admission.
// Inference builds only.
#ifndef GOFISH_FAST_UT_ADMISSION
#define GOFISH_FAST_UT_ADMISSION 0
#endif
#ifndef GOFISH_FAST_ADMISSION_SCALE
#define GOFISH_FAST_ADMISSION_SCALE 1.0f
#endif
// UT-invalid particles: 0 = dropped as in 3DGUT, 1 = "cone", a conservative bounding-sphere /
// tile-cone test, 5 = "cone_big", the cone test restricted to particles whose bounding sphere does
// not contain the camera and whose bounding cone has a half-angle in
// [GOFISH_FAST_FALLBACK_MIN_ANGLE, GOFISH_FAST_FALLBACK_MAX_ANGLE] (radians): large on-screen
// UT-invalid particles near the camera, not the many small ones beyond a wide-angle rim.
#ifndef GOFISH_FAST_FALLBACK
#define GOFISH_FAST_FALLBACK 1
#endif
// Real-time renderer partial UT (inference builds with the UT admission): a particle whose projected mean is valid
// but some sigma points are not (wide-angle rim) gets a footprint from its valid sigma points, each invalid
// point replaced by the reflection of its valid mirror point about the projected mean; the covariance is
// enlarged by GOFISH_FAST_PARTIAL_UT_SCALE^2.  Both points of an axis invalid -> no footprint.
#ifndef GOFISH_FAST_PARTIAL_UT
#define GOFISH_FAST_PARTIAL_UT 0
#endif
#ifndef GOFISH_FAST_PARTIAL_UT_SCALE
#define GOFISH_FAST_PARTIAL_UT_SCALE 1.5f
#endif
#ifndef GOFISH_FAST_FALLBACK_MIN_ANGLE
#define GOFISH_FAST_FALLBACK_MIN_ANGLE 0.0f
#endif
#ifndef GOFISH_FAST_FALLBACK_MAX_ANGLE
#define GOFISH_FAST_FALLBACK_MAX_ANGLE 1.5707964f
#endif
// Real-time renderer: per-tile sort key = the particle's camera radial distance
// (3DGUT-style global key) instead of SOF's per-tile maximum-response depth.
#ifndef GOFISH_FAST_GLOBAL_DEPTH_KEY
#define GOFISH_FAST_GLOBAL_DEPTH_KEY 0
#endif
template <typename Particles, typename Params, typename UTParams>
struct GUTProjector : Params, UTParams {

    using TFeaturesVec = typename Particles::TFeaturesVec;
    using TParticles = Particles;

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

#if GOFISH_FAST_UT_ADMISSION
    // Fallback for particles without a valid UT footprint (sigma points past
    // the calibrated domain, wide-angle rim, near the camera): a conservative
    // angular test of the bounding sphere (radius 3 ||B||_F, which encloses
    // the kernel support r < 3) against each tile's leaf ray cone.  A sphere
    // that contains the camera centre admits every retained tile.
    template <typename Visit, typename Skip>
    static inline __device__ uint32_t fastFallbackTiles(const tcnn::uvec2& tileGrid,
                                                        const gofish_physical_candidate::PhysicalTileView& physicalTiles,
                                                        const gofish_conservative_bound::ConservativeConeNode* __restrict__ coneTree,
                                                        uint32_t coneSide,
                                                        uint32_t coneLevels,
                                                        const tcnn::vec3& cameraCenter,
                                                        const tcnn::vec3& particleCenter,
                                                        const tcnn::mat3& factorTranspose,
                                                        float opacityScale,
                                                        Visit&& visit,
                                                        Skip&& skip) {
        // Bounding radius of the opacity support: sigma_max(B) times the canonical
        // radius where alpha = min(0.99, rho exp(-r^2/2)) falls to the minimum alpha
        // (capped by the kernel cutoff).  sigma_max^2 is the largest eigenvalue of
        // M = B B^T, from the closed-form symmetric 3x3 solution, inflated by 1e-3.
        const tcnn::mat3& bt = factorTranspose;  // columns are the rows of B
        float m[6];  // M = B B^T: m00 m01 m02 m11 m12 m22, B row i = (bt[0][i], bt[1][i], bt[2][i])
        {
            const tcnn::vec3 r0{bt[0][0], bt[1][0], bt[2][0]}, r1{bt[0][1], bt[1][1], bt[2][1]}, r2{bt[0][2], bt[1][2], bt[2][2]};
            m[0] = dot(r0, r0); m[1] = dot(r0, r1); m[2] = dot(r0, r2); m[3] = dot(r1, r1); m[4] = dot(r1, r2); m[5] = dot(r2, r2);
        }
        const float q = (m[0] + m[3] + m[5]) / 3.0f;
        const float p1 = m[1] * m[1] + m[2] * m[2] + m[4] * m[4];
        const float p2 = (m[0] - q) * (m[0] - q) + (m[3] - q) * (m[3] - q) + (m[5] - q) * (m[5] - q) + 2.0f * p1;
        float lambdaMax = q;
        if (p2 > 0.0f) {
            const float pp = sqrtf(p2 / 6.0f);
            const float b00 = (m[0] - q) / pp, b01 = m[1] / pp, b02 = m[2] / pp, b11 = (m[3] - q) / pp, b12 = m[4] / pp, b22 = (m[5] - q) / pp;
            const float detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
            const float phi = acosf(fmaxf(-1.0f, fminf(1.0f, 0.5f * detB))) / 3.0f;
            lambdaMax = q + 2.0f * pp * cosf(phi);
        }
        lambdaMax = fmaxf(lambdaMax, fmaxf(m[0], fmaxf(m[3], m[5]))) * 1.001f;
        const float rho = fminf(0.99f, opacityScale);
        const float canonical = fminf(2.994f, sqrtf(fmaxf(0.0f, 2.0f * logf(rho / Params::AlphaThreshold))));
        const float radius = canonical * sqrtf(lambdaMax) * 1.001f;
        const tcnn::vec3 offset = particleCenter - cameraCenter;
        const float distance = length(offset);
        if (!(isfinite(radius) && isfinite(distance))) return 0u;
        uint32_t n = 0u;
        if (GOFISH_FAST_FALLBACK == 0 || (GOFISH_FAST_FALLBACK == 5 && distance <= radius)) return 0u;
        if (distance <= radius) {  // the bounding sphere contains the camera centre
            for (uint32_t y = 0; y < tileGrid.y; ++y)
                for (uint32_t x = 0; x < tileGrid.x; ++x)
                    if (physicalTiles.retainXY(x, y) && !skip(x, y)) { visit(x, y, n); ++n; }
            return n;
        }
        const tcnn::vec3 axis = offset / distance;
        const float alpha = asinf(fminf(1.0f, radius / distance)) + 1.0e-4f;
        if (GOFISH_FAST_FALLBACK == 5 && (alpha > float(GOFISH_FAST_FALLBACK_MAX_ANGLE) || alpha < float(GOFISH_FAST_FALLBACK_MIN_ANGLE))) return 0u;
        // Depth-first traversal of the calibrated tile cone quadtree (root = level coneLevels).
        uint32_t stack[64];
        int top = 0;
        stack[top++] = coneLevels << 24;
        while (top > 0) {
            const uint32_t item = stack[--top];
            const uint32_t level = item >> 24, x = (item >> 12) & 0xfffu, y = item & 0xfffu;
            const uint32_t side = coneSide >> level;
            const float4 cone = coneTree[gofish_conservative_bound::levelOffset(coneSide, level) + y * side + x].angularCone;
            if (!(cone.w >= 0.0f)) continue;
            const float cosine = fmaxf(-1.0f, fminf(1.0f, cone.x * axis.x + cone.y * axis.y + cone.z * axis.z));
            if (acosf(cosine) > cone.w + alpha) continue;
            if (level == 0u) {
                if (x < tileGrid.x && y < tileGrid.y && physicalTiles.retainXY(x, y) && !skip(x, y)) { visit(x, y, n); ++n; }
                continue;
            }
            if (top + 4 > 64) continue;  // cannot happen for the certified stack depth (3 levels + 1 per level)
            for (int c = 3; c >= 0; --c)
                stack[top++] = ((level - 1u) << 24) | ((2u * x + (c & 1)) << 12) | (2u * y + (c >> 1));
        }
        return n;
    }

    static inline __device__ bool fastUtValid(const tcnn::vec4& conicOpacity, const tcnn::vec2& extent) {
        constexpr float eps = 1e-06f;
        return extent.x > eps && conicOpacity.w >= Params::AlphaThreshold;
    }

    // Visits the admitted tiles of one particle (y-major) from its stored
    // projection; the count pass and the emit pass share it exactly.
    template <typename Visit, typename Skip>
    static inline __device__ uint32_t fastAdmittedTiles(const tcnn::uvec2& tileGrid,
                                                        const gofish_physical_candidate::PhysicalTileView& physicalTiles,
                                                        const tcnn::vec2& center,
                                                        const tcnn::vec4& conicOpacity,
                                                        const tcnn::vec2& extent,
                                                        Visit&& visit,
                                                        Skip&& skip) {
        if (!fastUtValid(conicOpacity, extent)) return 0u;
        constexpr float s = float(GOFISH_FAST_ADMISSION_SCALE);
        const BoundingBox2D box = computeTileSpaceBBox(tileGrid, center, extent * s);
        const float maxPower = logf(conicOpacity.w / Params::AlphaThreshold) * (s * s);
        uint32_t n = 0u;
        for (uint32_t y = box.min.y; y < box.max.y; ++y) {
            for (uint32_t x = box.min.x; x < box.max.x; ++x) {
                if (!physicalTiles.retainXY(x, y)) continue;
                if (skip(x, y)) continue;
                if constexpr (Params::TileCulling) {
                    if (!(tileMinParticlePowerResponse(tcnn::vec2(x, y), conicOpacity, center) < maxPower)) continue;
                }
                visit(x, y, n);
                ++n;
            }
        }
        return n;
    }
#endif

    static inline __device__ uint64_t concatTileDepthKeys(uint32_t tileKey, uint32_t depthKey) {
        return (static_cast<uint64_t>(tileKey) << 32) | depthKey;
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
                                                                     float& radius,
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
            // therefore not allowed to delete/shrink the broad-phase support.
            // Covariance dilation and conic shape remain unchanged.
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
        radius                   = extentFactor * sqrtf(lambda);
        extent                   = Params::RectBounding ? min(extentFactor * sqrt(tcnn::vec2{dilatedCovariance.x, dilatedCovariance.z}), tcnn::vec2{radius}) : tcnn::vec2{radius};

        return radius > 0.f;
    }

    // A sufficient, never-necessary proof for the inherited 20-pixel
    // screen-radius rule when the native UT conic is invalid.  The six probes
    // are strictly inside the renderer's opacity support along the factor
    // axes.  Two calibrated projections more than 40 pixels apart prove that
    // every enclosing image-space radius exceeds 20 pixels.  Fewer than two
    // valid projections, non-finite arithmetic, or a smaller diameter are
    // Unknown and therefore return false (retain).
    static inline __device__ uint8_t supportProjectionProvenGt20(
        const tcnn::ivec2& resolution,
        const threedgut::TSensorModel& sensorModel,
        const threedgut::TSensorState& sensorShutterState,
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters) {
        constexpr float kInheritedRadiusPixels = 20.0f;
        constexpr float kDiameterSquared =
            (2.0f * kInheritedRadiusPixels) *
            (2.0f * kInheritedRadiusPixels);
        const float opacity = particles.opacity(particleParameters);
        if (!(opacity > Params::AlphaThreshold) || !isfinite(opacity)) {
            return 0u;
        }
        const float alphaRadius = sqrtf(fmaxf(
            0.0f, 2.0f * logf(opacity / Params::AlphaThreshold)));
        // The ray kernel has the inherited three-sigma density cutoff.  Move
        // one float inward so every witness is in the open accepted support.
        const float support = nextafterf(fminf(3.0f, alphaRadius), 0.0f);
        if (!(support > 0.0f) || !isfinite(support)) return 0u;

        const tcnn::vec3 mean = particles.position(particleParameters);
        tcnn::vec2 projected[2 * UTParams::D];
        uint8_t valid[2 * UTParams::D]{};
#pragma unroll
        for (int axis = 0; axis < UTParams::D; ++axis) {
            const tcnn::vec3 delta = support * particles.factorColumn(
                particleParameters, axis);
            const int positive = axis;
            const int negative = axis + UTParams::D;
            valid[positive] = threedgut::projectPointWithShutter<
                UTParams::NRollingShutterIterations>(
                    mean + delta, resolution, sensorModel,
                    sensorShutterState, UTParams::ImageMarginFactor,
                    projected[positive], false)
                && isfinite(projected[positive].x)
                && isfinite(projected[positive].y);
            valid[negative] = threedgut::projectPointWithShutter<
                UTParams::NRollingShutterIterations>(
                    mean - delta, resolution, sensorModel,
                    sensorShutterState, UTParams::ImageMarginFactor,
                    projected[negative], false)
                && isfinite(projected[negative].x)
                && isfinite(projected[negative].y);
        }
#pragma unroll
        for (int first = 0; first < 2 * UTParams::D; ++first) {
            if (!valid[first]) continue;
#pragma unroll
            for (int second = first + 1; second < 2 * UTParams::D; ++second) {
                if (!valid[second]) continue;
                const tcnn::vec2 delta = projected[first] - projected[second];
                const float distanceSquared = dot(delta, delta);
                if (isfinite(distanceSquared)
                    && distanceSquared > kDiameterSquared) {
                    return 1u;
                }
            }
        }
        return 0u;
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
        // Candidate completeness is a property of the finite 3-D support,
        // not of the native conic.  Establish cap eligibility before either
        // radial-near or center-projection admission can reject that conic.
        // The exact densityHit narrow phase remains the final acceptance test.
        // The calibrated ray-tree proof is camera-model agnostic. Centrality
        // is established from the actual origin LUT later; a noncentral table
        // disables pruning and therefore retains every tile.
        supportCapEligible =
            sensorModel.shutterType == threedgut::TSensorModel::GlobalShutter;
        if (cameraNativeNonlinear) {
            // A calibrated >90-degree ray can have negative camera z.  The
            // inherited metric near constant therefore bounds radial range.
            const tcnn::vec3 cameraNativeMeanRay = particleMean - sensorWorldPosition;
            if (!threedgut::cameraNativeRadialNearAdmitted(
                    cameraNativeMeanRay, Params::ParticleMinSensorZ)) {
                return false;
            }
        } else
        {
            // Pinhole near-plane predicate.
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
            // an invented principal-point circle. UT sigma support and the
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
        float gofishCovarianceScale = 1.0f;
        if (numValidPoints != (2 * UTParams::D + 1)) {
#if GOFISH_FAST_UT_ADMISSION && GOFISH_FAST_PARTIAL_UT
            bool repaired = cameraNativeCenterProjection;
            bool pointValid[2 * UTParams::D];
#pragma unroll
            for (int i = 0; i < UTParams::D; ++i) {
                const tcnn::vec3 delta = UTParams::Delta * particles.factorColumn(particleParameters, i);
                tcnn::vec2 q;
                pointValid[i] = threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean + delta, resolution, sensorModel, sensorShutterState, UTParams::ImageMarginFactor, q, false);
                pointValid[i + UTParams::D] = threedgut::projectPointWithShutter<UTParams::NRollingShutterIterations>(
                    particleMean - delta, resolution, sensorModel, sensorShutterState, UTParams::ImageMarginFactor, q, false);
                if (!pointValid[i] && !pointValid[i + UTParams::D]) repaired = false;
            }
            if (repaired) {
#pragma unroll
                for (int i = 0; i < UTParams::D; ++i) {
                    if (!pointValid[i]) projectedSigmaPoints[i + 1] = 2.0f * projectedSigmaPoints[0] - projectedSigmaPoints[i + 1 + UTParams::D];
                    if (!pointValid[i + UTParams::D]) projectedSigmaPoints[i + 1 + UTParams::D] = 2.0f * projectedSigmaPoints[0] - projectedSigmaPoints[i + 1];
                }
                particleProjCenter = projectedSigmaPoints[0] * (Lambda / (UTParams::D + Lambda));
#pragma unroll
                for (int i = 0; i < 2 * UTParams::D; ++i) particleProjCenter += weightI * projectedSigmaPoints[i + 1];
                gofishCovarianceScale = float(GOFISH_FAST_PARTIAL_UT_SCALE) * float(GOFISH_FAST_PARTIAL_UT_SCALE);
            } else
#endif
            {
            supportCapEligible = supportCapEligible || numValidPoints > 0;
            return false;
            }
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
        if (gofishCovarianceScale != 1.0f) particleProjCovariance *= gofishCovarianceScale;

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

    static inline __device__ bool supportAngularCap(
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        const tcnn::vec3& sensorWorldPosition,
        float3& center,
        float& halfAngle) {
        const float opacity = particles.opacity(particleParameters);
        if (!(opacity >= Params::AlphaThreshold)) return false;
        const tcnn::vec3 relative = particles.position(particleParameters) - sensorWorldPosition;
        const float distance = length(relative);
        if (!(distance > 0.0f) || !isfinite(distance)) return false;
        const tcnn::vec3 n = relative / distance;
        center = make_float3(n.x, n.y, n.z);
        float traceSigma = 0.0f;
        float traceSigma2 = 0.0f;
        float qParallel = 0.0f;
        float qSigma2 = 0.0f;
        const tcnn::vec3 btN = particles.factorTransposeTimes(particleParameters, n);
        const tcnn::vec3 sigmaN = particles.factorTimes(particleParameters, btN);
        qParallel = fmaxf(0.0f, dot(n, sigmaN));
        qSigma2 = dot(sigmaN, sigmaN);
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            const tcnn::vec3 factorColumn = particles.factorColumn(
                particleParameters, axis);
            traceSigma += dot(factorColumn, factorColumn);
            const tcnn::vec3 basis = axis == 0
                ? tcnn::vec3{1.0f, 0.0f, 0.0f}
                : (axis == 1 ? tcnn::vec3{0.0f, 1.0f, 0.0f}
                             : tcnn::vec3{0.0f, 0.0f, 1.0f});
            const tcnn::vec3 sigmaColumn = particles.factorTimes(
                particleParameters,
                particles.factorTransposeTimes(particleParameters, basis));
            traceSigma2 += dot(sigmaColumn, sigmaColumn);
        }
        qParallel = fmaxf(0.0f, qParallel);
        const float transverseTrace = fmaxf(0.0f, traceSigma - qParallel);
        const float transverseSquareTrace = fmaxf(
            0.0f, traceSigma2 - 2.0f * qSigma2 + qParallel * qParallel);
        const float discriminant = fmaxf(
            0.0f, 2.0f * transverseSquareTrace - transverseTrace * transverseTrace);
        const float lambdaPerpendicular = 0.5f * (transverseTrace + sqrtf(discriminant));
        const float support = sqrtf(fmaxf(0.0f, 2.0f * logf(opacity / Params::AlphaThreshold)));
        const float hParallel = support * sqrtf(qParallel);
        const float hPerpendicular = support * sqrtf(fmaxf(0.0f, lambdaPerpendicular));
        const float axialLower = distance - hParallel;
        halfAngle = axialLower > 0.0f
            ? nextafterf(atan2f(hPerpendicular, axialLower), gofish_conservative_bound::kFloatMax)
            : gofish_conservative_bound::kPi;
        return true;
    }

    static inline __device__ gofish_whitened_cone::FiniteEllipsoidConeCache
    finiteEllipsoidConeCache(
        const Particles& particles,
        const typename Particles::DensityParameters& particleParameters,
        const tcnn::vec3& sensorWorldPosition,
        bool centralGlobalShutter) {
        float inverseFactor[9];
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            inverseFactor[3 * row + 0] = particleParameters.inverseFactor[row].x;
            inverseFactor[3 * row + 1] = particleParameters.inverseFactor[row].y;
            inverseFactor[3 * row + 2] = particleParameters.inverseFactor[row].z;
        }
        const tcnn::vec3 originMinusMean =
            sensorWorldPosition - particles.position(particleParameters);
        return gofish_whitened_cone::makeFiniteEllipsoidConeCache(
            inverseFactor,
            make_float3(originMinusMean.x, originMinusMean.y, originMinusMean.z),
            centralGlobalShutter,
            true);
    }

    static inline __device__ uint32_t countConservativeOnlyTiles(
        tcnn::uvec2 tileGrid,
        const gofish_conservative_bound::ConservativeConeNode* tree,
        uint32_t side, uint32_t levels,
        const float3 center, float halfAngle, bool validProjection,
        const BoundingBox2D& tileBBox, const tcnn::vec4& conicOpacity,
        const tcnn::vec2& projectedCenter, float maxConicOpacityPower
        , const gofish_whitened_cone::FiniteEllipsoidConeCache& coneCache
        , float rendererDensity, float alphaThreshold
        ) {
        uint32_t stack[gofish_conservative_bound::kTraversalStack];
        uint32_t top = 0;
        stack[top++] = (levels << 24); // packed level,y,x; root is (levels,0,0)
        uint32_t count = 0;
        while (top) {
            const uint32_t packed = stack[--top];
            const uint32_t level = packed >> 24;
            const uint32_t y = (packed >> 12) & 0xFFFu;
            const uint32_t x = packed & 0xFFFu;
            const uint32_t levelSide = side >> level;
            const uint32_t index = gofish_conservative_bound::levelOffset(side, level)
                                 + y * levelSide + x;
            if (!gofish_conservative_bound::traversalNodeAdmitted(
                    center, halfAngle, tree[index], level, coneCache
                    , rendererDensity, alphaThreshold
                    )) continue;
            if (level == 0) {
                if (gofish_admission::conservativeOnlyAdmission(
                        x, y, tileGrid.x, tileGrid.y,
                        nativeTileAdmissionBase(static_cast<int>(x), static_cast<int>(y),
                                            validProjection, tileBBox, conicOpacity,
                                            projectedCenter, maxConicOpacityPower))) ++count;
                continue;
            }
            const uint32_t childLevel = level - 1;
            // A depth-first quadtree requires at most 3*levels+1 live slots.
            stack[top++] = (childLevel << 24) | ((2*y) << 12) | (2*x);
            stack[top++] = (childLevel << 24) | ((2*y) << 12) | (2*x+1);
            stack[top++] = (childLevel << 24) | ((2*y+1) << 12) | (2*x);
            stack[top++] = (childLevel << 24) | ((2*y+1) << 12) | (2*x+1);
        }
        return count;
    }

    static inline __device__ void eval(tcnn::uvec2 tileGrid,
                                       gofish_physical_candidate::PhysicalTileView physicalTiles,
                                       uint32_t numParticles,
                                       tcnn::ivec2 resolution,
                                       threedgut::TSensorModel sensorModel,
                                       tcnn::vec3 sensorWorldPosition,
                                       tcnn::mat4x3 sensorViewMatrix,
                                       threedgut::TSensorState sensorShutterState,
                                       const tcnn::vec3* __restrict__ /*sensorRayDirectionPtr*/,
                                       tcnn::mat4x3 sensorToWorldTransform,
                                       const gofish_conservative_bound::ConservativeConeNode*
                                           __restrict__ conservativeConeTree,
                                       uint32_t conservativeConeSide,
                                       uint32_t conservativeConeLevels,
                                       const uint32_t* __restrict__ conservativeCentralityFlag,
                                       // One-stream count ABI.
                                       const gofish_certified_remainder::CameraThetaBandHessianTable*
                                           __restrict__ certifiedTable,
                                       gofish_certified_remainder::SharedCountEmitRecord*
                                           __restrict__ certifiedRecords,
                                       const gofish_certified_remainder_provider::SelectedKbCameraDescriptor*
                                           __restrict__ certifiedProviderCamera,
                                       gofish_certified_remainder::ExactConeState*
                                           __restrict__ certifiedExactStates,
                                       uint32_t* __restrict__ certifiedIntegrityError,
                                       uint32_t* __restrict__ certifiedFinalCounts,
                                       tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                       float* __restrict__ particlesProjectedRadiusPtr,
                                       uint8_t* __restrict__ particlesSupportProvenGt20Ptr,
                                       float* __restrict__ particlesGlobalDepthPtr,
                                       float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       int* __restrict__ particlesVisibilityCudaPtr,
                                       threedgut::MemoryHandles parameters) {

        const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (particleIdx >= numParticles) {
            return;
        }

        Particles particles;
        particles.initializeDensity(parameters);
        const auto particleParameters = particles.fetchDensityParameters(particleIdx);
        // Every return path owns a defined conservative decision.  Zero means
        // Unknown/Not proven and can never cause pruning.
        particlesSupportProvenGt20Ptr[particleIdx] = 0u;

#if GOFISH_FAST_UT_ADMISSION
        certifiedFinalCounts[particleIdx] = 0u;
#else
        if (certifiedTable == nullptr || certifiedRecords == nullptr
            || certifiedProviderCamera == nullptr || certifiedExactStates == nullptr
            || certifiedIntegrityError == nullptr || certifiedFinalCounts == nullptr) {
            gofish_certified_remainder::hardIntegrityFailure(certifiedIntegrityError);
            return;
        }
        // Every enabled row is defined before any provider/proof return. Empty
        // support and integrity failures must never leave stale scan input.
        certifiedFinalCounts[particleIdx] = 0u;
        gofish_certified_remainder_provider::Source providerSource{};
        providerSource.parameters = parameters;
#pragma unroll
        for (int row = 0; row < 3; ++row) {
#pragma unroll
            for (int col = 0; col < 4; ++col) {
                providerSource.worldToCamera[4 * row + col] =
                    sensorViewMatrix[col][row];
            }
        }
        providerSource.camera = *certifiedProviderCamera;
        gofish_certified_remainder_provider::CalibratedProvider<Particles>
            provider{providerSource};
        gofish_certified_remainder::AdmissionInput admission{};
        uint32_t existingParticleIndex = particleIdx;
        gofish_certified_remainder::ProviderContractMarker marker{};
        const auto providerStatus = provider(
            particleIdx, admission, existingParticleIndex, marker);
        auto record = gofish_certified_remainder::SharedCountEmitRecord{};
        record.decision.route = gofish_certified_remainder::AdmissionRoute::ExactCone;
        record.particleIndex = existingParticleIndex;
        record.providerContractMarker = marker.providerContractMarker;
        record.depthKeyContractMarker = marker.depthKeyContractMarker;
        record.exactConeCardinalityReserved = marker.exactConeCardinalityReserved;
        if (providerStatus
                == gofish_certified_remainder::InputProviderStatus::NonfiniteGaussian
            || marker.certified != 1u) {
            certifiedRecords[particleIdx] = record;
            gofish_certified_remainder::hardIntegrityFailure(certifiedIntegrityError);
            return;
        }
        if (providerStatus
            == gofish_certified_remainder::InputProviderStatus::FiniteEmptySupport) {
            record.decision.route =
                gofish_certified_remainder::AdmissionRoute::EmptySupport;
        }
        if (providerStatus
            == gofish_certified_remainder::InputProviderStatus::FinitePrepared) {
            record = gofish_certified_remainder::prepareSharedRecord(
                *certifiedTable, admission, existingParticleIndex);
            record.providerContractMarker = marker.providerContractMarker;
            record.depthKeyContractMarker = marker.depthKeyContractMarker;
            record.exactConeCardinalityReserved = marker.exactConeCardinalityReserved;
        }
        if (record.decision.route == gofish_certified_remainder::AdmissionRoute::ExactCone
            && record.exactConeCardinalityReserved != 1u) {
            certifiedRecords[particleIdx] = record;
            gofish_certified_remainder::hardIntegrityFailure(certifiedIntegrityError);
            return;
        }
        record.certified = 1u;
        certifiedRecords[particleIdx] = record;
        gofish_certified_remainder::OneStreamBuffers countStream{
            certifiedFinalCounts, nullptr, nullptr, nullptr, 0u,
            certifiedIntegrityError};
        gofish_certified_remainder::CertifiedEllipseCountCallback<GUTProjector>
            ellipseCount{tileGrid, countStream};
        gofish_certified_remainder::CertifiedRectangleCountCallback<GUTProjector>
            rectangleCount{
                tileGrid, physicalTiles,
                certifiedProviderCamera->physicalSampleOffsetLower,
                certifiedProviderCamera->physicalSampleOffsetUpper,
                countStream};
        using FlatCap = gofish_flat_cone::CapProvider;
        using FlatK = gofish_flat_cone::SourceBoundKPredicate;
        gofish_flat_cone::FlatConeCountCallback<FlatCap, FlatK> exactCount{
            {tileGrid, conservativeConeTree, conservativeConeSide,
             FlatCap{certifiedExactStates}, FlatK{certifiedExactStates}},
            gofish_flat_cone::streamView(countStream)};
        switch (record.decision.route) {
        case gofish_certified_remainder::AdmissionRoute::CertifiedRectangle:
            rectangleCount(particleIdx, record.decision.rectangle);
            break;
        case gofish_certified_remainder::AdmissionRoute::CertifiedEllipse:
            ellipseCount(particleIdx, record.decision.ellipse);
            break;
        case gofish_certified_remainder::AdmissionRoute::ExactCone:
            // The post-route warp kernel prepares state and owns exact count.
            break;
        case gofish_certified_remainder::AdmissionRoute::EmptySupport:
            // The row was explicitly zeroed before provider dispatch.
            break;
        default:
            record.certified = 0u;
            certifiedRecords[particleIdx] = record;
            certifiedFinalCounts[particleIdx] = 0u;
            gofish_certified_remainder::hardIntegrityFailure(
                certifiedIntegrityError);
            return;
        }
#endif

        tcnn::vec2 particleProjCenter
            = tcnn::vec2::zero()
            ;
        float particleProjOpacity
            = 0.0f
            ;
        tcnn::vec3 particleSensorRay
            = particles.position(particleParameters) - sensorWorldPosition
            ;
        tcnn::vec3 particleProjCovariance
            = tcnn::vec3::zero()
            ;
        bool validProjection = false;
        bool supportCapEligible = false;
        {
            validProjection = unscentedParticleProjection(
                resolution,
                sensorModel,
                sensorWorldPosition,
                sensorViewMatrix,
                sensorShutterState,
                particles,
                particleParameters,
                particleSensorRay,
                particleProjOpacity,
                particleProjCenter,
                particleProjCovariance,
                supportCapEligible);
        }

        tcnn::vec2 particleProjExtent
            = tcnn::vec2::zero()
            ;
        tcnn::vec4 particleProjConicOpacity
            = tcnn::vec4::zero()
            ;
        float particleProjRadius = 0.0f;
        float particleMaxConicOpacityPower
            = 0.0f
            ;
        bool validConicEstimation = false;
        if (validProjection) {
            validConicEstimation = computeProjectedExtentConicOpacity(particleProjCovariance,
                                                                      particleProjOpacity,
                                                                      particleProjExtent,
                                                                      particleProjRadius,
                                                                      particleProjConicOpacity,
                                                                      particleMaxConicOpacityPower);
        }

        validProjection = validProjection && validConicEstimation;

        if (!validProjection && supportCapEligible) {
            particlesSupportProvenGt20Ptr[particleIdx] =
                supportProjectionProvenGt20(
                    resolution, sensorModel, sensorShutterState,
                    particles, particleParameters);
        }

        uint32_t nativeValidTiles = 0;
        BoundingBox2D tileBBox{{0, 0}, {0, 0}};
        if (validProjection) {
            tileBBox = computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent);
            if constexpr (Params::TileCulling) {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y) {
                    for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x) {
                        if (physicalTiles.retainXY(uint32_t(x), uint32_t(y))
                            && tileMinParticlePowerResponse(
                                tcnn::vec2(x, y), particleProjConicOpacity,
                                particleProjCenter) < particleMaxConicOpacityPower)
                            nativeValidTiles++;
                    }
                }
            } else {
                for (int y = tileBBox.min.y; y < tileBBox.max.y; ++y)
                    for (int x = tileBBox.min.x; x < tileBBox.max.x; ++x)
                        nativeValidTiles += physicalTiles.retainXY(
                            uint32_t(x), uint32_t(y)) ? 1u : 0u;
            }
        }
        // Certified exact-cone counts are finalized after this projector pass.
        // Prepare projection and radiance state for every particle now; the
        // post-count kernel owns the exact visibility bit.
        const uint32_t numValidTiles = 1u;
        if (numValidTiles == 0) {
            particlesProjectedPositionPtr[particleIdx]     = tcnn::vec2::zero();
            particlesProjectedConicOpacityPtr[particleIdx] = tcnn::vec4::zero();
            particlesProjectedExtentPtr[particleIdx]       = tcnn::vec2::zero();
            particlesProjectedRadiusPtr[particleIdx]       = 0.0f;
            particlesGlobalDepthPtr[particleIdx]           = 0.f;
            return;
        }

        const float particleSensorDistance = length(particleSensorRay);

        if constexpr (!Params::PerRayParticleFeatures) {
            particles.initializeFeatures(parameters);
            reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesPtr)[particleIdx] =
                particles.template featuresCustomFromBuffer<false>(particleIdx, particleSensorRay / particleSensorDistance);
        }

        particlesProjectedPositionPtr[particleIdx]     = validProjection ? particleProjCenter : tcnn::vec2::zero();
        particlesProjectedConicOpacityPtr[particleIdx] = validProjection ? particleProjConicOpacity : tcnn::vec4::zero();
        particlesProjectedExtentPtr[particleIdx]       = validProjection ? particleProjExtent
            : (supportCapEligible ? tcnn::vec2{-1.0f, -1.0f} : tcnn::vec2::zero());
        // SOF/3DGS screen-size pruning consumes the same opacity-adaptive
        // covariance radius as raster admission, rounded up in calibrated
        // pixels.  Invalid native projections remain zero; exact support that
        // encloses the camera is promoted to +inf by the collector.
        particlesProjectedRadiusPtr[particleIdx]       =
            validProjection ? ceilf(particleProjRadius) : 0.0f;
        if constexpr (Params::GlobalZOrder) {
            const tcnn::vec3& particleMean       = particles.position(particleParameters);
            particlesGlobalDepthPtr[particleIdx] = (particleMean.x * sensorViewMatrix[0][2] + particleMean.y * sensorViewMatrix[1][2] +
                                                    particleMean.z * sensorViewMatrix[2][2] + sensorViewMatrix[3][2]);
        } else {
            particlesGlobalDepthPtr[particleIdx] = particleSensorDistance;
        }
#if GOFISH_FAST_UT_ADMISSION
        (void)nativeValidTiles;
        {
            const tcnn::vec2 storedCenter = validProjection ? particleProjCenter : tcnn::vec2::zero();
            const tcnn::vec4 storedConic = validProjection ? particleProjConicOpacity : tcnn::vec4::zero();
            const tcnn::vec2 storedExtent = validProjection ? particleProjExtent : tcnn::vec2::zero();
            uint32_t count = 0u;
            auto noSkip = [](uint32_t, uint32_t) -> bool { return false; };
            if (fastUtValid(storedConic, storedExtent)) {
                count = fastAdmittedTiles(tileGrid, physicalTiles, storedCenter, storedConic, storedExtent,
                                          [](uint32_t, uint32_t, uint32_t) {}, noSkip);
            } else if (particles.opacity(particleParameters) >= Params::AlphaThreshold) {
                count = fastFallbackTiles(tileGrid, physicalTiles, conservativeConeTree, conservativeConeSide,
                                          conservativeConeLevels, sensorToWorldTransform * tcnn::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                                          particles.position(particleParameters),
                                          particles.packedFactorAsTranspose(particleParameters),
                                          particles.opacity(particleParameters),
                                          [](uint32_t, uint32_t, uint32_t) {}, noSkip);
            }
            certifiedFinalCounts[particleIdx] = count;
        }
#endif
    }

    static inline __device__ void expand(tcnn::uvec2 tileGrid,
                                         gofish_physical_candidate::PhysicalTileView physicalTiles,
                                         int numParticles,
                                         tcnn::ivec2 resolution,
                                         threedgut::TSensorModel sensorModel,
                                         threedgut::TSensorState /*sensorState*/,
                                         // One-stream emit ABI.
                                         const gofish_certified_remainder::SharedCountEmitRecord*
                                             __restrict__ certifiedRecords,
                                         const gofish_certified_remainder_provider::SelectedKbCameraDescriptor*
                                             __restrict__ certifiedProviderCamera,
                                         const gofish_certified_remainder::ExactConeState*
                                             __restrict__ certifiedExactStates,
                                         uint32_t* __restrict__ certifiedIntegrityError,
                                         const uint64_t* __restrict__ certifiedFinalOffsets,
                                         const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                         const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                         const tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                         const float* __restrict__ particlesGlobalDepthPtr,
                                         threedgut::MemoryHandles parameters,
                                         const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                         const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                         tcnn::mat4x3 sensorToWorldTransform,
                                         const gofish_conservative_bound::ConservativeConeNode*
                                             __restrict__ conservativeConeTree,
                                         uint32_t conservativeConeSide,
                                         uint32_t conservativeConeLevels,
                                         const uint32_t* __restrict__ conservativeCentralityFlag,
                                         uint64_t* __restrict__ certifiedFinalKeys,
                                         uint32_t* __restrict__ certifiedFinalParticleIds) {

        const int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (particleIdx >= numParticles) {
            return;
        }

        const tcnn::vec2 particleProjExtent = particlesProjectedExtentPtr[particleIdx];
        const bool supportFallbackEligible = particleProjExtent.x < -0.5f;
        // check the particle projected extent x
        constexpr float eps = 1e-06f;

        const tcnn::vec2 particleProjCenter = particlesProjectedPositionPtr[particleIdx];
        const tcnn::vec4 conicOpacity       = particlesProjectedConicOpacityPtr[particleIdx];
        const bool validProjection          = particleProjExtent.x > eps && conicOpacity.w >= Params::AlphaThreshold;
        const bool supportCapEligible = supportFallbackEligible
            || (validProjection
                && sensorModel.shutterType == threedgut::TSensorModel::GlobalShutter);
        const BoundingBox2D tileBBox        = validProjection
            ? computeTileSpaceBBox(tileGrid, particleProjCenter, particleProjExtent)
            : BoundingBox2D{{0, 0}, {0, 0}};
        const float maxConicOpacityPower    = validProjection
            ? logf(conicOpacity.w / Params::AlphaThreshold) : 0.0f;

        // ---- SOF mode-3 coarse sort key (sort_order = PER_TILE_DEPTH) ----
        // This compile gate affects only the non-differentiable coarse CUB key,
        // not projection, candidate admission, alpha/support tests, poses,
        // stencils, the exact per-ray response, or HEAD4/MID8/TAIL64.
        Particles particles;
        particles.initializeDensity(parameters);
        // The conservative-support stream below this key block always consumes
        // the effective full-shape parameters. Keep them in the enclosing
        // scope even when SOF's unconstrained ordering key additionally uses
        // raw B^{-1}.
        const auto particleParameters = particles.fetchDensityParameters(particleIdx);
        const auto particleRawParameters = particles.fetchDensityRawParameters(particleIdx);
        const tcnn::vec3 fallbackCameraWorld = sensorToWorldTransform
            * tcnn::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const tcnn::vec3 particleWorld = particles.position(particleParameters);
        const float cameraRadialCandidate = length(
            particleWorld - fallbackCameraWorld);
        const float cameraRadialFallback =
            gofish_sof_sort::finiteNonnegativeDepth(
                cameraRadialCandidate, Params::MaxDepthValue);
        const uint32_t cameraRadialFallbackKey =
            __float_as_uint(cameraRadialFallback);
        auto perTileDepthKey = [&](int tx, int ty) -> uint32_t {
            if constexpr (Params::PerTileDepth) {
                constexpr int BX = (int)threedgut::GUTParameters::Tiling::BlockX;
                constexpr int BY = (int)threedgut::GUTParameters::Tiling::BlockY;
                // A 16x16 tile center lies between pixels 7 and 8. Average the
                // four calibrated central rays instead of selecting the biased
                // discrete (+8,+8) pixel. Partial boundary tiles use their own
                // continuous pixel-center midpoint. This affects only the key.
                const int xBegin = tx * BX;
                const int yBegin = ty * BY;
                const int xEnd = min((int)resolution.x - 1, (tx + 1) * BX - 1);
                const int yEnd = min((int)resolution.y - 1, (ty + 1) * BY - 1);
                const int xLo = (xBegin + xEnd) / 2;
                const int xHi = (xBegin + xEnd + 1) / 2;
                const int yLo = (yBegin + yEnd) / 2;
                const int yHi = (yBegin + yEnd + 1) / 2;
                const uint32_t p00 = (uint32_t)xLo + (uint32_t)resolution.x * (uint32_t)yLo;
                const uint32_t p10 = (uint32_t)xHi + (uint32_t)resolution.x * (uint32_t)yLo;
                const uint32_t p01 = (uint32_t)xLo + (uint32_t)resolution.x * (uint32_t)yHi;
                const uint32_t p11 = (uint32_t)xHi + (uint32_t)resolution.x * (uint32_t)yHi;
                const tcnn::vec3 sensorOrigin = 0.25f * (
                    sensorRayOriginPtr[p00] + sensorRayOriginPtr[p10]
                    + sensorRayOriginPtr[p01] + sensorRayOriginPtr[p11]);
                tcnn::vec3 sensorDirection = 0.25f * (
                    sensorRayDirectionPtr[p00] + sensorRayDirectionPtr[p10]
                    + sensorRayDirectionPtr[p01] + sensorRayDirectionPtr[p11]);
                const tcnn::vec3 worldOrigin = sensorToWorldTransform
                    * tcnn::vec4(sensorOrigin, 1.0f);
                const float tileRadialCandidate = length(particleWorld - worldOrigin);
                const float tileRadialFallback =
                    gofish_sof_sort::finiteNonnegativeDepth(
                        tileRadialCandidate, cameraRadialFallback);
                const float sensorDirectionNorm2 = dot(sensorDirection, sensorDirection);
                if (!(sensorDirectionNorm2 > 0.0f) || !isfinite(sensorDirectionNorm2)) {
                    return __float_as_uint(gofish_sof_sort::takeDepthFallback(
                        tileRadialFallback, cameraRadialFallback));
                }
                sensorDirection *= rsqrtf(sensorDirectionNorm2);
                tcnn::vec3 worldDirection = tcnn::mat3(sensorToWorldTransform)
                    * sensorDirection;
                const float worldDirectionNorm2 = dot(worldDirection, worldDirection);
                if (!(worldDirectionNorm2 > 0.0f) || !isfinite(worldDirectionNorm2)) {
                    return __float_as_uint(gofish_sof_sort::takeDepthFallback(
                        tileRadialFallback, cameraRadialFallback));
                }
                worldDirection *= rsqrtf(worldDirectionNorm2);
                const float depth = gofish_sof_sort::maximumResponseDepthBInverse(
                    particleRawParameters, worldOrigin, worldDirection,
                    tileRadialFallback);
                return __float_as_uint(depth);
            } else {
                return cameraRadialFallbackKey;
            }
        };

#if GOFISH_FAST_UT_ADMISSION
        {
            const uint64_t end = certifiedFinalOffsets[particleIdx];
            const uint64_t begin = particleIdx == 0 ? 0u : certifiedFinalOffsets[particleIdx - 1];
            auto noSkip = [](uint32_t, uint32_t) -> bool { return false; };
            auto emit = [&](uint32_t x, uint32_t y, uint32_t k) {
                if (begin + k < end) {
#if   GOFISH_FAST_GLOBAL_DEPTH_KEY
                    certifiedFinalKeys[begin + k] = concatTileDepthKeys(y * tileGrid.x + x, cameraRadialFallbackKey);
#else
                    certifiedFinalKeys[begin + k] = concatTileDepthKeys(y * tileGrid.x + x, perTileDepthKey(int(x), int(y)));
#endif
                    certifiedFinalParticleIds[begin + k] = uint32_t(particleIdx);
                }
            };
            uint32_t emitted = 0u;
            if (fastUtValid(conicOpacity, particleProjExtent)) {
                emitted = fastAdmittedTiles(tileGrid, physicalTiles, particleProjCenter, conicOpacity,
                                            particleProjExtent, emit, noSkip);
            } else if (particles.opacity(particleParameters) >= Params::AlphaThreshold) {
                emitted = fastFallbackTiles(tileGrid, physicalTiles, conservativeConeTree, conservativeConeSide,
                                            conservativeConeLevels, fallbackCameraWorld, particleWorld,
                                            particles.packedFactorAsTranspose(particleParameters),
                                            particles.opacity(particleParameters), emit, noSkip);
            }
            // The count pass reserved end - begin slots.  Should a borderline tile
            // decision differ between the two passes, extra tiles are dropped and
            // unused slots are filled with invalid entries that sort to the end of
            // their tile (the renderer stops a tile's stream at an invalid entry).
            for (uint64_t slot = begin + emitted; slot < end; ++slot) {
                certifiedFinalKeys[slot] = concatTileDepthKeys(0u, 0xffffffffu);
                certifiedFinalParticleIds[slot] = threedgut::GUTParameters::InvalidParticleIdx;
            }
            (void)tileBBox; (void)maxConicOpacityPower; (void)supportCapEligible; (void)validProjection;
            return;
        }
#endif
        if (certifiedRecords == nullptr || certifiedProviderCamera == nullptr
            || certifiedExactStates == nullptr
            || certifiedIntegrityError == nullptr || certifiedFinalOffsets == nullptr
            || certifiedFinalKeys == nullptr || certifiedFinalParticleIds == nullptr) {
            gofish_certified_remainder::hardIntegrityFailure(certifiedIntegrityError);
            return;
        }
        const auto record = certifiedRecords[particleIdx];
        if (!gofish_certified_remainder::recordIntegrityValid(record)) {
            gofish_certified_remainder::hardIntegrityFailure(certifiedIntegrityError);
            return;
        }
        const uint64_t capacity = certifiedFinalOffsets[numParticles - 1];
        gofish_certified_remainder::OneStreamBuffers emitStream{
            nullptr, certifiedFinalOffsets, certifiedFinalKeys,
            certifiedFinalParticleIds, capacity, certifiedIntegrityError};
        using DepthKey = decltype(perTileDepthKey);
        gofish_certified_remainder::CertifiedEllipseEmitCallback<
            GUTProjector, DepthKey> ellipseEmit{
                tileGrid, emitStream, perTileDepthKey};
        gofish_certified_remainder::CertifiedRectangleEmitCallback<
            GUTProjector, DepthKey> rectangleEmit{
                tileGrid, physicalTiles,
                certifiedProviderCamera->physicalSampleOffsetLower,
                certifiedProviderCamera->physicalSampleOffsetUpper,
                emitStream, perTileDepthKey};
        using FlatCap = gofish_flat_cone::CapProvider;
        using FlatK = gofish_flat_cone::SourceBoundKPredicate;
        gofish_flat_cone::FlatConeEmitCallback<
            GUTProjector, DepthKey, FlatCap, FlatK> exactEmit{
                {tileGrid, conservativeConeTree, conservativeConeSide,
                 FlatCap{certifiedExactStates}, FlatK{certifiedExactStates}},
                gofish_flat_cone::streamView(emitStream),
                perTileDepthKey};
        switch (record.decision.route) {
        case gofish_certified_remainder::AdmissionRoute::CertifiedRectangle:
            // Warp-cooperative direct emission owns this route.
            break;
        case gofish_certified_remainder::AdmissionRoute::CertifiedEllipse:
            // Warp-cooperative direct emission owns this route.
            break;
        case gofish_certified_remainder::AdmissionRoute::ExactCone:
            // The post-scan one-warp kernel owns exact y-major emission.
            break;
        case gofish_certified_remainder::AdmissionRoute::EmptySupport:
            // Count is zero; emit intentionally writes no key or particle ID.
            break;
        default:
            gofish_certified_remainder::hardIntegrityFailure(
                certifiedIntegrityError);
            return;
        }
        return;

    }

    static inline __device__ void
    evalBackward(tcnn::uvec2 tileGrid,
                 uint32_t numParticles,
                 tcnn::ivec2 resolution,
                 threedgut::TSensorModel sensorModel,
                 tcnn::vec3 sensorWorldPosition,
                 tcnn::mat4x3 sensorViewMatrix,
                 // Backward reuses the final one-stream count only.
                 const gofish_certified_remainder::SharedCountEmitRecord*
                     __restrict__ certifiedRecords,
                 uint32_t* __restrict__ certifiedIntegrityError,
                 const uint32_t* __restrict__ certifiedFinalCounts,
                 threedgut::MemoryHandles parameters,
                 const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                 const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                 const float* __restrict__ particlesGlobalDepthGradPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesPtr,
                 const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                 threedgut::MemoryHandles parametersGradient) {
        if constexpr (!Params::PerRayParticleFeatures) {

            const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (particleIdx >= numParticles) {
                return;
            }
            if (certifiedRecords == nullptr || certifiedIntegrityError == nullptr
                || !gofish_certified_remainder::recordIntegrityValid(
                    certifiedRecords[particleIdx])) {
                gofish_certified_remainder::hardIntegrityFailure(
                    certifiedIntegrityError);
                return;
            }
            if (certifiedFinalCounts[particleIdx] == 0) {
                return;
            }

            Particles particles;
            particles.initializeDensity(parameters);
            particles.initializeDensityGradient(parametersGradient);
            const tcnn::vec3 incidentDirection = tcnn::normalize(particles.fetchPosition(particleIdx) - sensorWorldPosition);
            tcnn::vec3 incidentDirectionGrad   = tcnn::vec3(0.0f);

            particles.initializeFeatures(parameters);
            particles.initializeFeaturesGradient(parametersGradient);
            particles.featuresBwdToBuffer<true>(particleIdx,
                                                reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr)[particleIdx],
                                                incidentDirection,
                                                incidentDirectionGrad);

            particles.template densityIncidentDirectionBwdToBuffer<true>(particleIdx, sensorWorldPosition, incidentDirectionGrad);
        }
    }
};
