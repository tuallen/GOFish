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

#include <3dgut/kernels/cuda/common/rayPayload.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/utils/bounding_box.h>
#include <3dgut/kernels/cuda/renderers/conservativeOpacityBound.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>
#if FINE_GRAINED_LOAD_BALANCING
#error "physical full-tile pruning requires 16x16 raster blocks"
#endif
#include <3dgut/kernels/cuda/renderers/certifiedRemainderOneStream.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderProvider.cuh>

__global__ void projectOnTiles(tcnn::uvec2 tileGrid,
                               gofish_physical_candidate::PhysicalTileView physicalTiles,
                               uint32_t numParticles,
                               tcnn::ivec2 resolution,
                               threedgut::TSensorModel sensorModel,
                               tcnn::vec3 sensorWorldPosition,
                               tcnn::mat4x3 sensorViewMatrix,
                               threedgut::TSensorState sensorShutterState,
                               const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                               tcnn::mat4x3 sensorToWorldTransform,
                               const gofish_conservative_bound::ConservativeConeNode*
                                   __restrict__ conservativeConeTree,
                               uint32_t conservativeConeSide,
                               uint32_t conservativeConeLevels,
                               const uint32_t* __restrict__ conservativeCentralityFlag,
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
                               const uint64_t* __restrict__ parameterMemoryHandles) {

    TGUTProjector::eval(tileGrid,
                        physicalTiles,
                        numParticles,
                        resolution,
                        sensorModel,
                        sensorWorldPosition,
                        sensorViewMatrix,
                        sensorShutterState,
                        sensorRayDirectionPtr,
                        sensorToWorldTransform,
                        conservativeConeTree,
                        conservativeConeSide,
                        conservativeConeLevels,
                        conservativeCentralityFlag,
                        certifiedTable,
                        certifiedRecords,
                        certifiedProviderCamera,
                        certifiedExactStates,
                        certifiedIntegrityError,
                        certifiedFinalCounts,
                        particlesProjectedPositionPtr,
                        particlesProjectedConicOpacityPtr,
                        particlesProjectedExtentPtr,
                        particlesProjectedRadiusPtr,
                        particlesSupportProvenGt20Ptr,
                        particlesGlobalDepthPtr,
                        particlesPrecomputedFeaturesPtr,
                        particlesVisibilityCudaPtr,
                        {parameterMemoryHandles});
}

__global__ void expandTileProjections(tcnn::uvec2 tileGrid,
                                      gofish_physical_candidate::PhysicalTileView physicalTiles,
                                      uint32_t numParticles,
                                      tcnn::ivec2 resolution,
                                      threedgut::TSensorModel sensorModel,
                                      threedgut::TSensorState sensorState,
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
                                      const uint64_t* __restrict__ parameterMemoryHandles,
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

    TGUTProjector::expand(tileGrid,
                          physicalTiles,
                          numParticles,
                          resolution,
                          sensorModel,
                          sensorState,
                          certifiedRecords,
                          certifiedProviderCamera,
                          certifiedExactStates,
                          certifiedIntegrityError,
                          certifiedFinalOffsets,
                          particlesProjectedPositionPtr,
                          particlesProjectedConicOpacityPtr,
                          particlesProjectedExtentPtr,
                          particlesGlobalDepthPtr,
                          {parameterMemoryHandles},
                          sensorRayOriginPtr,
                          sensorRayDirectionPtr,
                          sensorToWorldTransform,
                          conservativeConeTree,
                          conservativeConeSide,
                          conservativeConeLevels,
                          conservativeCentralityFlag,
                          certifiedFinalKeys,
                          certifiedFinalParticleIds);
}

#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
// real-time renderer heavy-tile split scratch: [GOFISH_FAST_TILE_SPLIT][pixels][RAY_FEATURE_DIM + 2]
// (features, transmittance, hit distance) of each chunk; transmittance < 0 marks a pixel the render skips.
__device__ float* gofishSplitScratch;

__global__ void gofishSplitCombine(threedgut::RenderParameters params,
                                   const uint8_t* __restrict__ physicalTileMaskPtr,
                                   const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                   uint32_t tileGridX,
                                   float* __restrict__ worldHitDistancePtr,
                                   TFeatureDensityElem* __restrict__ featureDensityPtr) {
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= params.resolution.x || y >= params.resolution.y) return;
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (physicalTileMaskPtr[tile] == 0u) return;
    const tcnn::uvec2 range = sortedTileRangeIndicesPtr[tile];
    if (range.y - range.x < uint32_t(GOFISH_FAST_TILE_SPLIT_MIN)) return;
    const uint32_t idx = x + params.resolution.x * y, npix = params.resolution.x * params.resolution.y;
    constexpr uint32_t W = RAY_FEATURE_DIM + 2;
    const float* scratch = gofishSplitScratch;
    if (scratch[size_t(idx) * W + RAY_FEATURE_DIM] < 0.0f) return;
    float f[RAY_FEATURE_DIM];
#pragma unroll
    for (int i = 0; i < RAY_FEATURE_DIM; ++i) f[i] = 0.0f;
    float t = 1.0f, h = 0.0f;
    for (uint32_t z = 0; z < uint32_t(GOFISH_FAST_TILE_SPLIT); ++z) {
        const float* p = scratch + (size_t(z) * npix + idx) * W;
#pragma unroll
        for (int i = 0; i < RAY_FEATURE_DIM; ++i) f[i] += t * p[i];
        h += t * p[RAY_FEATURE_DIM + 1];
        t *= p[RAY_FEATURE_DIM];
    }
    const uint32_t base = idx * (RAY_FEATURE_DIM + 1);
#pragma unroll
    for (int i = 0; i < RAY_FEATURE_DIM; ++i) featureDensityPtr[base + i] = TFeatureDensityElem(f[i]);
    featureDensityPtr[base + RAY_FEATURE_DIM] = TFeatureDensityElem(1.0f - t);
    worldHitDistancePtr[idx] = h;
}
#endif

#if GOFISH_SOF_SCHEDULE
// Runs the SOF MID8 cascade once and publishes every MID group's emission sequence.
__global__ void sofMid8Schedule(threedgut::RenderParameters params,
                                const uint8_t* __restrict__ physicalTileMaskPtr,
                                const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                const uint32_t* __restrict__ sortedTileDataPtr,
                                const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                tcnn::mat4x3 sensorToWorldTransform,
                                const uint64_t* __restrict__ parameterMemoryHandles) {
    if (blockIdx.y < gSofRowBegin || blockIdx.y >= gSofRowEnd) return;
    if (physicalTileMaskPtr[blockIdx.y * gridDim.x + blockIdx.x] == 0u)
        return;
    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);
    const tcnn::uvec2 range = sortedTileRangeIndicesPtr[blockIdx.y * gridDim.x + blockIdx.x];
    model_::Particles particles;
    particles.initializeDensity({parameterMemoryHandles});
    TGUTRenderer::evalSOFMid8Schedule(params, ray, particles, range, 0u, 0u,
                                      threadIdx.y * blockDim.x + threadIdx.x,
                                      sortedTileDataPtr, nullptr, nullptr, nullptr, nullptr);
}
#endif

__global__ void render(threedgut::RenderParameters params,
                       const uint8_t* __restrict__ physicalTileMaskPtr,
                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                       const uint32_t* __restrict__ sortedTileDataPtr,
                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                       const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                       tcnn::mat4x3 sensorToWorldTransform,
                       float* __restrict__ worldHitCountPtr,
                       float* __restrict__ worldHitDistancePtr,
                       TFeatureDensityElem* __restrict__ featureDensityPtr,
                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                       const float* __restrict__ particlesGlobalDepthPtr,
                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                       float* __restrict__ worldNormalPtr,
                       float* __restrict__ worldDepthVarPtr,
                       float* __restrict__ worldQuantDepthPtr,
                       float* __restrict__ worldExtentPtr,
                       float* __restrict__ worldOpacityFieldPtr,
                       float* __restrict__ worldInvDPtr,
                       float* __restrict__ worldInvD2Ptr,
                       const uint64_t* __restrict__ parameterMemoryHandles) {

#if GOFISH_SOF_SCHEDULE
    if (blockIdx.y < gSofRowBegin || blockIdx.y >= gSofRowEnd) return;  // current tile-row batch
#endif
    const uint32_t physicalTile = blockIdx.y * gridDim.x + blockIdx.x;
    if (physicalTileMaskPtr[physicalTile] == 0u)
        return;
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
    const bool gofishSplitTile = (sortedTileRangeIndicesPtr[physicalTile].y - sortedTileRangeIndicesPtr[physicalTile].x)
        >= uint32_t(GOFISH_FAST_TILE_SPLIT_MIN);
    if (!gofishSplitTile && blockIdx.z > 0)
        return;
#endif
    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);

    TGUTRenderer::eval(params,
                       ray,
                       sortedTileRangeIndicesPtr,
                       sortedTileDataPtr,
                       particlesProjectedPositionPtr,
                       particlesProjectedConicOpacityPtr,
                       particlesGlobalDepthPtr,
                       particlesPrecomputedFeaturesPtr,
                       {parameterMemoryHandles});

    // TGUTModel::eval(params, ray, {parameterMemoryHandles});

    // NB : finalize ray is not differentiable (has to be no-op when used in a differentiable renderer)
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
    if (gofishSplitTile) {
        const uint32_t px = threadIdx.x + blockDim.x * blockIdx.x, py = threadIdx.y + blockDim.y * blockIdx.y;
        if (px < params.resolution.x && py < params.resolution.y) {
            constexpr uint32_t W = RAY_FEATURE_DIM + 2;
            const uint32_t idx = px + params.resolution.x * py, npix = params.resolution.x * params.resolution.y;
            float* p = gofishSplitScratch + (size_t(blockIdx.z) * npix + idx) * W;
            if (ray.isValid()) {
#pragma unroll
                for (int i = 0; i < RAY_FEATURE_DIM; ++i) p[i] = ray.features[i];
                p[RAY_FEATURE_DIM] = ray.transmittance;
                p[RAY_FEATURE_DIM + 1] = ray.hitT;
            } else if (blockIdx.z == 0) {
                p[RAY_FEATURE_DIM] = -1.0f;
            }
        }
        return;
    }
#endif
    finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr, worldHitDistancePtr, featureDensityPtr, sensorToWorldTransform, worldNormalPtr, worldDepthVarPtr, worldQuantDepthPtr, worldExtentPtr, worldOpacityFieldPtr, worldInvDPtr, worldInvD2Ptr);
}

// SOF's direct opacity loss is a replay at the detached analytic crossing
// rho*.  The replay field is the checkpoint-owned physical-Mip scene used by
// extraction; Q/m2 remains an image-formation operator and is intentionally
// absent here.  Truncating the finite ray interval before finding each
// closest point implements alpha_i(min(t_i,rho*)).  A no-crossing ray replays
// the complete finite renderer interval, exactly matching SOF's residual
// supervision rather than returning a constant zero.
__global__ void sofExactCrossingOpacityForward(
    threedgut::RenderParameters params,
    const uint8_t* __restrict__ physicalTileMaskPtr,
    const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
    const uint32_t* __restrict__ sortedTileDataPtr,
    const tcnn::vec3* __restrict__ sensorRayOriginPtr,
    const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
    tcnn::mat4x3 sensorToWorldTransform,
    const float* __restrict__ worldQuantDepthPtr,
    float* __restrict__ worldOpacityFieldPtr,
    const uint64_t* __restrict__ parameterMemoryHandles) {
    const uint32_t physicalTile = blockIdx.y * gridDim.x + blockIdx.x;
    if (physicalTileMaskPtr[physicalTile] == 0u) return;
    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr,
        sensorToWorldTransform);
    if (!ray.isAlive()) return;

    const float rho = worldQuantDepthPtr[
        ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 1)];
    const bool hasCrossing = rho > 0.0f && isfinite(rho);
    const float replayMax = hasCrossing
        ? fminf(rho, ray.tMinMax.y)
        : ray.tMinMax.y;
    if (!(replayMax >= ray.tMinMax.x)) {
        worldOpacityFieldPtr[ray.idx] = 0.0f;
        return;
    }

    const uint32_t tileIdx = blockIdx.y * gridDim.x + blockIdx.x;
    const tcnn::uvec2 range = sortedTileRangeIndicesPtr[tileIdx];
    model_::Particles particles;
    particles.initializeDensity({parameterMemoryHandles});

    float product = 1.0f;
    for (uint32_t offset = range.x; offset < range.y; ++offset) {
        const uint32_t particleIdx = sortedTileDataPtr[offset];
        if (particleIdx == threedgut::GUTParameters::InvalidParticleIdx)
            continue;
        const auto raw = particles.fetchDensityRawParameters(particleIdx);
        const auto sample = threedgut::particleSofIntrinsicOpacitySample(
            raw,
            reinterpret_cast<const float3&>(ray.origin),
            reinterpret_cast<const float3&>(ray.direction),
            ray.tMinMax.x, replayMax,
            model_::Particles::MinParticleKernelDensity,
            model_::Particles::AlphaThreshold);
        if (!sample.accepted) continue;
        product *= (1.0f - sample.alpha);
    }
    worldOpacityFieldPtr[ray.idx] = 1.0f - product;
}

__global__ void sofExactCrossingOpacityBackward(
    threedgut::RenderParameters params,
    const uint8_t* __restrict__ physicalTileMaskPtr,
    const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
    const uint32_t* __restrict__ sortedTileDataPtr,
    const tcnn::vec3* __restrict__ sensorRayOriginPtr,
    const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
    tcnn::mat4x3 sensorToWorldTransform,
    const float* __restrict__ worldQuantDepthPtr,
    const float* __restrict__ worldOpacityFieldPtr,
    const float* __restrict__ worldOpacityFieldGradientPtr,
    const uint64_t* __restrict__ parameterMemoryHandles,
    const uint64_t* __restrict__ parameterGradientMemoryHandles,
    float* __restrict__ particlesPositionAbsGradPtr) {
    const uint32_t physicalTile = blockIdx.y * gridDim.x + blockIdx.x;
    if (physicalTileMaskPtr[physicalTile] == 0u) return;
    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr,
        sensorToWorldTransform);
    if (!ray.isAlive()) return;

    const float upstream = worldOpacityFieldGradientPtr[ray.idx];
    if (upstream == 0.0f) return;
    const float rho = worldQuantDepthPtr[
        ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 1)];
    const bool hasCrossing = rho > 0.0f && isfinite(rho);
    const float replayMax = hasCrossing
        ? fminf(rho, ray.tMinMax.y)
        : ray.tMinMax.y;
    if (!(replayMax >= ray.tMinMax.x)) return;

    const float fullProduct = fminf(
        fmaxf(1.0f - worldOpacityFieldPtr[ray.idx], 0.0f), 1.0f);
    const uint32_t tileIdx = blockIdx.y * gridDim.x + blockIdx.x;
    const tcnn::uvec2 range = sortedTileRangeIndicesPtr[tileIdx];
    model_::Particles particles;
    particles.initializeDensity({parameterMemoryHandles});
    particles.initializeDensityGradient({parameterGradientMemoryHandles});

    for (uint32_t offset = range.x; offset < range.y; ++offset) {
        const uint32_t particleIdx = sortedTileDataPtr[offset];
        if (particleIdx == threedgut::GUTParameters::InvalidParticleIdx)
            continue;
        const auto raw = particles.fetchDensityRawParameters(particleIdx);
        const auto sample = threedgut::particleSofIntrinsicOpacitySample(
            raw,
            reinterpret_cast<const float3&>(ray.origin),
            reinterpret_cast<const float3&>(ray.direction),
            ray.tMinMax.x, replayMax,
            model_::Particles::MinParticleKernelDensity,
            model_::Particles::AlphaThreshold);
        if (!sample.accepted) continue;

        const float denominator = fmaxf(
            1.0f - sample.alpha,
            1.0f - GAUSSIAN_PARTICLE_MAX_ALPHA);
        const float exclusiveProduct = fminf(
            fmaxf(fullProduct / denominator, 0.0f), 1.0f);
        auto gradient = threedgut::particleSofIntrinsicOpacityVjp(
            raw,
            reinterpret_cast<const float3&>(ray.origin),
            reinterpret_cast<const float3&>(ray.direction),
            sample, upstream * exclusiveProduct);

        if (particlesPositionAbsGradPtr != nullptr) {
#if GOFISH_EXACT_SCREEN_ABSGRAD
            const float* row = particlesPositionAbsGradPtr
                + size_t(particleIdx) * 8u;
            const float screenU = row[0] * gradient.position.x
                + row[1] * gradient.position.y
                + row[2] * gradient.position.z;
            const float screenV = row[3] * gradient.position.x
                + row[4] * gradient.position.y
                + row[5] * gradient.position.z;
            atomicAdd(&particlesPositionAbsGradPtr[particleIdx * 8 + 6],
                      fabsf(screenU));
            atomicAdd(&particlesPositionAbsGradPtr[particleIdx * 8 + 7],
                      fabsf(screenV));
#else
            atomicAdd(&particlesPositionAbsGradPtr[particleIdx * 3 + 0],
                      fabsf(gradient.position.x));
            atomicAdd(&particlesPositionAbsGradPtr[particleIdx * 3 + 1],
                      fabsf(gradient.position.y));
            atomicAdd(&particlesPositionAbsGradPtr[particleIdx * 3 + 2],
                      fabsf(gradient.position.z));
#endif
        }
        particles.processHitBwdUpdateDensityGradient<false>(
            particleIdx, gradient, 0u);
    }
}

#if FINE_GRAINED_LOAD_BALANCING
// Fine-grained load balancing rendering kernel: static allocation per virtual tile
__global__ void renderBalanced(threedgut::RenderParameters params,
                               const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                               const uint32_t* __restrict__ sortedTileDataPtr,
                               const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                               const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                               tcnn::mat4x3 sensorToWorldTransform,
                               float* __restrict__ worldHitCountPtr,
                               float* __restrict__ worldHitDistancePtr,
                               TFeatureDensityElem* __restrict__ featureDensityPtr,
                               const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               const float* __restrict__ particlesGlobalDepthPtr,
                               const float* __restrict__ particlesPrecomputedFeaturesPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles,
                               const tcnn::uvec2 tileGrid) {

    // Static allocation: each block handles one virtual tile
    using namespace threedgut;
    constexpr uint32_t virtualTilesPerTile = GUTParameters::Tiling::VirtualTilesPerTile;
    const uint32_t virtualTileId           = blockIdx.x;

    // Calculate total virtual tiles across all original tiles
    const uint32_t totalVirtualTiles = tileGrid.x * tileGrid.y * virtualTilesPerTile;

    // Boundary check
    if (virtualTileId >= totalVirtualTiles)
        return;

    // Map virtual tile back to original tile coordinates and local position
    const uint32_t originalTileId        = virtualTileId / virtualTilesPerTile;
    const uint32_t virtualTileInOriginal = virtualTileId % virtualTilesPerTile;

    const uint32_t originalTileX = originalTileId % tileGrid.x;
    const uint32_t originalTileY = originalTileId / tileGrid.x;

    // Map virtual tile to pixel coordinates within original tile
    constexpr uint32_t virtualTilesPerTileX = GUTParameters::Tiling::VirtualTilesPerTileX;
    constexpr uint32_t virtualTileX         = GUTParameters::Tiling::VirtualTileX;
    constexpr uint32_t virtualTileY         = GUTParameters::Tiling::VirtualTileY;
    constexpr uint32_t warpSize             = GUTParameters::Tiling::WarpSize;

    const uint32_t virtualTileXPos = virtualTileInOriginal % virtualTilesPerTileX; // 0-7
    const uint32_t virtualTileYPos = virtualTileInOriginal / virtualTilesPerTileX; // 0-7

    // Calculate base pixel coordinates for this virtual tile
    const uint32_t basePixelX = virtualTileXPos * virtualTileX; // 0,2,4,6,8,10,12,14
    const uint32_t basePixelY = virtualTileYPos * virtualTileY; // 0,2,4,6,8,10,12,14

    // Warp-level processing: each warp handles one pixel in virtual tile
    const uint32_t warpId = threadIdx.x / warpSize;
    const uint32_t laneId = threadIdx.x & (warpSize - 1);

    // Each block processes 1 virtual tile = virtualTileSize pixels, each warp handles 1 pixel
    constexpr uint32_t virtualTileSize = GUTParameters::Tiling::VirtualTileSize;
    constexpr uint32_t blockX          = GUTParameters::Tiling::BlockX;
    constexpr uint32_t blockY          = GUTParameters::Tiling::BlockY;

    if (warpId < virtualTileSize) { // virtualTileSize warps per block (1 warp per pixel)
        // Arrange pixels in row-major order within virtualTileX x virtualTileY region
        // warp 0-3 maps to pixels: (0,0),(1,0),(0,1),(1,1) for 2x2 virtual tile
        const uint32_t pixelOffsetX = warpId % virtualTileX;
        const uint32_t pixelOffsetY = warpId / virtualTileX;

        const uint32_t pixelLocalX = basePixelX + pixelOffsetX;
        const uint32_t pixelLocalY = basePixelY + pixelOffsetY;

        const tcnn::uvec2 pixel = {
            originalTileX * blockX + pixelLocalX,
            originalTileY * blockY + pixelLocalY};

        // Initialize ray for current pixel
        auto ray = initializeRayPerPixel<TGUTRenderer::TRayPayload>(
            params, pixel, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);

        // Warp-level parallel rendering using original tile's particle data
        const tcnn::uvec2 originalTile = {originalTileX, originalTileY};

        TGUTRenderer::evalForwardNoKBufferBalanced(params,
                                                   ray,
                                                   sortedTileRangeIndicesPtr,
                                                   sortedTileDataPtr,
                                                   particlesProjectedPositionPtr,
                                                   particlesProjectedConicOpacityPtr,
                                                   particlesGlobalDepthPtr,
                                                   particlesPrecomputedFeaturesPtr,
                                                   originalTile,
                                                   tileGrid,
                                                   laneId, // warp lane for parallel processing
                                                   {parameterMemoryHandles});

        // Write final results to output buffers
        // Only lane 0 should write, as only it has accumulated the correct values
        if (laneId == 0) {
            finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr,
                        worldHitDistancePtr, featureDensityPtr, sensorToWorldTransform);
        }
    }
}
#endif // FINE_GRAINED_LOAD_BALANCING

// Optional register cap on the backward kernel, set at build time with both
// args, e.g.
//   -DNHT_BWD_LB_THREADS=256 -DNHT_BWD_LB_MIN_BLOCKS=2
// to force <=128 regs/thread at BlockSize=256. By default the compiler
// chooses the register count.
#if defined(NHT_BWD_LB_THREADS) && defined(NHT_BWD_LB_MIN_BLOCKS)
#define NHT_BWD_LB __launch_bounds__(NHT_BWD_LB_THREADS, NHT_BWD_LB_MIN_BLOCKS)
#else
#define NHT_BWD_LB
#endif

__global__ NHT_BWD_LB void renderBackward(threedgut::RenderParameters params,
                                          const uint8_t* __restrict__ physicalTileMaskPtr,
                                          const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                          const uint32_t* __restrict__ sortedTileDataPtr,
                                          const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                          const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                          tcnn::mat4x3 sensorToWorldTransform,
                                          const float* __restrict__ worldHitDistancePtr,
                                          const float* __restrict__ worldHitDistanceGradientPtr,
                                          const float* __restrict__ worldDepthVarPtr,
                                          const float* __restrict__ worldDepthVarGradientPtr,
                                          const float* __restrict__ worldExtentPtr,
                                          const float* __restrict__ worldExtentGradientPtr,
                                          const float* __restrict__ worldOpacityFieldPtr,
                                          const float* __restrict__ worldOpacityFieldGradientPtr,
                                          const float* __restrict__ worldQuantDepthPtr,
                                          const float* __restrict__ worldQuantDepthGradientPtr,
                                          const TFeatureDensityElem* __restrict__ featureDensityPtr,
                                          const float* __restrict__ featureDensityGradientPtr,
                                          tcnn::vec3* __restrict__ /*worldRayOriginGradientPtr*/,
                                          tcnn::vec3* __restrict__ /*worldRayDirectionGradientPtr*/,
                                          const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                          const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                          const float* __restrict__ particlesGlobalDepthPtr,
                                          const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                          const uint64_t* __restrict__ parameterMemoryHandles,
                                          tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                          tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                          float* __restrict__ particlesGlobalDepthGradPtr,
                                          float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                          const uint64_t* __restrict__ parameterGradientMemoryHandles,
                                          float* __restrict__ particlesPositionAbsGradPtr,
                                          const float* __restrict__ worldNormalPtr,
                                          const float* __restrict__ worldNormalGradientPtr,
                                          const float* __restrict__ worldInvDPtr,
                                          const float* __restrict__ worldInvDGradientPtr,
                                          const float* __restrict__ worldInvD2Ptr,
                                          const float* __restrict__ worldInvD2GradientPtr) {

#if GOFISH_SOF_SCHEDULE
    if (blockIdx.y < gSofRowBegin || blockIdx.y >= gSofRowEnd) return;  // current tile-row batch
#endif
    const uint32_t physicalTile = blockIdx.y * gridDim.x + blockIdx.x;
    if (physicalTileMaskPtr[physicalTile] == 0u)
        return;
    auto ray = initializeBackwardRay<TGUTRenderer::TRayPayloadBackward>(params,
                                                                        sensorRayOriginPtr,
                                                                        sensorRayDirectionPtr,
                                                                        worldHitDistancePtr,
                                                                        worldHitDistanceGradientPtr,
                                                                        worldDepthVarPtr,
                                                                        worldDepthVarGradientPtr,
                                                                        worldExtentPtr,
                                                                        worldExtentGradientPtr,
                                                                        worldOpacityFieldPtr,
                                                                        nullptr,
                                                                        worldQuantDepthPtr,
                                                                        worldQuantDepthGradientPtr,
                                                                        worldNormalPtr,
                                                                        worldNormalGradientPtr,
                                                                        featureDensityPtr,
                                                                        featureDensityGradientPtr,
                                                                        sensorToWorldTransform,
                                                                        worldInvDPtr,
                                                                        worldInvDGradientPtr,
                                                                        worldInvD2Ptr,
                                                                        worldInvD2GradientPtr
                                                                        );

    // TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles});

    TGUTBackwardRenderer::eval(params,
                               ray,
                               sortedTileRangeIndicesPtr,
                               sortedTileDataPtr,
                               particlesProjectedPositionPtr,
                               particlesProjectedConicOpacityPtr,
                               particlesGlobalDepthPtr,
                               particlesPrecomputedFeaturesPtr,
                               {parameterMemoryHandles},
                               particlesProjectedPositionGradPtr,
                               particlesProjectedConicOpacityGradPtr,
                               particlesGlobalDepthGradPtr,
                               particlesPrecomputedFeaturesGradPtr,
                               {parameterGradientMemoryHandles},
                               particlesPositionAbsGradPtr);
}

__global__ void projectBackward(tcnn::uvec2 tileGrid,
                                uint32_t numParticles,
                                tcnn::ivec2 resolution,
                                threedgut::TSensorModel sensorModel,
                                tcnn::vec3 sensorWorldPosition,
                                tcnn::mat4x3 sensorViewMatrix,
                                const gofish_certified_remainder::SharedCountEmitRecord*
                                    __restrict__ certifiedRecords,
                                uint32_t* __restrict__ certifiedIntegrityError,
                                const uint32_t* __restrict__ certifiedFinalCounts,
                                const uint64_t* __restrict__ parameterMemoryHandles,
                                const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                const float* __restrict__ particlesGlobalDepthGradPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    TGUTProjector::evalBackward(tileGrid,
                                numParticles,
                                resolution,
                                sensorModel,
                                sensorWorldPosition,
                                sensorViewMatrix,
                                certifiedRecords,
                                certifiedIntegrityError,
                                certifiedFinalCounts,
                                {parameterMemoryHandles},
                                particlesProjectedPositionGradPtr,
                                particlesProjectedConicOpacityGradPtr,
                                particlesGlobalDepthGradPtr,
                                particlesPrecomputedFeaturesPtr,
                                particlesPrecomputedFeaturesGradPtr,
                                {parameterGradientMemoryHandles});
}
