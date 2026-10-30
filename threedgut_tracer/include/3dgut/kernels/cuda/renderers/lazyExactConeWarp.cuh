// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderProjectorCallbacks.cuh>
#include <3dgut/kernels/cuda/renderers/flatConeAdapter.cuh>
#include <3dgut/kernels/cuda/renderers/sofMaximumResponseDepth.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>

namespace gofish_lazy_exact_cone {

constexpr uint32_t kWarp = 32u;
constexpr uint32_t kWarpsPerBlock = 8u;
constexpr uint32_t kThreadsPerBlock = kWarp * kWarpsPerBlock;

static inline __device__ uint32_t laneMaskBefore(uint32_t lane) {
    return lane == 0u ? 0u : 0xffffffffu >> (32u - lane);
}

template <typename Projector, typename Particles>
static inline __device__ bool prepareState(
    uint32_t particle, tcnn::vec3 cameraWorldOrigin,
    uint32_t centralGlobalShutter, threedgut::MemoryHandles parameters,
    gofish_certified_remainder::ExactConeState& state) {
    state = {};
    state.gaussianCenter = make_float3(0.0f, 0.0f, 0.0f);
    state.gaussianHalfAngle = 0.0f;
    state.alphaThreshold = Projector::AlphaThreshold;
    state.prepared = 0u;
    if (centralGlobalShutter > 1u
        || !gofish_certified_remainder::exactConeFinite3(cameraWorldOrigin)
        || !(Projector::AlphaThreshold > 0.0f)
        || !isfinite(Projector::AlphaThreshold)) return false;
    Particles particles;
    particles.initializeDensity(parameters);
    const auto density = particles.fetchDensityParameters(particle);
    const tcnn::vec3 position = particles.position(density);
    const float opacity = particles.opacity(density);
    bool finite = gofish_certified_remainder::exactConeFinite3(position)
        && isfinite(opacity);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        finite = finite && isfinite(density.inverseFactor[row].x)
            && isfinite(density.inverseFactor[row].y)
            && isfinite(density.inverseFactor[row].z);
    }
    if (!finite) return false;
    state.cache = Projector::finiteEllipsoidConeCache(
        particles, density, cameraWorldOrigin, centralGlobalShutter != 0u);
    state.rawOpacity = opacity;
    state.flatFp32 = gofish_fp32_certificate::makeCache(state.cache);
    state.flatAdmitAll = 1u;
    const auto cameraRelation =
        gofish_flat_cone_k_test::cameraRelation(state.cache);
    state.cameraRelation = static_cast<uint32_t>(cameraRelation);
    const tcnn::vec3 relative = position - cameraWorldOrigin;
    float factor[9];
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        const tcnn::vec3 value = particles.factorColumn(density, col);
        factor[col] = value.x;
        factor[3 + col] = value.y;
        factor[6 + col] = value.z;
    }
    state.flatCap = gofish_flat_cone_k_test::makeFlatConeCap(
        opacity, Projector::AlphaThreshold,
        make_float3(relative.x, relative.y, relative.z), factor);
    state.flatAdmitAll =
        (cameraRelation != gofish_flat_cone_k_test::CameraRelation::Outside
         || state.flatCap.certified != 1u || state.flatCap.allTiles != 0u)
        ? 1u : 0u;
    state.gaussianCenter = make_float3(0.0f, 0.0f, 0.0f);
    state.gaussianHalfAngle = gofish_conservative_bound::kPi;
    state.prepared = 1u;
    return true;
}

static inline __device__ bool leafAdmitted(
    tcnn::uvec2 tileGrid,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side,
    const gofish_certified_remainder::ExactConeState* states,
    uint32_t particle,
    uint32_t tile) {
    if (tile >= tileGrid.x * tileGrid.y) return false;
    const uint32_t y = tile / tileGrid.x;
    const uint32_t x = tile - y * tileGrid.x;
    const uint32_t index = gofish_conservative_bound::levelOffset(side, 0u)
        + y * side + x;
    return gofish_flat_cone::SourceBoundKPredicate{states}(
        particle, x, y, tree[index]);
}

template <typename Projector, typename Particles>
__global__ void prepareAndCount(
    uint32_t numParticles,
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side, uint32_t levels,
    tcnn::vec3 cameraWorldOrigin,
    uint32_t centralGlobalShutter,
    const gofish_certified_remainder::SharedCountEmitRecord* records,
    threedgut::MemoryHandles parameters,
    gofish_certified_remainder::ExactConeState* states,
    uint32_t* counts,
    uint32_t* integrityError) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t particle = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (particle >= numParticles) return;
    const auto record = records[particle];
    if (record.decision.route
        != gofish_certified_remainder::AdmissionRoute::ExactCone) return;
    const uint32_t fullMask = 0xffffffffu;
    if (!gofish_certified_remainder::recordIntegrityValid(record)
        || tree == nullptr || states == nullptr || counts == nullptr
        || integrityError == nullptr || levels > 12u
        || side != (1u << levels) || tileGrid.x > side || tileGrid.y > side) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    bool prepared = true;
    if (lane == 0u) {
        gofish_certified_remainder::ExactConeState state{};
        prepared = prepareState<Projector, Particles>(
            particle, cameraWorldOrigin, centralGlobalShutter,
            parameters, state);
        states[particle] = state;
    }
    prepared = __shfl_sync(fullMask, prepared, 0);
    if (!prepared) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    __syncwarp(fullMask);
    const uint32_t tiles = tileGrid.x * tileGrid.y;
    uint32_t count = 0u;
    for (uint32_t base = 0u; base < tiles; base += kWarp) {
        const uint32_t tile = base + lane;
        const bool keep = physicalTiles.retainTile(tile)
            && leafAdmitted(tileGrid, tree, side, states, particle, tile);
        const uint32_t mask = __ballot_sync(fullMask, keep);
        if (lane == 0u) count += uint32_t(__popc(mask));
    }
    if (lane == 0u) counts[particle] = count;
}

template <typename Projector, typename Particles>
struct ExactDepthKey {
    tcnn::ivec2 resolution;
    const tcnn::vec3* sensorRayOrigins;
    const tcnn::vec3* sensorRayDirections;
    tcnn::mat4x3 sensorToWorld;
    threedgut::MemoryHandles parameters;

    inline __device__ uint32_t operator()(
        uint32_t particle, uint32_t tx, uint32_t ty) const {
        Particles particles;
        particles.initializeDensity(parameters);
        const auto density = particles.fetchDensityParameters(particle);
        const auto raw = particles.fetchDensityRawParameters(particle);
        const tcnn::vec3 particleWorld = particles.position(density);
        const tcnn::vec3 cameraWorld = sensorToWorld
            * tcnn::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const float cameraFallback = gofish_sof_sort::finiteNonnegativeDepth(
            length(particleWorld - cameraWorld), Projector::MaxDepthValue);
        if constexpr (!Projector::PerTileDepth)
            return __float_as_uint(cameraFallback);
        constexpr int BX = int(threedgut::GUTParameters::Tiling::BlockX);
        constexpr int BY = int(threedgut::GUTParameters::Tiling::BlockY);
        const int xBegin = int(tx) * BX;
        const int yBegin = int(ty) * BY;
        const int xEnd = min(int(resolution.x) - 1, (int(tx) + 1) * BX - 1);
        const int yEnd = min(int(resolution.y) - 1, (int(ty) + 1) * BY - 1);
        const int xLo = (xBegin + xEnd) / 2;
        const int xHi = (xBegin + xEnd + 1) / 2;
        const int yLo = (yBegin + yEnd) / 2;
        const int yHi = (yBegin + yEnd + 1) / 2;
        const uint32_t p00 = uint32_t(xLo) + uint32_t(resolution.x) * uint32_t(yLo);
        const uint32_t p10 = uint32_t(xHi) + uint32_t(resolution.x) * uint32_t(yLo);
        const uint32_t p01 = uint32_t(xLo) + uint32_t(resolution.x) * uint32_t(yHi);
        const uint32_t p11 = uint32_t(xHi) + uint32_t(resolution.x) * uint32_t(yHi);
        const tcnn::vec3 sensorOrigin = 0.25f * (
            sensorRayOrigins[p00] + sensorRayOrigins[p10]
            + sensorRayOrigins[p01] + sensorRayOrigins[p11]);
        tcnn::vec3 sensorDirection = 0.25f * (
            sensorRayDirections[p00] + sensorRayDirections[p10]
            + sensorRayDirections[p01] + sensorRayDirections[p11]);
        const tcnn::vec3 worldOrigin = sensorToWorld
            * tcnn::vec4(sensorOrigin, 1.0f);
        const float tileFallback = gofish_sof_sort::finiteNonnegativeDepth(
            length(particleWorld - worldOrigin), cameraFallback);
        const float sensorNorm2 = dot(sensorDirection, sensorDirection);
        if (!(sensorNorm2 > 0.0f) || !isfinite(sensorNorm2))
            return __float_as_uint(gofish_sof_sort::takeDepthFallback(
                tileFallback, cameraFallback));
        sensorDirection *= rsqrtf(sensorNorm2);
        tcnn::vec3 worldDirection = tcnn::mat3(sensorToWorld)
            * sensorDirection;
        const float worldNorm2 = dot(worldDirection, worldDirection);
        if (!(worldNorm2 > 0.0f) || !isfinite(worldNorm2))
            return __float_as_uint(gofish_sof_sort::takeDepthFallback(
                tileFallback, cameraFallback));
        worldDirection *= rsqrtf(worldNorm2);
        return __float_as_uint(gofish_sof_sort::maximumResponseDepthBInverse(
            raw, worldOrigin, worldDirection, tileFallback));
    }
};

template <typename Projector, typename Particles>
__global__ void emit(
    uint32_t numParticles, tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side,
    const gofish_certified_remainder::SharedCountEmitRecord* records,
    const gofish_certified_remainder::ExactConeState* states,
    const uint64_t* offsets,
    uint64_t* keys, uint32_t* particleIds,
    ExactDepthKey<Projector, Particles> depthKey,
    uint32_t* integrityError) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t particle = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (particle >= numParticles) return;
    const auto record = records[particle];
    if (record.decision.route
        != gofish_certified_remainder::AdmissionRoute::ExactCone) return;
    const uint32_t fullMask = 0xffffffffu;
    if (!gofish_certified_remainder::recordIntegrityValid(record)
        || states[particle].prepared != 1u || offsets == nullptr
        || keys == nullptr || particleIds == nullptr || tree == nullptr) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const uint64_t begin = particle == 0u ? 0u : offsets[particle - 1u];
    const uint64_t end = offsets[particle];
    uint32_t emitted = 0u;
    const uint32_t tiles = tileGrid.x * tileGrid.y;
    for (uint32_t base = 0u; base < tiles; base += kWarp) {
        const uint32_t tile = base + lane;
        const bool keep = physicalTiles.retainTile(tile)
            && leafAdmitted(tileGrid, tree, side, states, particle, tile);
        const uint32_t mask = __ballot_sync(fullMask, keep);
        const uint32_t rank = uint32_t(__popc(mask & laneMaskBefore(lane)));
        if (keep) {
            const uint64_t destination = begin + uint64_t(emitted + rank);
            if (destination >= end) {
                gofish_certified_remainder::callbackHardIntegrityFailure(
                    integrityError);
            } else {
                const uint32_t y = tile / tileGrid.x;
                const uint32_t x = tile - y * tileGrid.x;
                keys[destination] = Projector::concatTileDepthKeys(
                    tile, depthKey(record.particleIndex, x, y));
                particleIds[destination] = record.particleIndex;
            }
        }
        emitted += uint32_t(__popc(mask));
    }
    if (lane == 0u && begin + emitted != end)
        gofish_certified_remainder::callbackHardIntegrityFailure(integrityError);
}

} // namespace gofish_lazy_exact_cone

