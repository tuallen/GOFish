// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#if !HIER_PER_TILE_DEPTH
#error "direct tile-cap count/emit requires the exact per-tile SOF depth key"
#endif

#include <3dgut/kernels/cuda/renderers/certifiedRemainderProjectorCallbacks.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>

namespace gofish_tile_cap_direct {

constexpr uint32_t kWarp = 32u;
constexpr uint32_t kWarpsPerBlock = 8u;
constexpr uint32_t kThreadsPerBlock = kWarp * kWarpsPerBlock;

static inline __device__ uint32_t laneMaskBefore(uint32_t lane) {
    return lane == 0u ? 0u : 0xffffffffu >> (32u - lane);
}

template <typename Projector>
static inline __device__ typename Projector::BoundingBox2D routeBounds(
    tcnn::uvec2 tileGrid,
    const gofish_certified_remainder::SharedCountEmitRecord& record,
    float2 sampleOffsetLower, float2 sampleOffsetUpper) {
    using Route = gofish_certified_remainder::AdmissionRoute;
    if (record.decision.route == Route::CertifiedEllipse) {
        const auto& ellipse = record.decision.ellipse;
        return Projector::computeTileSpaceBBox(
            tileGrid,
            tcnn::vec2{ellipse.center.x, ellipse.center.y},
            tcnn::vec2{ellipse.extent.x, ellipse.extent.y});
    }
    if (record.decision.route == Route::CertifiedRectangle) {
        return gofish_certified_remainder::
            CertifiedRectangleCountCallback<Projector>::bbox(
                tileGrid, record.decision.rectangle,
                sampleOffsetLower, sampleOffsetUpper);
    }
    return typename Projector::BoundingBox2D{{0u, 0u}, {0u, 0u}};
}

template <typename Projector>
static inline __device__ bool routeBaseAdmitted(
    const gofish_certified_remainder::SharedCountEmitRecord& record,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    uint32_t x, uint32_t y,
    const typename Projector::BoundingBox2D& bounds) {
    using Route = gofish_certified_remainder::AdmissionRoute;
    if (record.decision.route == Route::CertifiedEllipse) {
        return gofish_certified_remainder::
            CertifiedEllipseCountCallback<Projector>::admitted(
                record.decision.ellipse, x, y, bounds);
    }
    if (record.decision.route == Route::CertifiedRectangle)
        return physicalTiles.retainXY(x, y);
    return false;
}

template <typename Projector>
__global__ void count(
    uint32_t numParticles,
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    float2 sampleOffsetLower,
    float2 sampleOffsetUpper,
    uint32_t paddedSide,
    const gofish_conservative_bound::ConservativeConeNode* __restrict__ tree,
    const gofish_certified_remainder::ExactConeState* __restrict__ states,
    const gofish_certified_remainder::SharedCountEmitRecord* __restrict__ records,
    uint32_t* __restrict__ retainedCounts,
    uint32_t* __restrict__ integrityError) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t particle = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (particle >= numParticles) return;
    const auto record = records[particle];
    using Route = gofish_certified_remainder::AdmissionRoute;
    if (record.decision.route != Route::CertifiedEllipse
        && record.decision.route != Route::CertifiedRectangle) return;
    if (!gofish_certified_remainder::recordIntegrityValid(record)
        || retainedCounts == nullptr || integrityError == nullptr) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const auto bounds = routeBounds<Projector>(
        tileGrid, record, sampleOffsetLower, sampleOffsetUpper);
    if (bounds.max.x < bounds.min.x || bounds.max.y < bounds.min.y) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const uint32_t width = bounds.max.x - bounds.min.x;
    const uint32_t height = bounds.max.y - bounds.min.y;
    const uint32_t area = width * height;
    const gofish_certified_remainder::FlatConeTileCapView tileCap{
        numParticles, tileGrid, paddedSide, tree, states};
    uint32_t accepted = 0u;
    for (uint32_t base = 0u; base < area; base += kWarp) {
        const uint32_t linear = base + lane;
        bool keep = false;
        if (linear < area) {
            const uint32_t x = bounds.min.x + linear % width;
            const uint32_t y = bounds.min.y + linear / width;
            keep = routeBaseAdmitted<Projector>(
                record, physicalTiles, x, y, bounds)
                && tileCap.retain(particle, x, y);
        }
        accepted += uint32_t(__popc(__ballot_sync(0xffffffffu, keep)));
    }
    if (lane == 0u) retainedCounts[particle] = accepted;
}

template <typename Projector, typename DepthKeyProvider>
__global__ void emit(
    uint32_t numParticles,
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    float2 sampleOffsetLower,
    float2 sampleOffsetUpper,
    uint32_t paddedSide,
    const gofish_conservative_bound::ConservativeConeNode* __restrict__ tree,
    const gofish_certified_remainder::ExactConeState* __restrict__ states,
    const gofish_certified_remainder::SharedCountEmitRecord* __restrict__ records,
    const uint64_t* __restrict__ offsets,
    uint64_t capacity,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ particleIds,
    DepthKeyProvider depthKey,
    uint32_t* __restrict__ integrityError) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t particle = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (particle >= numParticles) return;
    const auto record = records[particle];
    using Route = gofish_certified_remainder::AdmissionRoute;
    if (record.decision.route != Route::CertifiedEllipse
        && record.decision.route != Route::CertifiedRectangle) return;
    if (!gofish_certified_remainder::recordIntegrityValid(record)
        || offsets == nullptr || keys == nullptr || particleIds == nullptr
        || integrityError == nullptr) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const uint64_t begin = particle == 0u ? 0u : offsets[particle - 1u];
    const uint64_t end = offsets[particle];
    if (begin > end || end > capacity) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const auto bounds = routeBounds<Projector>(
        tileGrid, record, sampleOffsetLower, sampleOffsetUpper);
    if (bounds.max.x < bounds.min.x || bounds.max.y < bounds.min.y) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    const uint32_t width = bounds.max.x - bounds.min.x;
    const uint32_t height = bounds.max.y - bounds.min.y;
    const uint32_t area = width * height;
    const gofish_certified_remainder::FlatConeTileCapView tileCap{
        numParticles, tileGrid, paddedSide, tree, states};
    uint32_t emitted = 0u;
    for (uint32_t base = 0u; base < area; base += kWarp) {
        const uint32_t linear = base + lane;
        uint32_t x = 0u, y = 0u;
        bool keep = false;
        if (linear < area) {
            x = bounds.min.x + linear % width;
            y = bounds.min.y + linear / width;
            keep = routeBaseAdmitted<Projector>(
                record, physicalTiles, x, y, bounds)
                && tileCap.retain(particle, x, y);
        }
        const uint32_t mask = __ballot_sync(0xffffffffu, keep);
        const uint32_t rank = uint32_t(__popc(mask & laneMaskBefore(lane)));
        if (keep) {
            const uint64_t destination = begin + uint64_t(emitted + rank);
            if (destination >= end) {
                gofish_certified_remainder::callbackHardIntegrityFailure(
                    integrityError);
            } else {
                const uint32_t tile = y * tileGrid.x + x;
                keys[destination] = Projector::concatTileDepthKeys(
                    tile, depthKey(record.particleIndex, x, y));
                particleIds[destination] = record.particleIndex;
            }
        }
        emitted += uint32_t(__popc(mask));
    }
    if (lane == 0u && begin + uint64_t(emitted) != end)
        gofish_certified_remainder::callbackHardIntegrityFailure(
            integrityError);
}

} // namespace gofish_tile_cap_direct

