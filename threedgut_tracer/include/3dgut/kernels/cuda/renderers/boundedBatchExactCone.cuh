// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/lazyExactConeWarp.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedSupportCap.cuh>
#include <3dgut/kernels/cuda/renderers/angularRangeIndexOnce.cuh>

namespace gofish_bounded_exact {

constexpr uint64_t kReservationBytes = 2ull << 30;
constexpr uint64_t kFixedWorkspaceBytes = 64ull << 20;
constexpr uint64_t kModeCapCompact = 0u;
constexpr uint64_t kModePhysicalRange = 1u;
constexpr uint32_t kWarp = 32u;
constexpr uint32_t kWarpsPerBlock = 8u;
constexpr uint32_t kThreadsPerBlock = kWarp * kWarpsPerBlock;

struct Chunk {
    uint32_t particle;
    uint32_t tileBegin;
    uint32_t tileEnd;
    uint32_t slotBegin;
    uint32_t reservation;
    uint32_t mode;
};

static inline __device__ uint32_t laneMaskBefore(uint32_t lane) {
    return lane == 0u ? 0u : 0xffffffffu >> (32u - lane);
}

static inline __device__ bool capOverlaps(
    tcnn::uvec2 tileGrid,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side,
    const gofish_certified_remainder::ExactConeState& state,
    uint32_t tile) {
    if (tile >= tileGrid.x * tileGrid.y) return false;
    if (state.flatAdmitAll != 0u || state.flatCap.certified != 1u)
        return true;
    const uint32_t y = tile / tileGrid.x;
    const uint32_t x = tile - y * tileGrid.x;
    const uint32_t index = gofish_conservative_bound::levelOffset(side, 0u)
        + y * side + x;
    const auto& leaf = tree[index];
    const float4 angular = gofish_conservative_bound::angularConeOf(leaf);
    const gofish_flat_cone_k_test::TileRayCone cone{
        tile,
        make_float3(angular.x, angular.y, angular.z),
        leaf.chordRadius,
        leaf.directionMaxAbs,
        leaf.directionOperandsCertified,
    };
    return gofish_flat_cone_k_test::capMayOverlap(state.flatCap, cone);
}

template <typename Projector, typename Particles>
__global__ void prepareAndCapCount(
    uint32_t numParticles,
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side, uint32_t levels,
    gofish_angular_range_once::View angularIndex,
    tcnn::vec3 cameraWorldOrigin,
    uint32_t centralGlobalShutter,
    const gofish_certified_remainder::SharedCountEmitRecord* records,
    threedgut::MemoryHandles parameters,
    gofish_certified_remainder::ExactConeState* states,
    uint32_t* capCounts,
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
        || tree == nullptr || states == nullptr || capCounts == nullptr
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
        prepared = gofish_lazy_exact_cone::prepareState<Projector, Particles>(
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
    __shared__ uint32_t angularWords[kWarpsPerBlock]
        [gofish_angular_range_once::kMaxWords];
    uint32_t* rowWords = angularWords[warpInBlock];
    const bool indexed = gofish_angular_range_once::buildQueryMask(
        angularIndex, states[particle], lane, rowWords);
    const uint32_t tiles = tileGrid.x * tileGrid.y;
    uint32_t count = 0u;
    for (uint32_t base = 0u; base < tiles; base += kWarp) {
        const uint32_t tile = base + lane;
        const bool ranged = !indexed || (tile < tiles
            && gofish_angular_range_once::contains(rowWords, tile));
        const bool keep = ranged && tile < tiles
            && physicalTiles.retainTile(tile)
            && capOverlaps(tileGrid, tree, side, states[particle], tile);
        count += uint32_t(__popc(__ballot_sync(fullMask, keep)));
    }
    if (lane == 0u) capCounts[particle] = count;
}

template <typename Projector, typename Particles>
__global__ void evaluateOnce(
    const Chunk* chunks, uint32_t numChunks,
    uint32_t numParticles,
    uint32_t slotCapacity,
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side,
    gofish_angular_range_once::View angularIndex,
    const gofish_certified_remainder::SharedCountEmitRecord* records,
    const gofish_certified_remainder::ExactConeState* states,
    uint8_t* flags,
    uint64_t* provisionalKeys,
    uint64_t* provisionalOwners,
    uint32_t* exactAcceptedCounts,
    gofish_lazy_exact_cone::ExactDepthKey<Projector, Particles> depthKey,
    uint32_t* integrityError) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warpInBlock = threadIdx.x >> 5u;
    const uint32_t chunkIndex = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (chunkIndex >= numChunks) return;
    const uint32_t fullMask = 0xffffffffu;
    const Chunk chunk = chunks[chunkIndex];
    if (chunk.particle >= numParticles
        || chunk.tileBegin > chunk.tileEnd
        || chunk.tileEnd > tileGrid.x * tileGrid.y
        || chunk.slotBegin > slotCapacity
        || chunk.reservation > slotCapacity - chunk.slotBegin
        || (chunk.mode != kModeCapCompact
            && chunk.mode != kModePhysicalRange)
        || !gofish_certified_remainder::recordIntegrityValid(
            records[chunk.particle])
        || states[chunk.particle].prepared != 1u) {
        if (lane == 0u)
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        return;
    }
    __shared__ uint32_t angularWords[kWarpsPerBlock]
        [gofish_angular_range_once::kMaxWords];
    uint32_t* rowWords = angularWords[warpInBlock];
    const bool indexed = gofish_angular_range_once::buildQueryMask(
        angularIndex, states[chunk.particle], lane, rowWords);
    uint32_t capSeen = 0u;
    uint32_t accepted = 0u;
    for (uint32_t base = chunk.tileBegin; base < chunk.tileEnd; base += kWarp) {
        const uint32_t tile = base + lane;
        const bool valid = tile < chunk.tileEnd;
        const bool ranged = !indexed || (valid
            && gofish_angular_range_once::contains(rowWords, tile));
        const bool cap = ranged && valid && physicalTiles.retainTile(tile)
            && capOverlaps(tileGrid, tree, side, states[chunk.particle], tile);
        const uint32_t capMask = __ballot_sync(fullMask, cap);
        const uint32_t capRank = uint32_t(__popc(
            capMask & laneMaskBefore(lane)));
        uint32_t slot = 0u;
        bool ownsSlot = false;
        if (chunk.mode == kModeCapCompact) {
            ownsSlot = cap;
            slot = chunk.slotBegin + capSeen + capRank;
        } else if (valid) {
            ownsSlot = true;
            slot = chunk.slotBegin + (tile - chunk.tileBegin);
        }
        const bool keep = cap && gofish_lazy_exact_cone::leafAdmitted(
            tileGrid, tree, side, states, chunk.particle, tile);
        const uint32_t keepMask = __ballot_sync(fullMask, keep);
        const uint32_t keepRank = uint32_t(__popc(
            keepMask & laneMaskBefore(lane)));
        if (ownsSlot) {
            if (slot >= slotCapacity) {
                gofish_certified_remainder::callbackHardIntegrityFailure(
                    integrityError);
            } else {
                flags[slot] = keep ? 1u : 0u;
                if (keep) {
                    const uint32_t y = tile / tileGrid.x;
                    const uint32_t x = tile - y * tileGrid.x;
                    provisionalKeys[slot] = Projector::concatTileDepthKeys(
                        tile, depthKey(chunk.particle, x, y));
                    provisionalOwners[slot] =
                        (uint64_t(chunk.particle) << 32)
                        | uint64_t(accepted + keepRank);
                }
            }
        }
        capSeen += uint32_t(__popc(capMask));
        accepted += uint32_t(__popc(keepMask));
    }
    if (lane == 0u) {
        const uint32_t enumerated = chunk.mode == kModeCapCompact
            ? capSeen : chunk.tileEnd - chunk.tileBegin;
        if (enumerated != chunk.reservation) {
            gofish_certified_remainder::callbackHardIntegrityFailure(
                integrityError);
        } else {
            atomicAdd(exactAcceptedCounts + chunk.particle, accepted);
        }
    }
}

__global__ void gatherSelected(
    uint32_t count,
    const uint32_t* selected,
    const uint64_t* provisionalKeys,
    const uint64_t* provisionalOwners,
    uint64_t* compactKeys,
    uint64_t* compactOwners) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t source = selected[i];
    compactKeys[i] = provisionalKeys[source];
    compactOwners[i] = provisionalOwners[source];
}

__global__ void scatterCompacted(
    uint32_t count,
    const uint64_t* compactKeys,
    const uint64_t* compactOwners,
    const uint64_t* finalOffsets,
    uint64_t finalCapacity,
    uint64_t* finalKeys,
    uint32_t* finalIds,
    uint32_t* integrityError) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint64_t owner = compactOwners[i];
    const uint32_t particle = uint32_t(owner >> 32);
    const uint32_t rank = uint32_t(owner);
    const uint64_t begin = particle == 0u ? 0u : finalOffsets[particle - 1u];
    const uint64_t destination = begin + uint64_t(rank);
    if (destination >= finalCapacity) {
        gofish_certified_remainder::callbackHardIntegrityFailure(
            integrityError);
        return;
    }
    finalKeys[destination] = compactKeys[i];
    finalIds[destination] = particle;
}

} // namespace gofish_bounded_exact

