// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderOneStream.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>
#include <3dgut/kernels/cuda/renderers/conservativeOpacityBound.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedSupportCap.cuh>
#include <3dgut/kernels/cuda/renderers/flatConeKTest.cuh>
#include <3dgut/renderer/renderParameters.h>

namespace gofish_certified_remainder {

struct OneStreamBuffers {
    uint32_t* counts;       // per-particle count, zeroed before count
    const uint64_t* offsets; // lossless inclusive scan of counts
    uint64_t* keys;
    uint32_t* particleIds;
    uint64_t capacity;
    uint32_t* integrityError;
};

struct ExactConeState {
    gofish_whitened_cone::FiniteEllipsoidConeCache cache;
    float3 gaussianCenter;
    float gaussianHalfAngle;
    float rawOpacity;
    float alphaThreshold;
    // Prepared means the renderer inputs were finite and structurally valid.
    // It is deliberately independent of cache.certified: an uncertified cache
    // makes both cone predicates fail open and therefore retains every node.
    uint32_t prepared;
    gofish_fp32_certificate::Cache flatFp32;
    gofish_flat_cone_k_test::FlatConeCap flatCap;
    uint32_t flatAdmitAll;
    // Preserve the three-way certified relation.  Admission may fail open on
    // Uncertain, but screen-size pruning may treat only proven Inside as
    // camera-enclosing.
    uint32_t cameraRelation;
};

// FlatConeCap tile predicate shared by the count and emit paths. Every
// uncertainty and every malformed input keeps the tile.
static inline __device__ bool flatConeTileCapRetains(
    uint32_t tile, uint32_t particle, uint32_t numParticles,
    tcnn::uvec2 tileGrid, uint32_t paddedSide,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    const ExactConeState* states,
    uint8_t* failOpen = nullptr, uint8_t* integrityFailure = nullptr) {
    if (failOpen != nullptr) *failOpen = 1u;
    if (integrityFailure != nullptr) *integrityFailure = 0u;
    if (tree == nullptr || states == nullptr || particle >= numParticles
        || tileGrid.x == 0u || tileGrid.y == 0u || paddedSide == 0u
        || tileGrid.x > paddedSide || tileGrid.y > paddedSide
        || tile >= tileGrid.x * tileGrid.y) {
        if (integrityFailure != nullptr) *integrityFailure = 1u;
        return true;
    }
    const ExactConeState& state = states[particle];
    const auto outside = gofish_flat_cone_k_test::CameraRelation::Outside;
    if (state.prepared != 1u
        || state.cameraRelation != static_cast<uint32_t>(outside)
        || state.flatAdmitAll != 0u || state.flatCap.certified != 1u) {
        return true;
    }
    const uint32_t x = tile % tileGrid.x;
    const uint32_t y = tile / tileGrid.x;
    const uint32_t index = gofish_conservative_bound::levelOffset(
        paddedSide, 0u) + y * paddedSide + x;
    const auto node = tree[index];
    const float4 angular = gofish_conservative_bound::angularConeOf(node);
    const gofish_flat_cone_k_test::TileRayCone cone{
        tile,
        make_float3(angular.x, angular.y, angular.z),
        node.chordRadius,
        node.directionMaxAbs,
        node.directionOperandsCertified,
    };
    if (failOpen != nullptr) *failOpen = 0u;
    return gofish_flat_cone_k_test::capMayOverlap(state.flatCap, cone);
}

struct FlatConeTileCapView {
    uint32_t numParticles;
    tcnn::uvec2 tileGrid;
    uint32_t paddedSide;
    const gofish_conservative_bound::ConservativeConeNode* tree;
    const ExactConeState* states;

    inline __device__ bool retain(
        uint32_t particle, uint32_t x, uint32_t y) const {
        return flatConeTileCapRetains(
            y * tileGrid.x + x, particle, numParticles, tileGrid,
            paddedSide, tree, states);
    }
};

static inline __device__ void callbackHardIntegrityFailure(
    uint32_t* __restrict__ integrityError) {
    // The integrity channel is mandatory for this adapter.  A missing channel
    // cannot be reported in-band, so fail the kernel instead of continuing.
    if (integrityError == nullptr) {
        asm volatile("trap;");
        return;
    }
    hardIntegrityFailure(integrityError);
}

static inline __device__ bool exactConeFinite3(const tcnn::vec3& value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

template <typename Projector, typename Particles>
__global__ void prepareExactConeStates(
    uint32_t numParticles,
    const tcnn::vec3 cameraWorldOrigin,
    uint32_t centralGlobalShutter,
    threedgut::MemoryHandles parameters,
    ExactConeState* __restrict__ states,
    uint32_t* __restrict__ integrityError) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    if (states == nullptr || integrityError == nullptr
        || centralGlobalShutter > 1u || !exactConeFinite3(cameraWorldOrigin)
        || !(Projector::AlphaThreshold > 0.0f)
        || !isfinite(Projector::AlphaThreshold)) {
        callbackHardIntegrityFailure(integrityError);
        return;
    }

    ExactConeState state{};
    state.gaussianCenter = make_float3(0.0f, 0.0f, 0.0f);
    state.gaussianHalfAngle = 0.0f;
    state.alphaThreshold = Projector::AlphaThreshold;
    state.prepared = 0u;
    state.cameraRelation = static_cast<uint32_t>(
        gofish_flat_cone_k_test::CameraRelation::Uncertain);

    Particles particles;
    particles.initializeDensity(parameters);
    const auto particleParameters = particles.fetchDensityParameters(particle);
    const tcnn::vec3 position = particles.position(particleParameters);
    const float rawOpacity = particles.opacity(particleParameters);
    bool finiteInput = exactConeFinite3(position) && isfinite(rawOpacity);
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        finiteInput = finiteInput
            && isfinite(particleParameters.inverseFactor[row].x)
            && isfinite(particleParameters.inverseFactor[row].y)
            && isfinite(particleParameters.inverseFactor[row].z);
    }
    if (!finiteInput) {
        states[particle] = state;
        callbackHardIntegrityFailure(integrityError);
        return;
    }

    // Cap constructor; uses the same zero angular
    // center/half-angle plus raw renderer opacity used by its all-level path.
    state.cache = Projector::finiteEllipsoidConeCache(
        particles, particleParameters, cameraWorldOrigin,
        centralGlobalShutter != 0u);
    state.rawOpacity = rawOpacity;
    state.flatFp32 = gofish_fp32_certificate::makeCache(state.cache);
    state.flatAdmitAll = 1u;
    const auto cameraRelation =
        gofish_flat_cone_k_test::cameraRelation(state.cache);
    state.cameraRelation = static_cast<uint32_t>(cameraRelation);
    const tcnn::vec3 relativeTcnn = position - cameraWorldOrigin;
    float factor[9];
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        const tcnn::vec3 value = particles.factorColumn(
            particleParameters, col);
        factor[col] = value.x;
        factor[3 + col] = value.y;
        factor[6 + col] = value.z;
    }
    state.flatCap = gofish_flat_cone_k_test::makeFlatConeCap(
        rawOpacity, Projector::AlphaThreshold,
        make_float3(relativeTcnn.x, relativeTcnn.y, relativeTcnn.z), factor);
    state.flatAdmitAll =
        (cameraRelation != gofish_flat_cone_k_test::CameraRelation::Outside
         || state.flatCap.certified != 1u || state.flatCap.allTiles != 0u)
        ? 1u : 0u;
    // Source-bound capMayOverlap owns angular rejection. Camera-inside,
    // uncertain, and uncertified states bypass both cap and K rejection.
    state.gaussianCenter = make_float3(0.0f, 0.0f, 0.0f);
    state.gaussianHalfAngle = gofish_conservative_bound::kPi;
    // cache.certified=false is not an error. Both certified rejection
    // predicates return false, so traversalNodeAdmitted retains the node.
    state.prepared = 1u;
    states[particle] = state;
}

__global__ inline void initializeCertifiedRemainderCounts(
    uint32_t numParticles,
    uint32_t* __restrict__ counts) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < numParticles) counts[particle] = 0u;
}

static inline __device__ bool reservation(
    const OneStreamBuffers& stream,
    uint32_t particle,
    uint64_t& begin,
    uint64_t& end) {
    if (stream.offsets == nullptr || stream.keys == nullptr
        || stream.particleIds == nullptr) return false;
    begin = particle == 0u ? 0u : stream.offsets[particle - 1u];
    end = stream.offsets[particle];
    return begin <= end && end <= stream.capacity;
}

template <typename Projector>
struct CertifiedEllipseCountCallback {
    tcnn::uvec2 tileGrid;
    OneStreamBuffers stream;

    static inline __device__ bool admitted(
        const DilatedProjectedEllipse& ellipse, uint32_t x, uint32_t y,
        const typename Projector::BoundingBox2D& bbox) {
        return Projector::nativeTileAdmissionBase(
            int(x), int(y), true, bbox,
            tcnn::vec4{ellipse.conicOpacity.x, ellipse.conicOpacity.y,
                       ellipse.conicOpacity.z, ellipse.conicOpacity.w},
            tcnn::vec2{ellipse.center.x, ellipse.center.y},
            ellipse.maxConicOpacityPower);
    }

    inline __device__ void operator()(
        uint32_t particle,
        const DilatedProjectedEllipse& ellipse) const {
        if (stream.integrityError == nullptr || stream.counts == nullptr
            || tileGrid.x == 0u || tileGrid.y == 0u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const auto bbox = Projector::computeTileSpaceBBox(
            tileGrid,
            tcnn::vec2{ellipse.center.x, ellipse.center.y},
            tcnn::vec2{ellipse.extent.x, ellipse.extent.y});
        uint64_t count = 0u;
        for (uint32_t y = bbox.min.y; y < bbox.max.y; ++y) {
            for (uint32_t x = bbox.min.x; x < bbox.max.x; ++x) {
                count += admitted(ellipse, x, y, bbox) ? 1u : 0u;
            }
        }
        if (count > 0xffffffffULL) {
            stream.counts[particle] = 0u;
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        stream.counts[particle] = static_cast<uint32_t>(count);
    }
};

template <typename Projector, typename DepthKeyProvider>
struct CertifiedEllipseEmitCallback {
    tcnn::uvec2 tileGrid;
    OneStreamBuffers stream;
    DepthKeyProvider depthKey;

    inline __device__ void operator()(
        uint32_t particle,
        uint32_t existingParticleId,
        const DilatedProjectedEllipse& ellipse) const {
        if (stream.integrityError == nullptr || tileGrid.x == 0u
            || tileGrid.y == 0u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        uint64_t write = 0u, end = 0u;
        if (!reservation(stream, particle, write, end)) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const auto bbox = Projector::computeTileSpaceBBox(
            tileGrid,
            tcnn::vec2{ellipse.center.x, ellipse.center.y},
            tcnn::vec2{ellipse.extent.x, ellipse.extent.y});
        for (uint32_t y = bbox.min.y; y < bbox.max.y; ++y) {
            for (uint32_t x = bbox.min.x; x < bbox.max.x; ++x) {
                if (!CertifiedEllipseCountCallback<Projector>::admitted(
                        ellipse, x, y, bbox)) continue;
                if (write >= end) {
                    callbackHardIntegrityFailure(stream.integrityError);
                    return;
                }
                stream.keys[write] = Projector::concatTileDepthKeys(
                    y * tileGrid.x + x,
                    depthKey(static_cast<int>(x), static_cast<int>(y)));
                stream.particleIds[write] = existingParticleId;
                ++write;
            }
        }
        if (write != end) callbackHardIntegrityFailure(stream.integrityError);
    }
};

template <typename Projector>
struct CertifiedRectangleCountCallback {
    tcnn::uvec2 tileGrid;
    gofish_physical_candidate::PhysicalTileView physicalTiles;
    float2 physicalSampleOffsetLower;
    float2 physicalSampleOffsetUpper;
    OneStreamBuffers stream;

    struct AxisTileRange {
        uint32_t first;
        uint32_t last;
        bool nonempty;
    };

    static inline __device__ AxisTileRange discreteAxisRange(
        uint32_t tileCount, uint32_t blockSize,
        float sampleOffsetLower, float sampleOffsetUpper,
        float center, float halfExtent) {
        if (tileCount == 0u || blockSize == 0u)
            return AxisTileRange{0u, 0u, false};
        if (!isfinite(center) || !isfinite(halfExtent)
            || !isfinite(sampleOffsetLower) || !isfinite(sampleOffsetUpper)
            || sampleOffsetLower > sampleOffsetUpper
            || !(halfExtent >= 0.0f)
            || tileCount > 0x7fffffffu / blockSize)
            return AxisTileRange{0u, tileCount, true};
        // Pixel i samples this projection chart at i+o.  Enclose the exact
        // calibration-derived o in [sampleOffsetLower,sampleOffsetUpper].
        const float lower = __fsub_rd(
            __fsub_rd(center, halfExtent), sampleOffsetUpper);
        const float upper = __fsub_ru(
            __fadd_ru(center, halfExtent), sampleOffsetLower);
        const float firstFloat = ceilf(lower);
        const float lastFloat = floorf(upper);
        const int totalPixels = static_cast<int>(tileCount * blockSize);
        if (!isfinite(firstFloat) || !isfinite(lastFloat))
            return AxisTileRange{0u, tileCount, true};
        if (lastFloat < 0.0f || firstFloat >= float(totalPixels)
            || firstFloat > lastFloat)
            return AxisTileRange{0u, 0u, false};
        const int firstPixel = max(0, static_cast<int>(firstFloat));
        const int lastPixel = min(
            totalPixels - 1, static_cast<int>(lastFloat));
        return AxisTileRange{
            static_cast<uint32_t>(firstPixel) / blockSize,
            static_cast<uint32_t>(lastPixel) / blockSize + 1u,
            true};
    }

    static inline __device__ typename Projector::BoundingBox2D bbox(
        tcnn::uvec2 tileGrid,
        const CertifiedProjectedRectangle& rectangle,
        float2 sampleOffsetLower, float2 sampleOffsetUpper) {
        const AxisTileRange x = discreteAxisRange(
            tileGrid.x, threedgut::GUTParameters::Tiling::BlockX,
            sampleOffsetLower.x, sampleOffsetUpper.x,
            rectangle.center.x, rectangle.halfExtent.x);
        const AxisTileRange y = discreteAxisRange(
            tileGrid.y, threedgut::GUTParameters::Tiling::BlockY,
            sampleOffsetLower.y, sampleOffsetUpper.y,
            rectangle.center.y, rectangle.halfExtent.y);
        if (!x.nonempty || !y.nonempty)
            return typename Projector::BoundingBox2D{{0u, 0u}, {0u, 0u}};
        return typename Projector::BoundingBox2D{
            {x.first, y.first}, {x.last, y.last}};
    }

    inline __device__ void operator()(
        uint32_t particle,
        const CertifiedProjectedRectangle& rectangle) const {
        if (stream.integrityError == nullptr || stream.counts == nullptr
            || tileGrid.x == 0u || tileGrid.y == 0u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const auto bounds = bbox(
            tileGrid, rectangle,
            physicalSampleOffsetLower, physicalSampleOffsetUpper);
        uint64_t count = 0u;
        for (uint32_t y = bounds.min.y; y < bounds.max.y; ++y)
            for (uint32_t x = bounds.min.x; x < bounds.max.x; ++x)
                count += physicalTiles.retainXY(x, y) ? 1u : 0u;
        if (count > 0xffffffffULL) {
            stream.counts[particle] = 0u;
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        stream.counts[particle] = static_cast<uint32_t>(count);
    }
};

template <typename Projector, typename DepthKeyProvider>
struct CertifiedRectangleEmitCallback {
    tcnn::uvec2 tileGrid;
    gofish_physical_candidate::PhysicalTileView physicalTiles;
    float2 physicalSampleOffsetLower;
    float2 physicalSampleOffsetUpper;
    OneStreamBuffers stream;
    DepthKeyProvider depthKey;

    inline __device__ void operator()(
        uint32_t particle,
        uint32_t existingParticleId,
        const CertifiedProjectedRectangle& rectangle) const {
        if (stream.integrityError == nullptr || tileGrid.x == 0u
            || tileGrid.y == 0u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        uint64_t write = 0u, end = 0u;
        if (!reservation(stream, particle, write, end)) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const auto bounds =
            CertifiedRectangleCountCallback<Projector>::bbox(
                tileGrid, rectangle,
                physicalSampleOffsetLower, physicalSampleOffsetUpper);
        for (uint32_t y = bounds.min.y; y < bounds.max.y; ++y) {
            for (uint32_t x = bounds.min.x; x < bounds.max.x; ++x) {
                if (!physicalTiles.retainXY(x, y)) continue;
                if (write >= end) {
                    callbackHardIntegrityFailure(stream.integrityError);
                    return;
                }
                stream.keys[write] = Projector::concatTileDepthKeys(
                    y * tileGrid.x + x,
                    depthKey(static_cast<int>(x), static_cast<int>(y)));
                stream.particleIds[write] = existingParticleId;
                ++write;
            }
        }
        if (write != end) callbackHardIntegrityFailure(stream.integrityError);
    }
};

template <typename Projector>
struct ExactConeCountCallback {
    tcnn::uvec2 tileGrid;
    const gofish_conservative_bound::ConservativeConeNode* tree;
    uint32_t side;
    uint32_t levels;
    const ExactConeState* states;
    OneStreamBuffers stream;

    inline __device__ void operator()(uint32_t particle) const {
        if (stream.integrityError == nullptr || stream.counts == nullptr
            || states == nullptr || tree == nullptr || tileGrid.x == 0u
            || tileGrid.y == 0u || levels > 12u || side != (1u << levels)
            || tileGrid.x > side || tileGrid.y > side
            || states[particle].prepared != 1u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const ExactConeState& state = states[particle];
        uint32_t stack[gofish_conservative_bound::kTraversalStack];
        uint32_t top = 0u;
        stack[top++] = levels << 24;
        uint64_t count = 0u;
        while (top != 0u) {
            const uint32_t packed = stack[--top];
            const uint32_t level = packed >> 24;
            const uint32_t y = (packed >> 12) & 0xFFFu;
            const uint32_t x = packed & 0xFFFu;
            const uint32_t levelSide = side >> level;
            const uint32_t index = gofish_conservative_bound::levelOffset(
                side, level) + y * levelSide + x;
            if (!gofish_conservative_bound::traversalNodeAdmitted(
                    state.gaussianCenter, state.gaussianHalfAngle,
                    tree[index], level, state.cache,
                    state.rawOpacity, state.alphaThreshold)) continue;
            if (level == 0u) {
                count += x < tileGrid.x && y < tileGrid.y ? 1u : 0u;
                continue;
            }
            if (top + 4u > gofish_conservative_bound::kTraversalStack) {
                callbackHardIntegrityFailure(stream.integrityError);
                return;
            }
            const uint32_t child = level - 1u;
            stack[top++] = (child << 24) | ((2u*y) << 12) | (2u*x);
            stack[top++] = (child << 24) | ((2u*y) << 12) | (2u*x+1u);
            stack[top++] = (child << 24) | ((2u*y+1u) << 12) | (2u*x);
            stack[top++] = (child << 24) | ((2u*y+1u) << 12) | (2u*x+1u);
        }
        if (count > 0xffffffffULL) {
            stream.counts[particle] = 0u;
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        stream.counts[particle] = static_cast<uint32_t>(count);
    }
};

template <typename Projector, typename DepthKeyProvider>
struct ExactConeEmitCallback {
    tcnn::uvec2 tileGrid;
    const gofish_conservative_bound::ConservativeConeNode* tree;
    uint32_t side;
    uint32_t levels;
    const ExactConeState* states;
    OneStreamBuffers stream;
    DepthKeyProvider depthKey;

    inline __device__ void operator()(
        uint32_t particle,
        uint32_t existingParticleId) const {
        if (stream.integrityError == nullptr || tileGrid.x == 0u
            || tileGrid.y == 0u || states == nullptr || tree == nullptr
            || levels > 12u
            || side != (1u << levels) || tileGrid.x > side
            || tileGrid.y > side
            || states[particle].prepared != 1u) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        uint64_t write = 0u, end = 0u;
        if (!reservation(stream, particle, write, end)) {
            callbackHardIntegrityFailure(stream.integrityError);
            return;
        }
        const ExactConeState& state = states[particle];
        uint32_t stack[gofish_conservative_bound::kTraversalStack];
        uint32_t top = 0u;
        stack[top++] = levels << 24;
        while (top != 0u) {
            const uint32_t packed = stack[--top];
            const uint32_t level = packed >> 24;
            const uint32_t y = (packed >> 12) & 0xFFFu;
            const uint32_t x = packed & 0xFFFu;
            const uint32_t levelSide = side >> level;
            const uint32_t index = gofish_conservative_bound::levelOffset(
                side, level) + y * levelSide + x;
            if (!gofish_conservative_bound::traversalNodeAdmitted(
                    state.gaussianCenter, state.gaussianHalfAngle,
                    tree[index], level, state.cache,
                    state.rawOpacity, state.alphaThreshold)) continue;
            if (level == 0u) {
                if (x >= tileGrid.x || y >= tileGrid.y) continue;
                if (write >= end) {
                    callbackHardIntegrityFailure(stream.integrityError);
                    return;
                }
                stream.keys[write] = Projector::concatTileDepthKeys(
                    y * tileGrid.x + x,
                    depthKey(static_cast<int>(x), static_cast<int>(y)));
                stream.particleIds[write] = existingParticleId;
                ++write;
                continue;
            }
            if (top + 4u > gofish_conservative_bound::kTraversalStack) {
                callbackHardIntegrityFailure(stream.integrityError);
                return;
            }
            const uint32_t child = level - 1u;
            stack[top++] = (child << 24) | ((2u*y) << 12) | (2u*x);
            stack[top++] = (child << 24) | ((2u*y) << 12) | (2u*x+1u);
            stack[top++] = (child << 24) | ((2u*y+1u) << 12) | (2u*x);
            stack[top++] = (child << 24) | ((2u*y+1u) << 12) | (2u*x+1u);
        }
        if (write != end) callbackHardIntegrityFailure(stream.integrityError);
    }
};

} // namespace gofish_certified_remainder

