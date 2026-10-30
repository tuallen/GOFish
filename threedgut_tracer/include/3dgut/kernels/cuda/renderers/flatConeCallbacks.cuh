// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderOneStream.cuh>
#include <3dgut/kernels/cuda/renderers/conservativeOpacityBound.cuh>

namespace gofish_flat_cone {

// Supplied by the representation/camera integration. The cap must contain
// every finite support direction for this particle/view. Camera-inside support
// is represented by halfAngle==pi and therefore enumerates every real tile.
struct AngularCapState {
    float3 center;
    float halfAngle;
    uint32_t prepared;
};

// Lossless one-stream ABI. Counts remain checked uint32 values because one
// calibrated tile grid is bounded by the renderer; the scan and reservations
// are uint64 so sums cannot wrap.
struct OneStreamBuffers {
    uint32_t* counts;
    const uint64_t* offsets;
    uint64_t* keys;
    uint32_t* particleIds;
    uint64_t capacity;
    uint32_t* integrityError;
};

static inline __device__ void hardFailure(
    uint32_t* __restrict__ integrityError) {
    if (integrityError == nullptr) {
        asm volatile("trap;");
        return;
    }
    gofish_certified_remainder::hardIntegrityFailure(integrityError);
}

static inline __device__ bool finiteCap(const AngularCapState& cap) {
    return cap.prepared == 1u
        && isfinite(cap.center.x) && isfinite(cap.center.y)
        && isfinite(cap.center.z) && isfinite(cap.halfAngle)
        && cap.halfAngle >= 0.0f
        && cap.halfAngle <= gofish_conservative_bound::kPi;
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

// CertifiedKPredicate contract:
//   bool operator()(uint32_t particle, uint32_t tileX, uint32_t tileY,
//                   const ConservativeConeNode& leaf) const;
// true means retain/admit. Any unavailable or uncertified K proof MUST return
// true. This adapter deliberately contains no K arithmetic.
template <typename AngularCapProvider, typename CertifiedKPredicate>
struct FlatConeAdmission {
    tcnn::uvec2 tileGrid;
    const gofish_conservative_bound::ConservativeConeNode* leaves;
    uint32_t leafSide;
    AngularCapProvider capForParticle;
    CertifiedKPredicate certifiedKAdmits;

    inline __device__ bool structurallyValid(uint32_t particle) const {
        return leaves != nullptr && leafSide != 0u
            && leafSide <= 4096u
            && tileGrid.x != 0u && tileGrid.y != 0u
            && tileGrid.x <= leafSide && tileGrid.y <= leafSide
            && finiteCap(capForParticle(particle));
    }

    inline __device__ bool admitted(
        uint32_t particle, uint32_t x, uint32_t y) const {
        const auto& leaf = leaves[y * leafSide + x];
        const AngularCapState cap = capForParticle(particle);
        if (!gofish_conservative_bound::overlap(
                cap.center, cap.halfAngle, leaf)) return false;
        return certifiedKAdmits(particle, x, y, leaf);
    }
};

template <typename AngularCapProvider, typename CertifiedKPredicate>
struct FlatConeCountCallback {
    FlatConeAdmission<AngularCapProvider, CertifiedKPredicate> admission;
    OneStreamBuffers stream;

    inline __device__ void operator()(uint32_t particle) const {
        if (stream.integrityError == nullptr || stream.counts == nullptr
            || !admission.structurallyValid(particle)) {
            hardFailure(stream.integrityError);
            return;
        }
        uint64_t count = 0u;
        // One canonical order is shared with emit: y-major, then x-major.
        for (uint32_t y = 0u; y < admission.tileGrid.y; ++y) {
            for (uint32_t x = 0u; x < admission.tileGrid.x; ++x) {
                if (!admission.admitted(particle, x, y)) continue;
                if (count == 0xffffffffffffffffULL) {
                    stream.counts[particle] = 0u;
                    hardFailure(stream.integrityError);
                    return;
                }
                ++count;
            }
        }
        if (count > 0xffffffffULL) {
            stream.counts[particle] = 0u;
            hardFailure(stream.integrityError);
            return;
        }
        stream.counts[particle] = static_cast<uint32_t>(count);
    }
};

template <typename Projector, typename DepthKeyProvider,
          typename AngularCapProvider, typename CertifiedKPredicate>
struct FlatConeEmitCallback {
    FlatConeAdmission<AngularCapProvider, CertifiedKPredicate> admission;
    OneStreamBuffers stream;
    DepthKeyProvider depthKey;

    inline __device__ void operator()(
        uint32_t particle, uint32_t existingParticleId) const {
        if (stream.integrityError == nullptr
            || !admission.structurallyValid(particle)) {
            hardFailure(stream.integrityError);
            return;
        }
        uint64_t write = 0u;
        uint64_t end = 0u;
        if (!reservation(stream, particle, write, end)) {
            hardFailure(stream.integrityError);
            return;
        }
        for (uint32_t y = 0u; y < admission.tileGrid.y; ++y) {
            for (uint32_t x = 0u; x < admission.tileGrid.x; ++x) {
                if (!admission.admitted(particle, x, y)) continue;
                if (write >= end) {
                    hardFailure(stream.integrityError);
                    return;
                }
                stream.keys[write] = Projector::concatTileDepthKeys(
                    y * admission.tileGrid.x + x,
                    depthKey(static_cast<int>(x), static_cast<int>(y)));
                stream.particleIds[write] = existingParticleId;
                ++write;
            }
        }
        if (write != end) hardFailure(stream.integrityError);
    }
};

} // namespace gofish_flat_cone

