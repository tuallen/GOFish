// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/flatConeCallbacks.cuh>
#include <3dgut/kernels/cuda/renderers/flatConeKTest.cuh>

namespace gofish_flat_cone {

using ExactConeState = gofish_certified_remainder::ExactConeState;

struct CapProvider {
    const ExactConeState* states;

    inline __device__ gofish_flat_cone::AngularCapState operator()(
        uint32_t particle) const {
        const ExactConeState& state = states[particle];
        // The cosine-free cap test is evaluated by the supplied predicate
        // below. The generic angular stage enumerates all real tiles so it
        // cannot reject before that certificate.
        return gofish_flat_cone::AngularCapState{
            make_float3(0.0f, 0.0f, 0.0f),
            gofish_conservative_bound::kPi,
            state.prepared,
        };
    }
};

// K-test binding. The exact FP32 cache is prepared once with the
// state; FP32 and directed-FP64 predicates come from flatConeKTest.cuh.
struct SourceBoundKPredicate {
    const ExactConeState* states;

    inline __device__ bool operator()(
        uint32_t particle, uint32_t x, uint32_t y,
        const gofish_conservative_bound::ConservativeConeNode& leaf) const {
        (void)x;
        (void)y;
        const ExactConeState& state = states[particle];
        if (state.flatAdmitAll != 0u || state.flatCap.certified != 1u)
            return true;
        const float4 angular = gofish_conservative_bound::angularConeOf(leaf);
        const gofish_flat_cone_k_test::Input input{
            state.cache,
            state.flatFp32,
            state.rawOpacity,
            state.alphaThreshold,
            state.flatCap.certified,
        };
        const gofish_flat_cone_k_test::TileRayCone cone{
            0u,
            make_float3(angular.x, angular.y, angular.z),
            leaf.chordRadius,
            leaf.directionMaxAbs,
            leaf.directionOperandsCertified,
        };
        if (!gofish_flat_cone_k_test::capMayOverlap(state.flatCap, cone))
            return false;
        return gofish_flat_cone_k_test::testTile(input, cone)
            == gofish_flat_cone_k_test::TileDecision::Retain;
    }
};

static inline __device__ gofish_flat_cone::OneStreamBuffers streamView(
    const gofish_certified_remainder::OneStreamBuffers& source) {
    return gofish_flat_cone::OneStreamBuffers{
        source.counts,
        source.offsets,
        source.keys,
        source.particleIds,
        source.capacity,
        source.integrityError,
    };
}

} // namespace gofish_flat_cone

