// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gofish_sof_sort {

// Finite, non-negative sort-depth fallbacks of the SOF hierarchical sort.

static inline __device__ float finiteNonnegativeDepth(float candidate, float finiteMaximum) {
    if (isfinite(candidate) && candidate >= 0.0f) return candidate;
    return isfinite(finiteMaximum) && finiteMaximum >= 0.0f
        ? finiteMaximum : 0.0f;
}

static inline __device__ float radialDepthFallback(
    const float3& rayOrigin,
    const float3& particlePosition,
    float finiteMaximum) {
    const float dx = particlePosition.x - rayOrigin.x;
    const float dy = particlePosition.y - rayOrigin.y;
    const float dz = particlePosition.z - rayOrigin.z;
    const float radial = sqrtf(dx * dx + dy * dy + dz * dz);
    return finiteNonnegativeDepth(radial, finiteMaximum);
}

static inline __device__ float takeDepthFallback(float finiteNonnegativeFallback, float finiteMaximum) {
    return finiteNonnegativeDepth(finiteNonnegativeFallback, finiteMaximum);
}

} // namespace gofish_sof_sort
