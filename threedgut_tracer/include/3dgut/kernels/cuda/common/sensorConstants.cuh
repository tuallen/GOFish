// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace threedgut {

// Inherited from 3DGS/3DGUT's near-plane test.  Keep one definition for
// projection and for losses whose coordinate is only defined beyond the
// sensor near plane.
static constexpr float kParticleMinSensorZ = 0.2f;

struct SofDistortionCoordinate {
    float inverseDepth;
    float inverseDepthSquared;
    bool directDepthGradientActive;
};

// One implementation shared by forward accumulation and backward replay.
// `distanceSquared` is supplied by each path so values above the near plane
// preserve that path's established arithmetic exactly.
#if defined(__CUDACC__)
__host__ __device__ __forceinline__
#else
inline
#endif
SofDistortionCoordinate sofDistortionCoordinate(
    float distance, float distanceSquared) {
    const bool active = distance > kParticleMinSensorZ;
    const float boundedDistance = active ? distance : kParticleMinSensorZ;
    const float boundedDistanceSquared = active
        ? distanceSquared
        : (kParticleMinSensorZ * kParticleMinSensorZ);
    return {
        1.0f / boundedDistance,
        1.0f / boundedDistanceSquared,
        active,
    };
}

} // namespace threedgut
