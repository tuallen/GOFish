// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

// The Slang forward is
//   alpha = min(GAUSSIAN_PARTICLE_MAX_ALPHA, density * response).
// Its derivative with respect to density/response is therefore active only on
// the strict, unclamped branch.

namespace threedgut {

static inline __device__ float alphaCapBackwardMask(float unclampedAlpha, float maxAlpha) {
    // Strict '<' selects the subgradient at the min() kink.
    return unclampedAlpha < maxAlpha ? 1.0f : 0.0f;
}

} // namespace threedgut
