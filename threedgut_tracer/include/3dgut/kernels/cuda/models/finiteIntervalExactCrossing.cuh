// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cmath>

namespace threedgut {

enum class FiniteIntervalCrossingMode : unsigned char {
    NoCrossing = 0,
    Interior = 1,
    NearBoundary = 2,
    FarBoundary = 3,
    Tangent = 4,
};

struct FiniteIntervalCrossingResult {
    float depth;
    float rate;
    float discriminant;
    float nearRoot;
    FiniteIntervalCrossingMode mode;
    bool valid;
};

static __forceinline__ __host__ __device__ FiniteIntervalCrossingResult
noFiniteIntervalCrossing() {
    return {-1.0f, 0.0f, NAN, NAN,
            FiniteIntervalCrossingMode::NoCrossing, false};
}

static __forceinline__ __host__ __device__ float finiteIntervalCrossingAlpha(
    float AA, float BB, float CC, float density, float t, float cap) {
    const float q = fmaxf(AA * t * t + BB * t + CC, 0.0f);
    return fminf(cap, density * expf(-0.5f * q));
}

// Exact fixed-topology quadratic crossing on the same closed interval used by
// the finite-ray peak response.  Interval bounds are renderer state/no_diff.
static __forceinline__ __host__ __device__ FiniteIntervalCrossingResult
finiteIntervalExactCrossing(
    float AA, float BB, float CC, float density,
    float transmittanceBefore, float minHitDistance, float maxHitDistance,
    float threshold = 0.5f, float cap = 0.99f) {
    const bool finite = isfinite(AA) && isfinite(BB) && isfinite(CC) &&
        isfinite(density) && isfinite(transmittanceBefore) &&
        isfinite(minHitDistance) && isfinite(maxHitDistance) &&
        isfinite(threshold) && isfinite(cap);
    if (!finite || !(AA > 0.0f) || !(density > 0.0f) ||
        !(maxHitDistance > minHitDistance) ||
        !(transmittanceBefore > threshold) || !(transmittanceBefore <= 1.0f) ||
        !(threshold > 0.0f) || !(cap > 0.0f)) {
        return noFiniteIntervalCrossing();
    }

    const float tLine = -BB / (2.0f * AA);
    const float tPeak = fminf(fmaxf(tLine, minHitDistance), maxHitDistance);
    const float peakAlpha = finiteIntervalCrossingAlpha(
        AA, BB, CC, density, tPeak, cap);
    const float requiredAlpha = 1.0f - threshold / transmittanceBefore;
    // The strict cap inequality keeps the active root on the smooth uncapped
    // branch.  For h=0.5, T<=1 and cap=0.99 this always holds for a crossing.
    if (!(requiredAlpha > 0.0f) || !(requiredAlpha < cap) ||
        !(peakAlpha >= requiredAlpha)) {
        return noFiniteIntervalCrossing();
    }
    const float responseLevel = requiredAlpha / density;
    if (!(responseLevel > 0.0f) || !(responseLevel <= 1.0f) ||
        !isfinite(responseLevel)) {
        return noFiniteIntervalCrossing();
    }
    const float con = CC + 2.0f * logf(responseLevel);
    const float discriminant = BB * BB - 4.0f * AA * con;
    if (!(discriminant >= 0.0f) || !isfinite(discriminant)) {
        return noFiniteIntervalCrossing();
    }
    const float root = sqrtf(discriminant);
    const float nearRoot = (-BB - root) / (2.0f * AA);
    if (!isfinite(nearRoot)) {
        return noFiniteIntervalCrossing();
    }

    if (nearRoot <= minHitDistance) {
        const float entryAlpha = finiteIntervalCrossingAlpha(
            AA, BB, CC, density, minHitDistance, cap);
        if (!(entryAlpha >= requiredAlpha)) {
            return noFiniteIntervalCrossing();
        }
        return {minHitDistance, 0.0f, discriminant, nearRoot,
                FiniteIntervalCrossingMode::NearBoundary, true};
    }
    if (nearRoot >= maxHitDistance) {
        const float exitAlpha = finiteIntervalCrossingAlpha(
            AA, BB, CC, density, maxHitDistance, cap);
        if (!(exitAlpha >= requiredAlpha)) {
            return noFiniteIntervalCrossing();
        }
        return {maxHitDistance, 0.0f, discriminant, nearRoot,
                FiniteIntervalCrossingMode::FarBoundary, true};
    }

    const float rate = -0.5f * (transmittanceBefore - threshold) *
        (2.0f * AA * nearRoot + BB);
    if (discriminant == 0.0f || !isfinite(rate) || !(rate > 0.0f)) {
        return {nearRoot, 0.0f, discriminant, nearRoot,
                FiniteIntervalCrossingMode::Tangent, true};
    }
    return {nearRoot, rate, discriminant, nearRoot,
            FiniteIntervalCrossingMode::Interior, true};
}

} // namespace threedgut
