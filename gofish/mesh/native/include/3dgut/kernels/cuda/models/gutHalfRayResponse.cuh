// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

// Exact interval-constrained 3DGUT particle response on a finite ray segment.
#pragma once


namespace threedgut {

struct GutHalfRayClosestPoint {
    float tauLine;
    float tau;
    float3 canonicalPoint;
    bool interior;
    bool valid;
};

__device__ __forceinline__ GutHalfRayClosestPoint gutHalfRayClosestPoint(
    const float3& canonicalOrigin,
    const float3& canonicalDirection,
    float minHitDistance,
    float maxHitDistance) {
    GutHalfRayClosestPoint result;
    const float denominator = dot(canonicalDirection, canonicalDirection);
    result.valid = denominator > 0.0f && isfinite(denominator) &&
                   isfinite(minHitDistance) && maxHitDistance > minHitDistance;
    if (!result.valid) {
        result.tauLine = 0.0f;
        result.tau = 0.0f;
        result.canonicalPoint = canonicalOrigin;
        result.interior = false;
        return result;
    }
    result.tauLine = -dot(canonicalOrigin, canonicalDirection) / denominator;
    result.tau = fminf(fmaxf(result.tauLine, minHitDistance), maxHitDistance);
    result.interior = result.tauLine > minHitDistance && result.tauLine < maxHitDistance;
    result.canonicalPoint = canonicalOrigin + result.tau * canonicalDirection;
    return result;
}

// VJP for q=|a+t*b|^2 and t=clamp(-a.b/(b.b), near, far).
// Bounds are renderer state and intentionally receive no gradient. At an exact
// clamp boundary we select the active-bound subgradient (interior=false).
__device__ __forceinline__ void gutHalfRayClosestPointVjp(
    const float3& canonicalOrigin,
    const float3& canonicalDirection,
    const GutHalfRayClosestPoint& closest,
    float squaredDistanceGrad,
    float tauGrad,
    float3& canonicalOriginGrad,
    float3& canonicalDirectionGrad) {
    const float3 pointGrad = 2.0f * squaredDistanceGrad * closest.canonicalPoint;
    canonicalOriginGrad = pointGrad;
    canonicalDirectionGrad = closest.tau * pointGrad;
    const float totalTauGrad = tauGrad + dot(pointGrad, canonicalDirection);
    if (closest.interior) {
        const float denominator = dot(canonicalDirection, canonicalDirection);
        const float numerator = dot(canonicalOrigin, canonicalDirection);
        const float3 tauOriginGrad = -canonicalDirection / denominator;
        const float3 tauDirectionGrad =
            -canonicalOrigin / denominator +
            (2.0f * numerator / (denominator * denominator)) * canonicalDirection;
        canonicalOriginGrad += totalTauGrad * tauOriginGrad;
        canonicalDirectionGrad += totalTauGrad * tauDirectionGrad;
    }
}

} // namespace threedgut

