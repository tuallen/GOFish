// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace gofish_certified_support_cap {

constexpr float kPiUpper = 3.1415927410125732421875f;
constexpr double kHalfPiUpper = 0x1.921fb54442d19p+0;
constexpr double kDirectionBudget = 0x1.0p-8;

static inline __device__ bool certifiedSupportCap(
    float opacity,
    float alphaThreshold,
    const float3 relative,
    const float factor[9],
    float3& center,
    float& halfAngle) {
    auto certifyAllTiles = [&]() {
        center = make_float3(1.0f, 0.0f, 0.0f);
        halfAngle = kPiUpper;
        return true;
    };

    // CUDA fminf(0.99, NaN) is 0.99, so exact densityHit can accept a NaN
    // density.  Nonfinite opacity must reach pi before threshold rejection.
    if (!isfinite(opacity)) return certifyAllTiles();
    if (!(opacity >= alphaThreshold)) return false;
    if (!isfinite(relative.x) || !isfinite(relative.y) || !isfinite(relative.z)) {
        return certifyAllTiles();
    }

    double rowSum[3] = {0.0, 0.0, 0.0};
    double colSum[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double entry = fabs((double)factor[3 * row + col]);
            if (!isfinite(entry)) return certifyAllTiles();
            rowSum[row] = __dadd_ru(rowSum[row], entry);
            colSum[col] = __dadd_ru(colSum[col], entry);
        }
    }

    const double normOneUpper = fmax(fmax(colSum[0], colSum[1]), colSum[2]);
    const double normInfUpper = fmax(fmax(rowSum[0], rowSum[1]), rowSum[2]);
    const double inducedNormUpper = __dmul_ru(
        0.5, __dadd_ru(normOneUpper, normInfUpper));
    const double worldRadiusUpper = __dmul_ru(4.0, inducedNormUpper);
    const double rx = (double)relative.x;
    const double ry = (double)relative.y;
    const double rz = (double)relative.z;
    const double distanceLower = fmax(fmax(fabs(rx), fabs(ry)), fabs(rz));
    if (!isfinite(worldRadiusUpper) || !isfinite(distanceLower)
        || !(distanceLower > worldRadiusUpper)) {
        return certifyAllTiles();
    }

    const double ratioUpper = __ddiv_ru(worldRadiusUpper, distanceLower);
    double angleUpper = __dmul_ru(kHalfPiUpper, ratioUpper);
    angleUpper = __dadd_ru(angleUpper, kDirectionBudget);
    if (!isfinite(angleUpper) || angleUpper >= (double)kPiUpper) {
        return certifyAllTiles();
    }

    const double distance = __dsqrt_rn(rx*rx + ry*ry + rz*rz);
    if (!isfinite(distance) || !(distance > 0.0)) return certifyAllTiles();
    center = make_float3((float)(rx / distance), (float)(ry / distance),
                         (float)(rz / distance));
    halfAngle = __double2float_ru(angleUpper);
    return isfinite(halfAngle) ? true : certifyAllTiles();
}

}  // namespace gofish_certified_support_cap

