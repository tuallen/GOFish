// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace gofish_certified_opacity_radius {

constexpr float kFixedRadius = 3.0f;
constexpr float kAlphaThreshold = 0x1.010102p-8f;
constexpr float kAlphaCap = 0x1.fae148p-1f;
constexpr float kKernelMinimum = 0x1.724746p-7f;
constexpr float kExpAbsoluteError = 0x1p-22f;
constexpr float kProductAbsoluteError = 0x1p-24f;
constexpr float kLogAbsoluteError = 0x1.8p-20f;
constexpr float kExpNegFourPointFiveUpper = 0x1.6c0506p-7f;
constexpr double kGammaFiveUpper = 0x1.40000640001f5p-22;
constexpr double kOneMinusGammaFiveLower = 0x1.fffff5ffffcdfp-1;
constexpr double kDotUnderflowAbsoluteUpper = 0x1.8p-125;

enum class RadiusReason : std::uint32_t {
    FixedFeatureDisabled = 0,
    FixedIntrinsicAbiUnknown = 1,
    FixedProfileMismatch = 2,
    FixedOpacityUncertain = 3,
    FixedFormulaAtLeastThree = 4,
    FixedArithmeticUncertain = 5,
    AdaptiveCertified = 6,
};

struct RadiusResult {
    float radius;
    RadiusReason reason;
    std::uint32_t adaptiveCertified;
};

static inline __device__ RadiusResult fixed(RadiusReason reason) {
    return RadiusResult{kFixedRadius, reason, 0u};
}

static inline __device__ bool exactProfile(
    int kernelDegree, float alphaThreshold, float alphaCap,
    float kernelMinimum) {
    return kernelDegree == 2
        && __float_as_uint(alphaThreshold) == 0x3b808081u
        && __float_as_uint(alphaCap) == 0x3f7d70a4u
        && __float_as_uint(kernelMinimum) == 0x3c3923a3u;
}

// Callers pass the literal densityHit profile and use the returned radius for
// every downstream support calculation.
static inline __device__ RadiusResult certifiedRadius(
    float opacity, int kernelDegree, float alphaThreshold,
    float alphaCap, float kernelMinimum) {
    if (!exactProfile(kernelDegree, alphaThreshold, alphaCap, kernelMinimum))
        return fixed(RadiusReason::FixedProfileMismatch);
    if (!isfinite(opacity) || !(opacity > 0.0f))
        return fixed(RadiusReason::FixedOpacityUncertain);

    const float alphaNumeratorLower = __fsub_rd(
        alphaThreshold, kProductAbsoluteError);
    const float responseThresholdLower = __fsub_rd(
        __fdiv_rd(alphaNumeratorLower, opacity), kExpAbsoluteError);
    if (!isfinite(responseThresholdLower)
        || !(responseThresholdLower > kExpNegFourPointFiveUpper))
        return fixed(RadiusReason::FixedFormulaAtLeastThree);
    if (!(responseThresholdLower < 1.0f))
        return fixed(RadiusReason::FixedOpacityUncertain);

    const float logApprox = __logf(responseThresholdLower);
    if (!isfinite(logApprox))
        return fixed(RadiusReason::FixedArithmeticUncertain);
    const float minusLogUpper = __fadd_ru(-logApprox, kLogAbsoluteError);
    const float gateSquaredUpper = __fmul_ru(2.0f, minusLogUpper);
    if (!isfinite(gateSquaredUpper) || !(gateSquaredUpper > 0.0f))
        return fixed(RadiusReason::FixedArithmeticUncertain);

    const double geometricSquaredUpper = __ddiv_ru(
        __dadd_ru(double(gateSquaredUpper), kDotUnderflowAbsoluteUpper),
        kOneMinusGammaFiveLower);
    const double radiusUpper = __dsqrt_ru(geometricSquaredUpper);
    if (!isfinite(radiusUpper) || !(radiusUpper > 0.0)
        || !(radiusUpper < double(kFixedRadius)))
        return fixed(RadiusReason::FixedFormulaAtLeastThree);
    const float radiusFloatUpper = __double2float_ru(radiusUpper);
    if (!isfinite(radiusFloatUpper) || !(radiusFloatUpper < kFixedRadius))
        return fixed(RadiusReason::FixedFormulaAtLeastThree);
    return RadiusResult{
        radiusFloatUpper, RadiusReason::AdaptiveCertified, 1u};
}

}  // namespace gofish_certified_opacity_radius
