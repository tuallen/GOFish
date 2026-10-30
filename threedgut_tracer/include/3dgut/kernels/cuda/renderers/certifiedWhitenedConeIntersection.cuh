// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace gofish_whitened_cone {

// A closed interval whose endpoints are outward-rounded binary64 values.
struct Interval {
    double lo;
    double hi;
};

// One Gaussian/camera cache shared by the count and emit traversals.  Matrix
// entries are exact promotions of the checkpoint's float B^{-1}; `a` encloses
// B^{-1}(o-mu), and all scalar fields are one-sided certificates.
struct FiniteEllipsoidConeCache {
    double inverseFactor[9];
    Interval a[3];
    double aNormUpper;
    double inverseTransposeANormUpper;
    double kLower;
    double inverseNormUpper;
    bool certified;
};

// The Slang/CUDA profile is hash-bound to KernelDegree=2 and
// MinParticleKernelDensity=0.0113.  At squared norm 16, even the documented
// __expf fast-math error is more than 30x below that threshold.  Squared norm
// 25 reserves a full canonical unit for the separately bounded float
// closest-point multiply/add.  If that unit cannot be certified per node, the
// predicate fails open.
constexpr double kResponseHitNormUpper = 4.0;
constexpr double kFiniteEllipsoidRadiusUpper = 5.0;
constexpr double kFiniteEllipsoidSupportSquaredUpper = 25.0;
constexpr double kClosestPointRoundBudget = 1.0;
constexpr double kFloatUnitRoundoff = 0x1.0p-24;
constexpr double kFloatMinNormal = 0x1.0p-126;
constexpr double kFloatMax = 0x1.fffffep+127;
constexpr double kDoubleMax = 0x1.fffffffffffffp+1023;
constexpr double kGamma2 = (2.0 * kFloatUnitRoundoff)
                         / (1.0 - 2.0 * kFloatUnitRoundoff);
constexpr double kGamma5 = (5.0 * kFloatUnitRoundoff)
                         / (1.0 - 5.0 * kFloatUnitRoundoff);

static inline __device__ Interval addInterval(const Interval x, const Interval y) {
    return Interval{__dadd_rd(x.lo, y.lo), __dadd_ru(x.hi, y.hi)};
}

static inline __device__ Interval multiplyInterval(const Interval x, const Interval y) {
    const double lo0 = __dmul_rd(x.lo, y.lo);
    const double lo1 = __dmul_rd(x.lo, y.hi);
    const double lo2 = __dmul_rd(x.hi, y.lo);
    const double lo3 = __dmul_rd(x.hi, y.hi);
    const double hi0 = __dmul_ru(x.lo, y.lo);
    const double hi1 = __dmul_ru(x.lo, y.hi);
    const double hi2 = __dmul_ru(x.hi, y.lo);
    const double hi3 = __dmul_ru(x.hi, y.hi);
    return Interval{
        fmin(fmin(lo0, lo1), fmin(lo2, lo3)),
        fmax(fmax(hi0, hi1), fmax(hi2, hi3)),
    };
}

static inline __device__ Interval rowTimesPoint(
    const double matrix[9], int row, const float3 value) {
    const double v[3] = {(double)value.x, (double)value.y, (double)value.z};
    Interval sum{0.0, 0.0};
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        const double entry = matrix[3 * row + col];
        sum.lo = __dadd_rd(sum.lo, __dmul_rd(entry, v[col]));
        sum.hi = __dadd_ru(sum.hi, __dmul_ru(entry, v[col]));
    }
    return sum;
}

static inline __device__ double absoluteLower(const Interval value) {
    if (value.lo > 0.0) return value.lo;
    if (value.hi < 0.0) return -value.hi;
    return 0.0;
}

static inline __device__ double absoluteUpper(const Interval value) {
    return fmax(fabs(value.lo), fabs(value.hi));
}

static inline __device__ double normSquaredLower(const Interval value[3]) {
    double sum = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double component = absoluteLower(value[axis]);
        sum = __dadd_rd(sum, __dmul_rd(component, component));
    }
    return sum;
}

static inline __device__ double normSquaredUpper(const Interval value[3]) {
    double sum = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double component = absoluteUpper(value[axis]);
        sum = __dadd_ru(sum, __dmul_ru(component, component));
    }
    return sum;
}

static inline __device__ void transposeTimesInterval(
    const double matrix[9], const Interval value[3], Interval result[3]) {
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        result[col] = Interval{0.0, 0.0};
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            const double entry = matrix[3 * row + col];
            result[col] = addInterval(
                result[col], multiplyInterval(Interval{entry, entry}, value[row]));
        }
    }
}

static inline __device__ double floatMatvecRowRoundUpper(
    const double matrix[9], int row, const double valueAbsUpper[3]) {
    double absoluteProductsUpper = 0.0;
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        absoluteProductsUpper = __dadd_ru(
            absoluteProductsUpper,
            __dmul_ru(fabs(matrix[3 * row + col]), valueAbsUpper[col]));
    }
    if (!(absoluteProductsUpper < __ddiv_rd(
            kFloatMax, __dadd_ru(1.0, kGamma5)))) {
        return kDoubleMax;
    }
    // gamma_5 covers three rounded products plus two rounded additions.  The
    // absolute floor covers FTZ of every elementary result under fast math.
    return __dadd_ru(
        __dmul_ru(kGamma5, absoluteProductsUpper),
        __dmul_ru(5.0, kFloatMinNormal));
}

static inline __device__ Interval floatMatvecRowInterval(
    const double matrix[9], int row, const float3 value) {
    const Interval exact = rowTimesPoint(matrix, row, value);
    const double valueAbs[3] = {
        fabs((double)value.x), fabs((double)value.y), fabs((double)value.z)};
    const double error = floatMatvecRowRoundUpper(matrix, row, valueAbs);
    return Interval{__dsub_rd(exact.lo, error), __dadd_ru(exact.hi, error)};
}

static inline __device__ double directionMatvecRoundNormUpper(
    const double matrix[9], double directionMaxAbs) {
    const double componentAbs[3] = {
        directionMaxAbs, directionMaxAbs, directionMaxAbs};
    double squaredUpper = 0.0;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        const double error = floatMatvecRowRoundUpper(
            matrix, row, componentAbs);
        squaredUpper = __dadd_ru(squaredUpper, __dmul_ru(error, error));
    }
    return __dsqrt_ru(squaredUpper);
}

static inline __device__ bool isSubnormalNonzero(float value) {
    const float magnitude = fabsf(value);
    return magnitude > 0.0f && magnitude < (float)kFloatMinNormal;
}

static inline __device__ double inverseSpectralNormUpper(const double matrix[9]) {
    double rowSum[3] = {0.0, 0.0, 0.0};
    double columnSum[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double entry = fabs(matrix[3 * row + col]);
            rowSum[row] = __dadd_ru(rowSum[row], entry);
            columnSum[col] = __dadd_ru(columnSum[col], entry);
        }
    }
    const double normOne = fmax(fmax(columnSum[0], columnSum[1]), columnSum[2]);
    const double normInfinity = fmax(fmax(rowSum[0], rowSum[1]), rowSum[2]);
    return __dsqrt_ru(__dmul_ru(normOne, normInfinity));
}

static inline __device__ FiniteEllipsoidConeCache makeFiniteEllipsoidConeCache(
    const float inverseFactor[9],
    const float3 originMinusMean,
    bool centralGlobalShutter,
    bool supportedKernelContract) {
    FiniteEllipsoidConeCache cache{};
    cache.certified = false;
    if (!centralGlobalShutter || !supportedKernelContract
        || !isfinite(originMinusMean.x) || !isfinite(originMinusMean.y)
        || !isfinite(originMinusMean.z)
        || isSubnormalNonzero(originMinusMean.x)
        || isSubnormalNonzero(originMinusMean.y)
        || isSubnormalNonzero(originMinusMean.z)) {
        return cache;
    }
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        if (!isfinite(inverseFactor[entry])
            || isSubnormalNonzero(inverseFactor[entry])) return cache;
        cache.inverseFactor[entry] = (double)inverseFactor[entry];
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        // Enclose the actual fast-math float Slang matvec, not merely the
        // real-valued matrix product.
        cache.a[row] = floatMatvecRowInterval(
            cache.inverseFactor, row, originMinusMean);
    }
    const double aSquaredLower = normSquaredLower(cache.a);
    cache.kLower = __dsub_rd(
        aSquaredLower, kFiniteEllipsoidSupportSquaredUpper);
    cache.aNormUpper = __dsqrt_ru(normSquaredUpper(cache.a));
    Interval inverseTransposeA[3];
    transposeTimesInterval(cache.inverseFactor, cache.a, inverseTransposeA);
    cache.inverseTransposeANormUpper = __dsqrt_ru(
        normSquaredUpper(inverseTransposeA));
    cache.inverseNormUpper = inverseSpectralNormUpper(cache.inverseFactor);
    cache.certified = isfinite(cache.kLower)
                   && isfinite(cache.aNormUpper)
                   && isfinite(cache.inverseTransposeANormUpper)
                   && isfinite(cache.inverseNormUpper);
    return cache;
}

// Common count/emit primitive.  It returns true only when every positive ray
// d in the node cone is certified to miss ||a+t B^{-1}d||^2 <= s^2.
// Invalid arithmetic and camera-inside/on-support cases conservatively retain.
static inline __device__ bool certifiedNoPositiveRayIntersection(
    const FiniteEllipsoidConeCache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified) {
    if (!cache.certified || !(cache.kLower > 0.0)
        || !directionOperandsCertified
        || !(nodeChordRadius >= 0.0f) || !isfinite(nodeCenter.x)
        || !isfinite(nodeCenter.y) || !isfinite(nodeCenter.z)
        || !isfinite(nodeChordRadius) || !(directionMaxAbs > 0.0f)
        || !isfinite(directionMaxAbs)
        || isSubnormalNonzero(nodeCenter.x)
        || isSubnormalNonzero(nodeCenter.y)
        || isSubnormalNonzero(nodeCenter.z)
        || isSubnormalNonzero(nodeChordRadius)
        || isSubnormalNonzero(directionMaxAbs)) {
        return false;
    }

    const double directionMatvecRound = directionMatvecRoundNormUpper(
        cache.inverseFactor, (double)directionMaxAbs);
    if (!isfinite(directionMatvecRound)) return false;

    Interval u[3];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        u[row] = rowTimesPoint(cache.inverseFactor, row, nodeCenter);
    }
    Interval dotInterval{0.0, 0.0};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        dotInterval = addInterval(dotInterval, multiplyInterval(cache.a[axis], u[axis]));
    }

    const double delta = (double)nodeChordRadius;
    const double etaUpper = __dadd_ru(
        __dmul_ru(cache.inverseTransposeANormUpper, delta),
        __dmul_ru(cache.aNormUpper,
                  directionMatvecRound));

    // If the lower bound is nonnegative, q(t) is increasing at t=0 for every
    // direction in the cone.  Since k>0, no positive ray can enter support.
    const double positiveRayLower = __dsub_rd(dotInterval.lo, etaUpper);
    const bool allRaysOutward = positiveRayLower >= 0.0;

    const double exactDirectionErrorUpper = __dmul_ru(
        cache.inverseNormUpper, delta);
    const double directionErrorUpper = __dadd_ru(
        exactDirectionErrorUpper,
        directionMatvecRound);
    const double uNormLower = __dsqrt_rd(normSquaredLower(u));
    const double normDifferenceLower = fmax(
        0.0, __dsub_rd(uNormLower, exactDirectionErrorUpper));
    const double radialSquareLower = __dmul_rd(
        normDifferenceLower, normDifferenceLower);

    Interval inverseTransposeU[3];
    transposeTimesInterval(cache.inverseFactor, u, inverseTransposeU);
    const double inverseTransposeUNormUpper = __dsqrt_ru(
        normSquaredUpper(inverseTransposeU));
    const double crossUpper = __dmul_ru(
        2.0, __dmul_ru(inverseTransposeUNormUpper, delta));
    const double expandedSquareLower = fmax(
        0.0, __dsub_rd(normSquaredLower(u), crossUpper));
    const double qExactLower = fmax(radialSquareLower, expandedSquareLower);
    const double bNormLower = fmax(
        0.0, __dsub_rd(__dsqrt_rd(qExactLower),
                       directionMatvecRound));
    const double qLower = __dmul_rd(bNormLower, bNormLower);

    // The Slang closest point performs one float multiply and one float add
    // per component.  AABB clamp endpoints need not be bounded by the
    // unconstrained closest-point division, so derive tau from the necessary
    // condition for an actual radius-4 hit instead.  For any t>=0,
    //   t*bLo-aHi <= ||a+t*b|| <= 4 + gamma2*(aHi+t*bHi)+FTZ.
    // Solve this inequality upward. This covers interior and both finite-AABB
    // endpoints without reading or approximating the AABB in the broad phase.
    const double uNormUpper = __dsqrt_ru(normSquaredUpper(u));
    const double bNormUpper = __dadd_ru(
        uNormUpper, directionErrorUpper);
    const double denominatorLower = __dsub_rd(
        bNormLower, __dmul_ru(kGamma2, bNormUpper));
    if (!(denominatorLower > 0.0) || !isfinite(denominatorLower)) return false;
    const double ftzVectorUpper = __dmul_ru(
        4.0, kFloatMinNormal); // > sqrt(3)*2*minNormal
    const double numeratorUpper = __dadd_ru(
        __dadd_ru(
            __dmul_ru(__dadd_ru(1.0, kGamma2), cache.aNormUpper),
            kResponseHitNormUpper),
        ftzVectorUpper);
    const double tauUpper = __ddiv_ru(numeratorUpper, denominatorLower);
    double closestPointRoundSquaredUpper = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double aComponentUpper = absoluteUpper(cache.a[axis]);
        const double uComponentUpper = absoluteUpper(u[axis]);
        const double bComponentUpper = __dadd_ru(
            uComponentUpper, directionErrorUpper);
        const double magnitudeUpper = __dadd_ru(
            aComponentUpper, __dmul_ru(tauUpper, bComponentUpper));
        const double componentRoundUpper = __dadd_ru(
            __dmul_ru(kGamma2, magnitudeUpper),
            __dmul_ru(2.0, kFloatMinNormal));
        closestPointRoundSquaredUpper = __dadd_ru(
            closestPointRoundSquaredUpper,
            __dmul_ru(componentRoundUpper, componentRoundUpper));
    }
    const double closestPointRoundNormUpper = __dsqrt_ru(
        closestPointRoundSquaredUpper);
    if (!(closestPointRoundNormUpper <= kClosestPointRoundBudget)) return false;
    if (allRaysOutward) return true;

    const double negativeLinearUpper = fmax(0.0, -positiveRayLower);
    const double leftUpper = __dmul_ru(
        negativeLinearUpper, negativeLinearUpper);
    const double rightLower = __dmul_rd(cache.kLower, qLower);
    return leftUpper < rightLower;
}

} // namespace gofish_whitened_cone

