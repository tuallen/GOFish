// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedWhitenedConeIntersection.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedEllipticLeafIntersection.cuh>

#include <cuda_runtime.h>
#include <cmath>

// A coarse outward-binary32 evaluation of the FP64 expression tree.
// It is rejection-only: any invalid, nonfinite, boundary, or inconclusive case
// returns false and therefore executes the FP64 predicates.
namespace gofish_fp32_certificate {

using gofish_whitened_cone::FiniteEllipsoidConeCache;

struct Interval {
    float lo;
    float hi;
};

struct Matrix3 {
    Interval value[9];
};

struct Cache {
    float inverseFactor[9];
    Interval a[3];
    float aNormUpper;
    float inverseTransposeANormUpper;
    float kLower;
    float inverseNormUpper;
    Matrix3 precision;
    Interval p[3];
    Interval cSquared;
    bool certified;
};

constexpr float kFloatMinNormal = 0x1.0p-126f;
constexpr float kFloatMax = 0x1.fffffep+127f;

static inline __device__ bool finiteInterval(const Interval value) {
    return isfinite(value.lo) && isfinite(value.hi) && value.lo <= value.hi;
}

static inline __device__ float lowerFloat(double value) {
    return __double2float_rd(value);
}

static inline __device__ float upperFloat(double value) {
    return __double2float_ru(value);
}

static inline __device__ Interval encloseDoubleInterval(
    const gofish_whitened_cone::Interval value) {
    return Interval{lowerFloat(value.lo), upperFloat(value.hi)};
}

static inline __device__ Interval add(const Interval left, const Interval right) {
    return Interval{
        __fadd_rd(left.lo, right.lo),
        __fadd_ru(left.hi, right.hi),
    };
}

static inline __device__ Interval subtract(
    const Interval left, const Interval right) {
    return Interval{
        __fsub_rd(left.lo, right.hi),
        __fsub_ru(left.hi, right.lo),
    };
}

static inline __device__ Interval negate(const Interval value) {
    return Interval{-value.hi, -value.lo};
}

static inline __device__ Interval multiply(
    const Interval left, const Interval right) {
    const float lo0 = __fmul_rd(left.lo, right.lo);
    const float lo1 = __fmul_rd(left.lo, right.hi);
    const float lo2 = __fmul_rd(left.hi, right.lo);
    const float lo3 = __fmul_rd(left.hi, right.hi);
    const float hi0 = __fmul_ru(left.lo, right.lo);
    const float hi1 = __fmul_ru(left.lo, right.hi);
    const float hi2 = __fmul_ru(left.hi, right.lo);
    const float hi3 = __fmul_ru(left.hi, right.hi);
    return Interval{
        fminf(fminf(lo0, lo1), fminf(lo2, lo3)),
        fmaxf(fmaxf(hi0, hi1), fmaxf(hi2, hi3)),
    };
}

static inline __device__ float absoluteLower(const Interval value) {
    if (value.lo > 0.0f) return value.lo;
    if (value.hi < 0.0f) return -value.hi;
    return 0.0f;
}

static inline __device__ float absoluteUpper(const Interval value) {
    return fmaxf(fabsf(value.lo), fabsf(value.hi));
}

static inline __device__ float normSquaredLower(const Interval value[3]) {
    float sum = 0.0f;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float component = absoluteLower(value[axis]);
        sum = __fadd_rd(sum, __fmul_rd(component, component));
    }
    return sum;
}

static inline __device__ float normSquaredUpper(const Interval value[3]) {
    float sum = 0.0f;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float component = absoluteUpper(value[axis]);
        sum = __fadd_ru(sum, __fmul_ru(component, component));
    }
    return sum;
}

static inline __device__ Interval rowTimesPoint(
    const float matrix[9], int row, const float3 value) {
    const float component[3] = {value.x, value.y, value.z};
    Interval result{0.0f, 0.0f};
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        const float entry = matrix[3 * row + col];
        result.lo = __fadd_rd(
            result.lo, __fmul_rd(entry, component[col]));
        result.hi = __fadd_ru(
            result.hi, __fmul_ru(entry, component[col]));
    }
    return result;
}

static inline __device__ void transposeTimesInterval(
    const float matrix[9], const Interval value[3], Interval result[3]) {
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        result[col] = Interval{0.0f, 0.0f};
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            const float entry = matrix[3 * row + col];
            result[col] = add(
                result[col], multiply(Interval{entry, entry}, value[row]));
        }
    }
}

static inline __device__ float vectorNormUpper(const Interval value[3]) {
    const float squared = normSquaredUpper(value);
    return squared >= 0.0f ? __fsqrt_ru(squared) : CUDART_INF_F;
}

static inline __device__ float matrixFrobeniusNormUpper(const Matrix3& matrix) {
    float squared = 0.0f;
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        const float upper = absoluteUpper(matrix.value[entry]);
        squared = __fadd_ru(squared, __fmul_ru(upper, upper));
    }
    return squared >= 0.0f ? __fsqrt_ru(squared) : CUDART_INF_F;
}

static inline __device__ Interval dotPoint(
    const float3 point, const Interval vector[3]) {
    const float component[3] = {point.x, point.y, point.z};
    Interval result{0.0f, 0.0f};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        result = add(
            result,
            multiply(Interval{component[axis], component[axis]}, vector[axis]));
    }
    return result;
}

static inline __device__ void matrixTimesPoint(
    const Matrix3& matrix, const float3 point, Interval result[3]) {
    const float component[3] = {point.x, point.y, point.z};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        result[row] = Interval{0.0f, 0.0f};
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            result[row] = add(
                result[row],
                multiply(
                    matrix.value[3 * row + col],
                    Interval{component[col], component[col]}));
        }
    }
}

static inline __device__ bool validNode(
    const Cache& cache,
    const float3 center,
    float chordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified) {
    return cache.certified && directionOperandsCertified
        && chordRadius >= 0.0f && directionMaxAbs > 0.0f
        && isfinite(center.x) && isfinite(center.y) && isfinite(center.z)
        && isfinite(chordRadius) && isfinite(directionMaxAbs)
        && !gofish_whitened_cone::isSubnormalNonzero(center.x)
        && !gofish_whitened_cone::isSubnormalNonzero(center.y)
        && !gofish_whitened_cone::isSubnormalNonzero(center.z)
        && !gofish_whitened_cone::isSubnormalNonzero(chordRadius)
        && !gofish_whitened_cone::isSubnormalNonzero(directionMaxAbs);
}

static inline __device__ bool directionMatvecRoundUpper(
    const float matrix[9], float directionMaxAbs, float& result) {
    const float gamma5Upper = upperFloat(gofish_whitened_cone::kGamma5);
    const float onePlusGamma5Upper = __fadd_ru(1.0f, gamma5Upper);
    const float safeProductLimitLower = __fdiv_rd(
        kFloatMax, onePlusGamma5Upper);
    const float minNormalUpper = upperFloat(
        gofish_whitened_cone::kFloatMinNormal);
    float squaredUpper = 0.0f;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        float absoluteProductsUpper = 0.0f;
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            absoluteProductsUpper = __fadd_ru(
                absoluteProductsUpper,
                __fmul_ru(fabsf(matrix[3 * row + col]), directionMaxAbs));
        }
        if (!(absoluteProductsUpper < safeProductLimitLower)) return false;
        const float errorUpper = __fadd_ru(
            __fmul_ru(gamma5Upper, absoluteProductsUpper),
            __fmul_ru(5.0f, minNormalUpper));
        squaredUpper = __fadd_ru(
            squaredUpper, __fmul_ru(errorUpper, errorUpper));
    }
    result = __fsqrt_ru(squaredUpper);
    return isfinite(result);
}

static inline __device__ Cache makeCache(
    const FiniteEllipsoidConeCache& source) {
    Cache result{};
    result.certified = false;
    if (!source.certified) return result;

#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        const float value = (float)source.inverseFactor[entry];
        if (!isfinite(value) || (double)value != source.inverseFactor[entry]) {
            return result;
        }
        result.inverseFactor[entry] = value;
    }
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        result.a[axis] = encloseDoubleInterval(source.a[axis]);
        if (!finiteInterval(result.a[axis])) return result;
    }
    result.aNormUpper = upperFloat(source.aNormUpper);
    result.inverseTransposeANormUpper = upperFloat(
        source.inverseTransposeANormUpper);
    result.kLower = lowerFloat(source.kLower);
    result.inverseNormUpper = upperFloat(source.inverseNormUpper);
    if (!isfinite(result.aNormUpper)
        || !isfinite(result.inverseTransposeANormUpper)
        || !isfinite(result.kLower)
        || !isfinite(result.inverseNormUpper)) return result;

    Interval inverseTransposeA[3];
    transposeTimesInterval(result.inverseFactor, result.a, inverseTransposeA);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        result.p[axis] = negate(inverseTransposeA[axis]);
        if (!finiteInterval(result.p[axis])) return result;
    }
    result.cSquared = Interval{
        normSquaredLower(result.a), normSquaredUpper(result.a)};
    if (!finiteInterval(result.cSquared)) return result;

#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            Interval sum{0.0f, 0.0f};
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                const float left = result.inverseFactor[3 * inner + row];
                const float right = result.inverseFactor[3 * inner + col];
                sum = add(
                    sum,
                    multiply(Interval{left, left}, Interval{right, right}));
            }
            result.precision.value[3 * row + col] = sum;
        }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = row + 1; col < 3; ++col) {
            const Interval left = result.precision.value[3 * row + col];
            const Interval right = result.precision.value[3 * col + row];
            const Interval hull{
                fminf(left.lo, right.lo), fmaxf(left.hi, right.hi)};
            result.precision.value[3 * row + col] = hull;
            result.precision.value[3 * col + row] = hull;
        }
    }
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        if (!finiteInterval(result.precision.value[entry])) return result;
    }
    result.certified = true;
    return result;
}

// Mirrors certifiedNoPositiveRayIntersection with coarser intervals. A true
// result implies the FP64 function returns true.
static inline __device__ bool whitenedProvesSameRejection(
    const Cache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified) {
    if (!validNode(cache, nodeCenter, nodeChordRadius, directionMaxAbs,
                   directionOperandsCertified)
        || !(cache.kLower > 0.0f)) return false;

    float directionRoundUpper = 0.0f;
    if (!directionMatvecRoundUpper(
            cache.inverseFactor, directionMaxAbs, directionRoundUpper)) {
        return false;
    }

    Interval u[3];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        u[row] = rowTimesPoint(cache.inverseFactor, row, nodeCenter);
        if (!finiteInterval(u[row])) return false;
    }
    Interval dot{0.0f, 0.0f};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        dot = add(dot, multiply(cache.a[axis], u[axis]));
    }
    if (!finiteInterval(dot)) return false;

    const float delta = nodeChordRadius;
    const float etaUpper = __fadd_ru(
        __fmul_ru(cache.inverseTransposeANormUpper, delta),
        __fmul_ru(cache.aNormUpper, directionRoundUpper));
    const float positiveRayLower = __fsub_rd(dot.lo, etaUpper);

    const float exactDirectionErrorUpper = __fmul_ru(
        cache.inverseNormUpper, delta);
    const float directionErrorUpper = __fadd_ru(
        exactDirectionErrorUpper, directionRoundUpper);
    const float uSquaredLower = normSquaredLower(u);
    const float uNormLower = __fsqrt_rd(uSquaredLower);
    const float normDifferenceLower = fmaxf(
        0.0f, __fsub_rd(uNormLower, exactDirectionErrorUpper));
    const float radialSquareLower = __fmul_rd(
        normDifferenceLower, normDifferenceLower);

    Interval inverseTransposeU[3];
    transposeTimesInterval(cache.inverseFactor, u, inverseTransposeU);
    const float inverseTransposeUNormUpper = vectorNormUpper(inverseTransposeU);
    const float crossUpper = __fmul_ru(
        2.0f, __fmul_ru(inverseTransposeUNormUpper, delta));
    const float expandedSquareLower = fmaxf(
        0.0f, __fsub_rd(uSquaredLower, crossUpper));
    const float qExactLower = fmaxf(radialSquareLower, expandedSquareLower);
    const float bNormLower = fmaxf(
        0.0f,
        __fsub_rd(__fsqrt_rd(qExactLower), directionRoundUpper));
    const float qLower = __fmul_rd(bNormLower, bNormLower);
    const float uNormUpper = __fsqrt_ru(normSquaredUpper(u));
    const float bNormUpper = __fadd_ru(uNormUpper, directionErrorUpper);
    const float gamma2Upper = upperFloat(gofish_whitened_cone::kGamma2);
    const float denominatorLower = __fsub_rd(
        bNormLower, __fmul_ru(gamma2Upper, bNormUpper));
    if (!(denominatorLower > 0.0f) || !isfinite(denominatorLower)) return false;

    const float minNormalUpper = upperFloat(
        gofish_whitened_cone::kFloatMinNormal);
    const float numeratorUpper = __fadd_ru(
        __fadd_ru(
            __fmul_ru(__fadd_ru(1.0f, gamma2Upper), cache.aNormUpper),
            upperFloat(gofish_whitened_cone::kResponseHitNormUpper)),
        __fmul_ru(4.0f, minNormalUpper));
    const float tauUpper = __fdiv_ru(numeratorUpper, denominatorLower);
    if (!(tauUpper >= 0.0f) || !isfinite(tauUpper)) return false;

    float closestRoundSquaredUpper = 0.0f;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float bComponentUpper = __fadd_ru(
            absoluteUpper(u[axis]), directionErrorUpper);
        const float magnitudeUpper = __fadd_ru(
            absoluteUpper(cache.a[axis]),
            __fmul_ru(tauUpper, bComponentUpper));
        const float componentRoundUpper = __fadd_ru(
            __fmul_ru(gamma2Upper, magnitudeUpper),
            __fmul_ru(2.0f, minNormalUpper));
        closestRoundSquaredUpper = __fadd_ru(
            closestRoundSquaredUpper,
            __fmul_ru(componentRoundUpper, componentRoundUpper));
    }
    const float closestRoundUpper = __fsqrt_ru(closestRoundSquaredUpper);
    if (!(closestRoundUpper <= upperFloat(
              gofish_whitened_cone::kClosestPointRoundBudget))) return false;

    if (positiveRayLower >= 0.0f) return true;
    const float negativeLinearUpper = fmaxf(0.0f, -positiveRayLower);
    const float leftUpper = __fmul_ru(
        negativeLinearUpper, negativeLinearUpper);
    const float rightLower = __fmul_rd(cache.kLower, qLower);
    return isfinite(leftUpper) && isfinite(rightLower)
        && leftUpper < rightLower;
}

// Coarse counterpart of nodeClosestPointRoundNormUpper for the radius-3
// elliptic predicate. It succeeds only when every FP64 precondition is proved.
static inline __device__ bool ellipticRoundUpper(
    const Cache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified,
    float& roundUpper) {
    roundUpper = 0.0f;
    if (!validNode(cache, nodeCenter, nodeChordRadius, directionMaxAbs,
                   directionOperandsCertified)) return false;

    float directionRoundUpper = 0.0f;
    if (!directionMatvecRoundUpper(
            cache.inverseFactor, directionMaxAbs, directionRoundUpper)) {
        return false;
    }
    Interval u[3];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        u[row] = rowTimesPoint(cache.inverseFactor, row, nodeCenter);
        if (!finiteInterval(u[row])) return false;
    }
    const float delta = nodeChordRadius;
    const float exactDirectionErrorUpper = __fmul_ru(
        cache.inverseNormUpper, delta);
    const float directionErrorUpper = __fadd_ru(
        exactDirectionErrorUpper, directionRoundUpper);
    const float uSquaredLower = normSquaredLower(u);
    const float uNormLower = __fsqrt_rd(uSquaredLower);
    const float normDifferenceLower = fmaxf(
        0.0f, __fsub_rd(uNormLower, exactDirectionErrorUpper));
    const float radialSquareLower = __fmul_rd(
        normDifferenceLower, normDifferenceLower);
    Interval inverseTransposeU[3];
    transposeTimesInterval(cache.inverseFactor, u, inverseTransposeU);
    const float crossUpper = __fmul_ru(
        2.0f, __fmul_ru(vectorNormUpper(inverseTransposeU), delta));
    const float expandedSquareLower = fmaxf(
        0.0f, __fsub_rd(uSquaredLower, crossUpper));
    const float qExactLower = fmaxf(radialSquareLower, expandedSquareLower);
    const float bNormLower = fmaxf(
        0.0f,
        __fsub_rd(__fsqrt_rd(qExactLower), directionRoundUpper));
    const float uNormUpper = __fsqrt_ru(normSquaredUpper(u));
    const float bNormUpper = __fadd_ru(uNormUpper, directionErrorUpper);
    const float gamma2Upper = upperFloat(gofish_whitened_cone::kGamma2);
    const float denominatorLower = __fsub_rd(
        bNormLower, __fmul_ru(gamma2Upper, bNormUpper));
    if (!(denominatorLower > 0.0f) || !isfinite(denominatorLower)) return false;

    const float minNormalUpper = upperFloat(
        gofish_whitened_cone::kFloatMinNormal);
    const float numeratorUpper = __fadd_ru(
        __fadd_ru(
            __fmul_ru(__fadd_ru(1.0f, gamma2Upper), cache.aNormUpper),
            upperFloat(gofish_elliptic_leaf::kComputedKernelRadiusUpper)),
        __fmul_ru(4.0f, minNormalUpper));
    const float tauUpper = __fdiv_ru(numeratorUpper, denominatorLower);
    if (!(tauUpper >= 0.0f) || !isfinite(tauUpper)) return false;

    float roundSquaredUpper = 0.0f;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float bComponentUpper = __fadd_ru(
            absoluteUpper(u[axis]), directionErrorUpper);
        const float magnitudeUpper = __fadd_ru(
            absoluteUpper(cache.a[axis]),
            __fmul_ru(tauUpper, bComponentUpper));
        const float componentRoundUpper = __fadd_ru(
            __fmul_ru(gamma2Upper, magnitudeUpper),
            __fmul_ru(2.0f, minNormalUpper));
        roundSquaredUpper = __fadd_ru(
            roundSquaredUpper,
            __fmul_ru(componentRoundUpper, componentRoundUpper));
    }
    roundUpper = __fsqrt_ru(roundSquaredUpper);
    return isfinite(roundUpper);
}

// Mirrors certifiedNoEllipticNodeIntersection with coarser intervals. A true
// result implies the FP64 function returns true.
static inline __device__ bool ellipticProvesSameRejection(
    const Cache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified) {
    float roundUpper = 0.0f;
    if (!ellipticRoundUpper(
            cache, nodeCenter, nodeChordRadius, directionMaxAbs,
            directionOperandsCertified, roundUpper)) return false;

    const float radiusUpper = __fadd_ru(
        upperFloat(gofish_elliptic_leaf::kComputedKernelRadiusUpper),
        roundUpper);
    const float supportSquaredUpper = __fmul_ru(radiusUpper, radiusUpper);
    if (!isfinite(supportSquaredUpper)) return false;
    // The FP64 code stores its upward-computed support square as a singleton.
    // The coarse path knows only an upper bound on that value; radius >= 3
    // supplies the matching lower enclosure. Using the upper value as both
    // endpoints would not contain the FP64 singleton.
    const Interval supportSquared{9.0f, supportSquaredUpper};
    const Interval kappa = subtract(cache.cSquared, supportSquared);
    if (!finiteInterval(kappa) || !(kappa.lo > 0.0f)) return false;

    Matrix3 discriminant;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const Interval pp = multiply(cache.p[row], cache.p[col]);
            const Interval kappaP = multiply(
                kappa, cache.precision.value[3 * row + col]);
            discriminant.value[3 * row + col] = subtract(pp, kappaP);
        }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = row + 1; col < 3; ++col) {
            const Interval left = discriminant.value[3 * row + col];
            const Interval right = discriminant.value[3 * col + row];
            const Interval hull{
                fminf(left.lo, right.lo), fmaxf(left.hi, right.hi)};
            discriminant.value[3 * row + col] = hull;
            discriminant.value[3 * col + row] = hull;
        }
    }
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        if (!finiteInterval(discriminant.value[entry])) return false;
    }

    const float delta = nodeChordRadius;
    const Interval centerDotP = dotPoint(nodeCenter, cache.p);
    const float pNormUpper = vectorNormUpper(cache.p);
    const float linearUpper = __fadd_ru(
        centerDotP.hi, __fmul_ru(delta, pNormUpper));
    if (!isfinite(linearUpper)) return false;
    if (linearUpper <= 0.0f) return true;

    Interval kc[3];
    matrixTimesPoint(discriminant, nodeCenter, kc);
    const Interval centerKCenter = dotPoint(nodeCenter, kc);
    const float twoDeltaKc = __fmul_ru(
        __fmul_ru(2.0f, delta), vectorNormUpper(kc));
    const float deltaSquared = __fmul_ru(delta, delta);
    const float quadraticRemainder = __fmul_ru(
        deltaSquared, matrixFrobeniusNormUpper(discriminant));
    const float discriminantUpper = __fadd_ru(
        __fadd_ru(centerKCenter.hi, twoDeltaKc), quadraticRemainder);
    return isfinite(discriminantUpper) && discriminantUpper < 0.0f;
}

static inline __device__ bool provesSameRejection(
    const Cache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified) {
    if (whitenedProvesSameRejection(
            cache, nodeCenter, nodeChordRadius, directionMaxAbs,
            directionOperandsCertified)) return true;
    return ellipticProvesSameRejection(
        cache, nodeCenter, nodeChordRadius, directionMaxAbs,
        directionOperandsCertified);
}

} // namespace gofish_fp32_certificate

