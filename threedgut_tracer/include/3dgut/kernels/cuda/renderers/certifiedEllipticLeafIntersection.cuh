// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedWhitenedConeIntersection.cuh>

#include <cuda_runtime.h>

#include <cmath>

namespace gofish_elliptic_leaf {

using gofish_whitened_cone::FiniteEllipsoidConeCache;
using gofish_whitened_cone::Interval;

// Generated Slang accepts only response > 0.0113. Under fast math, a computed
// squared norm q>=9 cannot pass that test, so the fixed radius 3 is a
// conservative support for every opacity.
constexpr double kComputedKernelRadiusUpper = 3.0;
constexpr bool kOpacityAdaptiveRadiusCertified = false;

struct ComputedSupportRadiusContract {
    double radiusUpper;
    bool certified;
    bool opacityAdaptive;
};

struct IntervalMatrix3 {
    Interval value[9];
};

static inline __device__ bool finiteInterval(const Interval value) {
    return isfinite(value.lo) && isfinite(value.hi) && value.lo <= value.hi;
}

static inline __device__ Interval negateInterval(const Interval value) {
    return Interval{-value.hi, -value.lo};
}

static inline __device__ Interval subtractInterval(
    const Interval left, const Interval right) {
    return Interval{
        __dsub_rd(left.lo, right.hi),
        __dsub_ru(left.hi, right.lo),
    };
}

static inline __device__ double intervalAbsoluteUpper(const Interval value) {
    return fmax(fabs(value.lo), fabs(value.hi));
}

static inline __device__ Interval dotPointInterval(
    const float3 point, const Interval vector[3]) {
    const double component[3] = {
        (double)point.x, (double)point.y, (double)point.z};
    Interval result{0.0, 0.0};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        result = gofish_whitened_cone::addInterval(
            result,
            gofish_whitened_cone::multiplyInterval(
                Interval{component[axis], component[axis]}, vector[axis]));
    }
    return result;
}

static inline __device__ void matrixTimesPointInterval(
    const IntervalMatrix3& matrix, const float3 point, Interval result[3]) {
    const double component[3] = {
        (double)point.x, (double)point.y, (double)point.z};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        result[row] = Interval{0.0, 0.0};
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            result[row] = gofish_whitened_cone::addInterval(
                result[row],
                gofish_whitened_cone::multiplyInterval(
                    matrix.value[3 * row + col],
                    Interval{component[col], component[col]}));
        }
    }
}

static inline __device__ double intervalVectorNormUpper(
    const Interval vector[3]) {
    double squaredUpper = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double componentUpper = intervalAbsoluteUpper(vector[axis]);
        squaredUpper = __dadd_ru(
            squaredUpper,
            __dmul_ru(componentUpper, componentUpper));
    }
    return __dsqrt_ru(squaredUpper);
}

static inline __device__ double intervalMatrixFrobeniusNormUpper(
    const IntervalMatrix3& matrix) {
    double squaredUpper = 0.0;
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        const double componentUpper = intervalAbsoluteUpper(matrix.value[entry]);
        squaredUpper = __dadd_ru(
            squaredUpper,
            __dmul_ru(componentUpper, componentUpper));
    }
    return __dsqrt_ru(squaredUpper);
}

static inline __device__ ComputedSupportRadiusContract
computedSupportRadiusUpper(float density, float alphaThreshold) {
    (void)density;
    (void)alphaThreshold;
    // The fixed radius-3 response gate is conservative for every opacity, so
    // no opacity-dependent radius (directed log) is required.
    return ComputedSupportRadiusContract{
        kComputedKernelRadiusUpper,
        true,
        kOpacityAdaptiveRadiusCertified,
    };
}

// Replays the finite-hit error analysis with the tighter computed
// radius 3. It returns an outward norm bound e for the float closest-point
// multiply/add. Invalid/ill-conditioned arithmetic fails open.
static inline __device__ bool nodeClosestPointRoundNormUpper(
    const FiniteEllipsoidConeCache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified,
    double& roundNormUpper) {
    roundNormUpper = 0.0;
    if (!cache.certified || !directionOperandsCertified
        || !(nodeChordRadius >= 0.0f) || !isfinite(nodeCenter.x)
        || !isfinite(nodeCenter.y) || !isfinite(nodeCenter.z)
        || !isfinite(nodeChordRadius) || !(directionMaxAbs > 0.0f)
        || !isfinite(directionMaxAbs)
        || gofish_whitened_cone::isSubnormalNonzero(nodeCenter.x)
        || gofish_whitened_cone::isSubnormalNonzero(nodeCenter.y)
        || gofish_whitened_cone::isSubnormalNonzero(nodeCenter.z)
        || gofish_whitened_cone::isSubnormalNonzero(nodeChordRadius)
        || gofish_whitened_cone::isSubnormalNonzero(directionMaxAbs)) {
        return false;
    }

    const double directionMatvecRound =
        gofish_whitened_cone::directionMatvecRoundNormUpper(
            cache.inverseFactor, (double)directionMaxAbs);
    if (!isfinite(directionMatvecRound)) return false;

    Interval u[3];
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        u[row] = gofish_whitened_cone::rowTimesPoint(
            cache.inverseFactor, row, nodeCenter);
    }

    const double delta = (double)nodeChordRadius;
    const double exactDirectionErrorUpper = __dmul_ru(
        cache.inverseNormUpper, delta);
    const double directionErrorUpper = __dadd_ru(
        exactDirectionErrorUpper, directionMatvecRound);
    const double uNormLower = __dsqrt_rd(
        gofish_whitened_cone::normSquaredLower(u));
    const double normDifferenceLower = fmax(
        0.0, __dsub_rd(uNormLower, exactDirectionErrorUpper));
    const double radialSquareLower = __dmul_rd(
        normDifferenceLower, normDifferenceLower);

    Interval inverseTransposeU[3];
    gofish_whitened_cone::transposeTimesInterval(
        cache.inverseFactor, u, inverseTransposeU);
    const double inverseTransposeUNormUpper = __dsqrt_ru(
        gofish_whitened_cone::normSquaredUpper(inverseTransposeU));
    const double crossUpper = __dmul_ru(
        2.0, __dmul_ru(inverseTransposeUNormUpper, delta));
    const double expandedSquareLower = fmax(
        0.0,
        __dsub_rd(gofish_whitened_cone::normSquaredLower(u), crossUpper));
    const double qExactLower = fmax(radialSquareLower, expandedSquareLower);
    const double bNormLower = fmax(
        0.0,
        __dsub_rd(__dsqrt_rd(qExactLower), directionMatvecRound));
    const double uNormUpper = __dsqrt_ru(
        gofish_whitened_cone::normSquaredUpper(u));
    const double bNormUpper = __dadd_ru(uNormUpper, directionErrorUpper);
    const double denominatorLower = __dsub_rd(
        bNormLower,
        __dmul_ru(gofish_whitened_cone::kGamma2, bNormUpper));
    if (!(denominatorLower > 0.0) || !isfinite(denominatorLower)) return false;

    const double ftzVectorUpper = __dmul_ru(
        4.0, gofish_whitened_cone::kFloatMinNormal);
    const double numeratorUpper = __dadd_ru(
        __dadd_ru(
            __dmul_ru(
                __dadd_ru(1.0, gofish_whitened_cone::kGamma2),
                cache.aNormUpper),
            kComputedKernelRadiusUpper),
        ftzVectorUpper);
    const double tauUpper = __ddiv_ru(numeratorUpper, denominatorLower);
    if (!(tauUpper >= 0.0) || !isfinite(tauUpper)) return false;

    double roundSquaredUpper = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double aComponentUpper =
            gofish_whitened_cone::absoluteUpper(cache.a[axis]);
        const double uComponentUpper =
            gofish_whitened_cone::absoluteUpper(u[axis]);
        const double bComponentUpper = __dadd_ru(
            uComponentUpper, directionErrorUpper);
        const double magnitudeUpper = __dadd_ru(
            aComponentUpper, __dmul_ru(tauUpper, bComponentUpper));
        const double componentRoundUpper = __dadd_ru(
            __dmul_ru(gofish_whitened_cone::kGamma2, magnitudeUpper),
            __dmul_ru(2.0, gofish_whitened_cone::kFloatMinNormal));
        roundSquaredUpper = __dadd_ru(
            roundSquaredUpper,
            __dmul_ru(componentRoundUpper, componentRoundUpper));
    }
    roundNormUpper = __dsqrt_ru(roundSquaredUpper);
    return isfinite(roundNormUpper);
}

static inline __device__ bool makePrecisionInterval(
    const FiniteEllipsoidConeCache& cache,
    IntervalMatrix3& precision) {
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            Interval sum{0.0, 0.0};
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                const double left = cache.inverseFactor[3 * inner + row];
                const double right = cache.inverseFactor[3 * inner + col];
                sum = gofish_whitened_cone::addInterval(
                    sum,
                    gofish_whitened_cone::multiplyInterval(
                        Interval{left, left}, Interval{right, right}));
            }
            precision.value[3 * row + col] = sum;
        }
    }
    // Explicit hull preserves symmetry even if directed evaluation paths differ.
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = row + 1; col < 3; ++col) {
            const Interval left = precision.value[3 * row + col];
            const Interval right = precision.value[3 * col + row];
            const Interval hull{
                fmin(left.lo, right.lo), fmax(left.hi, right.hi)};
            precision.value[3 * row + col] = hull;
            precision.value[3 * col + row] = hull;
        }
    }
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        if (!finiteInterval(precision.value[entry])) return false;
    }
    return true;
}

// True only if every actual ray enclosed by this hierarchy-node chord misses the enlarged
// renderer support. The quadratic uses a chord-ball bound:
//   (c+h)^T K(c+h) <= c^T Kc + 2 delta ||Kc|| + delta^2 ||K||_F.
// It avoids all uncertified trigonometric/transcendental operations.
static inline __device__ bool certifiedNoEllipticNodeIntersection(
    const FiniteEllipsoidConeCache& cache,
    const float3 nodeCenter,
    float nodeChordRadius,
    float directionMaxAbs,
    bool directionOperandsCertified,
    float density,
    float alphaThreshold) {
    const ComputedSupportRadiusContract computedSupport =
        computedSupportRadiusUpper(density, alphaThreshold);
    if (!computedSupport.certified) return false;

    double closestPointRoundUpper = 0.0;
    if (!nodeClosestPointRoundNormUpper(
            cache, nodeCenter, nodeChordRadius, directionMaxAbs,
            directionOperandsCertified, closestPointRoundUpper)) return false;

    const double exactSupportRadiusUpper = __dadd_ru(
        computedSupport.radiusUpper, closestPointRoundUpper);
    const double supportSquaredUpper = __dmul_ru(
        exactSupportRadiusUpper, exactSupportRadiusUpper);
    if (!isfinite(supportSquaredUpper)) return false;

    IntervalMatrix3 precision;
    if (!makePrecisionInterval(cache, precision)) return false;

    Interval inverseTransposeA[3];
    gofish_whitened_cone::transposeTimesInterval(
        cache.inverseFactor, cache.a, inverseTransposeA);
    Interval p[3];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        p[axis] = negateInterval(inverseTransposeA[axis]);
        if (!finiteInterval(p[axis])) return false;
    }

    const Interval cSquared{
        gofish_whitened_cone::normSquaredLower(cache.a),
        gofish_whitened_cone::normSquaredUpper(cache.a)};
    const Interval supportSquared{
        supportSquaredUpper, supportSquaredUpper};
    const Interval kappa = subtractInterval(cSquared, supportSquared);
    if (!finiteInterval(kappa) || !(kappa.lo > 0.0)) return false;

    IntervalMatrix3 discriminant;
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const Interval pp = gofish_whitened_cone::multiplyInterval(
                p[row], p[col]);
            const Interval kappaP = gofish_whitened_cone::multiplyInterval(
                kappa, precision.value[3 * row + col]);
            discriminant.value[3 * row + col] = subtractInterval(pp, kappaP);
        }
    }
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = row + 1; col < 3; ++col) {
            const Interval left = discriminant.value[3 * row + col];
            const Interval right = discriminant.value[3 * col + row];
            const Interval hull{
                fmin(left.lo, right.lo), fmax(left.hi, right.hi)};
            discriminant.value[3 * row + col] = hull;
            discriminant.value[3 * col + row] = hull;
        }
    }
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        if (!finiteInterval(discriminant.value[entry])) return false;
    }

    const double delta = (double)nodeChordRadius;
    if (!(delta >= 0.0) || !isfinite(delta)) return false;

    const Interval centerDotP = dotPointInterval(nodeCenter, p);
    const double pNormUpper = intervalVectorNormUpper(p);
    const double linearUpper = __dadd_ru(
        centerDotP.hi, __dmul_ru(delta, pNormUpper));
    if (!isfinite(linearUpper)) return false;
    if (linearUpper <= 0.0) return true;

    Interval kc[3];
    matrixTimesPointInterval(discriminant, nodeCenter, kc);
    const Interval centerKCenter = dotPointInterval(nodeCenter, kc);
    const double kcNormUpper = intervalVectorNormUpper(kc);
    const double kNormUpper = intervalMatrixFrobeniusNormUpper(discriminant);
    const double twoDeltaKc = __dmul_ru(
        __dmul_ru(2.0, delta), kcNormUpper);
    const double deltaSquared = __dmul_ru(delta, delta);
    const double quadraticRemainder = __dmul_ru(
        deltaSquared, kNormUpper);
    const double discriminantUpper = __dadd_ru(
        __dadd_ru(centerKCenter.hi, twoDeltaKc), quadraticRemainder);
    return isfinite(discriminantUpper) && discriminantUpper < 0.0;
}

} // namespace gofish_elliptic_leaf

