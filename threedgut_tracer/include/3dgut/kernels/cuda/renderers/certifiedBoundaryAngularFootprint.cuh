// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gofish_boundary_angular {

constexpr double kPi = 3.14159265358979323846264338327950288;
// CUDA binary64 trigonometric/inverse-trigonometric proof budget. This is a
// fixed implementation accuracy bound, not a scene or method tolerance.
constexpr uint32_t kTranscendentalUlpBudget = 4u;

enum class Route : uint32_t {
    ExactCone = 0u,
    Interior = 1u,
    CertifiedRectangle = 2u,
};

struct Interval {
    double lo;
    double hi;
};

struct Camera {
    double fx, fy, cx, cy;
    double k1, k2, k3, k4;
    double thetaMax;
};

struct Rectangle {
    float2 center;
    float2 halfExtent;
    double thetaLower;
    double thetaUpper;
    double phiHalfUpper;
    double supportBallRadiusUpper;
};

struct Result {
    Route route;
    Rectangle rectangle;
};

static inline __device__ Interval point(double value) {
    return Interval{
        nextafter(value, -CUDART_INF),
        nextafter(value, CUDART_INF),
    };
}

static inline __device__ Interval add(Interval left, Interval right) {
    return Interval{
        __dadd_rd(left.lo, right.lo),
        __dadd_ru(left.hi, right.hi),
    };
}

static inline __device__ Interval subtract(Interval left, Interval right) {
    return Interval{
        __dsub_rd(left.lo, right.hi),
        __dsub_ru(left.hi, right.lo),
    };
}

static inline __device__ Interval multiply(Interval left, Interval right) {
    const double lo[4] = {
        __dmul_rd(left.lo, right.lo), __dmul_rd(left.lo, right.hi),
        __dmul_rd(left.hi, right.lo), __dmul_rd(left.hi, right.hi),
    };
    const double hi[4] = {
        __dmul_ru(left.lo, right.lo), __dmul_ru(left.lo, right.hi),
        __dmul_ru(left.hi, right.lo), __dmul_ru(left.hi, right.hi),
    };
    return Interval{
        fmin(fmin(lo[0], lo[1]), fmin(lo[2], lo[3])),
        fmax(fmax(hi[0], hi[1]), fmax(hi[2], hi[3])),
    };
}

static inline __device__ bool finite(Interval value) {
    return isfinite(value.lo) && isfinite(value.hi) && value.lo <= value.hi;
}

static inline __device__ double absoluteUpper(Interval value) {
    return nextafter(fmax(fabs(value.lo), fabs(value.hi)), CUDART_INF);
}

static inline __device__ Interval outwardTranscendental(double value) {
    if (!isfinite(value)) return Interval{-CUDART_INF, CUDART_INF};
    double lower = value;
    double upper = value;
#pragma unroll
    for (uint32_t index = 0u; index < kTranscendentalUlpBudget; ++index) {
        lower = nextafter(lower, -CUDART_INF);
        upper = nextafter(upper, CUDART_INF);
    }
    return Interval{lower, upper};
}

static inline __device__ double transcendentalUpper(double value) {
    if (!isfinite(value)) return CUDART_INF;
#pragma unroll
    for (uint32_t index = 0u; index < kTranscendentalUlpBudget; ++index)
        value = nextafter(value, CUDART_INF);
    return value;
}

static inline __device__ double transcendentalLower(double value) {
    if (!isfinite(value)) return -CUDART_INF;
#pragma unroll
    for (uint32_t index = 0u; index < kTranscendentalUlpBudget; ++index)
        value = nextafter(value, -CUDART_INF);
    return value;
}

static inline __device__ Interval kbRadiusInterval(
    const Camera& camera, Interval theta) {
    const Interval theta2 = multiply(theta, theta);
    Interval horner = add(point(camera.k3),
                          multiply(theta2, point(camera.k4)));
    horner = add(point(camera.k2), multiply(theta2, horner));
    horner = add(point(camera.k1), multiply(theta2, horner));
    return multiply(theta, add(point(1.0), multiply(theta2, horner)));
}

static inline __device__ Interval kbPrimeInterval(
    const Camera& camera, Interval theta) {
    const Interval theta2 = multiply(theta, theta);
    Interval horner = add(point(7.0 * camera.k3),
                          multiply(theta2, point(9.0 * camera.k4)));
    horner = add(point(5.0 * camera.k2), multiply(theta2, horner));
    horner = add(point(3.0 * camera.k1), multiply(theta2, horner));
    return add(point(1.0), multiply(theta2, horner));
}

static inline __device__ Interval kbSecondInterval(
    const Camera& camera, Interval theta) {
    const Interval theta2 = multiply(theta, theta);
    Interval horner = add(point(42.0 * camera.k3),
                          multiply(theta2, point(72.0 * camera.k4)));
    horner = add(point(20.0 * camera.k2), multiply(theta2, horner));
    horner = add(point(6.0 * camera.k1), multiply(theta2, horner));
    return multiply(theta, horner);
}

static inline __device__ double kbRadius(
    const Camera& camera, double theta) {
    const double theta2 = theta * theta;
    return theta * (1.0 + theta2 * (
        camera.k1 + theta2 * (
            camera.k2 + theta2 * (
                camera.k3 + theta2 * camera.k4))));
}

static inline __device__ double kbPrime(
    const Camera& camera, double theta) {
    const double theta2 = theta * theta;
    return 1.0 + theta2 * (
        3.0 * camera.k1 + theta2 * (
            5.0 * camera.k2 + theta2 * (
                7.0 * camera.k3 + theta2 * 9.0 * camera.k4)));
}

static inline __device__ bool distanceLower(
    float3 mean, double& output) {
    const double component[3] = {
        double(mean.x), double(mean.y), double(mean.z),
    };
    double squared = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        squared = __dadd_rd(
            squared, __dmul_rd(component[axis], component[axis]));
    }
    if (!(squared > 0.0) || !isfinite(squared)) return false;
    output = __dsqrt_rd(squared);
    return output > 0.0 && isfinite(output);
}

static inline __device__ bool qLowerAt(
    double theta, double thetaCenter, double rho,
    double& output) {
    const Interval sineTheta = outwardTranscendental(sin(theta));
    const Interval sineCenter = outwardTranscendental(sin(thetaCenter));
    const Interval denominator = multiply(sineTheta, sineCenter);
    if (!(denominator.lo > 0.0) || !finite(denominator)) return false;
    const Interval numerator = subtract(
        outwardTranscendental(cos(rho)),
        multiply(outwardTranscendental(cos(theta)),
                 outwardTranscendental(cos(thetaCenter))));
    if (!finite(numerator)) return false;
    output = __ddiv_rd(numerator.lo, denominator.hi);
    return isfinite(output);
}

static inline __device__ bool certifiedAzimuthHalf(
    double thetaLower, double thetaUpper,
    double thetaCenter, double rho,
    double& phiHalfUpper) {
    double qLower = 0.0;
    double qUpper = 0.0;
    if (!qLowerAt(thetaLower, thetaCenter, rho, qLower)
        || !qLowerAt(thetaUpper, thetaCenter, rho, qUpper)) return false;
    double qMinimum = fmin(qLower, qUpper);

    const Interval cosineRho = outwardTranscendental(cos(rho));
    const Interval cosineCenter = outwardTranscendental(cos(thetaCenter));
    if (!(cosineRho.lo > 0.0)
        || !finite(cosineRho) || !finite(cosineCenter)) return false;
    const double ratioLo[4] = {
        __ddiv_rd(cosineCenter.lo, cosineRho.lo),
        __ddiv_rd(cosineCenter.lo, cosineRho.hi),
        __ddiv_rd(cosineCenter.hi, cosineRho.lo),
        __ddiv_rd(cosineCenter.hi, cosineRho.hi),
    };
    const double ratioHi[4] = {
        __ddiv_ru(cosineCenter.lo, cosineRho.lo),
        __ddiv_ru(cosineCenter.lo, cosineRho.hi),
        __ddiv_ru(cosineCenter.hi, cosineRho.lo),
        __ddiv_ru(cosineCenter.hi, cosineRho.hi),
    };
    const Interval ratio{
        fmin(fmin(ratioLo[0], ratioLo[1]), fmin(ratioLo[2], ratioLo[3])),
        fmax(fmax(ratioHi[0], ratioHi[1]), fmax(ratioHi[2], ratioHi[3])),
    };
    if (!finite(ratio)) return false;
    if (ratio.lo <= 1.0 && ratio.hi >= -1.0) {
        if (ratio.lo < -1.0 || ratio.hi > 1.0) return false;
        const Interval stationary{
            transcendentalLower(acos(ratio.hi)),
            transcendentalUpper(acos(ratio.lo)),
        };
        if (!finite(stationary)) return false;
        if (stationary.hi < thetaLower || stationary.lo > thetaUpper) {
            // Endpoints own the minimum q.
        } else if (stationary.lo >= thetaLower
                   && stationary.hi <= thetaUpper) {
            const Interval square = subtract(
                multiply(cosineRho, cosineRho),
                multiply(cosineCenter, cosineCenter));
            const Interval sineCenter = outwardTranscendental(
                sin(thetaCenter));
            if (!(square.lo >= 0.0) || !(sineCenter.lo > 0.0)
                || !finite(square) || !finite(sineCenter)) return false;
            const double stationaryQLower = __ddiv_rd(
                __dsqrt_rd(square.lo), sineCenter.hi);
            if (!isfinite(stationaryQLower)) return false;
            qMinimum = fmin(qMinimum, stationaryQLower);
        } else {
            // A stationary point that cannot be classified is uncertainty.
            return false;
        }
    }
    if (!(qMinimum <= 1.0) || qMinimum <= -1.0
        || !isfinite(qMinimum)) return false;
    phiHalfUpper = transcendentalUpper(
        acos(fmax(-1.0, qMinimum)));
    return phiHalfUpper >= 0.0 && phiHalfUpper < kPi
        && isfinite(phiHalfUpper);
}

// Certifies only boundary-crossing support. Interior support is handled by
// the residual provider. Outside or uncertain support is ExactCone;
// this certificate never emits EmptySupport and therefore never prunes.
static inline __device__ Result certify(
    const Camera& camera,
    float3 meanFromCamera,
    double supportBallRadiusUpper) {
    Result result{};
    result.route = Route::ExactCone;
    if (!(camera.fx > 0.0) || !(camera.fy > 0.0)
        || !(camera.thetaMax > 0.0) || !(camera.thetaMax < kPi)
        || !isfinite(camera.fx) || !isfinite(camera.fy)
        || !isfinite(camera.cx) || !isfinite(camera.cy)
        || !isfinite(camera.k1) || !isfinite(camera.k2)
        || !isfinite(camera.k3) || !isfinite(camera.k4)
        || !isfinite(camera.thetaMax)
        || !(supportBallRadiusUpper >= 0.0)
        || !isfinite(supportBallRadiusUpper)
        || !isfinite(meanFromCamera.x)
        || !isfinite(meanFromCamera.y)
        || !isfinite(meanFromCamera.z)) return result;

    double distance = 0.0;
    if (!distanceLower(meanFromCamera, distance)
        || !(supportBallRadiusUpper < distance)) return result;
    const double ratioUpper = __ddiv_ru(supportBallRadiusUpper, distance);
    if (!(ratioUpper >= 0.0) || !(ratioUpper < 1.0)
        || !isfinite(ratioUpper)) return result;
    const Interval ballRho = outwardTranscendental(asin(ratioUpper));

    const double x = double(meanFromCamera.x);
    const double y = double(meanFromCamera.y);
    const double z = double(meanFromCamera.z);
    const double radialSquaredLower = __dadd_rd(
        __dmul_rd(x, x), __dmul_rd(y, y));
    const double radialSquaredUpper = __dadd_ru(
        __dmul_ru(x, x), __dmul_ru(y, y));
    if (radialSquaredLower < 0.0 || radialSquaredUpper < radialSquaredLower
        || !isfinite(radialSquaredLower)
        || !isfinite(radialSquaredUpper)) return result;
    const Interval radialInterval{
        __dsqrt_rd(radialSquaredLower),
        __dsqrt_ru(radialSquaredUpper),
    };
    if (!finite(radialInterval)) return result;
    if (radialInterval.hi == 0.0) {
        if (z > 0.0 && ballRho.hi <= camera.thetaMax)
            result.route = Route::Interior;
        return result;
    }
    if (!(radialInterval.lo > 0.0)) return result;
    const double radial = sqrt(x * x + y * y);
    const double thetaCenter = atan2(radial, z);
    const double phiCenter = atan2(y, x);
    const Interval thetaAtLower = outwardTranscendental(
        atan2(radialInterval.lo, z));
    const Interval thetaAtUpper = outwardTranscendental(
        atan2(radialInterval.hi, z));
    const Interval thetaEnclosure{
        fmin(thetaAtLower.lo, thetaAtUpper.lo),
        fmax(thetaAtLower.hi, thetaAtUpper.hi),
    };
    const Interval phiEnclosure = outwardTranscendental(phiCenter);
    if (!finite(ballRho) || !finite(thetaEnclosure) || !finite(phiEnclosure))
        return result;
    const double thetaError = fmax(
        __dsub_ru(thetaCenter, thetaEnclosure.lo),
        __dsub_ru(thetaEnclosure.hi, thetaCenter));
    const double phiError = fmax(
        __dsub_ru(phiCenter, phiEnclosure.lo),
        __dsub_ru(phiEnclosure.hi, phiCenter));
    const double rho = __dadd_ru(
        ballRho.hi, __dadd_ru(thetaError, phiError));
    if (!(rho >= 0.0) || !(rho < 0.5 * kPi) || !isfinite(rho)) return result;

    const double thetaLower = fmax(
        0.0, __dsub_rd(thetaCenter, rho));
    const double thetaUpper = __dadd_ru(thetaCenter, rho);
    if (!isfinite(thetaLower) || !isfinite(thetaUpper)
        || thetaLower > thetaUpper) return result;
    if (thetaUpper <= camera.thetaMax) {
        result.route = Route::Interior;
        return result;
    }
    // Strict outside is deliberately ExactCone: this certificate does not prune.
    if (thetaLower > camera.thetaMax || !(thetaLower > 0.0)) return result;

    double phiHalf = 0.0;
    if (!certifiedAzimuthHalf(
            thetaLower, camera.thetaMax,
            thetaCenter, rho, phiHalf)) return result;
    // Keep one non-wrapping angular chart; branch-cut uncertainty is ExactCone.
    if (!(phiCenter - phiHalf > -kPi)
        || !(phiCenter + phiHalf < kPi)) return result;

    const double deltaTheta = __dsub_ru(camera.thetaMax, thetaLower);
    const double deltaPhi = phiHalf;
    if (!(deltaTheta >= 0.0) || !(deltaPhi >= 0.0)
        || !(deltaPhi < kPi)) return result;

    const Interval thetaBand{thetaLower, camera.thetaMax};
    const Interval radiusInterval = kbRadiusInterval(camera, thetaBand);
    const Interval primeInterval = kbPrimeInterval(camera, thetaBand);
    const Interval secondInterval = kbSecondInterval(camera, thetaBand);
    if (!finite(radiusInterval) || !finite(primeInterval)
        || !finite(secondInterval)) return result;
    const double radiusAbs = absoluteUpper(radiusInterval);
    const double primeAbs = absoluteUpper(primeInterval);
    const double secondAbs = absoluteUpper(secondInterval);
    const Interval boundaryTheta = point(camera.thetaMax);
    const Interval radius0Interval = kbRadiusInterval(camera, boundaryTheta);
    const Interval prime0Interval = kbPrimeInterval(camera, boundaryTheta);
    const double radius0 = kbRadius(camera, camera.thetaMax);
    const double prime0 = kbPrime(camera, camera.thetaMax);
    if (!isfinite(radiusAbs) || !isfinite(primeAbs)
        || !isfinite(secondAbs) || !isfinite(radius0)
        || !isfinite(prime0) || !finite(radius0Interval)
        || !finite(prime0Interval)) return result;

    const double cosinePhi = cos(phiCenter);
    const double sinePhi = sin(phiCenter);
    const Interval cosinePhiInterval = outwardTranscendental(cosinePhi);
    const Interval sinePhiInterval = outwardTranscendental(sinePhi);
    if (!finite(cosinePhiInterval) || !finite(sinePhiInterval)) return result;
    const double centerX = camera.cx + camera.fx * radius0 * cosinePhi;
    const double centerY = camera.cy + camera.fy * radius0 * sinePhi;
    const Interval centerXInterval = add(
        point(camera.cx), multiply(
            point(camera.fx), multiply(radius0Interval, cosinePhiInterval)));
    const Interval centerYInterval = add(
        point(camera.cy), multiply(
            point(camera.fy), multiply(radius0Interval, sinePhiInterval)));
    if (!finite(centerXInterval) || !finite(centerYInterval)) return result;
    const double duTheta = __dmul_ru(
        fabs(camera.fx), __dmul_ru(
            absoluteUpper(prime0Interval), absoluteUpper(cosinePhiInterval)));
    const double duPhi = __dmul_ru(
        fabs(camera.fx), __dmul_ru(
            absoluteUpper(radius0Interval), absoluteUpper(sinePhiInterval)));
    const double dvTheta = __dmul_ru(
        fabs(camera.fy), __dmul_ru(
            absoluteUpper(prime0Interval), absoluteUpper(sinePhiInterval)));
    const double dvPhi = __dmul_ru(
        fabs(camera.fy), __dmul_ru(
            absoluteUpper(radius0Interval), absoluteUpper(cosinePhiInterval)));
    const double deltaSquared = __dadd_ru(
        __dmul_ru(deltaTheta, deltaTheta),
        __dmul_ru(deltaPhi, deltaPhi));
    const double hessianCommon = __dsqrt_ru(__dadd_ru(
        __dmul_ru(secondAbs, secondAbs),
        __dadd_ru(
            __dmul_ru(2.0, __dmul_ru(primeAbs, primeAbs)),
            __dmul_ru(radiusAbs, radiusAbs))));
    const double remainderX = __dmul_ru(
        __dmul_ru(0.5, fabs(camera.fx)),
        __dmul_ru(hessianCommon, deltaSquared));
    const double remainderY = __dmul_ru(
        __dmul_ru(0.5, fabs(camera.fy)),
        __dmul_ru(hessianCommon, deltaSquared));
    const float centerXFloat = __double2float_rn(centerX);
    const float centerYFloat = __double2float_rn(centerY);
    const double centerErrorX = fmax(
        __dsub_ru(double(centerXFloat), centerXInterval.lo),
        __dsub_ru(centerXInterval.hi, double(centerXFloat)));
    const double centerErrorY = fmax(
        __dsub_ru(double(centerYFloat), centerYInterval.lo),
        __dsub_ru(centerYInterval.hi, double(centerYFloat)));
    const double extentX = __dadd_ru(
        __dadd_ru(
            __dmul_ru(duTheta, deltaTheta),
            __dmul_ru(duPhi, deltaPhi)),
        __dadd_ru(remainderX, centerErrorX));
    const double extentY = __dadd_ru(
        __dadd_ru(
            __dmul_ru(dvTheta, deltaTheta),
            __dmul_ru(dvPhi, deltaPhi)),
        __dadd_ru(remainderY, centerErrorY));
    if (!isfinite(centerX) || !isfinite(centerY)
        || !isfinite(extentX) || !isfinite(extentY)
        || !(extentX > 0.0) || !(extentY > 0.0)) return result;

    result.rectangle.center = make_float2(centerXFloat, centerYFloat);
    result.rectangle.halfExtent = make_float2(
        __double2float_ru(extentX), __double2float_ru(extentY));
    result.rectangle.thetaLower = thetaLower;
    result.rectangle.thetaUpper = camera.thetaMax;
    result.rectangle.phiHalfUpper = phiHalf;
    result.rectangle.supportBallRadiusUpper = supportBallRadiusUpper;
    result.route = Route::CertifiedRectangle;
    return result;
}

} // namespace gofish_boundary_angular

