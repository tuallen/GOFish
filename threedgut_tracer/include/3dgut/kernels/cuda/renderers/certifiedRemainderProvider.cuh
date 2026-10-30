// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderAdmission.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedBoundaryAngularFootprint.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedOpacityRadius.cuh>
#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>
#include <3dgut/renderer/renderParameters.h>
#include <3dgut/sensors/cameraModels.h>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>

namespace gofish_certified_remainder_provider {

// The provider consumes the renderer's exact OpenCV-fisheye or ideal-pinhole
// payload directly; no retyped camera model or approximate projection exists.
using gofish_certified_remainder::AdmissionInput;
using gofish_certified_remainder::EffectiveShapeInput;
using gofish_certified_remainder::InputProviderStatus;
using gofish_certified_remainder::Interval;
using gofish_certified_remainder::ProviderContractMarker;

// Local carrier populated directly from Particles::fetchDensityParameters.
// No producer-side factor/opacity/position arrays exist.
struct RendererEffectiveShape {
    float factorWorld[9];
    float inverseFactorWorld[9];
    uint64_t shapeContract[4];
};

// One host-selected table descriptor and its exact native/active KB transport.
// There is no camera index or fallback: the provider accepts this descriptor
// only when all active parameters are the exact full-raster affine image of
// the native calibration and distortion/max-angle bits are unchanged.
struct SelectedKbCameraDescriptor {
    const gofish::kb_hessian_abi::CameraDescriptor* descriptor;
    threedgut::OpenCVFisheyeProjectionParameters nativeProjection;
    threedgut::OpenCVFisheyeProjectionParameters activeProjection;
    tcnn::ivec2 activeResolution;
    // Outward float enclosure of the physical ray sample in the active
    // projection chart. Native and active COLMAP charts share the corner
    // origin, and every active raster sample is exactly i + 1/2.
    float2 physicalSampleOffsetLower;
    float2 physicalSampleOffsetUpper;
    uint64_t linearizationContract[4];
    uint32_t hostSelectionValidated;
    // The descriptor layout is append-only: every fisheye field keeps
    // its offset and byte representation.
    uint32_t modelId;
    // 0: native COLMAP corner calibration scaled to the active corner chart.
    // 1: selector and descriptor are already in the active corner chart.
    uint32_t chartMode;
    threedgut::OpenCVPinholeProjectionParameters nativePinholeProjection;
    threedgut::OpenCVPinholeProjectionParameters activePinholeProjection;
};

// Provider source has no UT fields and no renderer-produced transport arrays.
// worldToCamera is row-major [R|t]. Density state is fetched from MemoryHandles
// through the Particles implementation exactly as gutProjector does.
struct Source {
    threedgut::MemoryHandles parameters;
    float worldToCamera[12];
    SelectedKbCameraDescriptor camera;
    uint64_t shapeContract[4];
};

static inline __device__ bool finiteNormal(float value) {
    const float magnitude = fabsf(value);
    return isfinite(value)
        && !(magnitude > 0.0f && magnitude < 0x1.0p-126f);
}

static inline __device__ Interval point(double value) {
    return Interval{value, value};
}

static inline __device__ Interval intervalAdd(Interval a, Interval b) {
    return Interval{__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)};
}

static inline __device__ Interval intervalSub(Interval a, Interval b) {
    return Interval{__dsub_rd(a.lo, b.hi), __dsub_ru(a.hi, b.lo)};
}

static inline __device__ Interval intervalMul(Interval a, Interval b) {
    const double lo0 = __dmul_rd(a.lo, b.lo);
    const double lo1 = __dmul_rd(a.lo, b.hi);
    const double lo2 = __dmul_rd(a.hi, b.lo);
    const double lo3 = __dmul_rd(a.hi, b.hi);
    const double hi0 = __dmul_ru(a.lo, b.lo);
    const double hi1 = __dmul_ru(a.lo, b.hi);
    const double hi2 = __dmul_ru(a.hi, b.lo);
    const double hi3 = __dmul_ru(a.hi, b.hi);
    return Interval{
        fmin(fmin(lo0, lo1), fmin(lo2, lo3)),
        fmax(fmax(hi0, hi1), fmax(hi2, hi3)),
    };
}

static inline __device__ Interval intervalDivPositive(Interval a, Interval b) {
    if (!(b.lo > 0.0) || !isfinite(b.lo) || !isfinite(b.hi)) {
        return Interval{-CUDART_INF, CUDART_INF};
    }
    const Interval reciprocal{__ddiv_rd(1.0, b.hi), __ddiv_ru(1.0, b.lo)};
    return intervalMul(a, reciprocal);
}

static inline __device__ Interval intervalSqrtPositive(Interval value) {
    if (value.lo < 0.0 || !isfinite(value.lo) || !isfinite(value.hi)) {
        return Interval{-CUDART_INF, CUDART_INF};
    }
    return Interval{__dsqrt_rd(value.lo), __dsqrt_ru(value.hi)};
}

static inline __device__ Interval outwardTranscendental(double value) {
    if (!isfinite(value)) return Interval{-CUDART_INF, CUDART_INF};
    return Interval{
        nextafter(value, -CUDART_INF), nextafter(value, CUDART_INF)};
}

// Exact broad-phase rejection against the calibration-derived angular domain.
// The renderer support is contained in the world-space ball B(w,R), where R
// uses the same opacity cutoff and the residual-certified ||A^{-1}|| bound.
// For ||w||>R, every support direction lies within asin(R/||w||) of w.
// If even the lower edge of that cap is beyond maxAngle, no physical pixel ray
// can meet the support.  Uncertain and boundary cases remain on the complete
// ellipse/cone path; this routine can only prove EmptySupport.
static inline __device__ bool provablyOutsideActiveAngularDomain(
    const AdmissionInput& input, float maxAngle) {
    double distanceLower = 0.0;
    double supportBallRadiusUpper = 0.0;
    double dMinLower = 0.0;
    if (!gofish_certified_remainder::cameraOutsideProof(
            input, distanceLower,
            supportBallRadiusUpper, dMinLower)) return false;
    if (!(maxAngle > 0.0f) || !isfinite(maxAngle)) return false;

    const float3 w = input.meanFromCamera;
    const double rho = hypot(double(w.x), double(w.y));
    const Interval theta = outwardTranscendental(
        atan2(rho, double(w.z)));
    const double ratioUpper = __ddiv_ru(
        supportBallRadiusUpper, distanceLower);
    if (!(ratioUpper >= 0.0) || !(ratioUpper < 1.0)
        || !isfinite(ratioUpper)) return false;
    const Interval halfAngle = outwardTranscendental(asin(ratioUpper));
    const double supportThetaLower = __dsub_rd(theta.lo, halfAngle.hi);
    return isfinite(supportThetaLower)
        && supportThetaLower > double(maxAngle);
}

static inline __device__ Interval scaledCoefficient(float coefficient, int scale) {
    return Interval{
        __dmul_rd(double(coefficient), double(scale)),
        __dmul_ru(double(coefficient), double(scale)),
    };
}

static inline __device__ Interval horner4(
    Interval x, Interval c0, Interval c1, Interval c2, Interval c3) {
    Interval value = c3;
    value = intervalAdd(intervalMul(x, value), c2);
    value = intervalAdd(intervalMul(x, value), c1);
    value = intervalAdd(intervalMul(x, value), c0);
    return value;
}

static inline __device__ bool sameFloatBits(float left, float right) {
    return __float_as_uint(left) == __float_as_uint(right);
}

static inline __device__ bool selectedPinholeCameraMatchesActive(
    const SelectedKbCameraDescriptor& camera) {
    using gofish::kb_hessian_abi::kModelIdealPinhole;
    const auto* descriptor = camera.descriptor;
    if (camera.hostSelectionValidated != 1u || descriptor == nullptr
        || camera.modelId != kModelIdealPinhole || camera.chartMode > 1u
        || descriptor->model_id != kModelIdealPinhole
        || descriptor->native_width == 0u || descriptor->native_height == 0u
        || descriptor->reserved != 0u
        || !isfinite(descriptor->theta_max_radians)
        || camera.activeResolution.x <= 0 || camera.activeResolution.y <= 0)
        return false;
    if (camera.chartMode == 1u
        && (camera.activeResolution.x != int(descriptor->native_width)
            || camera.activeResolution.y != int(descriptor->native_height)))
        return false;
    const float xScale = camera.chartMode == 1u ? 1.0f : __fdiv_rn(
        float(camera.activeResolution.x), float(descriptor->native_width));
    const float yScale = camera.chartMode == 1u ? 1.0f : __fdiv_rn(
        float(camera.activeResolution.y), float(descriptor->native_height));
    if (!finiteNormal(xScale) || !finiteNormal(yScale)
        || !(xScale > 0.0f) || !(yScale > 0.0f)) return false;
    const auto& native = camera.nativePinholeProjection;
    const auto& active = camera.activePinholeProjection;
    if (!finiteNormal(native.focalLength.x)
        || !finiteNormal(native.focalLength.y)
        || !finiteNormal(native.principalPoint.x)
        || !finiteNormal(native.principalPoint.y)
        || !(native.focalLength.x > 0.0f)
        || !(native.focalLength.y > 0.0f)) return false;
#pragma unroll
    for (int index = 0; index < 6; ++index) {
        if (!sameFloatBits(native.radialCoeffs[index], 0.0f)
            || !sameFloatBits(active.radialCoeffs[index], 0.0f)) return false;
    }
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        if (!sameFloatBits(native.tangentialCoeffs[index], 0.0f)
            || !sameFloatBits(active.tangentialCoeffs[index], 0.0f)) return false;
    }
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        if (!sameFloatBits(native.thinPrismCoeffs[index], 0.0f)
            || !sameFloatBits(active.thinPrismCoeffs[index], 0.0f)) return false;
    }
    return sameFloatBits(
               active.focalLength.x, native.focalLength.x * xScale)
        && sameFloatBits(
               active.focalLength.y, native.focalLength.y * yScale)
        && sameFloatBits(active.principalPoint.x, camera.chartMode == 1u
               ? native.principalPoint.x
               : native.principalPoint.x * xScale)
        && sameFloatBits(active.principalPoint.y, camera.chartMode == 1u
               ? native.principalPoint.y
               : native.principalPoint.y * yScale);
}

static inline __device__ bool selectedCameraMatchesActive(
    const SelectedKbCameraDescriptor& camera) {
    if (camera.modelId == gofish::kb_hessian_abi::kModelIdealPinhole) {
        return selectedPinholeCameraMatchesActive(camera);
    }
    using gofish::kb_hessian_abi::kModelOpenCvFisheyeKb4;
    const auto* descriptor = camera.descriptor;
    if (camera.hostSelectionValidated != 1u || descriptor == nullptr
        || descriptor->model_id != kModelOpenCvFisheyeKb4
        || camera.chartMode > 1u
        || descriptor->native_width == 0u || descriptor->native_height == 0u
        || descriptor->reserved != 0u
        || !isfinite(descriptor->theta_max_radians)
        || camera.activeResolution.x <= 0 || camera.activeResolution.y <= 0)
        return false;
    const auto& native = camera.nativeProjection;
    const auto& active = camera.activeProjection;
    if (!finiteNormal(native.focalLength.x)
        || !finiteNormal(native.focalLength.y)
        || !finiteNormal(native.principalPoint.x)
        || !finiteNormal(native.principalPoint.y)
        || !finiteNormal(native.maxAngle)
        || !finiteNormal(active.maxAngle)
        || !(native.focalLength.x > 0.0f)
        || !(native.focalLength.y > 0.0f)
        || !(active.maxAngle > 0.0f)
        || !sameFloatBits(native.maxAngle,
                          float(descriptor->theta_max_radians))
        || !sameFloatBits(active.maxAngle, native.maxAngle)) return false;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        if (!finiteNormal(native.radialCoeffs[index])
            || !sameFloatBits(native.radialCoeffs[index],
                              active.radialCoeffs[index])) return false;
    }
    // COLMAP native-corner to active-corner transport:
    // p_active=diag(Wa/Wn,Ha/Hn)*p_native. Compare the exact float
    // operation used to produce active intrinsics; no tolerance or fallback.
    if (camera.chartMode == 1u
        && (camera.activeResolution.x != int(descriptor->native_width)
            || camera.activeResolution.y != int(descriptor->native_height)))
        return false;
    const float xScale = camera.chartMode == 1u ? 1.0f : __fdiv_rn(
        float(camera.activeResolution.x), float(descriptor->native_width));
    const float yScale = camera.chartMode == 1u ? 1.0f : __fdiv_rn(
        float(camera.activeResolution.y), float(descriptor->native_height));
    return finiteNormal(xScale) && finiteNormal(yScale)
        && xScale > 0.0f && yScale > 0.0f
        && sameFloatBits(active.focalLength.x, native.focalLength.x * xScale)
        && sameFloatBits(active.focalLength.y, native.focalLength.y * yScale)
        && sameFloatBits(active.principalPoint.x, camera.chartMode == 1u
                         ? native.principalPoint.x
                         : native.principalPoint.x * xScale)
        && sameFloatBits(active.principalPoint.y, camera.chartMode == 1u
                         ? native.principalPoint.y
                         : native.principalPoint.y * yScale);
}

static inline __device__ float3 transformMean(
    const float matrix[12], float3 world) {
    return make_float3(
        matrix[0] * world.x + matrix[1] * world.y
            + matrix[2] * world.z + matrix[3],
        matrix[4] * world.x + matrix[5] * world.y
            + matrix[6] * world.z + matrix[7],
        matrix[8] * world.x + matrix[9] * world.y
            + matrix[10] * world.z + matrix[11]);
}

static inline __device__ void cameraFrameShape(
    const float matrix[12], const RendererEffectiveShape& source,
    EffectiveShapeInput& output) {
    // B_cam=R_cw B_world.
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            float value = 0.0f;
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                value += matrix[4 * row + inner]
                    * source.factorWorld[3 * inner + col];
            }
            output.factor[3 * row + col] = value;
        }
    }
    // B_cam^{-1}=B_world^{-1} R_cw^T.
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            float value = 0.0f;
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                value += source.inverseFactorWorld[3 * row + inner]
                    * matrix[4 * col + inner];
            }
            output.inverseFactor[3 * row + col] = value;
        }
    }
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        output.shapeContract[index] = source.shapeContract[index];
    }
}

struct KbIntervals {
    Interval center[2];
    Interval jacobian[6];
    Interval theta;
};

// Directed interval enclosure of the exact ideal-pinhole projection and its
// Jacobian at the renderer's float camera-space mean.  The table carries an
// analytic Hessian bound over the finite calibrated raster; support that
// approaches/crosses z=0 is sent to the exact ray-cone path instead.
static inline __device__ bool continuousPinholeIntervals(
    const threedgut::OpenCVPinholeProjectionParameters& camera,
    float3 w, KbIntervals& output) {
    if (!(w.z > 0.0f) || !finiteNormal(w.x)
        || !finiteNormal(w.y) || !finiteNormal(w.z)) return false;
    const Interval x = point(double(w.x));
    const Interval y = point(double(w.y));
    const Interval z = point(double(w.z));
    const Interval invZ = intervalDivPositive(point(1.0), z);
    const Interval invZ2 = intervalMul(invZ, invZ);
    const Interval fx = point(double(camera.focalLength.x));
    const Interval fy = point(double(camera.focalLength.y));
    output.center[0] = intervalAdd(
        intervalMul(fx, intervalMul(x, invZ)),
        point(double(camera.principalPoint.x)));
    output.center[1] = intervalAdd(
        intervalMul(fy, intervalMul(y, invZ)),
        point(double(camera.principalPoint.y)));
    output.jacobian[0] = intervalMul(fx, invZ);
    output.jacobian[1] = point(0.0);
    output.jacobian[2] = intervalMul(
        point(-1.0), intervalMul(fx, intervalMul(x, invZ2)));
    output.jacobian[3] = point(0.0);
    output.jacobian[4] = intervalMul(fy, invZ);
    output.jacobian[5] = intervalMul(
        point(-1.0), intervalMul(fy, intervalMul(y, invZ2)));
    const double rho = hypot(double(w.x), double(w.y));
    output.theta = outwardTranscendental(atan2(rho, double(w.z)));
#pragma unroll
    for (int index = 0; index < 2; ++index) {
        if (!gofish_certified_remainder::finiteInterval(output.center[index]))
            return false;
    }
#pragma unroll
    for (int index = 0; index < 6; ++index) {
        if (!gofish_certified_remainder::finiteInterval(output.jacobian[index]))
            return false;
    }
    return gofish_certified_remainder::finiteInterval(output.theta)
        && output.theta.lo >= 0.0;
}

// Continuous KB centre/J interval at the same float camera-space mean consumed
// by the active projector. Transcendental point evaluations are expanded by one
// binary64 neighbour before directed algebra.
static inline __device__ bool continuousKbIntervals(
    const threedgut::OpenCVFisheyeProjectionParameters& camera,
    float3 w, KbIntervals& output) {
    const Interval x = point(double(w.x));
    const Interval y = point(double(w.y));
    const Interval z = point(double(w.z));
    Interval rhoSquared = intervalAdd(intervalMul(x, x), intervalMul(y, y));
    const Interval rho = intervalSqrtPositive(rhoSquared);
    if (!isfinite(rho.lo) || !isfinite(rho.hi)) return false;

    if (rho.hi == 0.0) {
        if (!(w.z > 0.0f)) return false;
        output.center[0] = point(double(camera.principalPoint.x));
        output.center[1] = point(double(camera.principalPoint.y));
        const Interval invZ = intervalDivPositive(point(1.0), z);
        output.jacobian[0] = intervalMul(point(double(camera.focalLength.x)), invZ);
        output.jacobian[1] = point(0.0);
        output.jacobian[2] = point(0.0);
        output.jacobian[3] = point(0.0);
        output.jacobian[4] = intervalMul(point(double(camera.focalLength.y)), invZ);
        output.jacobian[5] = point(0.0);
        output.theta = point(0.0);
        return true;
    }
    if (!(rho.lo > 0.0)) return false;

    const double rhoNominal = hypot(double(w.x), double(w.y));
    const double thetaNominal = atan2(rhoNominal, double(w.z));
    output.theta = outwardTranscendental(thetaNominal);
    if (!isfinite(output.theta.lo) || !isfinite(output.theta.hi)
        || output.theta.lo < 0.0
        || output.theta.hi >= double(camera.maxAngle)) return false;

    const Interval t2 = intervalMul(output.theta, output.theta);
    const Interval k0 = point(double(camera.radialCoeffs[0]));
    const Interval k1 = point(double(camera.radialCoeffs[1]));
    const Interval k2 = point(double(camera.radialCoeffs[2]));
    const Interval k3 = point(double(camera.radialCoeffs[3]));
    const Interval radial = horner4(t2, k0, k1, k2, k3);
    const Interval onePlus = intervalAdd(point(1.0), intervalMul(t2, radial));
    const Interval q = intervalMul(output.theta, onePlus);
    // q'=1+t^2(3k1+t^2(5k2+t^2(7k3+t^2 9k4))).
    Interval derivative = scaledCoefficient(camera.radialCoeffs[3], 9);
    derivative = intervalAdd(
        intervalMul(t2, derivative),
        scaledCoefficient(camera.radialCoeffs[2], 7));
    derivative = intervalAdd(
        intervalMul(t2, derivative),
        scaledCoefficient(camera.radialCoeffs[1], 5));
    derivative = intervalAdd(
        intervalMul(t2, derivative),
        scaledCoefficient(camera.radialCoeffs[0], 3));
    derivative = intervalAdd(intervalMul(t2, derivative), point(1.0));
    const Interval scale = intervalDivPositive(q, rho);
    output.center[0] = intervalAdd(
        intervalMul(
            point(double(camera.focalLength.x)),
            intervalMul(x, scale)),
        point(double(camera.principalPoint.x)));
    output.center[1] = intervalAdd(
        intervalMul(
            point(double(camera.focalLength.y)),
            intervalMul(y, scale)),
        point(double(camera.principalPoint.y)));

    const Interval r2 = intervalAdd(rhoSquared, intervalMul(z, z));
    const Interval dThetaX = intervalDivPositive(
        intervalMul(z, x), intervalMul(rho, r2));
    const Interval dThetaY = intervalDivPositive(
        intervalMul(z, y), intervalMul(rho, r2));
    const Interval dThetaZ = intervalMul(
        point(-1.0), intervalDivPositive(rho, r2));
    const Interval dRhoX = intervalDivPositive(x, rho);
    const Interval dRhoY = intervalDivPositive(y, rho);
    const Interval rho2 = intervalMul(rho, rho);
    const Interval dScaleX = intervalDivPositive(
        intervalSub(
            intervalMul(intervalMul(derivative, dThetaX), rho),
            intervalMul(q, dRhoX)), rho2);
    const Interval dScaleY = intervalDivPositive(
        intervalSub(
            intervalMul(intervalMul(derivative, dThetaY), rho),
            intervalMul(q, dRhoY)), rho2);
    const Interval dScaleZ = intervalDivPositive(
        intervalMul(intervalMul(derivative, dThetaZ), rho), rho2);

    output.jacobian[0] = intervalMul(
        point(double(camera.focalLength.x)),
        intervalAdd(scale, intervalMul(x, dScaleX)));
    output.jacobian[1] = intervalMul(
        point(double(camera.focalLength.x)), intervalMul(x, dScaleY));
    output.jacobian[2] = intervalMul(
        point(double(camera.focalLength.x)), intervalMul(x, dScaleZ));
    output.jacobian[3] = intervalMul(
        point(double(camera.focalLength.y)), intervalMul(y, dScaleX));
    output.jacobian[4] = intervalMul(
        point(double(camera.focalLength.y)),
        intervalAdd(scale, intervalMul(y, dScaleY)));
    output.jacobian[5] = intervalMul(
        point(double(camera.focalLength.y)), intervalMul(y, dScaleZ));

#pragma unroll
    for (int index = 0; index < 2; ++index) {
        if (!gofish_certified_remainder::finiteInterval(output.center[index]))
            return false;
    }
#pragma unroll
    for (int index = 0; index < 6; ++index) {
        if (!gofish_certified_remainder::finiteInterval(output.jacobian[index]))
            return false;
    }
    return true;
}

static inline __device__ double centerErrorUpper(
    float reference, Interval enclosure) {
    const double ref = double(reference);
    const double left = __dsub_ru(ref, enclosure.lo);
    const double right = __dsub_ru(enclosure.hi, ref);
    return fmax(fmax(left, right), 0.0);
}

static inline __device__ bool thetaSupport(
    float3 w, Interval centerTheta,
    double canonicalRadiusUpper, double shapeSupportSigmaUpper,
    float maxAngle, double& supportRadiusUpper,
    double& thetaLower, double& thetaUpper) {
    if (!isfinite(shapeSupportSigmaUpper)
        || !(shapeSupportSigmaUpper > 0.0)) return false;
    supportRadiusUpper = __dmul_ru(
        canonicalRadiusUpper, shapeSupportSigmaUpper);
    double distanceSquaredLower = 0.0;
    const double component[3] = {double(w.x), double(w.y), double(w.z)};
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        distanceSquaredLower = __dadd_rd(
            distanceSquaredLower,
            __dmul_rd(component[axis], component[axis]));
    }
    const double distanceLower = __dsqrt_rd(distanceSquaredLower);
    if (!(supportRadiusUpper < distanceLower)
        || !isfinite(supportRadiusUpper)) return false;
    const double ratioUpper = __ddiv_ru(supportRadiusUpper, distanceLower);
    if (!(ratioUpper >= 0.0 && ratioUpper < 1.0)) return false;
    const double rhoUpper = nextafter(asin(ratioUpper), CUDART_INF);
    if (!isfinite(rhoUpper)) return false;
    thetaLower = fmax(0.0, __dsub_rd(centerTheta.lo, rhoUpper));
    thetaUpper = __dadd_ru(centerTheta.hi, rhoUpper);
    return isfinite(thetaLower) && isfinite(thetaUpper)
        && thetaLower <= thetaUpper && thetaUpper <= double(maxAngle);
}

static inline __device__ void copyDigest(
    uint8_t destination[32], const uint8_t source[32]) {
#pragma unroll
    for (int index = 0; index < 32; ++index) destination[index] = source[index];
}

static inline __device__ void copyContract(
    uint64_t destination[4], const uint64_t source[4]) {
#pragma unroll
    for (int index = 0; index < 4; ++index) destination[index] = source[index];
}

// Called only when the interior projector/support proof would
// return FiniteExactCone. It never changes a successful interior route.
static inline __device__ InputProviderStatus prepareBoundaryProjection(
    const Source& source, AdmissionInput& output) {
    if (source.camera.modelId
            != gofish::kb_hessian_abi::kModelOpenCvFisheyeKb4) {
        // The finite pinhole raster has no radial boundary chart.  Its
        // near-plane or edge-straddling support is evaluated exactly by the
        // shared calibrated ray-cone test.
        return InputProviderStatus::FiniteExactCone;
    }
    gofish_certified_remainder::ShapeResidualCertificate residual{};
    if (!gofish_certified_remainder::certifyShapeResidual(
            output.shape, residual)) return InputProviderStatus::FiniteExactCone;
    const double supportRadiusUpper = __dmul_ru(
        output.supportRadiusUpper, residual.supportSigmaUpper);
    const auto& active = source.camera.activeProjection;
    const gofish_boundary_angular::Camera camera{
        double(active.focalLength.x), double(active.focalLength.y),
        double(active.principalPoint.x), double(active.principalPoint.y),
        double(active.radialCoeffs[0]), double(active.radialCoeffs[1]),
        double(active.radialCoeffs[2]), double(active.radialCoeffs[3]),
        double(active.maxAngle),
    };
    const auto boundary = gofish_boundary_angular::certify(
        camera, output.meanFromCamera, supportRadiusUpper);
    if (boundary.route
            != gofish_boundary_angular::Route::CertifiedRectangle) {
        return InputProviderStatus::FiniteExactCone;
    }
    const auto* descriptor = source.camera.descriptor;
    if (descriptor == nullptr) return InputProviderStatus::FiniteExactCone;
    output.boundaryProjection.center = boundary.rectangle.center;
    output.boundaryProjection.rectangleHalfExtent =
        boundary.rectangle.halfExtent;
    output.boundaryProjection.supportBallRadiusUpper =
        boundary.rectangle.supportBallRadiusUpper;
    output.boundaryProjection.thetaLower = boundary.rectangle.thetaLower;
    output.boundaryProjection.thetaUpper = boundary.rectangle.thetaUpper;
    copyDigest(output.boundaryProjection.cameraIdentitySha256,
               descriptor->camera_identity_sha256);
    copyDigest(output.boundaryProjection.activeCanonicalDomainSha256,
               descriptor->active_canonical_domain_sha256);
    copyContract(output.boundaryProjection.shapeContract,
                 output.shape.shapeContract);
    output.boundaryProjection.activeWidth =
        uint32_t(source.camera.activeResolution.x);
    output.boundaryProjection.activeHeight =
        uint32_t(source.camera.activeResolution.y);
    output.boundaryProjection.certified = 1u;
    return InputProviderStatus::FinitePrepared;
}

template <typename Particles>
struct CalibratedProvider {
    Source source;

    __device__ InputProviderStatus operator()(
        uint32_t particle, AdmissionInput& output,
        uint32_t& existingParticleIndex,
        ProviderContractMarker& marker) const {
        marker.providerContractMarker =
            gofish_certified_remainder::kProviderContractMarker;
        marker.depthKeyContractMarker =
            gofish_certified_remainder::kDepthKeyContractMarker;
        marker.exactConeCardinalityReserved = 1u;
        marker.certified = 1u;
        existingParticleIndex = particle;

        if (!selectedCameraMatchesActive(source.camera)) {
            marker.certified = 0u;
            return InputProviderStatus::NonfiniteGaussian;
        }
#pragma unroll
        for (int index = 0; index < 12; ++index) {
            if (!finiteNormal(source.worldToCamera[index]))
                return InputProviderStatus::NonfiniteGaussian;
        }

        Particles particles;
        particles.initializeDensity(source.parameters);
        const auto parameters = particles.fetchDensityParameters(particle);
        const tcnn::vec3 meanWorldTcnn = particles.position(parameters);
        const float3 meanWorld = make_float3(
            meanWorldTcnn.x, meanWorldTcnn.y, meanWorldTcnn.z);
        // H2 replays the intrinsic physical-Mip field while image formation
        // keeps the transient Q/m2 response.  Sigma_sensor=Sigma_intrinsic+Q
        // contains the intrinsic ellipsoid because Q is PSD.  Pairing that
        // sensor-shape envelope with max(sensor,intrinsic) peak opacity makes
        // the complete-admission stream conservative for both consumers;
        // intrinsic-only candidates remain alpha-inert in the image renderer.
        const float opacity = fmaxf(
            particles.opacity(parameters), particles.intrinsicOpacity(particle));
        if (!finiteNormal(meanWorld.x) || !finiteNormal(meanWorld.y)
            || !finiteNormal(meanWorld.z) || !finiteNormal(opacity)) {
            return InputProviderStatus::NonfiniteGaussian;
        }

        output = AdmissionInput{};
        output.meanFromCamera = transformMean(source.worldToCamera, meanWorld);
        output.opacity = opacity;
        const auto certifiedRadius =
            gofish_certified_opacity_radius::certifiedRadius(
                opacity, 2,
                gofish_certified_opacity_radius::kAlphaThreshold,
                gofish_certified_opacity_radius::kAlphaCap,
                gofish_certified_opacity_radius::kKernelMinimum);
        output.supportRadiusUpper = double(certifiedRadius.radius);
        output.finiteGaussian = 1u;
        output.activeAdmissionPaddingVariance = 0.0f;
        RendererEffectiveShape rendererShape{};
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const tcnn::vec3 value = particles.factorColumn(parameters, col);
            rendererShape.factorWorld[col] = value.x;
            rendererShape.factorWorld[3 + col] = value.y;
            rendererShape.factorWorld[6 + col] = value.z;
        }
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            rendererShape.inverseFactorWorld[3 * row + 0] =
                parameters.inverseFactor[row].x;
            rendererShape.inverseFactorWorld[3 * row + 1] =
                parameters.inverseFactor[row].y;
            rendererShape.inverseFactorWorld[3 * row + 2] =
                parameters.inverseFactor[row].z;
        }
        copyContract(rendererShape.shapeContract, source.shapeContract);
        cameraFrameShape(
            source.worldToCamera, rendererShape, output.shape);
        gofish_certified_remainder::ShapeResidualCertificate shapeResidual{};
        if (!gofish_certified_remainder::certifyShapeResidual(
                output.shape, shapeResidual)) {
            return InputProviderStatus::FiniteExactCone;
        }
        output.shapeSupportSigmaUpper = shapeResidual.supportSigmaUpper;
        output.shapeInverseErrorSigmaUpper =
            shapeResidual.inverseErrorSigmaUpper;
        output.shapeResidualCertified = 1u;
        if (source.camera.modelId
                == gofish::kb_hessian_abi::kModelIdealPinhole) {
            const float pinholeMaxAngle = source.camera.descriptor == nullptr
                ? 0.0f : float(source.camera.descriptor->theta_max_radians);
            if (provablyOutsideActiveAngularDomain(output, pinholeMaxAngle)) {
                return InputProviderStatus::FiniteEmptySupport;
            }
        } else if (provablyOutsideActiveAngularDomain(
                output, source.camera.activeProjection.maxAngle)) {
            return InputProviderStatus::FiniteEmptySupport;
        }

        tcnn::vec2 exactCenter = tcnn::vec2::zero();
        const tcnn::vec3 meanCamera{
            output.meanFromCamera.x,
            output.meanFromCamera.y,
            output.meanFromCamera.z};
        if (source.camera.modelId
                == gofish::kb_hessian_abi::kModelIdealPinhole) {
            if (!threedgut::projectPoint(
                    source.camera.activePinholeProjection,
                    source.camera.activeResolution,
                    meanCamera, 0.0f, exactCenter, false)) {
                return prepareBoundaryProjection(source, output);
            }
        } else if (!threedgut::projectPoint(
                source.camera.activeProjection, source.camera.activeResolution,
                meanCamera, 0.0f, exactCenter, false)) {
            return prepareBoundaryProjection(source, output);
        }

        KbIntervals enclosure{};
        if (source.camera.modelId
                == gofish::kb_hessian_abi::kModelIdealPinhole) {
            if (!continuousPinholeIntervals(
                    source.camera.activePinholeProjection,
                    output.meanFromCamera, enclosure)) {
                return prepareBoundaryProjection(source, output);
            }
        } else if (!continuousKbIntervals(
                source.camera.activeProjection, output.meanFromCamera,
                enclosure)) {
            return prepareBoundaryProjection(source, output);
        }
        double supportRadiusUpper = 0.0;
        double thetaLower = 0.0;
        double thetaUpper = 0.0;
        if (source.camera.modelId
                == gofish::kb_hessian_abi::kModelIdealPinhole) {
            const float pinholeMaxAngle = source.camera.descriptor == nullptr
                ? 0.0f : float(source.camera.descriptor->theta_max_radians);
            if (!thetaSupport(
                    output.meanFromCamera, enclosure.theta,
                    output.supportRadiusUpper, output.shapeSupportSigmaUpper,
                    pinholeMaxAngle,
                    supportRadiusUpper, thetaLower, thetaUpper)) {
                return prepareBoundaryProjection(source, output);
            }
        } else if (!thetaSupport(
                output.meanFromCamera, enclosure.theta,
                output.supportRadiusUpper, output.shapeSupportSigmaUpper,
                source.camera.activeProjection.maxAngle,
                supportRadiusUpper, thetaLower, thetaUpper)) {
            return prepareBoundaryProjection(source, output);
        }
        const auto* descriptor = source.camera.descriptor;

        output.linearization.evaluationPoint = output.meanFromCamera;
        output.linearization.centerReference = make_float2(
            exactCenter.x, exactCenter.y);
        output.linearization.centerErrorUpper[0] = centerErrorUpper(
            exactCenter.x, enclosure.center[0]);
        output.linearization.centerErrorUpper[1] = centerErrorUpper(
            exactCenter.y, enclosure.center[1]);
#pragma unroll
        for (int index = 0; index < 6; ++index) {
            output.linearization.jacobian[index] = enclosure.jacobian[index];
        }
        copyDigest(output.linearization.cameraIdentitySha256,
                   descriptor->camera_identity_sha256);
        copyDigest(output.linearization.activeCanonicalDomainSha256,
                   descriptor->active_canonical_domain_sha256);
        copyContract(output.linearization.linearizationContract,
                     source.camera.linearizationContract);
        copyContract(output.linearization.shapeContract,
                     output.shape.shapeContract);
        output.linearization.activeWidth =
            uint32_t(source.camera.activeResolution.x);
        output.linearization.activeHeight =
            uint32_t(source.camera.activeResolution.y);
        output.linearization.certified = 1u;

        output.thetaSupport.evaluationPoint = output.meanFromCamera;
        output.thetaSupport.supportBallRadiusUpper = supportRadiusUpper;
        output.thetaSupport.thetaLower = thetaLower;
        output.thetaSupport.thetaUpper = thetaUpper;
        copyDigest(output.thetaSupport.cameraIdentitySha256,
                   descriptor->camera_identity_sha256);
        copyDigest(output.thetaSupport.activeCanonicalDomainSha256,
                   descriptor->active_canonical_domain_sha256);
        copyContract(output.thetaSupport.linearizationContract,
                     source.camera.linearizationContract);
        output.thetaSupport.activeWidth =
            uint32_t(source.camera.activeResolution.x);
        output.thetaSupport.activeHeight =
            uint32_t(source.camera.activeResolution.y);
        output.thetaSupport.certified = 1u;
        return InputProviderStatus::FinitePrepared;
    }
};

} // namespace gofish_certified_remainder_provider

