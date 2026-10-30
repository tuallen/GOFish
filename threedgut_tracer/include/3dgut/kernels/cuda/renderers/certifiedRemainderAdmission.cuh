// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>
#include <3dgut/kernels/cuda/renderers/kb_hessian_table_abi.h>
#include <3dgut/kernels/cuda/renderers/certifiedBoundaryAngularFootprint.cuh>

#include <cmath>
#include <cstdint>

namespace gofish_certified_remainder {

constexpr double kRendererSupportRadius = 3.0;
constexpr double kRendererSupportSquared = 9.0;
constexpr double kPixelCellHalfExtent = 0.5;
constexpr double kFloatMinNormal = 0x1.0p-126;
constexpr float kRendererAlphaThreshold = 0.00392156885936856f;
constexpr float kRendererAlphaCap = 0.99000000953674316f;
constexpr uint64_t kProviderContractMarker = 0x31564f5250435247ULL;
constexpr uint64_t kDepthKeyContractMarker = 0x3159454b44524347ULL;

struct Interval {
    double lo;
    double hi;
};

struct CameraThetaBandHessianTable {
    gofish::kb_hessian_abi::TableView view;
    const gofish::kb_hessian_abi::CameraDescriptor* camera;
    uint32_t activeWidth;
    uint32_t activeHeight;
    uint8_t activeCameraIdentitySha256[32];
    uint8_t activeCanonicalDomainSha256[32];
    uint32_t initializedCertified;
};

// Renderer shape of one Gaussian. B and B^{-1} are the effective
// factor bytes used by densityHit; the kernel verifies their interval product
// contains identity before using either one.
struct EffectiveShapeInput {
    float factor[9];
    float inverseFactor[9];
    uint64_t shapeContract[4];
};

// Proof-carrying output of one calibrated projector evaluation at exactly w.
// pi(w) is enclosed around centerReference by coordinate-wise errors; J is an
// interval Jacobian from the same evaluation and camera contract.
struct CertifiedProjectionLinearization {
    float3 evaluationPoint;
    float2 centerReference;
    double centerErrorUpper[2];
    Interval jacobian[6];
    uint8_t cameraIdentitySha256[32];
    uint8_t activeCanonicalDomainSha256[32];
    uint64_t linearizationContract[4];
    uint64_t shapeContract[4];
    uint32_t activeWidth;
    uint32_t activeHeight;
    uint32_t certified;
};

// Generated from the same w and support-ball radius. The
// kernel accepts a larger radius but never a smaller one than its own q.
struct CertifiedThetaSupport {
    float3 evaluationPoint;
    double supportBallRadiusUpper;
    double thetaLower;
    double thetaUpper;
    uint8_t cameraIdentitySha256[32];
    uint8_t activeCanonicalDomainSha256[32];
    uint64_t linearizationContract[4];
    uint32_t activeWidth;
    uint32_t activeHeight;
    uint32_t certified;
};

// Produced only for support that intersects the finite theta boundary. The
// half extents already enclose the pixel cell and angular Taylor remainder.
struct CertifiedBoundaryProjection {
    float2 center;
    float2 rectangleHalfExtent;
    double supportBallRadiusUpper;
    double thetaLower;
    double thetaUpper;
    uint8_t cameraIdentitySha256[32];
    uint8_t activeCanonicalDomainSha256[32];
    uint64_t shapeContract[4];
    uint32_t activeWidth;
    uint32_t activeHeight;
    uint32_t certified;
};

struct AdmissionInput {
    float3 meanFromCamera;
    EffectiveShapeInput shape;
    CertifiedProjectionLinearization linearization;
    CertifiedThetaSupport thetaSupport;
    CertifiedBoundaryProjection boundaryProjection;
    float opacity;
    double supportRadiusUpper;
    // One Neumann residual certificate is produced with the camera-frame
    // factor/inverse and consumed by every later bound.  Recomputing it cannot
    // strengthen the proof.
    double shapeSupportSigmaUpper;
    double shapeInverseErrorSigmaUpper;
    uint32_t shapeResidualCertified;
    uint32_t finiteGaussian;
    // Must be exactly zero. Certified payloads bypass the covariance
    // padding path and are appended as final conic/extent values.
    float activeAdmissionPaddingVariance;
};

struct DilatedProjectedEllipse {
    float2 center;
    float2 extent;
    float4 conicOpacity;
    float maxConicOpacityPower;
    uint32_t firstThetaBand;
    uint32_t lastThetaBand;
};

struct CertifiedProjectedRectangle {
    float2 center;
    float2 halfExtent;
};

enum class AdmissionRoute : uint32_t {
    ExactCone = 0,
    CertifiedEllipse = 1,
    EmptySupport = 2,
    CertifiedRectangle = 3,
};

enum class InputProviderStatus : uint32_t {
    FinitePrepared = 0,
    FiniteExactCone = 1,
    NonfiniteGaussian = 2,
    FiniteEmptySupport = 3,
};

struct AdmissionDecision {
    AdmissionRoute route;
    DilatedProjectedEllipse ellipse;
    CertifiedProjectedRectangle rectangle;
};

// Persist this exact route/cardinality/particle record between count and emit.
// Per-tile SOF depth keys are deliberately not persisted here.
struct SharedCountEmitRecord {
    AdmissionDecision decision;
    uint32_t particleIndex;
    uint64_t providerContractMarker;
    uint64_t depthKeyContractMarker;
    uint32_t exactConeCardinalityReserved;
    uint32_t certified;
};

struct ProviderContractMarker {
    uint64_t providerContractMarker;
    uint64_t depthKeyContractMarker;
    uint32_t exactConeCardinalityReserved;
    uint32_t certified;
};

static inline __device__ bool normalFinite(float value) {
    const float magnitude = fabsf(value);
    return isfinite(value)
        && !(magnitude > 0.0f && magnitude < float(kFloatMinNormal));
}

static inline __device__ bool finiteInterval(Interval value) {
    return isfinite(value.lo) && isfinite(value.hi) && value.lo <= value.hi;
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

static inline __device__ double absoluteUpper(Interval value) {
    return fmax(fabs(value.lo), fabs(value.hi));
}

static inline __device__ bool contains(Interval value, double target) {
    return finiteInterval(value) && value.lo <= target && target <= value.hi;
}

static inline __device__ bool contractsEqual(
    const uint64_t left[4], const uint64_t right[4]) {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

static inline __device__ bool digestEqual(
    const uint8_t left[32], const uint8_t right[32]) {
#pragma unroll
    for (int index = 0; index < 32; ++index) {
        if (left[index] != right[index]) return false;
    }
    return true;
}

static inline __device__ bool samePoint(float3 left, float3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

static inline __device__ bool validatedTableHbarForActiveRaster(
    const CameraThetaBandHessianTable& table,
    const CertifiedThetaSupport& support,
    double& hbarActivePixelsUpper,
    uint32_t& firstBand,
    uint32_t& lastBand) {
    using namespace gofish::kb_hessian_abi;
    if (table.initializedCertified != 1u
        || table.view.header == nullptr || table.view.cameras == nullptr
        || table.view.bands == nullptr || table.camera == nullptr
        || table.view.header->abi_version != kVersion
        || table.view.header->endian_tag != kEndianTag
        || table.view.header->header_bytes != kHeaderBytes
        || table.view.header->camera_stride != kCameraBytes
        || table.view.header->band_stride != kBandBytes
        || table.view.header->flags != kRequiredFlags
        || (table.camera->model_id != kModelOpenCvFisheyeKb4
            && table.camera->model_id != kModelIdealPinhole)
        || table.camera->native_width == 0u || table.camera->native_height == 0u
        || table.activeWidth == 0u || table.activeHeight == 0u
        || !digestEqual(table.activeCameraIdentitySha256,
                        table.camera->camera_identity_sha256)
        || !digestEqual(table.activeCanonicalDomainSha256,
                        table.camera->active_canonical_domain_sha256)
        || support.certified != 1u
        || !isfinite(support.thetaLower) || !isfinite(support.thetaUpper)
        || support.thetaLower < 0.0
        || support.thetaLower > support.thetaUpper
        || support.thetaUpper > table.camera->theta_max_radians
        || support.activeWidth != table.activeWidth
        || support.activeHeight != table.activeHeight
        || !digestEqual(support.cameraIdentitySha256,
                        table.activeCameraIdentitySha256)
        || !digestEqual(support.activeCanonicalDomainSha256,
                        table.activeCanonicalDomainSha256)) return false;
    AggregatedBands aggregate{};
    if (!aggregateIntersectedBands(
            table.view.bands, *table.camera,
            support.thetaLower, support.thetaUpper, &aggregate)) return false;
    const double scaled = hbarForRaster(
        aggregate.hbar_native_pixels,
        table.camera->native_width, table.camera->native_height,
        table.activeWidth, table.activeHeight);
    if (!(scaled > 0.0) || !isfinite(scaled)) return false;
    hbarActivePixelsUpper = scaled;
    firstBand = aggregate.first_global_band;
    lastBand = aggregate.last_global_band;
    return true;
}

struct ShapeResidualCertificate {
    double etaUpper;
    double factorNormUpper;
    double supportSigmaUpper;
    double inverseErrorSigmaUpper;
    uint32_t certified;
};

static inline __device__ double sigmaMaxUpper(
    const EffectiveShapeInput& shape) {
    double rowSumUpper[3] = {0.0, 0.0, 0.0};
    double colSumUpper[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double value = fabs(double(shape.factor[3 * row + col]));
            rowSumUpper[row] = __dadd_ru(rowSumUpper[row], value);
            colSumUpper[col] = __dadd_ru(colSumUpper[col], value);
        }
    }
    const double normInfUpper = fmax(
        fmax(rowSumUpper[0], rowSumUpper[1]), rowSumUpper[2]);
    const double normOneUpper = fmax(
        fmax(colSumUpper[0], colSumUpper[1]), colSumUpper[2]);
    return __dsqrt_ru(__dmul_ru(normInfUpper, normOneUpper));
}

static inline __device__ bool certifyShapeResidual(
    const EffectiveShapeInput& shape,
    ShapeResidualCertificate& result) {
    result = ShapeResidualCertificate{};
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        if (!normalFinite(shape.factor[index])
            || !normalFinite(shape.inverseFactor[index])) return false;
    }
    double rowSumUpper[3] = {0.0, 0.0, 0.0};
    double colSumUpper[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            Interval product{0.0, 0.0};
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                const double left = double(shape.factor[3 * row + inner]);
                const double right = double(shape.inverseFactor[3 * inner + col]);
                product = add(product, multiply(
                    Interval{left, left}, Interval{right, right}));
            }
            const Interval residual = subtract(
                Interval{row == col ? 1.0 : 0.0,
                         row == col ? 1.0 : 0.0}, product);
            if (!finiteInterval(residual)) return false;
            const double magnitude = absoluteUpper(residual);
            rowSumUpper[row] = __dadd_ru(rowSumUpper[row], magnitude);
            colSumUpper[col] = __dadd_ru(colSumUpper[col], magnitude);
        }
    }
    const double normInfUpper = fmax(
        fmax(rowSumUpper[0], rowSumUpper[1]), rowSumUpper[2]);
    const double normOneUpper = fmax(
        fmax(colSumUpper[0], colSumUpper[1]), colSumUpper[2]);
    result.etaUpper = __dsqrt_ru(__dmul_ru(normInfUpper, normOneUpper));
    result.factorNormUpper = sigmaMaxUpper(shape);
    const double denominatorLower = __dsub_rd(1.0, result.etaUpper);
    if (!(result.etaUpper >= 0.0) || !(result.etaUpper < 1.0)
        || !(denominatorLower > 0.0)
        || !isfinite(result.factorNormUpper)) return false;
    result.supportSigmaUpper = __ddiv_ru(
        result.factorNormUpper, denominatorLower);
    result.inverseErrorSigmaUpper = __ddiv_ru(
        __dmul_ru(result.factorNormUpper, result.etaUpper),
        denominatorLower);
    result.certified = isfinite(result.supportSigmaUpper)
        && isfinite(result.inverseErrorSigmaUpper) ? 1u : 0u;
    return result.certified == 1u;
}

static inline __device__ bool shapeIsMatched(
    const EffectiveShapeInput& shape) {
    ShapeResidualCertificate residual{};
    return certifyShapeResidual(shape, residual);
}

static inline __device__ bool cameraOutsideProof(
    const AdmissionInput& input,
    double& distanceLower,
    double& supportBallRadiusUpper,
    double& dMinLower) {
    const float3 w = input.meanFromCamera;
    if (!normalFinite(w.x) || !normalFinite(w.y) || !normalFinite(w.z)
        || input.shapeResidualCertified != 1u
        || !isfinite(input.shapeSupportSigmaUpper)
        || !isfinite(input.shapeInverseErrorSigmaUpper)) return false;
    if (!isfinite(input.supportRadiusUpper)
        || !(input.supportRadiusUpper > 0.0)
        || input.supportRadiusUpper > kRendererSupportRadius) return false;
    const double supportSquaredUpper = __dmul_ru(
        input.supportRadiusUpper, input.supportRadiusUpper);

    Interval canonical[3];
    const double component[3] = {double(w.x), double(w.y), double(w.z)};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
        canonical[row] = Interval{0.0, 0.0};
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double value = double(input.shape.inverseFactor[3 * row + col]);
            canonical[row] = add(canonical[row], multiply(
                Interval{value, value}, Interval{component[col], component[col]}));
        }
    }
    double canonicalSquaredLower = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        double magnitudeLower = 0.0;
        if (canonical[axis].lo > 0.0) magnitudeLower = canonical[axis].lo;
        else if (canonical[axis].hi < 0.0) magnitudeLower = -canonical[axis].hi;
        canonicalSquaredLower = __dadd_rd(
            canonicalSquaredLower,
            __dmul_rd(magnitudeLower, magnitudeLower));
    }
    const double kappaLower = __dsub_rd(
        canonicalSquaredLower, supportSquaredUpper);
    if (!(kappaLower > 0.0) || !isfinite(kappaLower)) return false;

    double distanceSquaredLower = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        distanceSquaredLower = __dadd_rd(
            distanceSquaredLower,
            __dmul_rd(component[axis], component[axis]));
    }
    distanceLower = __dsqrt_rd(distanceSquaredLower);
    supportBallRadiusUpper = __dmul_ru(
        input.supportRadiusUpper, input.shapeSupportSigmaUpper);
    if (!(supportBallRadiusUpper < distanceLower)
        || !isfinite(supportBallRadiusUpper)) return false;
    dMinLower = __dsub_rd(distanceLower, supportBallRadiusUpper);
    return dMinLower > 0.0 && isfinite(dMinLower);
}

static inline __device__ bool linearizationIsCoLocated(
    const AdmissionInput& input,
    const CameraThetaBandHessianTable& table) {
    const CertifiedProjectionLinearization& linear = input.linearization;
    const CertifiedThetaSupport& theta = input.thetaSupport;
    if (linear.certified != 1u
        || !samePoint(linear.evaluationPoint, input.meanFromCamera)
        || !samePoint(theta.evaluationPoint, input.meanFromCamera)
        || !digestEqual(linear.cameraIdentitySha256,
                        table.activeCameraIdentitySha256)
        || !digestEqual(linear.activeCanonicalDomainSha256,
                        table.activeCanonicalDomainSha256)
        || !digestEqual(theta.cameraIdentitySha256,
                        table.activeCameraIdentitySha256)
        || !digestEqual(theta.activeCanonicalDomainSha256,
                        table.activeCanonicalDomainSha256)
        || !contractsEqual(theta.linearizationContract,
                           linear.linearizationContract)
        || !contractsEqual(linear.shapeContract, input.shape.shapeContract)
        || linear.activeWidth != table.activeWidth
        || linear.activeHeight != table.activeHeight
        || theta.activeWidth != table.activeWidth
        || theta.activeHeight != table.activeHeight
        || !normalFinite(linear.centerReference.x)
        || !normalFinite(linear.centerReference.y)) return false;
#pragma unroll
    for (int axis = 0; axis < 2; ++axis) {
        if (!isfinite(linear.centerErrorUpper[axis])
            || linear.centerErrorUpper[axis] < 0.0) return false;
    }
#pragma unroll
    for (int entry = 0; entry < 6; ++entry) {
        if (!finiteInterval(linear.jacobian[entry])) return false;
    }
    return true;
}

static inline __device__ bool decideBoundaryProjection(
    const CameraThetaBandHessianTable& table,
    const AdmissionInput& input,
    AdmissionDecision& decision) {
    const CertifiedBoundaryProjection& boundary = input.boundaryProjection;
    if (boundary.certified != 1u) return false;

    // Inline the typed table/identity proof; no Taylor value is consumed here.
    if (table.initializedCertified != 1u
        || table.view.header == nullptr || table.view.cameras == nullptr
        || table.view.bands == nullptr || table.camera == nullptr
        || boundary.activeWidth != table.activeWidth
        || boundary.activeHeight != table.activeHeight
        || !digestEqual(boundary.cameraIdentitySha256,
                        table.activeCameraIdentitySha256)
        || !digestEqual(boundary.activeCanonicalDomainSha256,
                        table.activeCanonicalDomainSha256)
        || !contractsEqual(boundary.shapeContract, input.shape.shapeContract)
        || !normalFinite(boundary.center.x)
        || !normalFinite(boundary.center.y)
        || !normalFinite(boundary.rectangleHalfExtent.x)
        || !normalFinite(boundary.rectangleHalfExtent.y)
        || !(boundary.rectangleHalfExtent.x > 0.0f)
        || !(boundary.rectangleHalfExtent.y > 0.0f)
        || !(boundary.supportBallRadiusUpper >= 0.0)
        || !isfinite(boundary.supportBallRadiusUpper)
        || !(boundary.thetaLower >= 0.0)
        || !(boundary.thetaLower <= boundary.thetaUpper)
        || !isfinite(boundary.thetaLower)
        || !isfinite(boundary.thetaUpper)
        || boundary.thetaUpper > table.camera->theta_max_radians)
        return false;

    ShapeResidualCertificate residual{};
    if (!(input.supportRadiusUpper > 0.0)
        || !isfinite(input.supportRadiusUpper)
        || !certifyShapeResidual(input.shape, residual)) return false;
    const double requiredSupportRadiusUpper = __dmul_ru(
        input.supportRadiusUpper, residual.supportSigmaUpper);
    if (!isfinite(requiredSupportRadiusUpper)
        || boundary.supportBallRadiusUpper < requiredSupportRadiusUpper)
        return false;

    using namespace gofish::kb_hessian_abi;
    if (table.view.header->abi_version != kVersion
        || table.view.header->endian_tag != kEndianTag
        || table.view.header->header_bytes != kHeaderBytes
        || table.view.header->camera_stride != kCameraBytes
        || table.view.header->band_stride != kBandBytes
        || table.view.header->flags != kRequiredFlags
        || table.camera->model_id != kModelOpenCvFisheyeKb4
        || !digestEqual(table.activeCameraIdentitySha256,
                        table.camera->camera_identity_sha256)
        || !digestEqual(table.activeCanonicalDomainSha256,
                        table.camera->active_canonical_domain_sha256))
        return false;
    AggregatedBands aggregate{};
    if (!aggregateIntersectedBands(
            table.view.bands, *table.camera,
            boundary.thetaLower, boundary.thetaUpper, &aggregate)) return false;
    decision.rectangle.center = boundary.center;
    decision.rectangle.halfExtent = boundary.rectangleHalfExtent;
    decision.route = AdmissionRoute::CertifiedRectangle;
    return true;
}

// Every failed proof condition returns ExactCone.
static inline __device__ AdmissionDecision decide(
    const CameraThetaBandHessianTable& table,
    const AdmissionInput& input) {
    AdmissionDecision decision{};
    decision.route = AdmissionRoute::ExactCone;
    if (input.finiteGaussian != 1u || !normalFinite(input.opacity))
        return decision;
    if (!(input.supportRadiusUpper > 0.0)
        || !isfinite(input.supportRadiusUpper)) return decision;
    // Exact renderer comparator at peak response one. Equality is empty
    // because densityHit requires alpha strictly greater than the threshold.
    const float peakAlpha = fminf(kRendererAlphaCap, input.opacity);
    if (!(peakAlpha > kRendererAlphaThreshold)) {
        decision.route = AdmissionRoute::EmptySupport;
        return decision;
    }
    if (input.activeAdmissionPaddingVariance != 0.0f) return decision;
    if (input.boundaryProjection.certified != 0u) {
        if (!decideBoundaryProjection(table, input, decision))
            return AdmissionDecision{};
        return decision;
    }
    if (!linearizationIsCoLocated(input, table)) return decision;

    double distanceLower = 0.0;
    double supportBallRadiusUpper = 0.0;
    double dMinLower = 0.0;
    if (!cameraOutsideProof(
            input, distanceLower,
            supportBallRadiusUpper, dMinLower)) return decision;
    if (!isfinite(input.thetaSupport.supportBallRadiusUpper)
        || input.thetaSupport.supportBallRadiusUpper < supportBallRadiusUpper)
        return decision;

    double hbarActivePixelsUpper = 0.0;
    uint32_t firstBand = 0u;
    uint32_t lastBand = 0u;
    if (!validatedTableHbarForActiveRaster(
            table, input.thetaSupport, hbarActivePixelsUpper,
            firstBand, lastBand)) return decision;

    const double dMinSquaredLower = __dmul_rd(dMinLower, dMinLower);
    if (!(dMinSquaredLower > 0.0) || !isfinite(dMinSquaredLower)) return decision;
    const double hMaxUpper = gofish::kb_hessian_abi::upperFromRange(
        hbarActivePixelsUpper, dMinLower);
    const double supportBallSquaredUpper = __dmul_ru(
        supportBallRadiusUpper, supportBallRadiusUpper);
    const double remainderRadiusUpper = __dmul_ru(
        0.5, __dmul_ru(hMaxUpper, supportBallSquaredUpper));
    if (!isfinite(hMaxUpper) || !isfinite(remainderRadiusUpper)) return decision;

    // C=J*B with J enclosed at the same w as pi(w).
    Interval projectedFactor[6];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            Interval sum{0.0, 0.0};
#pragma unroll
            for (int inner = 0; inner < 3; ++inner) {
                const double factor = double(input.shape.factor[3 * inner + col]);
                sum = add(sum, multiply(
                    input.linearization.jacobian[3 * row + inner],
                    Interval{factor, factor}));
            }
            if (!finiteInterval(sum)) return decision;
            projectedFactor[3 * row + col] = sum;
        }
    }

    double projectedVarianceUpper[2] = {0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double component = absoluteUpper(
                projectedFactor[3 * row + col]);
            projectedVarianceUpper[row] = __dadd_ru(
                projectedVarianceUpper[row], __dmul_ru(component, component));
        }
    }

    // Linear support + Taylor disk + exact pi(w) enclosure + square pixel cell.
    // B_infinity(1/2) is norm-correct and is not covariance padding.
    const double residualWorldRadiusUpper = __dmul_ru(
        input.supportRadiusUpper, input.shapeInverseErrorSigmaUpper);
    double rectangleHalfExtent[2];
#pragma unroll
    for (int axis = 0; axis < 2; ++axis) {
        double jacobianNormSquaredUpper = 0.0;
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double component = absoluteUpper(
                input.linearization.jacobian[3 * axis + col]);
            jacobianNormSquaredUpper = __dadd_ru(
                jacobianNormSquaredUpper, __dmul_ru(component, component));
        }
        const double residualImageRadiusUpper = __dmul_ru(
            __dsqrt_ru(jacobianNormSquaredUpper), residualWorldRadiusUpper);
        const double linearAxisUpper = __dmul_ru(
            input.supportRadiusUpper,
            __dsqrt_ru(projectedVarianceUpper[axis]));
        rectangleHalfExtent[axis] = __dadd_ru(
            __dadd_ru(__dadd_ru(linearAxisUpper, residualImageRadiusUpper),
                      remainderRadiusUpper),
            input.linearization.centerErrorUpper[axis]);
        if (!(rectangleHalfExtent[axis] > 0.0)
            || !isfinite(rectangleHalfExtent[axis])) return decision;
    }

    decision.rectangle.center = input.linearization.centerReference;
    decision.rectangle.halfExtent = make_float2(
        __double2float_ru(rectangleHalfExtent[0]),
        __double2float_ru(rectangleHalfExtent[1]));
    if (!isfinite(decision.rectangle.halfExtent.x)
        || !isfinite(decision.rectangle.halfExtent.y))
        return AdmissionDecision{};
    decision.route = AdmissionRoute::CertifiedRectangle;
    return decision;
}

static inline __device__ SharedCountEmitRecord prepareSharedRecord(
    const CameraThetaBandHessianTable& table,
    const AdmissionInput& input,
    uint32_t particleIndex) {
    SharedCountEmitRecord record{};
    record.decision = decide(table, input);
    record.particleIndex = particleIndex;
    record.certified = 1u;
    return record;
}

template <typename CountEllipse, typename CountRectangle,
          typename CountExactCone>
static inline __device__ void countOneStream(
    const SharedCountEmitRecord& record,
    CountEllipse countEllipse,
    CountRectangle countRectangle,
    CountExactCone countExactCone) {
    if (record.decision.route == AdmissionRoute::CertifiedRectangle) {
        countRectangle(record.decision.rectangle);
    } else if (record.decision.route == AdmissionRoute::CertifiedEllipse) {
        countEllipse(record.decision.ellipse);
    } else {
        countExactCone();
    }
}

template <typename EmitEllipse, typename EmitRectangle,
          typename EmitExactCone>
static inline __device__ void emitOneStream(
    const SharedCountEmitRecord& record,
    EmitEllipse emitEllipse,
    EmitRectangle emitRectangle,
    EmitExactCone emitExactCone) {
    if (record.decision.route == AdmissionRoute::CertifiedRectangle) {
        emitRectangle(record.particleIndex, record.decision.rectangle);
    } else if (record.decision.route == AdmissionRoute::CertifiedEllipse) {
        emitEllipse(record.particleIndex, record.decision.ellipse);
    } else {
        emitExactCone(record.particleIndex);
    }
}

} // namespace gofish_certified_remainder

