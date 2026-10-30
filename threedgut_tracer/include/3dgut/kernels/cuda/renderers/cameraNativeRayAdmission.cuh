// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>

#include <cstdint>

namespace threedgut {

enum CameraNativeAdmissionCounter : uint32_t {
    AdmissionExamined = 0,
    AdmissionLegacyAccepted,
    AdmissionNativeAccepted,
    AdmissionBothAccepted,
    AdmissionNativeOnly,
    AdmissionLegacyOnly,
    AdmissionRadialNearRejected,
    AdmissionProjectionRejected,
    AdmissionPhysicalDomainRejected,
    AdmissionNegativeZNativeAccepted,
    AdmissionCounterCount,
};

static inline __device__ bool isNonlinearFisheye(const TSensorModel& sensorModel) {
    return sensorModel.modelType == TSensorModel::OpenCVFisheyeModel ||
           sensorModel.modelType == TSensorModel::FThetaModel;
}

static inline __device__ bool cameraNativeRadialNearAdmitted(
    const tcnn::vec3& cameraRay, float inheritedNearDistance) {
    const float radius2 = dot(cameraRay, cameraRay);
    const float near2 = inheritedNearDistance * inheritedNearDistance;
    return isfinite(radius2) && isfinite(near2) && inheritedNearDistance >= 0.0f &&
           radius2 >= near2;
}

static inline __device__ bool cameraNativeContinuousRasterAdmitted(
    const tcnn::ivec2& resolution, const tcnn::vec2& projected) {
    return isfinite(projected.x) && isfinite(projected.y) &&
           projected.x >= 0.0f && projected.y >= 0.0f &&
           projected.x < static_cast<float>(resolution.x) &&
           projected.y < static_cast<float>(resolution.y);
}

static inline __device__ bool cameraNativePhysicalDomainAdmitted(
    const TSensorModel& sensorModel,
    const tcnn::ivec2& resolution,
    const tcnn::vec2& projected) {
    if (!cameraNativeContinuousRasterAdmitted(resolution, projected)) {
        return false;
    }
    if (sensorModel.modelType != TSensorModel::OpenCVFisheyeModel) {
        // FTheta supplies an explicit calibrated maxAngle but does not imply a
        // circular aperture.  Do not invent one; retain its native raster.
        return sensorModel.modelType == TSensorModel::FThetaModel;
    }

    const tcnn::vec2 principal = sensorModel.ocvFisheyeParams.principalPoint;
    const float sensorRadius = fminf(
        fminf(principal.x, static_cast<float>(resolution.x) - principal.x),
        fminf(principal.y, static_cast<float>(resolution.y) - principal.y));
    const tcnn::vec2 delta = projected - principal;
    const float radius2 = dot(delta, delta);
    return isfinite(sensorRadius) && sensorRadius >= 0.0f && isfinite(radius2) &&
           radius2 <= sensorRadius * sensorRadius;
}

static inline __device__ void recordCameraNativeAdmission(
    unsigned long long* counters,
    bool legacyAccepted,
    bool radialNearAccepted,
    bool projectionAccepted,
    bool physicalDomainAccepted,
    float cameraZ) {
    if (counters == nullptr) return;
    const bool nativeAccepted = radialNearAccepted && projectionAccepted && physicalDomainAccepted;
    atomicAdd(counters + AdmissionExamined, 1ull);
    if (legacyAccepted) atomicAdd(counters + AdmissionLegacyAccepted, 1ull);
    if (nativeAccepted) atomicAdd(counters + AdmissionNativeAccepted, 1ull);
    if (legacyAccepted && nativeAccepted) atomicAdd(counters + AdmissionBothAccepted, 1ull);
    if (!legacyAccepted && nativeAccepted) atomicAdd(counters + AdmissionNativeOnly, 1ull);
    if (legacyAccepted && !nativeAccepted) atomicAdd(counters + AdmissionLegacyOnly, 1ull);
    if (!radialNearAccepted) atomicAdd(counters + AdmissionRadialNearRejected, 1ull);
    else if (!projectionAccepted) atomicAdd(counters + AdmissionProjectionRejected, 1ull);
    else if (!physicalDomainAccepted) atomicAdd(counters + AdmissionPhysicalDomainRejected, 1ull);
    if (nativeAccepted && cameraZ < 0.0f) atomicAdd(counters + AdmissionNegativeZNativeAccepted, 1ull);
}

} // namespace threedgut

