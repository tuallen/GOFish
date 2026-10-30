// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>

#include <cstdint>



namespace threedgut {

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

} // namespace threedgut

