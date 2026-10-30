// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cstdint>

// Camera-local near test (mirrors threedgrut.camera_local_validity).
// Camera model IDs follow threedgrut.physical_mip.MODEL_IDS. ID 2 is
// OPENCV_FISHEYE; ideal and distorted pinhole IDs use view-space z.
// Raster and camera-support tests are applied separately.
namespace gofish {

constexpr double kMinimumCameraLocalDistance = 0.2;

__device__ inline double cameraLocalCoordinate(
    const int32_t camera_model, const double view_z, const double radial_distance) {
    return camera_model == 2 ? radial_distance : view_z;
}

__device__ inline bool cameraLocalNearValid(
    const int32_t camera_model, const double view_z, const double radial_distance) {
    return cameraLocalCoordinate(camera_model, view_z, radial_distance)
        > kMinimumCameraLocalDistance;
}

}  // namespace gofish
