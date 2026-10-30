// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define GOFISH_ADMISSION_HD __host__ __device__
#else
#define GOFISH_ADMISSION_HD
#endif

namespace gofish_admission {

// Complete admission is defined for central, global-shutter cameras only.
GOFISH_ADMISSION_HD inline bool globalShutterAccepted(int shutter, int global) {
    return shutter == global;
}

// A tile inside the grid that the camera-native ray test did not admit.
GOFISH_ADMISSION_HD inline bool conservativeOnlyAdmission(
        uint32_t x, uint32_t y, uint32_t gridX, uint32_t gridY,
        bool nativeAdmission) {
    return x < gridX && y < gridY && !nativeAdmission;
}

} // namespace gofish_admission

#undef GOFISH_ADMISSION_HD
