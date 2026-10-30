// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/renderer/renderParameters.h>
#include <3dgut/utils/logger.h>
#include <3dgut/utils/status.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <exception>

namespace threedgut::fast_sof_persistent_camera_table {

// Opaque owner of one immutable camera table and one reusable cooperative
// workspace.  Camera payload pointers remain valid because the SplatRaster
// binding retains their owning CUDA tensors until destroyTable().
struct Table;

struct CameraStageTelemetry {
    std::uint32_t intersections = 0;
    std::uint32_t boundedReasonMask = 0;
    std::uint32_t boundedFirst = UINT32_MAX;
    std::uint32_t boundedCount = 0;
};

struct RoundTelemetry {
    std::uint32_t failureCode = 0;
    std::uint32_t failureCamera = UINT32_MAX;
    std::uint32_t failureIndex = UINT32_MAX;
    std::uint32_t finalActive = 0;
    std::uint32_t finalObserved = 0;
    std::uint32_t exactEqual = 0;
};

struct DestroyResult {
    cudaError_t getDevice = cudaSuccess;
    cudaError_t setDevice = cudaSuccess;
    cudaError_t synchronize = cudaSuccess;
    cudaError_t restoreDevice = cudaSuccess;
    bool freed = false;

    bool ok() const {
        return getDevice == cudaSuccess && setDevice == cudaSuccess &&
            synchronize == cudaSuccess && restoreDevice == cudaSuccess && freed;
    }
};

Table* createTable(
    std::uint32_t particleCount,
    std::uint32_t queryCount,
    const float* particleSigmaMax,
    std::uint64_t* parameterBuffer,
    int device,
    cudaStream_t stream,
    const Logger& logger);

Status appendCamera(
    Table& table,
    const RenderParameters& params,
    const tcnn::vec3* rayOrigins,
    const tcnn::vec3* rayDirections,
    const std::uint8_t* physicalMask,
    tcnn::mat4x3 worldToSensor,
    tcnn::mat4x3 sensorToWorld,
    CameraStageTelemetry& telemetry,
    const Logger& logger);

Status sealTable(Table& table, const Logger& logger);

// Enqueues every camera in immutable ordinal order.  The implementation does
// not synchronize, allocate, hash, compact on the host, or read a camera
// result.  Every per-camera decision and the camera minimum are represented by
// active/observed/equal device bytes.  The caller enqueues one final D2H copy
// and performs exactly one stream synchronization for the round.
Status enqueueRound(
    Table& table,
    const tcnn::vec3* queryPoints,
    std::uint8_t* active,
    std::uint8_t* observed,
    std::uint8_t* exactEqual,
    std::uint32_t* cameraCounters,
    std::uint32_t* deviceFailure,
    std::uint8_t* packedRoundOutput,
    size_t packedRoundBytes,
    const Logger& logger);

std::uint32_t cameraCount(const Table& table);
std::uint32_t queryCount(const Table& table);
std::uint32_t pointBlockBound(const Table& table);
DestroyResult destroyTable(Table* table) noexcept;

} // namespace threedgut::fast_sof_persistent_camera_table
