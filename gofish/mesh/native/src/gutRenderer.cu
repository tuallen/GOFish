// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright Amazon.com, Inc. or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <3dgut/threedgut.cuh>

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/kernels/cuda/common/opacityField.cuh>
#include <3dgut/kernels/cuda/common/bisectionTableImpl.inl>
#include <3dgut/sensors/sensors.h>

#include <tiny-cuda-nn/common_host.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <limits>


using namespace tcnn;

namespace {

using namespace threedgut;

} // namespace

struct GUTRenderer::GutRenderForwardContext {
    cudaStream_t cudaStream;

    GutRenderForwardContext(cudaStream_t iCudaStream)
        : cudaStream(iCudaStream) {
    }

    ~GutRenderForwardContext() {
        Logger logger;
        fastSofTntFullShapeWorkspace.clear(cudaStream, logger);
    }

    fast_sof_tnt_full_shape::CooperativeWorkspace fastSofTntFullShapeWorkspace;
};


threedgut::GUTRenderer::GUTRenderer(const nlohmann::json& config, const Logger& logger)
    : m_logger(logger) {
}

threedgut::GUTRenderer::~GUTRenderer() {
}

threedgut::Status threedgut::GUTRenderer::evaluateFastSofTntFullShapeCamera(
    const RenderParameters& params,
    const vec3* sensorRayOriginCudaPtr,
    const vec3* sensorRayDirectionCudaPtr,
    const uint8_t* physicalMaskCudaPtr,
    const vec3* queryPointsCudaPtr,
    uint32_t queryCount,
    const float* particleSigmaMaxCudaPtr,
    float* crossViewMinimumOpacityCudaPtr,
    uint8_t* crossViewObservedCudaPtr,
    uint32_t* fallbackCameraCounterCudaPtr,
    uint32_t* hostSelectiveFallbackQueryCount,
    uint32_t* hostSelectiveFallbackFirstQuery,
    uint32_t* hostFallbackTelemetry,
    bool thresholdTieAware,
    Parameters& parameters,
    int cudaDeviceIndex,
    cudaStream_t cudaStream
    ) {
    int currentCudaDevice = -1;
    const cudaError_t currentDeviceStatus = cudaGetDevice(&currentCudaDevice);
    if (currentDeviceStatus != cudaSuccess || currentCudaDevice != cudaDeviceIndex) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "full-shape field renderer current CUDA device differs");
    }
    if (parameters.values.numParticles == 0 || queryCount == 0 ||
        sensorRayOriginCudaPtr == nullptr || sensorRayDirectionCudaPtr == nullptr ||
        queryPointsCudaPtr == nullptr ||
        particleSigmaMaxCudaPtr == nullptr ||
        crossViewMinimumOpacityCudaPtr == nullptr ||
        crossViewObservedCudaPtr == nullptr || fallbackCameraCounterCudaPtr == nullptr ||
        hostSelectiveFallbackQueryCount == nullptr ||
        hostSelectiveFallbackFirstQuery == nullptr ||
        hostFallbackTelemetry == nullptr) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "full-shape field received null/empty input");
    }
    *hostSelectiveFallbackQueryCount = 0;
    *hostSelectiveFallbackFirstQuery = UINT32_MAX;
    const TSensorPose worldToSensorPose = interpolatedSensorPose(
        params.sensorState.startPose, params.sensorState.endPose, 0.5f);
    const TSensorPose sensorToWorldPose = sensorPoseInverse(worldToSensorPose);
    if (!m_forwardContext || m_forwardContext->cudaStream != cudaStream) {
        m_forwardContext = std::make_unique<GutRenderForwardContext>(cudaStream);
    }
    return fast_sof_tnt_full_shape::evaluateCooperative(
        m_forwardContext->fastSofTntFullShapeWorkspace, params,
        sensorRayOriginCudaPtr, sensorRayDirectionCudaPtr, physicalMaskCudaPtr,
        queryPointsCudaPtr, queryCount, parameters.values.numParticles,
        particleSigmaMaxCudaPtr, crossViewMinimumOpacityCudaPtr,
        crossViewObservedCudaPtr, fallbackCameraCounterCudaPtr,
        hostSelectiveFallbackQueryCount, hostSelectiveFallbackFirstQuery,
        hostFallbackTelemetry, thresholdTieAware,
        parameters.m_dptrParametersBuffer, sensorPoseToMat(worldToSensorPose),
        sensorPoseToMat(sensorToWorldPose), cudaStream, m_logger
        );
}
