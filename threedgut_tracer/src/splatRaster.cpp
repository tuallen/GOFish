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
//
// Portions derived from SOF and CoMe (https://github.com/r4dl/SOF, https://github.com/r4dl/CoMe):
// the SOF opacity-field and geometry terms and the CoMe confidence paths, distributed under the Gaussian-Splatting License (Inria / MPII;
// non-commercial research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgut/splatRaster.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/core/ScalarType.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>

//------------------------------------------------------------------------------
// CUDA / OPTIX macros
//------------------------------------------------------------------------------

#define CUDA_CHECK_LAST(logger) \
    CUDA_CHECK(logger, cudaGetLastError())

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------

using namespace threedgut;

namespace {

static void THREEDGUT_LOGGER_CB logCallback(uint8_t level, const char* msg, void* data) {
    std::ostream& stream = (level > LoggerParameters::Error) ? std::cout : std::cerr;
    stream << "[3dgut][" << LoggerParameters::levelToString(level) << "] ::: "
           << msg << std::flush << std::endl;
}

const char* errorCodeName(ErrorCode status) noexcept {
    switch (status) {
    case ErrorCode::None:
        return "success";
    case ErrorCode::InvalidResourcesError:
        return "invalid resources";
    case ErrorCode::BadInput:
        return "invalid inputs";
    case ErrorCode::OutOfMemory:
        return "out of memory";
    case ErrorCode::NotImplemented:
        return "missing implementation";
    case ErrorCode::Runtime:
        return "runtime error";
    case ErrorCode::Num:
        return "invalid error-code sentinel";
    }
    return "unknown error code";
}

void requireStatus(ErrorCode status, const char* operation) {
    TORCH_CHECK(
        THREEDGUT_SUCCESS(status), "[3dgut] ", operation,
        " failed with renderer status ", static_cast<int>(status), " (",
        errorCodeName(status), ")");
}

void requireStatus(const Status& status, const char* operation) {
    requireStatus(static_cast<ErrorCode>(status), operation);
}

} // namespace

//------------------------------------------------------------------------------
// SplatRaster
//------------------------------------------------------------------------------
inline void* voidDataPtr(torch::Tensor& tensor) {
    if (tensor.size(0) == 0) {
        return nullptr;
    }
    switch (tensor.scalar_type()) {
    case torch::kFloat32:
        return tensor.contiguous().data_ptr<float>();
    case torch::kHalf:
        return tensor.contiguous().data_ptr<torch::Half>();
    case torch::kInt32:
        return tensor.contiguous().data_ptr<int32_t>();
    case torch::kInt64:
        return tensor.contiguous().data_ptr<int64_t>();
    // case torch::kUInt32:
    //     return tensor.contiguous().data_ptr<uint32_t>();
    // case torch::kUInt64:
    //     return tensor.contiguous().data_ptr<uint64_t>();
    default:
        throw std::runtime_error{"[3dgut] Unknown precision torch->void: " + std::string(c10::toString(tensor.scalar_type()))};
    }
}

torch::Tensor particleRadianceKernelTensor(const torch::Tensor& particleRadiance) {
#if PARTICLE_FEATURE_HALF
    return particleRadiance.scalar_type() == torch::kHalf ? particleRadiance.contiguous()
                                                          : particleRadiance.to(torch::kHalf).contiguous();
#else
    return particleRadiance.scalar_type() == torch::kFloat32 ? particleRadiance.contiguous()
                                                             : particleRadiance.to(torch::kFloat32).contiguous();
#endif
}

using DeviceQueueHandle = uint64_t;

threedgut::TSensorPose poseInverse(const TSensorPose& pose) {
    static_assert(sizeof(TSensorPose) == sizeof(threedgut::TSensorPose));
    const threedgut::TSensorPose iPose = threedgut::sensorPoseInverse({pose.elems[0], pose.elems[1], pose.elems[2], pose.elems[3], pose.elems[4], pose.elems[5], pose.elems[6]});
    return {iPose[0], iPose[1], iPose[2], iPose[3], iPose[4], iPose[5], iPose[6]};
}

TSensorState toSensorState(threedgut::TTimestamp startTs, torch::Tensor sensorsStartPose, threedgut::TTimestamp endTs, torch::Tensor sensorsEndPose) {
    const torch::Tensor sensorsStartPoseHost = sensorsStartPose.cpu();
    const torch::Tensor sensorsEndPoseHost   = sensorsEndPose.cpu();

    auto startInvPosePtr = reinterpret_cast<const threedgut::TSensorPose*>(sensorsStartPoseHost.data_ptr<float>());
    auto endInvPosePtr   = reinterpret_cast<const threedgut::TSensorPose*>(sensorsEndPoseHost.data_ptr<float>());

    return threedgut::TSensorState{startTs, /*poseInverse*/ (*startInvPosePtr), endTs, /*poseInverse*/ (*endInvPosePtr)};
}

struct SplatRaster::CudaTimer {
    const char* _tag;
    cudaStream_t _stream;
    cudaEvent_t _start = 0, _stop = 0;
    bool _valid = false, _stopped = false;

    CudaTimer(const char* tag, cudaStream_t stream)
        : _tag(tag), _stream(stream) {
        _valid = cudaEventCreate(&_start) == cudaSuccess;
        _valid = _valid && (cudaEventCreate(&_stop) == cudaSuccess);
        if (_valid) {
            cudaEventRecord(_start, _stream);
        }
    }
    ~CudaTimer() {
        if (_stop) {
            cudaEventDestroy(_stop);
        }
        if (_start) {
            cudaEventDestroy(_start);
        }
    }

    std::string tag() const {
        return _tag;
    }

    void stop() {
        if (_valid && !_stopped) {
            cudaEventRecord(_stop, _stream);
            _stopped = true;
        }
    }

    float collect() {
        float milliseconds = 1e09f;
        if (_valid) {
            stop();
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&milliseconds, _start, _stop);
        }
        return milliseconds;
    }
};

SplatRaster::SplatRaster(const nlohmann::json& config)
    : m_logLevel(static_cast<uint8_t>(LoggerParameters::Debug))
    , m_logger(LoggerParameters{m_logLevel, logCallback, nullptr, nullptr, nullptr})
    , m_renderer(std::make_unique<GUTRenderer>(config, m_logger)) {

    const auto& renderConfig = config["render"];
    m_enableKernelTimings    = renderConfig.value("enable_kernel_timings", false);

    requireStatus(
        m_parameters.valuesBuffer.resize(
            sizeof(m_parameters.values), 0, m_logger),
        "SplatRaster parameter-value buffer initialization");
    requireStatus(
        m_parameters.parametersBuffer.resize(
            sizeof(m_parameters.parameters), 0, m_logger),
        "SplatRaster parameter-pointer buffer initialization");
    requireStatus(
        m_parameters.gradientsBuffer.resize(
            sizeof(m_parameters.gradients), 0, m_logger),
        "SplatRaster gradient-pointer buffer initialization");

    m_parameters.parameters.dptrValuesBuffer = m_parameters.valuesBuffer.data();
    m_parameters.m_dptrParametersBuffer      = (uint64_t*)m_parameters.parametersBuffer.data();
    m_parameters.m_dptrGradientsBuffer       = (uint64_t*)m_parameters.gradientsBuffer.data();
}

SplatRaster::~SplatRaster(void) {
}

void SplatRaster::registerPhysicalTileMask(
    torch::Tensor mask, const std::string& token) {
    TORCH_CHECK(token.size() == 64u
                && std::all_of(token.begin(), token.end(), [](char value) {
                       return (value >= '0' && value <= '9')
                           || (value >= 'a' && value <= 'f');
                   }), "physical tile token must be lowercase SHA256");
    TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == torch::kUInt8
                && mask.is_contiguous() && mask.dim() == 2,
                "physical tile mask must be contiguous CUDA uint8 [Ty,Tx]");
    TORCH_CHECK(m_physicalTileMasks.find(token) == m_physicalTileMasks.end(),
                "physical tile token was already registered");
    // Values were validated once when the cached binding was constructed.
    // Registration and trace/traceBwd contain no device reduction or scalar read.
    m_physicalTileMasks.emplace(token, mask);
}

torch::Tensor SplatRaster::requirePhysicalTileMask(
    const std::string& token, int device, int height, int width) const {
    const auto found = m_physicalTileMasks.find(token);
    TORCH_CHECK(found != m_physicalTileMasks.end(),
                "physical tile token is not registered on this SplatRaster");
    const torch::Tensor& mask = found->second;
    TORCH_CHECK(mask.get_device() == device
                && mask.size(0) == (height + 15) / 16
                && mask.size(1) == (width + 15) / 16,
                "physical tile token does not match current raster/device");
    return mask;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SplatRaster::trace(uint32_t frameNumber, bool sofExactOpacityActive,
                   int numActiveFeatures,
                   torch::Tensor particleDensity,
                   torch::Tensor particleRadiance,
                   torch::Tensor rayOrigin,
                   torch::Tensor rayDirection,
                   torch::Tensor rayTimestamp,
                   TSensorModel sensorModel,
                   TTimestamp startTimestamp,
                   TTimestamp endTimestamp,
                   torch::Tensor sensorsStartPose,
                   torch::Tensor sensorsEndPose,
                   const std::string& physicalTileToken) {

    const int cudaDeviceIndex = rayOrigin.get_device();
    m_lastCudaDeviceIndex = cudaDeviceIndex;
    cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

    const int width  = rayOrigin.size(2);
    const int height = rayOrigin.size(1);
    const torch::Tensor physicalTileMask = requirePhysicalTileMask(
        physicalTileToken, cudaDeviceIndex, height, width);

    const uint32_t numParticles = particleDensity.size(0);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // Feature output dtype: fp16 halves memory bandwidth for NHT feature buffers
#if FEATURE_OUTPUT_HALF
    const torch::TensorOptions featureOpts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
#else
    const torch::TensorOptions featureOpts = opts;
#endif

    torch::Tensor rayRadianceDensity     = torch::zeros({height, width, static_cast<int64_t>(RAY_FEATURE_DIM + 1)}, featureOpts);
    torch::Tensor rayHitDistance         = torch::ones({height, width, 1}, opts).multiply(1e06f);
    torch::Tensor rayHitCount            = torch::zeros({height, width, 1}, opts);
    torch::Tensor particleVisibility     = torch::zeros({numParticles, 1}, opts);
    torch::Tensor rayNormal              = torch::zeros({height, width, 3}, opts);
    torch::Tensor rayDepthVar            = torch::zeros({height, width, 1}, opts);
    // Front-surface / quantile depth readout: FRONT_NQ(=7) opacity-threshold crossing
    // depths, the max-contribution (tau_max) depth, the exact 0.5-crossing t* and its
    // rate. Keep in sync with FRONT_NQ in rayPayload.cuh.
#ifndef GOFISH_COME_CONFIDENCE
#define GOFISH_COME_CONFIDENCE 0
#endif
    torch::Tensor rayQuantDepth          = torch::zeros({height, width, 10 + GOFISH_COME_CONFIDENCE}, opts);  // FRONT_NQ+3: quantile depths, tau, t*, crossing rate (+1 CoMe confidence)
    // Differentiable along-ray EXTENT: RAW sum_i w_i * sigma_ray_i per ray (1 channel).
    torch::Tensor rayExtent              = torch::zeros({height, width, 1}, opts);
    // Differentiable OPACITY-FIELD: O_N = 1 - prod_i (1 - a_i@min(t_i,t*)) per ray (1 channel).
    torch::Tensor rayOpacityField        = torch::zeros({height, width, 1}, opts);
    // Differentiable SOF inverse-depth distortion numerators: sum_i w_i*(1/d_i), sum_i w_i*(1/d_i^2).
    torch::Tensor rayInvD                = torch::zeros({height, width, 1}, opts);
    torch::Tensor rayInvD2               = torch::zeros({height, width, 1}, opts);
    torch::Tensor particleRadianceKernel = particleRadianceKernelTensor(particleRadiance);

    m_parameters.values.numParticles               = numParticles;
    m_parameters.values.radianceSphDegree          = numActiveFeatures;
    m_parameters.parameters.dptrDensityParameters  = voidDataPtr(particleDensity);
    m_parameters.parameters.dptrRadianceParameters = voidDataPtr(particleRadianceKernel);
    requireStatus(
        m_parameters.valuesBuffer.setFromHost(
            &m_parameters.values, sizeof(m_parameters.values),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::trace parameter-value upload");
    requireStatus(
        m_parameters.parametersBuffer.setFromHost(
            &m_parameters.parameters, sizeof(m_parameters.parameters),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::trace parameter-pointer upload");

    std::shared_ptr<CudaTimer> timer;
    if (m_enableKernelTimings) {
        timer = std::make_shared<CudaTimer>("forward_render", cudaStream);
        m_timers.emplace_back(timer);
        // keep only the last m_maxNumTimers timers
        if (m_timers.size() > m_maxNumTimers) {
            m_timers.pop_front();
        }
    }

    threedgut::RenderParameters renderParameters;
    renderParameters.id          = frameNumber;
    renderParameters.resolution  = tcnn::ivec2{width, height};
    renderParameters.sensorModel = sensorModel;
    renderParameters.sensorState = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
    renderParameters.objectAABB  = threedgut::BoundingBox{tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    m_renderer->setPhysicalTileMask(
        physicalTileMask.data_ptr<uint8_t>(),
        static_cast<uint32_t>(physicalTileMask.numel()));
    ErrorCode status = m_renderer->renderForward(
        renderParameters,
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayOrigin)),
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayDirection)),
        reinterpret_cast<float*>(voidDataPtr(rayHitCount)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
        reinterpret_cast<float*>(voidDataPtr(rayNormal)),
        reinterpret_cast<float*>(voidDataPtr(rayDepthVar)),
        reinterpret_cast<float*>(voidDataPtr(rayQuantDepth)),
        reinterpret_cast<float*>(voidDataPtr(rayExtent)),
        sofExactOpacityActive
            ? reinterpret_cast<float*>(voidDataPtr(rayOpacityField))
            : nullptr,
        reinterpret_cast<float*>(voidDataPtr(rayInvD)),
        reinterpret_cast<float*>(voidDataPtr(rayInvD2)),
        reinterpret_cast<TFeatureDensityElem*>(voidDataPtr(rayRadianceDensity)),
        reinterpret_cast<int*>(voidDataPtr(particleVisibility)),
        m_parameters,
        cudaDeviceIndex,
        cudaStream);

    requireStatus(status, "GUTRenderer::renderForward");

    CUDA_CHECK_LAST(m_logger);

    if (timer) {
        timer->stop();
    }

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(rayRadianceDensity, rayHitDistance, rayHitCount, particleVisibility, rayNormal, rayDepthVar, rayQuantDepth, rayExtent, rayOpacityField, rayInvD, rayInvD2);
}

#if GOFISH_INFERENCE_ONLY
std::tuple<torch::Tensor, torch::Tensor>
SplatRaster::traceInference(int numActiveFeatures,
                            torch::Tensor particleDensity,
                            torch::Tensor particleRadiance,
                            torch::Tensor rayOrigin,
                            torch::Tensor rayDirection,
                            TSensorModel sensorModel,
                            TTimestamp startTimestamp,
                            TTimestamp endTimestamp,
                            torch::Tensor sensorsStartPose,
                            torch::Tensor sensorsEndPose,
                            const std::string& physicalTileToken) {
    const int cudaDeviceIndex = rayOrigin.get_device();
    m_lastCudaDeviceIndex = cudaDeviceIndex;
    cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

    const int width  = rayOrigin.size(2);
    const int height = rayOrigin.size(1);
    const torch::Tensor physicalTileMask = requirePhysicalTileMask(
        physicalTileToken, cudaDeviceIndex, height, width);

    const uint32_t numParticles = particleDensity.size(0);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
#if FEATURE_OUTPUT_HALF
    const torch::TensorOptions featureOpts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
#else
    const torch::TensorOptions featureOpts = opts;
#endif
    // Same initial values as trace(): pixels of tiles outside the camera
    // domain keep them.
    torch::Tensor rayRadianceDensity = torch::zeros({height, width, static_cast<int64_t>(RAY_FEATURE_DIM + 1)}, featureOpts);
    torch::Tensor rayHitDistance     = torch::ones({height, width, 1}, opts).multiply(1e06f);
    if (!m_inferenceVisibility.defined() || m_inferenceVisibility.size(0) != numParticles
        || m_inferenceVisibility.get_device() != cudaDeviceIndex) {
        m_inferenceVisibility = torch::zeros({numParticles, 1}, opts);
    } else {
        m_inferenceVisibility.zero_();
    }
    torch::Tensor particleRadianceKernel = particleRadianceKernelTensor(particleRadiance);

    m_parameters.values.numParticles               = numParticles;
    m_parameters.values.radianceSphDegree          = numActiveFeatures;
    m_parameters.parameters.dptrDensityParameters  = voidDataPtr(particleDensity);
    m_parameters.parameters.dptrRadianceParameters = voidDataPtr(particleRadianceKernel);
    requireStatus(
        m_parameters.valuesBuffer.setFromHost(
            &m_parameters.values, sizeof(m_parameters.values),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::traceInference parameter-value upload");
    requireStatus(
        m_parameters.parametersBuffer.setFromHost(
            &m_parameters.parameters, sizeof(m_parameters.parameters),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::traceInference parameter-pointer upload");

    threedgut::RenderParameters renderParameters;
    renderParameters.id          = 0;
    renderParameters.resolution  = tcnn::ivec2{width, height};
    renderParameters.sensorModel = sensorModel;
    renderParameters.sensorState = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
    renderParameters.objectAABB  = threedgut::BoundingBox{tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    m_renderer->setPhysicalTileMask(
        physicalTileMask.data_ptr<uint8_t>(),
        static_cast<uint32_t>(physicalTileMask.numel()));
    ErrorCode status = m_renderer->renderForward(
        renderParameters,
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayOrigin)),
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayDirection)),
        /*worldHitCount=*/nullptr,
        reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
        /*normal=*/nullptr, /*depthVar=*/nullptr, /*quantDepth=*/nullptr,
        /*extent=*/nullptr, /*opacityField=*/nullptr,
        /*invD=*/nullptr, /*invD2=*/nullptr,
        reinterpret_cast<TFeatureDensityElem*>(voidDataPtr(rayRadianceDensity)),
        reinterpret_cast<int*>(voidDataPtr(m_inferenceVisibility)),
        m_parameters,
        cudaDeviceIndex,
        cudaStream);
    requireStatus(status, "GUTRenderer::renderForward (inference)");
    CUDA_CHECK_LAST(m_logger);
    return {rayRadianceDensity, rayHitDistance};
}
#endif

std::tuple<torch::Tensor, torch::Tensor>
SplatRaster::traceBwd(uint32_t frameNumber, bool sofExactOpacityActive,
                      int numActiveFeatures,
                      torch::Tensor particleDensity,
                      torch::Tensor particleRadiance,
                      torch::Tensor rayOrigin,
                      torch::Tensor rayDirection,
                      torch::Tensor rayTimestamp,
                      TSensorModel sensorModel,
                      TTimestamp startTimestamp,
                      TTimestamp endTimestamp,
                      torch::Tensor sensorsStartPose,
                      torch::Tensor sensorsEndPose,
                      // Gradients
                      torch::Tensor rayRadianceDensity,
                      torch::Tensor rayRadianceDensityGradient,
                      torch::Tensor rayHitDistance,
                      torch::Tensor rayHitDistanceGradient,
                      torch::Tensor rayDepthVar,
                      torch::Tensor rayDepthVarGradient,
                      torch::Tensor rayExtent,
                      torch::Tensor rayExtentGradient,
                      torch::Tensor rayOpacityField,
                      torch::Tensor rayOpacityFieldGradient,
                      torch::Tensor rayQuantDepth,
                      torch::Tensor rayQuantDepthGradient,
                      torch::Tensor rayNormal,
                      torch::Tensor rayNormalGradient,
                      torch::Tensor rayInvD,
                      torch::Tensor rayInvDGradient,
                      torch::Tensor rayInvD2,
                      torch::Tensor rayInvD2Gradient,
                      const std::string& physicalTileToken) {

    const int cudaDeviceIndex = rayOrigin.get_device();
    cudaStream_t cudaStream   = at::cuda::getCurrentCUDAStream(cudaDeviceIndex);

    const int width  = rayOrigin.size(2);
    const int height = rayOrigin.size(1);
    const torch::Tensor physicalTileMask = requirePhysicalTileMask(
        physicalTileToken, cudaDeviceIndex, height, width);

    const uint32_t numParticles = particleDensity.size(0);

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor particleDensityGradient  = torch::zeros({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor particleRadianceGradient = torch::zeros({particleRadiance.size(0), particleRadiance.size(1)}, opts);
    torch::Tensor particlePositionAbsGradient;
    float* particlePositionAbsGradientPtr = nullptr;
#if GOFISH_EXACT_SCREEN_ABSGRAD
    if (m_exactScreenAbsGradWorkspace.defined()) {
        TORCH_CHECK(m_exactScreenAbsGradWorkspace.is_cuda()
                    && m_exactScreenAbsGradWorkspace.scalar_type() == torch::kFloat32
                    && m_exactScreenAbsGradWorkspace.is_contiguous()
                    && m_exactScreenAbsGradWorkspace.dim() == 2
                    && m_exactScreenAbsGradWorkspace.size(0) == particleDensity.size(0)
                    && m_exactScreenAbsGradWorkspace.size(1) == 8,
                    "exact screen AbsGrad workspace differs");
        particlePositionAbsGradientPtr =
            reinterpret_cast<float*>(voidDataPtr(m_exactScreenAbsGradWorkspace));
    }
#else
    particlePositionAbsGradient = torch::zeros({particleDensity.size(0), 3}, opts);
    particlePositionAbsGradientPtr =
        reinterpret_cast<float*>(voidDataPtr(particlePositionAbsGradient));
#endif
    torch::Tensor particleRadianceKernel   = particleRadianceKernelTensor(particleRadiance);

    const bool rayBackpropagation = false;

    torch::Tensor rayOriginGradient;
    torch::Tensor rayDirectionGradient;
    if (rayBackpropagation) {
        rayOriginGradient    = torch::zeros({rayOrigin.size(0), rayOrigin.size(1), rayOrigin.size(2), 3}, opts);
        rayDirectionGradient = torch::zeros({rayDirection.size(0), rayDirection.size(1), rayDirection.size(2), 3}, opts);
    }

    m_parameters.values.numParticles               = numParticles;
    m_parameters.values.radianceSphDegree          = numActiveFeatures;
    m_parameters.parameters.dptrDensityParameters  = voidDataPtr(particleDensity);
    m_parameters.parameters.dptrRadianceParameters = voidDataPtr(particleRadianceKernel);
    m_parameters.gradients.dptrDensityGradients    = voidDataPtr(particleDensityGradient);
    m_parameters.gradients.dptrRadianceGradients   = voidDataPtr(particleRadianceGradient);
    requireStatus(
        m_parameters.valuesBuffer.setFromHost(
            &m_parameters.values, sizeof(m_parameters.values),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::traceBwd parameter-value upload");
    requireStatus(
        m_parameters.parametersBuffer.setFromHost(
            &m_parameters.parameters, sizeof(m_parameters.parameters),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::traceBwd parameter-pointer upload");
    requireStatus(
        m_parameters.gradientsBuffer.setFromHost(
            &m_parameters.gradients, sizeof(m_parameters.gradients),
            reinterpret_cast<uint64_t>(cudaStream), m_logger),
        "SplatRaster::traceBwd gradient-pointer upload");

    std::shared_ptr<CudaTimer> timer;
    if (m_enableKernelTimings) {
        timer = std::make_shared<CudaTimer>("backward_render", cudaStream);
        m_timers.emplace_back(timer);
        // keep only the last m_maxNumTimers timers
        if (m_timers.size() > m_maxNumTimers) {
            m_timers.pop_front();
        }
    }

    threedgut::RenderParameters renderParameters;
    renderParameters.id          = frameNumber;
    renderParameters.resolution  = tcnn::ivec2{width, height};
    renderParameters.sensorModel = sensorModel;
    renderParameters.sensorState = toSensorState(startTimestamp, sensorsStartPose, endTimestamp, sensorsEndPose);
    renderParameters.objectAABB  = threedgut::BoundingBox{tcnn::vec3{-1e06f, -1e06f, -1e06f}, tcnn::vec3{1e06f, 1e06f, 1e06f}};

    m_renderer->setPhysicalTileMask(
        physicalTileMask.data_ptr<uint8_t>(),
        static_cast<uint32_t>(physicalTileMask.numel()));
    ErrorCode status = m_renderer->renderBackward(
        renderParameters,
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayOrigin)),
        reinterpret_cast<const tcnn::vec3*>(voidDataPtr(rayDirection)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistance)),
        reinterpret_cast<float*>(voidDataPtr(rayHitDistanceGradient)),
        reinterpret_cast<const float*>(voidDataPtr(rayDepthVar)),
        reinterpret_cast<const float*>(voidDataPtr(rayDepthVarGradient)),
        reinterpret_cast<const float*>(voidDataPtr(rayExtent)),
        reinterpret_cast<const float*>(voidDataPtr(rayExtentGradient)),
        sofExactOpacityActive
            ? reinterpret_cast<const float*>(voidDataPtr(rayOpacityField))
            : nullptr,
        sofExactOpacityActive
            ? reinterpret_cast<const float*>(voidDataPtr(rayOpacityFieldGradient))
            : nullptr,
        reinterpret_cast<const float*>(voidDataPtr(rayQuantDepth)),
        reinterpret_cast<const float*>(voidDataPtr(rayQuantDepthGradient)),
        reinterpret_cast<const TFeatureDensityElem*>(voidDataPtr(rayRadianceDensity)),
        reinterpret_cast<const float*>(voidDataPtr(rayRadianceDensityGradient)),
        rayBackpropagation ? reinterpret_cast<tcnn::vec3*>(voidDataPtr(rayOriginGradient)) : nullptr,
        rayBackpropagation ? reinterpret_cast<tcnn::vec3*>(voidDataPtr(rayDirectionGradient)) : nullptr,
        particlePositionAbsGradientPtr,
        reinterpret_cast<const float*>(voidDataPtr(rayNormal)),
        reinterpret_cast<const float*>(voidDataPtr(rayNormalGradient)),
        reinterpret_cast<const float*>(voidDataPtr(rayInvD)),
        reinterpret_cast<const float*>(voidDataPtr(rayInvDGradient)),
        reinterpret_cast<const float*>(voidDataPtr(rayInvD2)),
        reinterpret_cast<const float*>(voidDataPtr(rayInvD2Gradient)),
        m_parameters, cudaDeviceIndex, cudaStream);

    requireStatus(status, "GUTRenderer::renderBackward");

    CUDA_CHECK_LAST(m_logger);

    if (m_enableKernelTimings) {
        timer->stop();
    }

#if GOFISH_EXACT_SCREEN_ABSGRAD
    if (m_exactScreenAbsGradWorkspace.defined()) {
        m_lastPositionAbsGradient =
            m_exactScreenAbsGradWorkspace.slice(1, 6, 8).contiguous();
        m_exactScreenAbsGradWorkspace = torch::Tensor();
    } else {
        m_lastPositionAbsGradient = torch::Tensor();
    }
#else
    m_lastPositionAbsGradient = particlePositionAbsGradient;
#endif
    return std::tuple<torch::Tensor, torch::Tensor>(particleDensityGradient, particleRadianceGradient);
}

torch::Tensor SplatRaster::collectPositionAbsGrad() {
#if GOFISH_EXACT_SCREEN_ABSGRAD
    TORCH_CHECK(m_lastPositionAbsGradient.defined()
                && m_lastPositionAbsGradient.dim() == 2
                && m_lastPositionAbsGradient.size(1) == 2,
                "exact screen AbsGrad was not armed for the last backward");
#endif
    return m_lastPositionAbsGradient;
}

torch::Tensor SplatRaster::collectSofMaxRadii2D() {
    TORCH_CHECK(m_lastCudaDeviceIndex >= 0,
                "SOF radius collection requires a completed forward pass");
    const int64_t count = static_cast<int64_t>(m_renderer->lastNumParticles());
    const auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::Device(torch::kCUDA, m_lastCudaDeviceIndex));
    torch::Tensor radii = torch::empty({count}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(m_lastCudaDeviceIndex);
    ErrorCode status = m_renderer->collectSofMaxRadii2D(
        radii.data_ptr<float>(), stream);
    requireStatus(status, "GUTRenderer::collectSofMaxRadii2D");
    CUDA_CHECK_LAST(m_logger);
    return radii;
}

std::tuple<torch::Tensor, torch::Tensor>
SplatRaster::collectSofScreenPruneState() {
    TORCH_CHECK(m_lastCudaDeviceIndex >= 0,
                "SOF prune-state collection requires a completed forward pass");
    const int64_t count = static_cast<int64_t>(m_renderer->lastNumParticles());
    const auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::Device(torch::kCUDA, m_lastCudaDeviceIndex));
    torch::Tensor provenGt20 = torch::empty({count}, options);
    torch::Tensor cameraRelation = torch::empty({count}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(m_lastCudaDeviceIndex);
    ErrorCode status = m_renderer->collectSofScreenPruneState(
        provenGt20.data_ptr<uint8_t>(), cameraRelation.data_ptr<uint8_t>(),
        stream);
    requireStatus(status, "GUTRenderer::collectSofScreenPruneState");
    CUDA_CHECK_LAST(m_logger);
    return {provenGt20, cameraRelation};
}

#if GOFISH_EXACT_SCREEN_ABSGRAD
void SplatRaster::setExactScreenAbsGradMap(torch::Tensor map) {
    TORCH_CHECK(map.is_cuda() && map.scalar_type() == torch::kFloat32
                && map.is_contiguous() && map.dim() == 3
                && map.size(1) == 2 && map.size(2) == 3,
                "exact screen AbsGrad map must be contiguous CUDA float32 [N,2,3]");
    TORCH_CHECK(!m_exactScreenAbsGradWorkspace.defined(),
                "exact screen AbsGrad map was armed twice before backward");
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(map.device());
    m_exactScreenAbsGradWorkspace = torch::zeros({map.size(0), 8}, opts);
    m_exactScreenAbsGradWorkspace.slice(1, 0, 6).copy_(map.reshape({map.size(0), 6}));
}

#endif

std::map<std::string, float>
SplatRaster::collectTimes() {

    if (!m_timers.empty()) {

        std::map<std::string, float> timings;
        std::map<std::string, int> counts;

        for (auto& timer : m_timers) {
            const auto tag = timer->tag();

            if (timings.count(tag) == 0) {
                timings[tag] = timer->collect();
                counts[tag]  = 1;
            }

            else {
                timings[tag] += timer->collect();
                counts[tag] += 1;
            }
        }

        for (auto& [tag, time] : timings) {
            m_timings[tag] = time / counts[tag];
        }

        m_timers.clear();
    }

    return m_timings;
}
