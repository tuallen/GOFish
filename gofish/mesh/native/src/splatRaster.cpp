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

// Portions derived from SOF (https://github.com/r4dl/SOF): the SOF opacity-field and geometry terms,
// distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial research and evaluation use):
// third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

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

} // namespace

//------------------------------------------------------------------------------
// SplatRaster
//------------------------------------------------------------------------------
TSensorState toSensorState(threedgut::TTimestamp startTs, torch::Tensor sensorsStartPose, threedgut::TTimestamp endTs, torch::Tensor sensorsEndPose) {
    const torch::Tensor sensorsStartPoseHost = sensorsStartPose.cpu();
    const torch::Tensor sensorsEndPoseHost   = sensorsEndPose.cpu();

    auto startInvPosePtr = reinterpret_cast<const threedgut::TSensorPose*>(sensorsStartPoseHost.data_ptr<float>());
    auto endInvPosePtr   = reinterpret_cast<const threedgut::TSensorPose*>(sensorsEndPoseHost.data_ptr<float>());

    return threedgut::TSensorState{startTs, /*poseInverse*/ (*startInvPosePtr), endTs, /*poseInverse*/ (*endInvPosePtr)};
}

#include <3dgut/kernels/cuda/common/opacityFieldBinding.inl>

SplatRaster::SplatRaster(const nlohmann::json& config)
    : m_logLevel(static_cast<uint8_t>(LoggerParameters::Debug))
    , m_logger(LoggerParameters{m_logLevel, logCallback, nullptr, nullptr, nullptr})
    , m_renderer(std::make_unique<GUTRenderer>(config, m_logger)) {

    m_parameters.valuesBuffer.resize(sizeof(m_parameters.values), 0, m_logger);
    m_parameters.parametersBuffer.resize(sizeof(m_parameters.parameters), 0, m_logger);

    m_parameters.parameters.dptrValuesBuffer = m_parameters.valuesBuffer.data();
    m_parameters.m_dptrParametersBuffer      = (uint64_t*)m_parameters.parametersBuffer.data();
}

SplatRaster::~SplatRaster(void) {
    // Context destruction synchronizes its owning stream without throwing.
    m_opacityFieldContext.reset();
}
