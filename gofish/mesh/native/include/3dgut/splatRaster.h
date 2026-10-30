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

#pragma once

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/utils/logger.h>

#include <json/json.hpp>
#include <pybind11_json/pybind11_json.hpp>

#include <string>

using TTimestamp = int64_t;

class SplatRaster final {
private:
    uint8_t m_logLevel;
    threedgut::Logger m_logger;

    std::unique_ptr<threedgut::GUTRenderer> m_renderer;

    threedgut::GUTRenderer::Parameters m_parameters;

    struct OpacityFieldContext;
    std::unique_ptr<OpacityFieldContext> m_opacityFieldContext;
    uint64_t m_opacityFieldNextToken = 1;

    bool prepareOpacityFieldCamera(OpacityFieldContext& context,
                                   uint32_t frameNumber,
                                   torch::Tensor rayOrigin,
                                   torch::Tensor rayDirection,
                                   torch::Tensor physicalMask,
                                   threedgut::TSensorModel sensorModel,
                                   TTimestamp startTimestamp,
                                   TTimestamp endTimestamp,
                                   torch::Tensor sensorsStartPose,
                                   torch::Tensor sensorsEndPose,
                                   threedgut::RenderParameters& params);

public:
    SplatRaster(const nlohmann::json& config);

    ~SplatRaster();

    // Matched camera-min opacity field (mesh extraction).
    int64_t beginOpacityField(torch::Tensor particleDensity,
                              torch::Tensor particleSigmaMax,
                              torch::Tensor queryPoints);
    void integrateOpacityFieldCamera(int64_t contextToken,
                                     uint32_t frameNumber,
                                     torch::Tensor rayOrigin,
                                     torch::Tensor rayDirection,
                                     torch::Tensor physicalMask,
                                     threedgut::TSensorModel sensorModel,
                                     TTimestamp startTimestamp,
                                     TTimestamp endTimestamp,
                                     torch::Tensor sensorsStartPose,
                                     torch::Tensor sensorsEndPose);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    finishOpacityField(int64_t contextToken);
    void releaseOpacityField(int64_t contextToken);

    // Persistent camera table for the fixed bisection rounds.
    int64_t beginBisectionTable(torch::Tensor particleDensity,
                                torch::Tensor particleSigmaMax,
                                torch::Tensor probeQueryPoints);
    void stageBisectionCamera(int64_t contextToken,
                              uint32_t frameNumber,
                              torch::Tensor rayOrigin,
                              torch::Tensor rayDirection,
                              torch::Tensor physicalMask,
                              threedgut::TSensorModel sensorModel,
                              TTimestamp startTimestamp,
                              TTimestamp endTimestamp,
                              torch::Tensor sensorsStartPose,
                              torch::Tensor sensorsEndPose);
    void sealBisectionTable(int64_t contextToken);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluateBisectionRound(int64_t contextToken, torch::Tensor queryPoints, uint32_t roundOrdinal);
    void resetBisectionBatch(int64_t contextToken);
    void releaseBisectionTable(int64_t contextToken);
};
