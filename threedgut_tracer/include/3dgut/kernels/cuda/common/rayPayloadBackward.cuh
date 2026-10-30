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

#pragma once

#include <3dgut/kernels/cuda/common/rayPayload.cuh>

template <int FeatN>
struct RayPayloadBackward : public RayPayload<FeatN> {
    float transmittanceBackward;
    float transmittanceGradient;
    float hitTBackward;
    float hitTGradient;
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    float sumWD2Backward;
    float sumWD2Gradient;
    float sumExtBackward;        // integrated (final) raw extent = sum_i w_i sigma_ray_i (read from fwd output)
    float sumExtGradient;        // dL/d(pred_extent)
    float sumWinvDBackward;      // integrated (final) sum_i w_i*(1/d_i) (read from fwd output)
    float sumWinvDGradient;      // dL/d(pred_invd)
    float sumWinvD2Backward;     // integrated (final) sum_i w_i*(1/d_i^2) (read from fwd output)
    float sumWinvD2Gradient;     // dL/d(pred_invd2)
    float opaFieldBackward;      // integrated (final) O_N = 1 - prod_i (1 - a_i@min(t_i,t*)) (read from fwd output)
    float opaFieldGradient;      // dL/d(pred_opacity_field)
    float tStar;                 // detached median depth = qDepth[FRONT_NQ-1] (0.5 crossing) for O_N re-eval
    float crossingTStar;         // EXACT 0.5-crossing t* (ch FRONT_NQ+1) for the IFT differentiable-crossing bwd
    float crossingGrad;          // per-ray IFT scale g = -(dL/dt*)/Odot* (formed in initializeBackwardRay)
#if GOFISH_COME_CONFIDENCE
    float sumConfGradient;       // dL/d(pred_confidence composite)
#endif
    tcnn::vec3 normalBackward;   // integrated (final) raw normal = sum_i w_i n_i
    tcnn::vec3 normalGradient;   // dL/d(pred raw normal)
#endif
    tcnn::vec<FeatN> featuresBackward;
    tcnn::vec<FeatN> featuresGradient;
};

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeBackwardRay(const threedgut::RenderParameters& params,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const float* __restrict__ worldHitDistancePtr,
                                                        const float* __restrict__ worldHitDistanceGradientPtr,
                                                        const float* __restrict__ worldDepthVarPtr,
                                                        const float* __restrict__ worldDepthVarGradientPtr,
                                                        const float* __restrict__ worldExtentPtr,
                                                        const float* __restrict__ worldExtentGradientPtr,
                                                        const float* __restrict__ worldOpacityFieldPtr,
                                                        const float* __restrict__ worldOpacityFieldGradientPtr,
                                                        const float* __restrict__ worldQuantDepthPtr,
                                                        const float* __restrict__ worldQuantDepthGradientPtr,
                                                        const float* __restrict__ worldNormalPtr,
                                                        const float* __restrict__ worldNormalGradientPtr,
                                                        const TFeatureDensityElem* __restrict__ featuresDensityPtr,
                                                        const float* __restrict__ featuresDensityGradientPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform,
                                                        const float* __restrict__ worldInvDPtr = nullptr,
                                                        const float* __restrict__ worldInvDGradientPtr = nullptr,
                                                        const float* __restrict__ worldInvD2Ptr = nullptr,
                                                        const float* __restrict__ worldInvD2GradientPtr = nullptr
                                                        ) {

    // NB : no backpropagation through the forward ray initialization / finalization
    RayPayloadT ray = initializeRay<RayPayloadT>(params,
                                                 sensorRayOriginPtr,
                                                 sensorRayDirectionPtr,
                                                 sensorToWorldTransform);

    if (ray.isAlive()) {
        constexpr uint32_t stride = RayPayloadT::FeatDim + 1;
        const uint32_t base       = ray.idx * stride;
        // Forward features: fp16 when FEATURE_OUTPUT_HALF=1, fp32 otherwise.
        // Gradient buffer: always fp32 — keeps backward numerically stable regardless of forward dtype.
#if FEATURE_OUTPUT_HALF
#pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = __half2float(featuresDensityPtr[base + i]);
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - __half2float(featuresDensityPtr[base + RayPayloadT::FeatDim]);
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#else
#pragma unroll
        for (int i = 0; i < RayPayloadT::FeatDim; ++i) {
            ray.featuresBackward[i] = featuresDensityPtr[base + i];
            ray.featuresGradient[i] = featuresDensityGradientPtr[base + i];
        }
        ray.transmittanceBackward = 1.f - featuresDensityPtr[base + RayPayloadT::FeatDim];
        ray.transmittanceGradient = -1.f * featuresDensityGradientPtr[base + RayPayloadT::FeatDim];
#endif
        ray.hitTBackward = worldHitDistancePtr[ray.idx];
        ray.hitTGradient = worldHitDistanceGradientPtr[ray.idx];
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
        ray.sumWD2Backward = (worldDepthVarPtr != nullptr) ? worldDepthVarPtr[ray.idx] : 0.0f;
        ray.sumWD2Gradient = (worldDepthVarGradientPtr != nullptr) ? worldDepthVarGradientPtr[ray.idx] : 0.0f;
        ray.sumExtBackward = (worldExtentPtr != nullptr) ? worldExtentPtr[ray.idx] : 0.0f;
        ray.sumExtGradient = (worldExtentGradientPtr != nullptr) ? worldExtentGradientPtr[ray.idx] : 0.0f;
        ray.sumWinvDBackward = (worldInvDPtr != nullptr) ? worldInvDPtr[ray.idx] : 0.0f;
        ray.sumWinvDGradient = (worldInvDGradientPtr != nullptr) ? worldInvDGradientPtr[ray.idx] : 0.0f;
        ray.sumWinvD2Backward = (worldInvD2Ptr != nullptr) ? worldInvD2Ptr[ray.idx] : 0.0f;
        ray.sumWinvD2Gradient = (worldInvD2GradientPtr != nullptr) ? worldInvD2GradientPtr[ray.idx] : 0.0f;
        ray.opaFieldBackward = (worldOpacityFieldPtr != nullptr) ? worldOpacityFieldPtr[ray.idx] : 0.0f;
        ray.opaFieldGradient = (worldOpacityFieldGradientPtr != nullptr) ? worldOpacityFieldGradientPtr[ray.idx] : 0.0f;
        ray.tStar = (worldQuantDepthPtr != nullptr) ? worldQuantDepthPtr[ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ - 1)] : 0.0f;
        // ---- IFT differentiable-crossing backward inputs (exact t*, Odot*, and dL/dt*) ----
        ray.crossingTStar = (worldQuantDepthPtr != nullptr) ? worldQuantDepthPtr[ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 1)] : 0.0f;
        {
            const float odot = (worldQuantDepthPtr != nullptr) ? worldQuantDepthPtr[ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 2)] : 0.0f;
            const float dLdt = (worldQuantDepthGradientPtr != nullptr) ? worldQuantDepthGradientPtr[ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 1)] : 0.0f;
            ray.crossingGrad = (fabsf(odot) > 1e-8f && ray.crossingTStar > 0.0f) ? (-dLdt / odot) : 0.0f;
        }
#if GOFISH_COME_CONFIDENCE
        ray.sumConfGradient = (worldQuantDepthGradientPtr != nullptr)
            ? worldQuantDepthGradientPtr[ray.idx * GOFISH_QUANT_STRIDE + (FRONT_NQ + 3)] : 0.0f;
#endif
        ray.normalBackward = (worldNormalPtr != nullptr)
            ? tcnn::vec3(worldNormalPtr[ray.idx*3+0], worldNormalPtr[ray.idx*3+1], worldNormalPtr[ray.idx*3+2])
            : tcnn::vec3(0.0f);
        ray.normalGradient = (worldNormalGradientPtr != nullptr)
            ? tcnn::vec3(worldNormalGradientPtr[ray.idx*3+0], worldNormalGradientPtr[ray.idx*3+1], worldNormalGradientPtr[ray.idx*3+2])
            : tcnn::vec3(0.0f);
#endif
    }

    return ray;
}
