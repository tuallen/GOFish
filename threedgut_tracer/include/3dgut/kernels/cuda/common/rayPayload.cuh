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

#include <3dgut/kernels/cuda/common/cudaMath.cuh>
#include <3dgut/kernels/cuda/common/random.cuh>
#include <3dgut/renderer/renderParameters.h>

// ---- Front-surface / quantile depth readout ----
// Number of absolute cumulative-opacity thresholds at which we record the
// front-to-back crossing depth. Output buffer packs FRONT_NQ quantile depths
// followed by 1 max-contribution (tau_max) depth => (FRONT_NQ + 1) channels.
// Thresholds are defined in gutKBufferRenderer.cuh (processHitParticle).
#ifndef FRONT_NQ
#define FRONT_NQ 7
#endif
#ifndef GOFISH_COME_CONFIDENCE
#define GOFISH_COME_CONFIDENCE 0
#endif
// Quant buffer channels: FRONT_NQ crossings, tau_max, exact t*, rate, [CoMe confidence composite].
#define GOFISH_QUANT_STRIDE (FRONT_NQ + 3 + GOFISH_COME_CONFIDENCE)

template <int FeatN>
struct RayPayload {
    static constexpr uint32_t FeatDim = FeatN;

    threedgut::TTimestamp timestamp;
    tcnn::vec3 origin;
    tcnn::vec3 direction;
    tcnn::vec2 tMinMax;
    float hitT;
    float transmittance;
    enum {
        Default = 0,
        Valid   = 1 << 0,
        Alive   = 1 << 2,
        // BackHit             = 1 << 3,
        // BackHitProxySurface = 1 << 4,
        // FrontHit            = 1 << 5,
    };
    uint32_t flags;
    uint32_t idx;
    tcnn::vec<FeatN> features;
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    tcnn::vec3 normal;
    float sumW;
    float sumWD2;
    float sumExt;             // differentiable along-ray extent: sum_i w_i * sigma_ray_i (sigma_ray = |gscl*rayDirR|)
    float distortionDepthScale; // pinhole: camera-ray dz; fisheye: 1 (radial)
    float sumWinvD;           // differentiable Sum w_i*(1/d_i) numerator (SOF inverse-depth-mapped distortion)
    float sumWinvD2;          // differentiable Sum w_i*(1/d_i^2) numerator (SOF inverse-depth-mapped distortion)
    float opaFieldT;          // opacity-field product accumulator prod_i (1 - a_i@min(t_i,t*)); O_N = 1 - opaFieldT
    float qDepth[FRONT_NQ];   // front-to-back crossing depth per opacity threshold (0 = not crossed)
    float tauMaxW;            // largest per-hit weight seen (max-contribution)
    float tauMaxD;            // depth of the largest-weight hit (tau_max readout)
    float exactDepth;         // SOF ANALYTIC continuous 0.5-opacity-crossing depth (0 = not crossed)
    float exactDepthRate;     // Odot* = dO/dt at the 0.5-crossing (local opacity rate) for IFT crossing backward
#if GOFISH_COME_CONFIDENCE
    float sumConf;            // CoMe confidence composite sum_i w_i conf_i (weights detached in the VJP)
#endif
#endif

#if GAUSSIAN_ENABLE_HIT_COUNT
    uint32_t hitN;
#endif

    __device__ __forceinline__ bool isAlive() const {
        return flags & Alive;
    }

    __device__ __forceinline__ void kill() {
        flags &= ~Alive;
    }

    __device__ __forceinline__ bool isValid() const {
        return flags & Valid;
    }

    // __device__ __forceinline__ bool isFrontHit() const {
    //     return flags & FrontHit;
    // }

    // __device__ __forceinline__ void hitFront() {
    //     flags |= FrontHit;
    // }

    __device__ __forceinline__ void countHit(uint32_t count = 1) {
#if GAUSSIAN_ENABLE_HIT_COUNT
        hitN += count;
#endif
    }
};

template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRay(const threedgut::RenderParameters& params,
                                                const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                const tcnn::mat4x3& sensorToWorldTransform) {
    const uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;
    if ((x >= params.resolution.x) || (y >= params.resolution.y)) {
        return ray;
    }
    ray.idx           = x + params.resolution.x * y;
    ray.hitT          = 0.0f;
    ray.transmittance = 1.0f;
    ray.features      = tcnn::vec<RayPayloadT::FeatDim>::zero();
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    ray.normal = tcnn::vec3::zero();
    ray.sumW = 0.0f;
    ray.sumWD2 = 0.0f;
    ray.sumExt = 0.0f;
    ray.distortionDepthScale =
        (params.sensorModel.modelType == threedgut::TSensorModel::OpenCVPinholeModel)
        ? sensorRayDirectionPtr[ray.idx].z
        : 1.0f;
    ray.sumWinvD = 0.0f;
    ray.sumWinvD2 = 0.0f;
    ray.opaFieldT = 1.0f;
#pragma unroll
    for (int qi = 0; qi < FRONT_NQ; ++qi) ray.qDepth[qi] = 0.0f;
    ray.tauMaxW = 0.0f;
    ray.tauMaxD = 0.0f;
    ray.exactDepth = 0.0f;
    ray.exactDepthRate = 0.0f;
#if GOFISH_COME_CONFIDENCE
    ray.sumConf = 0.0f;
#endif
#endif

    ray.origin    = sensorToWorldTransform * tcnn::vec4(sensorRayOriginPtr[ray.idx], 1.0f);
    ray.direction = tcnn::mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx];

    ray.tMinMax   = params.objectAABB.ray_intersect(ray.origin, ray.direction);
    ray.tMinMax.x = fmaxf(ray.tMinMax.x, 0.0f);

    if (ray.tMinMax.y > ray.tMinMax.x) {
        ray.flags |= RayPayloadT::Valid | RayPayloadT::Alive;
    }

#if GAUSSIAN_ENABLE_HIT_COUNT
    ray.hitN = 0;
#endif

    return ray;
}

// Output element type for the feature+density buffer.
// FEATURE_OUTPUT_HALF=1: write __half (fp16) to halve memory bandwidth.
// FEATURE_OUTPUT_HALF=0: write float (fp32, default).
#ifndef FEATURE_OUTPUT_HALF
#define FEATURE_OUTPUT_HALF 0
#endif

#if FEATURE_OUTPUT_HALF
#include <cuda_fp16.h>
using TFeatureDensityElem = __half;
#else
using TFeatureDensityElem = float;
#endif

// Initialize ray based on given pixel coordinates (load-balanced mode)
template <typename RayPayloadT>
__device__ __inline__ RayPayloadT initializeRayPerPixel(const threedgut::RenderParameters& params,
                                                        const tcnn::uvec2& pixel,
                                                        const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                                        const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                                                        const tcnn::mat4x3& sensorToWorldTransform) {
    RayPayloadT ray;
    ray.flags = RayPayloadT::Default;

    if ((pixel.x >= params.resolution.x) || (pixel.y >= params.resolution.y)) {
        return ray;
    }

    ray.idx           = pixel.x + params.resolution.x * pixel.y;
    ray.hitT          = 0.0f;
    ray.transmittance = 1.0f;
    ray.features      = tcnn::vec<RayPayloadT::FeatDim>::zero();
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    ray.normal = tcnn::vec3::zero();
    ray.sumW = 0.0f;
    ray.sumWD2 = 0.0f;
    ray.sumExt = 0.0f;
    ray.distortionDepthScale =
        (params.sensorModel.modelType == threedgut::TSensorModel::OpenCVPinholeModel)
        ? sensorRayDirectionPtr[ray.idx].z
        : 1.0f;
    ray.sumWinvD = 0.0f;
    ray.sumWinvD2 = 0.0f;
    ray.opaFieldT = 1.0f;
#pragma unroll
    for (int qi = 0; qi < FRONT_NQ; ++qi) ray.qDepth[qi] = 0.0f;
    ray.tauMaxW = 0.0f;
    ray.tauMaxD = 0.0f;
    ray.exactDepth = 0.0f;
    ray.exactDepthRate = 0.0f;
#if GOFISH_COME_CONFIDENCE
    ray.sumConf = 0.0f;
#endif
#endif

    ray.origin    = sensorToWorldTransform * tcnn::vec4(sensorRayOriginPtr[ray.idx], 1.0f);
    ray.direction = tcnn::mat3(sensorToWorldTransform) * sensorRayDirectionPtr[ray.idx];

    ray.tMinMax   = params.objectAABB.ray_intersect(ray.origin, ray.direction);
    ray.tMinMax.x = fmaxf(ray.tMinMax.x, 0.0f);

    if (ray.tMinMax.y > ray.tMinMax.x) {
        ray.flags |= RayPayloadT::Valid | RayPayloadT::Alive;
    }

#if GAUSSIAN_ENABLE_HIT_COUNT
    ray.hitN = 0;
#endif

    return ray;
}

template <typename TRayPayload>
__device__ __inline__ void finalizeRay(const TRayPayload& ray,
                                       const threedgut::RenderParameters& params,
                                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                                       float* __restrict__ worldCountPtr,
                                       float* __restrict__ worldHitDistancePtr,
                                       TFeatureDensityElem* __restrict__ featureDensityPtr,
                                       const tcnn::mat4x3& sensorToWorldTransform,
                                       float* __restrict__ worldNormalPtr = nullptr,
                                       float* __restrict__ worldDepthVarPtr = nullptr,
                                       float* __restrict__ worldQuantDepthPtr = nullptr,
                                       float* __restrict__ worldExtentPtr = nullptr,
                                       float* __restrict__ worldOpacityFieldPtr = nullptr,
                                       float* __restrict__ worldInvDPtr = nullptr,
                                       float* __restrict__ worldInvD2Ptr = nullptr) {
    if (!ray.isValid()) {
        return;
    }

    static_assert(RAY_FEATURE_DIM == TRayPayload::FeatDim, "RAY_FEATURE_DIM must equal TRayPayload::FeatDim");
    const uint32_t base = ray.idx * (RAY_FEATURE_DIM + 1);
#if FEATURE_OUTPUT_HALF
#pragma unroll
    for (int i = 0; i < TRayPayload::FeatDim; ++i) {
        featureDensityPtr[base + i] = __float2half(ray.features[i]);
    }
    featureDensityPtr[base + RAY_FEATURE_DIM] = __float2half(1.0f - ray.transmittance);
#else
#pragma unroll
    for (int i = 0; i < TRayPayload::FeatDim; ++i) {
        featureDensityPtr[base + i] = ray.features[i];
    }
    featureDensityPtr[base + RAY_FEATURE_DIM] = (1.0f - ray.transmittance);
#endif

    worldHitDistancePtr[ray.idx] = ray.hitT;
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    if (worldNormalPtr) {
        // Emit RAW alpha-weighted normal sum_i w_i n_i (linear accumulator with exact CUDA backward).
        // Normalized in Python, as in SOF.
        worldNormalPtr[ray.idx*3+0]=ray.normal.x; worldNormalPtr[ray.idx*3+1]=ray.normal.y; worldNormalPtr[ray.idx*3+2]=ray.normal.z;
    }
    if (worldDepthVarPtr) {
        // Differentiable distortion: emit RAW 2nd-moment numerator sum_i w_i d_i^2 (linear accumulator with an
        // exact CUDA backward). Variance E[d^2]-E[d]^2 is reconstructed differentiably in Python from this +
        // pred_dist (sum w d) + pred_opacity (sum w).
        worldDepthVarPtr[ray.idx]=ray.sumWD2;
    }
    if (worldExtentPtr) {
        // Differentiable along-ray extent: emit RAW sum_i w_i * sigma_ray_i (linear accumulator with exact
        // CUDA backward). sigma_ray = |gscl (elementwise*) rayDirR| = along-ray Gaussian std. Python loss =
        // mean(pred_extent[valid]) shrinks fat Gaussians (SOF T&T extent regularizer). Fisheye-native.
        worldExtentPtr[ray.idx]=ray.sumExt;
    }
    if (worldOpacityFieldPtr) {
        // Differentiable opacity-field: O_N = 1 - prod_i (1 - a_i@min(t_i,t*)). Python loss = (O_N-0.5)^2.
        worldOpacityFieldPtr[ray.idx]=1.0f - ray.opaFieldT;
    }
    if (worldInvDPtr) {
        // Differentiable SOF inverse-depth distortion numerator: RAW sum_i w_i*(1/d_i). Python reconstructs
        // Var[1/t]=E[1/t^2]-E[1/t]^2, mapped to SOF distortion C2^2*Var[1/t], C2=far*near/(far-near).
        worldInvDPtr[ray.idx]=ray.sumWinvD;
    }
    if (worldInvD2Ptr) {
        worldInvD2Ptr[ray.idx]=ray.sumWinvD2;
    }
    if (worldQuantDepthPtr) {
        // Buffer packs (FRONT_NQ + 2) channels per ray: FRONT_NQ quantile-crossing depths,
        // then tau_max depth, then the SOF ANALYTIC continuous 0.5-crossing exact-depth.
        const uint32_t qb = ray.idx * GOFISH_QUANT_STRIDE;
#pragma unroll
        for (int qi = 0; qi < FRONT_NQ; ++qi) worldQuantDepthPtr[qb + qi] = ray.qDepth[qi];
        worldQuantDepthPtr[qb + FRONT_NQ]     = ray.tauMaxD;    // channel FRONT_NQ   = max-contribution depth
        worldQuantDepthPtr[qb + FRONT_NQ + 1] = ray.exactDepth; // channel FRONT_NQ+1 = SOF analytic exact-depth
        worldQuantDepthPtr[qb + FRONT_NQ + 2] = ray.exactDepthRate; // channel FRONT_NQ+2 = Odot* (crossing rate) for IFT bwd
#if GOFISH_COME_CONFIDENCE
        worldQuantDepthPtr[qb + FRONT_NQ + 3] = ray.sumConf;        // channel FRONT_NQ+3 = CoMe confidence composite
#endif
    }
#endif

#if GAUSSIAN_ENABLE_HIT_COUNT
    worldCountPtr[ray.idx] = (float)ray.hitN;
#endif
}
