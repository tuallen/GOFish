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

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/kernels/cuda/common/sensorConstants.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

#ifndef GOFISH_EXACT_SCREEN_ABSGRAD
#define GOFISH_EXACT_SCREEN_ABSGRAD 0
#endif
#if GOFISH_COME_CONFIDENCE && (!GAUSSIAN_HIERARCHICAL_SORT || !GAUSSIAN_PARTICLE_ENABLE_NORMAL)
#error "CoMe confidence is implemented on the hierarchical processHitParticle path only"
#endif

// Inference-only build (render.py --renderer exact-forward and realtime): the
// forward pass skips the training-only geometric channels (normals, distortion
// moments, extent, quantile/exact-crossing depth).  Colour, opacity and hit
// distance are computed by exactly the same code as the reference renderer.
#ifndef GOFISH_INFERENCE_ONLY
#define GOFISH_INFERENCE_ONLY 0
#endif

// Scheduled SOF hierarchy (training builds, see sofMid8Schedule.inl): the SOF
// cascade runs once per forward pass instead of once per pixel pass.
#ifndef GOFISH_SOF_SCHEDULE
#define GOFISH_SOF_SCHEDULE 0
#endif
#if GOFISH_SOF_SCHEDULE && (GOFISH_INFERENCE_ONLY || FINE_GRAINED_LOAD_BALANCING || !GAUSSIAN_HIERARCHICAL_SORT || !GAUSSIAN_SOF_MID8_HIERARCHY)
#error "GOFISH_SOF_SCHEDULE requires the SOF MID8 hierarchy on 16x16 raster blocks in a training build"
#endif


#if GOFISH_EXACT_SCREEN_ABSGRAD
// Workspace row: B_ndc[2,3] in slots 0..5, |u|/|v| accumulators in 6..7.
__forceinline__ __device__ float2 gofishExactScreenAbsGrad(
    const float* workspace, int particle, const float3& gradient) {
    const float* row = workspace + size_t(particle) * 8u;
    const float u = row[0] * gradient.x + row[1] * gradient.y + row[2] * gradient.z;
    const float v = row[3] * gradient.x + row[4] * gradient.y + row[5] * gradient.z;
    return make_float2(fabsf(u), fabsf(v));
}
#endif

// HitParticle stores per-ray hit state inside the k-buffer. The
// `canonicalIntersection` is only consumed by the NHT feature path; for the
// SH path (`PerRayParticleFeatures=false`) we drop it from the struct so the
// k-buffer pays no storage cost in registers / shared memory per hit.
template <bool HasCanonical>
struct HitParticleT;

// Base / SH case: common per-hit state only.
template <>
struct HitParticleT<false> {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
    float3 normal                      = make_float3(0.f, 0.f, 0.f);
#endif
};

// NHT case extends with the per-ray canonical intersection point used by the
// feature evaluation. Plain (non-virtual) inheritance: layout is base fields
// then the extra `float3`.
template <>
struct HitParticleT<true> : HitParticleT<false> {
    float3 canonicalIntersection = make_float3(0.f, 0.f, 0.f);
};

// Helper returning a writable `float3&` slot usable as the `densityHit`
// canonical-intersection out-parameter. Routes to the struct field when
// present, otherwise to a caller-provided stack scratch (which the compiler
// elides when unused downstream).
template <bool HasCanonical>
__forceinline__ __device__ float3& canonicalIntersectionSlot(HitParticleT<HasCanonical>& hit, float3& scratch);

template <>
__forceinline__ __device__ float3& canonicalIntersectionSlot<true>(HitParticleT<true>& hit, float3& /*scratch*/) {
    return hit.canonicalIntersection;
}

template <>
__forceinline__ __device__ float3& canonicalIntersectionSlot<false>(HitParticleT<false>& /*hit*/, float3& scratch) {
    return scratch;
}

#if GOFISH_SOF_SCHEDULE
// Per-forward SOF emission schedule, set by GUTRenderer::renderForward: the
// emitted particle ids and counts of every MID8 group, the tile-row batch
// [gSofRowBegin, gSofRowEnd) and its first tile-intersection offset.
static __device__ uint32_t* gSofSchedIds = nullptr;
static __device__ uint32_t* gSofSchedCounts = nullptr;
static __device__ uint32_t gSofSchedBase = 0u;
static __device__ uint32_t gSofRowBegin = 0u;
static __device__ uint32_t gSofRowEnd = 0xffffffffu;
#endif

template <int K, bool HasCanonical>
struct HitParticleKBufferT {
    using HitParticle = HitParticleT<HasCanonical>;

    __device__ HitParticleKBufferT() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <bool HasCanonical>
struct HitParticleKBufferT<0, HasCanonical> {
    using HitParticle = HitParticleT<HasCanonical>;
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const { return HitParticle(); }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { return hitParticle; }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec;

    using TRayPayload         = RayPayload<Particles::RayFeatureDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::RayFeatureDim>;

    // Storage-optimized hit types: the per-ray canonical intersection is kept
    // only when the feature model needs it (NHT / PerRayParticleFeatures).
    using HitParticle        = HitParticleT<Params::PerRayParticleFeatures>;
    using HitParticleKBuffer = HitParticleKBufferT<Params::KHitBufferSize, Params::PerRayParticleFeatures>;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedRawParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityRawParameters densityParameters;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        Particles& particles,
        const TFeaturesVec* __restrict__ particleFeatures,
        TFeaturesVec* __restrict__ particleFeaturesGradient,
        float* __restrict__ particlesPositionAbsGradPtr = nullptr) {

        if constexpr (Backward) {
            if constexpr (Params::PerRayParticleFeatures) {
                // NHT path: Slang autodiff backward (features re-eval + density-to-buffer).
                float hitAlphaGrad               = 0.f;
                float3 canonicalIntersectionGrad = make_float3(0.f, 0.f, 0.f);
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                              hitParticle.canonicalIntersection,
                                                              canonicalIntersectionGrad,
                                                              hitParticle.alpha,
                                                              hitAlphaGrad,
                                                              hitParticle.idx,
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction, hitParticle.canonicalIntersection),
                                                              ray.featuresBackward,
                                                              ray.featuresGradient);
                particles.densityProcessHitBwdToBufferInterval<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient,
                                                          canonicalIntersectionGrad,
                                                          ray.tMinMax.x, ray.tMinMax.y);
                ray.transmittance *= (1.0 - hitParticle.alpha);
            } else {
                // ---- SH SORTED backward ----
                // Use the SAME hand-written processHitBwd as the UNSORTED path (evalBackwardNoKBuffer),
                // so ALL geometric-channel gradients (distortion sumWD2/invd/invd2, dnc normal, esmooth,
                // opacity-field, extent) flow under sorting AND the abs position grad is available to
                // snapshot. Accumulate PER-HIT (processHitBwdUpdate*<false> = no warp reduce): the
                // k-buffer gives each thread a DIFFERENT particle, so the unsorted same-particle-per-warp
                // warp reduction would be incorrect. processHitBwd walks ray.transmittance + ray.hitT by
                // reference (identical scheme to the unsorted path) -> NO manual transmittance update here.
                DensityRawParameters dgrad;
                dgrad.density    = 0.f;
                dgrad.position   = make_float3(0.f);
                dgrad.quaternion = make_float4(0.f);
                dgrad.scale      = make_float3(0.f);
                dgrad.intrinsicDensity = 0.0f;
                dgrad.factor[0] = make_float3(0.0f);
                dgrad.factor[1] = make_float3(0.0f);
                dgrad.factor[2] = make_float3(0.0f);
                dgrad.inverseFactor[0] = make_float3(0.0f);
                dgrad.inverseFactor[1] = make_float3(0.0f);
                dgrad.inverseFactor[2] = make_float3(0.0f);
                dgrad.fullShapePadding = make_float2(0.0f, 0.0f);
                TFeaturesVec fgrad = TFeaturesVec::zero();
#if GOFISH_COME_CONFIDENCE
                const float comeTransmittanceBefore = ray.transmittance;
#endif
                particles.processHitBwd<false>(
                    ray.origin, ray.direction, hitParticle.idx,
                    particles.fetchDensityRawParameters(hitParticle.idx), &dgrad,
                    tcnn::max(particleFeatures[hitParticle.idx], 0.f), &fgrad,
                    ray.transmittance, ray.transmittanceBackward, ray.transmittanceGradient,
                    ray.features, ray.featuresBackward, ray.featuresGradient,
                    ray.hitT, ray.hitTBackward, ray.hitTGradient,
                    &ray.sumWD2, ray.sumWD2Backward, ray.sumWD2Gradient,
                    &ray.normal, ray.normalBackward, ray.normalGradient,
                    &ray.sumExt, ray.sumExtBackward, ray.sumExtGradient,
                    ray.opaFieldBackward, ray.opaFieldGradient, ray.tStar,
                    &ray.sumWinvD, ray.sumWinvDBackward, ray.sumWinvDGradient,
                    &ray.sumWinvD2, ray.sumWinvD2Backward, ray.sumWinvD2Gradient,
                    ray.crossingGrad, ray.crossingTStar,
                    ray.tMinMax.x, ray.tMinMax.y,
                    ray.distortionDepthScale);
                // Detached densification statistic; it does not alter the optimizer gradient below.
                if (particlesPositionAbsGradPtr != nullptr) {
#if GOFISH_EXACT_SCREEN_ABSGRAD
                    const float2 screen = gofishExactScreenAbsGrad(
                        particlesPositionAbsGradPtr, hitParticle.idx, dgrad.position);
                    atomicAdd(&particlesPositionAbsGradPtr[hitParticle.idx * 8 + 6], screen.x);
                    atomicAdd(&particlesPositionAbsGradPtr[hitParticle.idx * 8 + 7], screen.y);
#else
                    atomicAdd(&particlesPositionAbsGradPtr[hitParticle.idx * 3 + 0], fabsf(dgrad.position.x));
                    atomicAdd(&particlesPositionAbsGradPtr[hitParticle.idx * 3 + 1], fabsf(dgrad.position.y));
                    atomicAdd(&particlesPositionAbsGradPtr[hitParticle.idx * 3 + 2], fabsf(dgrad.position.z));
#endif
                }
                particles.processHitBwdUpdateFeaturesGradient<false>(hitParticle.idx, fgrad, particleFeaturesGradient, 0);
                particles.processHitBwdUpdateDensityGradient<false>(hitParticle.idx, dgrad, 0);
#if GOFISH_COME_CONFIDENCE
                // CoMe confidence VJP with detached weights: d(sum_i w_i c_i)/dc_i = w_i, where w_i is this
                // backward walk's own alpha*T (T_before - T_after; 0 for a rejected hit). No alpha gradient.
                particles.processHitBwdUpdateComeConfidenceGradient(
                    hitParticle.idx, (comeTransmittanceBefore - ray.transmittance) * ray.sumConfGradient);
#endif
            }

        } else {
            // Transmittance BEFORE this hit (densityIntegrateHit multiplies it by (1-alpha)).
            // Needed for the SOF analytic exact-depth 0.5-crossing solve below.
            const float transmittanceBeforeHit = ray.transmittance;
#if GOFISH_INFERENCE_ONLY
            // The operations of particleDensityIntegrateHit (front-to-back).
            // In the reference build the depth product weight * t_i is shared
            // with the training-only second-moment channel and is rounded before
            // it is accumulated; round it here too so the hit distance is
            // bit-identical without that channel.
            const float hitWeight = hitParticle.alpha * ray.transmittance;
            ray.hitT = __fadd_rn(ray.hitT, __fmul_rn(hitParticle.hitT, hitWeight));
            ray.transmittance = ray.transmittance * (1.0f - hitParticle.alpha);
            (void)transmittanceBeforeHit;
#else
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT);
#endif
#if GOFISH_COME_CONFIDENCE
            // CoMe confidence composite: conf_i = exp(gamma_i) rides in the full-shape ABI padding word.
            ray.sumConf += hitWeight * particles.fetchDensityRawParameters(hitParticle.idx).fullShapePadding.x;
#endif
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL && !GOFISH_INFERENCE_ONLY
            ray.normal.x += hitWeight * hitParticle.normal.x;
            ray.normal.y += hitWeight * hitParticle.normal.y;
            ray.normal.z += hitWeight * hitParticle.normal.z;
            ray.sumW += hitWeight;
            ray.sumWD2 += hitWeight * hitParticle.hitT * hitParticle.hitT;
            // Differentiable SOF inverse-depth distortion numerators. SOF
            // never evaluates this coordinate below its 0.2 near plane. GOFish
            // renders there, so only the auxiliary loss coordinate is clipped. Ray parameterization is
            // camera-native: rays are unit length. For pinhole,
            // view-space z = radial hitT * camera-ray dz; calibrated fisheye
            // uses radial hitT directly.
            const float distortionDepth =
                hitParticle.hitT * ray.distortionDepthScale;
            const threedgut::SofDistortionCoordinate distortionCoordinate =
                threedgut::sofDistortionCoordinate(
                    distortionDepth,
                    hitParticle.hitT * hitParticle.hitT
                        * ray.distortionDepthScale * ray.distortionDepthScale);
            ray.sumWinvD += hitWeight * distortionCoordinate.inverseDepth;
            ray.sumWinvD2 +=
                hitWeight * distortionCoordinate.inverseDepthSquared;
            // Differentiable along-ray EXTENT uses only the stored scene's
            // intrinsic physical-Mip shape.  The current-view footprint/Q factor is
            // deliberately excluded; processHitBwd reconstructs this exact
            // expression from the same packed quaternion and effective scale.
            ray.sumExt += hitWeight * threedgut::particleAlongRaySigmaIntrinsicPhysicalMip(
                particles.fetchDensityRawParameters(hitParticle.idx),
                reinterpret_cast<const float3&>(ray.direction));
            // Front-surface / quantile depth: hits arrive front-to-back, so record
            // the depth at which cumulative opacity (1 - transmittance) FIRST reaches
            // each threshold. densityIntegrateHit has already multiplied ray.transmittance
            // by (1-alpha), so (1 - ray.transmittance) is the opacity accumulated up to
            // and including this hit. Robust to lone near-floaters: a single low-opacity
            // hit won't reach the smaller thresholds. Also track max-contribution (tau_max).
            {
                const float FRONT_QT[FRONT_NQ] = {0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.40f, 0.50f};
                const float oAcc = 1.0f - ray.transmittance;
#pragma unroll
                for (int qi = 0; qi < FRONT_NQ; ++qi) {
                    if (ray.qDepth[qi] == 0.0f && oAcc >= FRONT_QT[qi]) ray.qDepth[qi] = hitParticle.hitT;
                }
                if (hitWeight > ray.tauMaxW) { ray.tauMaxW = hitWeight; ray.tauMaxD = hitParticle.hitT; }
            }
            // SOF ANALYTIC continuous exact-depth (as in SOF's hierarchical_render.cuh):
            // at the Gaussian that carries cumulative transmittance across 0.5 (before>0.5, after<=0.5),
            // solve the 0.5-crossing IN CLOSED FORM inside this crossing Gaussian (NOT a discrete knot,
            // NOT interpolation between stored hits). Set once at the first crossing. Falls back to the
            // hit knot ONLY when the solve is numerically degenerate (helper returns <0).
            if (ray.exactDepth == 0.0f && transmittanceBeforeHit > 0.5f && ray.transmittance <= 0.5f) {
                const threedgut::FiniteIntervalCrossingResult crossing =
                    threedgut::particleExactDepthCrossingRaw(
                        particles.fetchDensityRawParameters(hitParticle.idx),
                        reinterpret_cast<const float3&>(ray.origin),
                        reinterpret_cast<const float3&>(ray.direction),
                        transmittanceBeforeHit, ray.tMinMax.x, ray.tMinMax.y);
                ray.exactDepth = crossing.valid ? crossing.depth : hitParticle.hitT;
                ray.exactDepthRate = crossing.valid ? crossing.rate : 0.0f;
            }
            // SOF's opacity target is the analytic rho*=exactDepth, not the
            // responsible-hit qDepth knot; the intrinsic field is replayed at
            // rho* in a separate kernel after this pass.
#endif

            // `if constexpr` branches so the SH specialization of Hit (which
            // has no `canonicalIntersection` member) is not instantiated with
            // a missing field reference.
            if constexpr (Params::PerRayParticleFeatures) {
                particles.featureIntegrateFwd(hitWeight,
                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction, hitParticle.canonicalIntersection),
                                              ray.features);
            } else {
                particles.featureIntegrateFwd(hitWeight,
                                              tcnn::max(particleFeatures[hitParticle.idx], 0.f),
                                              ray.features);
            }

            if (hitWeight > 0.0f)
                ray.countHit();
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                       TRay& ray,
                                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                       const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                       const float* __restrict__ particlesGlobalDepthPtr,
                                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                       threedgut::MemoryHandles parameters,
                                       tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                       tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                       float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                       float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                       threedgut::MemoryHandles parametersGradient                        = {},
                                       float* __restrict__ particlesPositionAbsGradPtr                     = nullptr) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        tcnn::uvec2 tileParticleRangeIndices         = sortedTileRangeIndicesPtr[tileIdx];
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
        // real-time renderer heavy-tile split: a tile with >= GOFISH_FAST_TILE_SPLIT_MIN entries is processed by
        // GOFISH_FAST_TILE_SPLIT blocks (blockIdx.z), each over one contiguous chunk of its depth-sorted list;
        // the chunks are composited front to back by gofishSplitCombine.
        if constexpr (!Backward) {
            const uint32_t splitLength = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
            if (splitLength >= uint32_t(GOFISH_FAST_TILE_SPLIT_MIN)) {
                const uint32_t chunk = tcnn::div_round_up(splitLength, uint32_t(GOFISH_FAST_TILE_SPLIT));
                const uint32_t begin = min(tileParticleRangeIndices.y, tileParticleRangeIndices.x + blockIdx.z * chunk);
                tileParticleRangeIndices = tcnn::uvec2{begin, min(tileParticleRangeIndices.y, begin + chunk)};
            }
        }
#endif
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

#if GAUSSIAN_HIERARCHICAL_SORT
#if GAUSSIAN_SOF_MID8_HIERARCHY
        // SOF spatial/window semantics. The processHitParticle path serves
        // both forward and backward templates.
        evalSOFMid8Hierarchy(params, ray, particles, tileParticleRangeIndices,
                             tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                             sortedTileParticleIdxPtr, particlesGlobalDepthPtr,
                             particleFeaturesBuffer, particleFeaturesGradientBuffer,
                             particlesPositionAbsGradPtr);
#else
        // SOF hierarchical StopThePop cascade. Streams the CUB-sorted particles through a
        // shared per-tile TAIL staging window into a per-ray register HEAD (exact-hitT, GAUSSIAN_HEAD_SIZE
        // deep), compositing front-to-back via the SAME processHitParticle blend (forward AND backward),
        // so all GOF geometric channels (color/normal/distortion/opacity-field/extent/exact-depth) and the
        // replay backward share one blend; only the traversal order differs.
        evalHierarchical(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                         sortedTileParticleIdxPtr, particlesGlobalDepthPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
#endif
#else
        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                  sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
        } else {
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_SLIM_PREFETCH
            if constexpr (!Backward) {
                evalKBufferSlim(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr,
                                particlesProjectedPositionPtr, particlesProjectedConicOpacityPtr);
            } else
#endif
            evalKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
        }
#endif
    }

#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_SLIM_PREFETCH
    // Real-time renderer (inference forward only): the tile prefetch keeps only what the forward hit test reads
    // (position, inverse factor, density; 56 B instead of 92 B per particle), so a block holds less shared
    // memory and loads less.  The factor B is used only for normals, which the inference forward never
    // computes.  Same arithmetic as evalKBuffer: identical output.
    struct SlimPrefetchedParticleData {
        uint32_t idx;
        float3 position;
        float density;
        float33 inverseFactor;
    };

    template <typename TRay>
    static inline __device__ void evalKBufferSlim(TRay& ray,
                                                  Particles& particles,
                                                  const tcnn::uvec2& tileParticleRangeIndices,
                                                  uint32_t tileNumBlocksToProcess,
                                                  uint32_t tileNumParticlesToProcess,
                                                  const uint32_t tileThreadIdx,
                                                  const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                  const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                  TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
                                                  float* __restrict__ particlesPositionAbsGradPtr,
                                                  const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                                  const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr) {
        using namespace threedgut;
        __shared__ SlimPrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer hitParticleKBuffer;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    const DensityParameters full = particles.fetchDensityParameters(particleIdx);
                    SlimPrefetchedParticleData& slot = prefetchedParticlesData[tileThreadIdx];
                    slot.idx = particleIdx;
                    slot.position = full.position;
                    slot.density = full.density;
                    for (int r = 0; r < 3; ++r) slot.inverseFactor[r] = full.inverseFactor[r];
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const SlimPrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }
                DensityParameters densityParameters;
                densityParameters.position = particleData.position;
                for (int r = 0; r < 3; ++r) densityParameters.inverseFactor[r] = particleData.inverseFactor[r];
                densityParameters.density = particleData.density;

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                float3 canonicalScratch = make_float3(0.f, 0.f, 0.f);
                if (particles.densityHitInterval(ray.origin,
                                         ray.direction,
                                         densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT,
                                         canonicalIntersectionSlot<Params::PerRayParticleFeatures>(hitParticle, canonicalScratch),
                                         ray.tMinMax.x, ray.tMinMax.y
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
                                         , nullptr
#endif
                                         ) &&
                    (hitParticle.hitT >= ray.tMinMax.x) &&
                    (hitParticle.hitT <= ray.tMinMax.y)) {

                    if (hitParticleKBuffer.full()) {
                        processHitParticle(ray,
                                           hitParticleKBuffer.closestHit(hitParticle),
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer,
                                           particlesPositionAbsGradPtr);
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                   hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer,
                                   particlesPositionAbsGradPtr);
            }
        }
    }
#endif

    template <typename TRay>
    static inline __device__ void evalKBuffer(TRay& ray,
                                              Particles& particles,
                                              const tcnn::uvec2& tileParticleRangeIndices,
                                              uint32_t tileNumBlocksToProcess,
                                              uint32_t tileNumParticlesToProcess,
                                              const uint32_t tileThreadIdx,
                                              const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                              const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                              TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
                                              float* __restrict__ particlesPositionAbsGradPtr = nullptr) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer hitParticleKBuffer;

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                HitParticle hitParticle;
                hitParticle.idx = particleData.idx;
                // `canonicalScratch` is only written when the SH specialization
                // of Hit has no canonicalIntersection field; compiler elides it.
                float3 canonicalScratch = make_float3(0.f, 0.f, 0.f);
                if (particles.densityHitInterval(ray.origin,
                                         ray.direction,
                                         particleData.densityParameters,
                                         hitParticle.alpha,
                                         hitParticle.hitT,
                                         canonicalIntersectionSlot<Params::PerRayParticleFeatures>(hitParticle, canonicalScratch),
                                         ray.tMinMax.x, ray.tMinMax.y
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
                                         , ((Backward || !GOFISH_INFERENCE_ONLY) ? reinterpret_cast<tcnn::vec3*>(&hitParticle.normal) : nullptr)
#endif
                                         ) &&
                    (hitParticle.hitT >= ray.tMinMax.x) &&
                    (hitParticle.hitT <= ray.tMinMax.y)) {

                    if (hitParticleKBuffer.full()) {
                        processHitParticle(ray,
                                           hitParticleKBuffer.closestHit(hitParticle),
                                           particles,
                                           particleFeaturesBuffer,
                                           particleFeaturesGradientBuffer,
                                           particlesPositionAbsGradPtr);
                    }
                    hitParticleKBuffer.insert(hitParticle);
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                   hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer,
                                   particlesPositionAbsGradPtr);
            }
        }
    }


#if GAUSSIAN_HIERARCHICAL_SORT
    // Register HEAD window: shallow, per-ray, EXACT-hitT sorted (SOF per_pixel level). Sized by the
    // compile constant GAUSSIAN_HEAD_SIZE independent of the flat KHitBufferSize. With the TAIL disabled
    // this is byte-for-byte the flat k-buffer algorithm at K=GAUSSIAN_HEAD_SIZE => hierarchical-HEAD==flat-K
    // holds by construction.
    using HitParticleHeadBuffer = HitParticleKBufferT<GAUSSIAN_HEAD_SIZE, Params::PerRayParticleFeatures>;

    // Feed one (already-tail-ordered) particle through this ray's register HEAD (exact-hitT reorder),
    // compositing the HEAD front via processHitParticle when the HEAD overflows.
    template <typename TRay>
    static inline __device__ void feedHead(TRay& ray,
                                           HitParticleHeadBuffer& hitParticleHead,
                                           Particles& particles,
                                           uint32_t idx,
                                           const DensityParameters& densityParameters,
                                           const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                           TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
                                           float* __restrict__ particlesPositionAbsGradPtr) {
        HitParticle hitParticle;
        hitParticle.idx        = idx;
        float3 canonicalScratch = make_float3(0.f, 0.f, 0.f);
        if (particles.densityHitInterval(ray.origin,
                                 ray.direction,
                                 densityParameters,
                                 hitParticle.alpha,
                                 hitParticle.hitT,
                                 canonicalIntersectionSlot<Params::PerRayParticleFeatures>(hitParticle, canonicalScratch),
                                         ray.tMinMax.x, ray.tMinMax.y
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
                                 , ((Backward || !GOFISH_INFERENCE_ONLY) ? reinterpret_cast<tcnn::vec3*>(&hitParticle.normal) : nullptr)
#endif
                                 ) &&
            (hitParticle.hitT >= ray.tMinMax.x) &&
            (hitParticle.hitT <= ray.tMinMax.y)) {
            if (hitParticleHead.full()) {
                processHitParticle(ray,
                                   hitParticleHead.closestHit(hitParticle),
                                   particles,
                                   particleFeaturesBuffer,
                                   particleFeaturesGradientBuffer,
                                   particlesPositionAbsGradPtr);
            }
            hitParticleHead.insert(hitParticle);
        }
    }

    template <typename TRay>
    static inline __device__ void evalHierarchical(TRay& ray,
                                                   Particles& particles,
                                                   const tcnn::uvec2& tileParticleRangeIndices,
                                                   uint32_t tileNumBlocksToProcess,
                                                   uint32_t tileNumParticlesToProcess,
                                                   const uint32_t tileThreadIdx,
                                                   const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                   const float* __restrict__ particlesGlobalDepthPtr,
                                                   const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                   TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
                                                   float* __restrict__ particlesPositionAbsGradPtr = nullptr) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleHeadBuffer hitParticleHead;

#if GAUSSIAN_TAIL_SIZE > 0
        // ================= DEEP per-4x4-SUB-TILE selection-window cascade (SOF tail level) =========
        // 16 sub-tiles of 4x4 pixels per 16x16 block. Each sub-tile owns a shared W-slot selection window
        // keyed by its REPRESENTATIVE pixel's (local (1,1)) exact hitT (coherence key; falls back to the
        // particle global CUB depth when the rep ray misses so the key is ALWAYS finite -> no junk-fill).
        // Correctness invariants:
        //  * NO DROP: every streamed particle is pushed to every sub-tile window and emitted EXACTLY once
        //    (steady state emits min(window U new); end-drain empties). Only ORDER is approximate, never
        //    membership.
        //  * EXACT INTEGRATE: emit == feedHead -> processHitParticle (SOF/GOF blend), and the
        //    per-pixel HEAD refines the emit stream to per-pixel EXACT hitT over its H window.
        //  * DETERMINISTIC + UNIFORM __syncthreads (no conditional syncs) -> backward re-runs the identical
        //    cascade -> exact gradient of the realized forward.
        constexpr int W = GAUSSIAN_TAIL_SIZE;
        __shared__ int               s_winId[16][W];
        __shared__ float             s_winKey[16][W];
        __shared__ int               s_winCnt[16];
        __shared__ DensityParameters s_poppedParam[16];
        __shared__ int               s_poppedIdx[16];
        __shared__ int               s_poppedHas[16];

        const uint32_t subTile = (threadIdx.y >> 2) * 4u + (threadIdx.x >> 2); // 4x4 sub-tile id [0,16)
        const uint32_t localRank = (threadIdx.y & 3u) * 4u + (threadIdx.x & 3u); // pixel-in-subtile [0,16)
        const bool isRep       = (localRank == 5u);                             // local (1,1) representative

        if (tileThreadIdx < 16u) {
            s_winCnt[tileThreadIdx] = 0;
        }
        __syncthreads();

        bool streamDone = false;
        for (uint32_t i = 0; (i < tileNumBlocksToProcess) && !streamDone; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            const int jmax = min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess);
            for (int j = 0; j < jmax; j++) {
                const uint32_t idx_j = prefetchedParticlesData[j].idx; // UNIFORM across all threads
                if (idx_j == GUTParameters::InvalidParticleIdx) {       // uniform: end of valid stream
                    streamDone = true;
                    break;
                }

                // --- Rep manages this sub-tile's selection window (serial per sub-tile; 16 reps in parallel).
                if (isRep) {
                    s_poppedHas[subTile] = 0;
                    // Rep-ray exact hitT key (finite fallback = particle CUB global depth if rep ray misses).
                    HitParticle rk;
                    float3 sc = make_float3(0.f, 0.f, 0.f);
                    const bool repHit = particles.densityHitInterval(ray.origin, ray.direction,
                                                             prefetchedParticlesData[j].densityParameters,
                                                             rk.alpha, rk.hitT,
                                                             canonicalIntersectionSlot<Params::PerRayParticleFeatures>(rk, sc),
                                                             ray.tMinMax.x, ray.tMinMax.y
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
                                                             , ((Backward || !GOFISH_INFERENCE_ONLY) ? reinterpret_cast<tcnn::vec3*>(&rk.normal) : nullptr)
#endif
                    );
                    const float key = (repHit && (rk.hitT >= ray.tMinMax.x)
                                   && (rk.hitT <= ray.tMinMax.y))
                                          ? rk.hitT
                                          : particlesGlobalDepthPtr[idx_j];
                    const int cnt = s_winCnt[subTile];
                    if (cnt < W) {
                        // fill phase: buffer, emit nothing
                        s_winKey[subTile][cnt] = key;
                        s_winId[subTile][cnt]  = (int)idx_j;
                        s_winCnt[subTile]      = cnt + 1;
                    } else {
                        // steady state: emit min(window U new) to guarantee bounded-displacement order
                        int mslot = 0;
                        float mk = s_winKey[subTile][0];
                        for (int q = 1; q < W; ++q) {
                            if (s_winKey[subTile][q] < mk) { mk = s_winKey[subTile][q]; mslot = q; }
                        }
                        if (key <= mk) {
                            // new particle is the front -> emit it directly, window unchanged
                            s_poppedIdx[subTile]   = (int)idx_j;
                            s_poppedParam[subTile] = prefetchedParticlesData[j].densityParameters;
                            s_poppedHas[subTile]   = 1;
                        } else {
                            // pop window min, insert new into its slot
                            s_poppedIdx[subTile]   = s_winId[subTile][mslot];
                            s_poppedParam[subTile] = particles.fetchDensityParameters((uint32_t)s_winId[subTile][mslot]);
                            s_poppedHas[subTile]   = 1;
                            s_winKey[subTile][mslot] = key;
                            s_winId[subTile][mslot]  = (int)idx_j;
                        }
                    }
                }
                __syncthreads();
                if (s_poppedHas[subTile] && ray.isAlive()) {
                    feedHead(ray, hitParticleHead, particles, (uint32_t)s_poppedIdx[subTile], s_poppedParam[subTile],
                             particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
                }
                __syncthreads();
            }
        }

        // DRAIN: emit the remaining buffered particles in min-key order (uniform W iterations).
        for (int d = 0; d < W; ++d) {
            if (isRep) {
                s_poppedHas[subTile] = 0;
                const int cnt = s_winCnt[subTile];
                if (cnt > 0) {
                    int mslot = 0;
                    float mk = s_winKey[subTile][0];
                    for (int q = 1; q < cnt; ++q) {
                        if (s_winKey[subTile][q] < mk) { mk = s_winKey[subTile][q]; mslot = q; }
                    }
                    s_poppedIdx[subTile]   = s_winId[subTile][mslot];
                    s_poppedParam[subTile] = particles.fetchDensityParameters((uint32_t)s_winId[subTile][mslot]);
                    s_poppedHas[subTile]   = 1;
                    // compact: move last valid entry into the emitted slot
                    s_winKey[subTile][mslot] = s_winKey[subTile][cnt - 1];
                    s_winId[subTile][mslot]  = s_winId[subTile][cnt - 1];
                    s_winCnt[subTile]        = cnt - 1;
                }
            }
            __syncthreads();
            if (s_poppedHas[subTile] && ray.isAlive()) {
                feedHead(ray, hitParticleHead, particles, (uint32_t)s_poppedIdx[subTile], s_poppedParam[subTile],
                         particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
            }
            __syncthreads();
        }

#else  // GAUSSIAN_TAIL_SIZE == 0  : HEAD-only (equivalent to a flat k-buffer of size HEAD)
        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {
            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {
                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }
                feedHead(ray, hitParticleHead, particles, particleData.idx, particleData.densityParameters,
                         particleFeaturesBuffer, particleFeaturesGradientBuffer, particlesPositionAbsGradPtr);
            }
        }
#endif // GAUSSIAN_TAIL_SIZE

        // Drain the remaining HEAD window front-to-back (common to both paths).
        for (int k = 0; ray.isAlive() && (k < hitParticleHead.numHits()); ++k) {
            processHitParticle(ray,
                               hitParticleHead[GAUSSIAN_HEAD_SIZE - hitParticleHead.numHits() + k],
                               particles,
                               particleFeaturesBuffer,
                               particleFeaturesGradientBuffer,
                               particlesPositionAbsGradPtr);
        }
    }

    // Included inside GUTKBufferRenderer so it reuses the exact same templated
    // forward/backward types and processHitParticle implementation.
#if GOFISH_SOF_SCHEDULE
#include <3dgut/kernels/cuda/renderers/sofMid8Schedule.inl>
#else
#include <3dgut/kernels/cuda/renderers/sofMid8Hierarchy.inl>
#endif
#endif // GAUSSIAN_HIERARCHICAL_SORT

    // Fine-grained balanced forward rendering: Gaussian-wise parallelism with warp-level optimization
    template <typename TRay>
    static inline __device__ void evalForwardNoKBufferBalanced(
        const threedgut::RenderParameters& params,
        TRay& ray,
        const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
        const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
        const float* __restrict__ particlesGlobalDepthPtr,
        const float* __restrict__ particlesPrecomputedFeaturesPtr,
        const tcnn::uvec2& tile,
        const tcnn::uvec2& tileGrid,
        const int laneId,
        threedgut::MemoryHandles parameters,
        tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr     = nullptr,
        tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr = nullptr,
        float* __restrict__ particlesGlobalDepthGradPtr                = nullptr,
        float* __restrict__ particlesPrecomputedFeaturesGradPtr        = nullptr,
        threedgut::MemoryHandles parametersGradient                    = {}) {

        using namespace threedgut;

        // Get tile data: each warp processes particles from a single 16x16 tile
        const uint32_t tileIdx                     = tile.y * tileGrid.x + tile.x;
        const tcnn::uvec2 tileParticleRangeIndices = sortedTileRangeIndicesPtr[tileIdx];

        uint32_t tileNumParticlesToProcess = tileParticleRangeIndices.y - tileParticleRangeIndices.x;

        // Setup feature buffers based on rendering mode
        const TFeaturesVec* particleFeaturesBuffer =
            Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer =
            (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        // Initialize particle system
        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }

        static_assert(Params::KHitBufferSize == 0, "evalForwardNoKBufferBalanced only supports K=0 (no hit buffer). Use evalKBuffer for K>0 cases.");

        // Warp-aligned processing: round up to multiple of WarpSize to avoid divergence
        constexpr uint32_t WarpSize   = GUTParameters::Tiling::WarpSize; // 32 threads per warp
        uint32_t alignedParticleCount = ((tileNumParticlesToProcess + WarpSize - 1) / WarpSize) * WarpSize;

        // Main loop: Gaussian-wise parallelism - WarpSize threads process Gaussians, single ray
        for (uint32_t j = laneId; j < alignedParticleCount; j += WarpSize) {
            if (!ray.isAlive())
                break;

            float hitAlpha                  = 0.0f;
            float3 hitCanonicalIntersection = make_float3(0.f, 0.f, 0.f);
            float hitT                      = 0.0f;
            TFeaturesVec hitFeatures        = TFeaturesVec::zero();
            bool validHit                   = false;

            // Step 1: Each thread tests one Gaussian intersection
            if (j < tileNumParticlesToProcess) {
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + j;
                const uint32_t particleIdx          = sortedTileParticleIdxPtr[toProcessSortedIndex];

                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    auto densityParams = particles.fetchDensityParameters(particleIdx);

                    if (particles.densityHitInterval(ray.origin,
                                             ray.direction,
                                             densityParams,
                                             hitAlpha,
                                             hitT,
                                             hitCanonicalIntersection,
                                             ray.tMinMax.x, ray.tMinMax.y) &&
                        (hitT >= ray.tMinMax.x) &&
                        (hitT <= ray.tMinMax.y)) {

                        validHit = true;

                        // Get Gaussian features
                        if constexpr (Params::PerRayParticleFeatures) {
                            hitFeatures = particles.featuresFromBuffer(particleIdx, ray.direction, hitCanonicalIntersection);
                        } else {
                            hitFeatures = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                        }
                    }
                }
            }

            // Skip if no hits in this warp batch
            constexpr uint32_t WarpMask = GUTParameters::Tiling::WarpMask; // 0xFFFFFFFF for full warp
            if (__all_sync(WarpMask, !validHit))
                continue;

            // Step 2: Compute per-thread transmittance contribution
            float localTransmittance = validHit ? (1.0f - hitAlpha) : 1.0f;

            // Step 3: Warp-level prefix scan for cumulative transmittance
            for (uint32_t offset = 1; offset < WarpSize; offset <<= 1) {
                float n = __shfl_up_sync(WarpMask, localTransmittance, offset);
                if (laneId >= offset) {
                    localTransmittance *= n;
                }
            }

            // Get overall batch transmittance impact
            float batchTransmittance = __shfl_sync(WarpMask, localTransmittance, WarpSize - 1);
            float newTransmittance   = ray.transmittance * batchTransmittance;

            // Step 4: Early termination detection - find exact termination point
            unsigned int earlyTerminationMask = __ballot_sync(WarpMask,
                                                              validHit && (ray.transmittance * localTransmittance) < Particles::MinTransmittanceThreshold);

            bool shouldTerminate = false;
            int terminationLane  = -1;

            if (earlyTerminationMask) {
                terminationLane = __ffs(earlyTerminationMask) - 1; // Find first terminating lane
                shouldTerminate = true;
                ray.kill();
            }

            // Step 5: Warp reduction for feature accumulation
            TFeaturesVec accumulatedFeatures = TFeaturesVec::zero();
            float accumulatedHitT            = 0.0f;
            uint32_t accumulatedHitCount     = 0;

            // Only accumulate contributions before (and including) termination point
            bool shouldContribute = validHit && (!shouldTerminate || laneId <= terminationLane);

            if (shouldContribute) {
                // Use precomputed prefix transmittance, excluding current particle
                float prefixTransmittance   = (laneId > 0) ? (localTransmittance / (1.0f - hitAlpha)) : 1.0f;
                float particleTransmittance = ray.transmittance * prefixTransmittance;
                float hitWeight             = hitAlpha * particleTransmittance;

                // Compute weighted contributions
                for (int featIdx = 0; featIdx < Particles::RayFeatureDim; ++featIdx) {
                    accumulatedFeatures[featIdx] = hitFeatures[featIdx] * hitWeight;
                }
                accumulatedHitT     = hitT * hitWeight;
                accumulatedHitCount = (hitWeight > 0.0f) ? 1 : 0;
            }

            // Step 6: Warp-level reduction (tree-based sum)
            for (int featIdx = 0; featIdx < Particles::RayFeatureDim; ++featIdx) {
                for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                    accumulatedFeatures[featIdx] += __shfl_down_sync(WarpMask, accumulatedFeatures[featIdx], offset);
                }
            }

            for (uint32_t offset = WarpSize / 2; offset > 0; offset >>= 1) {
                accumulatedHitT += __shfl_down_sync(WarpMask, accumulatedHitT, offset);
                accumulatedHitCount += __shfl_down_sync(WarpMask, accumulatedHitCount, offset);
            }

            // Step 7: Only lane 0 updates ray state (avoid race conditions)
            if (laneId == 0) {
                for (int featIdx = 0; featIdx < Particles::RayFeatureDim; ++featIdx) {
                    ray.features[featIdx] += accumulatedFeatures[featIdx];
                }
                ray.hitT += accumulatedHitT;
                ray.countHit(accumulatedHitCount);
            }

            // Step 8: Update ray transmittance
            ray.transmittance = newTransmittance;

            // Break on early termination
            if (shouldTerminate) {
                break;
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
                                                        float* __restrict__ particlesPositionAbsGradPtr = nullptr) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;

        if constexpr (Params::PerRayParticleFeatures) {
            // NHT path: features are re-evaluated per-ray from buffer (no pre-computed feature cache).
            // Gradients accumulate into a thread-private local buffer, then warp-reduced before a single
            // atomicAdd per feature dim — reduces global memory traffic by up to 32× vs. per-hit atomics.
            __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

            for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

                if (__syncthreads_and(!ray.isAlive())) {
                    break;
                }

                // Collectively fetch density parameters only (features fetched per-ray below)
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
                if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                    const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                    if (particleIdx != GUTParameters::InvalidParticleIdx) {
                        prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                    } else {
                        prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                    }
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
                __syncthreads();

                // Process fetched particles
                for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                    if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                        break;
                    }

                    const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                    if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                        ray.kill();
                        break;
                    }

                    // Thread-private feature gradient buffer for warp reduction.
                    // Zero-initialized: non-hitting threads contribute 0 to the warp sum.
                    float featureLocalGrad[Particles::ParticleFeatureDim] = {};

                    if (ray.isAlive()) {
                        float hitAlpha               = 0.f;
                        float hitT                   = 0.f;
                        float3 canonicalIntersection = make_float3(0.f, 0.f, 0.f);

                        if (particles.densityHitInterval(ray.origin, ray.direction, particleData.densityParameters,
                                                 hitAlpha, hitT, canonicalIntersection,
                                                 ray.tMinMax.x, ray.tMinMax.y) &&
                            (hitT >= ray.tMinMax.x) &&
                            (hitT <= ray.tMinMax.y)) {
                            // Re-evaluate NHT features at canonical intersection point (cheap: barycentric interp)
                            const TFeaturesVec hitFeatures = particles.featuresFromBuffer(particleData.idx, ray.direction, canonicalIntersection);

                            float hitAlphaGrad               = 0.f;
                            float3 canonicalIntersectionGrad = make_float3(0.f, 0.f, 0.f);

                            // Write feature grad to thread-private local buffer (no atomics); warp reduction follows below.
                            particles.featuresIntegrateBwdToLocalGrad(ray.direction,
                                                                      canonicalIntersection,
                                                                      canonicalIntersectionGrad,
                                                                      hitAlpha,
                                                                      hitAlphaGrad,
                                                                      particleData.idx,
                                                                      hitFeatures,
                                                                      ray.featuresBackward,
                                                                      ray.featuresGradient,
                                                                      featureLocalGrad);

                            particles.densityProcessHitBwdToBufferInterval<false>(ray.origin,
                                                                                   ray.direction,
                                                                                   particleData.idx,
                                                                                   hitAlpha,
                                                                                   hitAlphaGrad,
                                                                                   ray.transmittanceBackward,
                                                                                   ray.transmittanceGradient,
                                                                                   hitT,
                                                                                   ray.hitTBackward,
                                                                                   ray.hitTGradient,
                                                                                   canonicalIntersectionGrad,
                                                          ray.tMinMax.x, ray.tMinMax.y);

                            ray.transmittance *= (1.0f - hitAlpha);
                        }

                        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                            ray.kill();
                        }
                    }

                    // Warp reduction: all 32 threads (alive or not) participate in __shfl_xor_sync.
                    // Non-hitting threads contribute featureLocalGrad=0 → no spurious gradient.
                    // Reduces 32×ParticleFeatureDim atomics per particle to ParticleFeatureDim atomics.
                    particles.featureLocalGradWarpReduceAndWrite(particleData.idx, featureLocalGrad, tileThreadIdx);
                }
            }
        } else {
            // SH path: pre-computed features fetched from shared memory, gradients via precomputed buffer.
            __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];

            for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

                if (__syncthreads_and(!ray.isAlive())) {
                    break;
                }

                // Collectively fetch particle data
                const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
                if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                    const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                    if (particleIdx != GUTParameters::InvalidParticleIdx) {
                        prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                        prefetchedRawParticlesData[tileThreadIdx].features          = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                        prefetchedRawParticlesData[tileThreadIdx].idx               = particleIdx;
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                    }
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
                __syncthreads();

                // Process fetched particles
                for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                    if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                        break;
                    }

                    const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                    if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                        ray.kill();
                        break;
                    }

                    DensityRawParameters densityRawParametersGrad;
                    densityRawParametersGrad.density    = 0.0f;
                    densityRawParametersGrad.position   = make_float3(0.0f);
                    densityRawParametersGrad.quaternion = make_float4(0.0f);
                    densityRawParametersGrad.scale      = make_float3(0.0f);
                    densityRawParametersGrad.intrinsicDensity = 0.0f;
                    densityRawParametersGrad.factor[0] = make_float3(0.0f);
                    densityRawParametersGrad.factor[1] = make_float3(0.0f);
                    densityRawParametersGrad.factor[2] = make_float3(0.0f);
                    densityRawParametersGrad.inverseFactor[0] = make_float3(0.0f);
                    densityRawParametersGrad.inverseFactor[1] = make_float3(0.0f);
                    densityRawParametersGrad.inverseFactor[2] = make_float3(0.0f);
                    densityRawParametersGrad.fullShapePadding = make_float2(0.0f, 0.0f);

                    TFeaturesVec featuresGrad = TFeaturesVec::zero();

                    if (ray.isAlive()) {
                        particles.processHitBwd<false>(
                            ray.origin,
                            ray.direction,
                            particleData.idx,
                            particleData.densityParameters,
                            &densityRawParametersGrad,
                            particleData.features,
                            &featuresGrad,
                            ray.transmittance,
                            ray.transmittanceBackward,
                            ray.transmittanceGradient,
                            ray.features,
                            ray.featuresBackward,
                            ray.featuresGradient,
                            ray.hitT,
                            ray.hitTBackward,
                            ray.hitTGradient
#if GAUSSIAN_PARTICLE_ENABLE_NORMAL
                            , &ray.sumWD2
                            , ray.sumWD2Backward
                            , ray.sumWD2Gradient
                            , &ray.normal
                            , ray.normalBackward
                            , ray.normalGradient
                            , &ray.sumExt
                            , ray.sumExtBackward
                            , ray.sumExtGradient
                            , ray.opaFieldBackward
                            , ray.opaFieldGradient
                            , ray.tStar
                            , &ray.sumWinvD
                            , ray.sumWinvDBackward
                            , ray.sumWinvDGradient
                            , &ray.sumWinvD2
                            , ray.sumWinvD2Backward
                            , ray.sumWinvD2Gradient
#endif
                            , 0.0f, 0.0f
                            , ray.tMinMax.x, ray.tMinMax.y
                            , ray.distortionDepthScale
                            );
                        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                            ray.kill();
                        }
                    }

                    // Snapshot the detached statistic before signed reduction mutates the local grad.
#if GOFISH_EXACT_SCREEN_ABSGRAD
                    float2 _absScreen = make_float2(0.0f, 0.0f);
                    if (particlesPositionAbsGradPtr != nullptr) {
                        _absScreen = gofishExactScreenAbsGrad(
                            particlesPositionAbsGradPtr, particleData.idx,
                            densityRawParametersGrad.position);
                    }
#else
                    float3 _absPos = make_float3(fabsf(densityRawParametersGrad.position.x),
                                                 fabsf(densityRawParametersGrad.position.y),
                                                 fabsf(densityRawParametersGrad.position.z));
#endif
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                    particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
                    if (particlesPositionAbsGradPtr != nullptr) {
#pragma unroll
                        for (int _m = 1; _m < warpSize; _m *= 2) {
#if GOFISH_EXACT_SCREEN_ABSGRAD
                            _absScreen.x += __shfl_xor_sync(0xffffffff, _absScreen.x, _m);
                            _absScreen.y += __shfl_xor_sync(0xffffffff, _absScreen.y, _m);
#else
                            _absPos.x += __shfl_xor_sync(0xffffffff, _absPos.x, _m);
                            _absPos.y += __shfl_xor_sync(0xffffffff, _absPos.y, _m);
                            _absPos.z += __shfl_xor_sync(0xffffffff, _absPos.z, _m);
#endif
                        }
                        if ((tileThreadIdx & (warpSize - 1)) == 0) {
#if GOFISH_EXACT_SCREEN_ABSGRAD
                            atomicAdd(&particlesPositionAbsGradPtr[particleData.idx * 8 + 6], _absScreen.x);
                            atomicAdd(&particlesPositionAbsGradPtr[particleData.idx * 8 + 7], _absScreen.y);
#else
                            atomicAdd(&particlesPositionAbsGradPtr[particleData.idx * 3 + 0], _absPos.x);
                            atomicAdd(&particlesPositionAbsGradPtr[particleData.idx * 3 + 1], _absPos.y);
                            atomicAdd(&particlesPositionAbsGradPtr[particleData.idx * 3 + 2], _absPos.z);
#endif
                        }
                    }
                }
            }
        }
    }
};
