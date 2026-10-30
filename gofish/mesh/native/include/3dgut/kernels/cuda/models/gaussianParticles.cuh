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

#include <3dgut/kernels/cuda/common/mathUtils.cuh>
#include <3dgut/kernels/cuda/models/gutHalfRayResponse.cuh>
#include <3dgut/kernels/cuda/models/alphaCapBackward.cuh>

#if PARTICLE_FEATURE_HALF
#include <cuda_fp16.h>
#endif

// Following SOF (detach_alpha_extent=True): the extent penalty back-propagates
// into geometry (scale/rotation through sigma_ray) only, not into alpha/opacity.
#include <3dgut/kernels/cuda/models/finiteIntervalExactCrossing.cuh>

namespace threedgut {

struct ParticleDensity {
    float3 position;
    float density;
    float4 quaternion;
    float3 scale;
    float padding;
    float33 factor;
    float33 inverseFactor;
    float2 fullShapePadding;
};

struct ParticeFetchedDensity {
    float3 position;
    float33 factor;
    float33 inverseFactor;
    float density;
};

__forceinline__ __device__ void quaternionWXYZToMatrix(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float rx = r * x;
    const float ry = r * y;
    const float rz = r * z;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
    ret[1] = make_float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
    ret[2] = make_float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));
}

// Exact matrix VJPs for the two row-major CUDA helpers in mathUtils.cuh.
// `input * matrix` evaluates matrix*input, so row i receives
// outputGradient[i]*input. `matrix * input` evaluates matrix^T*input, so row
// i receives input[i]*outputGradient.  Keeping both operations explicit
// avoids relying on S/S^-1 symmetry in the backward implementation.
__forceinline__ __device__ void addMatrixVectorVjp(
    float33& matrixGradient,
    const float3& input,
    const float3& outputGradient) {
    matrixGradient[0] += outputGradient.x * input;
    matrixGradient[1] += outputGradient.y * input;
    matrixGradient[2] += outputGradient.z * input;
}

__forceinline__ __device__ void addTransposeMatrixVectorVjp(
    float33& matrixGradient,
    const float3& input,
    const float3& outputGradient) {
    matrixGradient[0] += input.x * outputGradient;
    matrixGradient[1] += input.y * outputGradient;
    matrixGradient[2] += input.z * outputGradient;
}

// Along-ray Gaussian std used by the differentiable EXTENT channel (fisheye-native).
// sigma_ray = |gscl (elementwise*) rayDirR| where rayDirR = rayDirection * R(quaternion).
// Uses the RAW density parameters (quaternion + scale) exactly as processHitBwd does, so the
// forward accumulation ray.sumExt matches the backward recompute bit-for-bit.
static inline __device__ float particleAlongRaySigmaRaw(const ParticleDensity& pd, const float3& rayDirection) {
    // sigma_ray^2=d^T B B^T d=||B^T d||^2. For row-major float33,
    // `matrix * vector` evaluates matrix^T*vector.
    const float3 factorDirection = pd.factor * rayDirection;
    return sqrtf(dot(factorDirection, factorDirection));
}

static constexpr __device__ float SpHCoeff0   = 0.28209479177387814f;
static constexpr __device__ float SpHCoeff1   = 0.4886025119029199f;
static constexpr __device__ float SpHCoeff2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                                 -1.0925484305920792f, 0.5462742152960396f};
static constexpr __device__ float SpHCoeff3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                                 -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

static inline __device__ float3
radianceFromSpH(int deg, const float3* sphCoefficients, const float3& rdir, bool clamped = true) {
    float3 rad = SpHCoeff0 * sphCoefficients[0];
    if (deg > 0) {
        const float3& dir = rdir;

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad           = rad - SpHCoeff1 * y * sphCoefficients[1] + SpHCoeff1 * z * sphCoefficients[2] -
              SpHCoeff1 * x * sphCoefficients[3];

        if (deg > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SpHCoeff2[0] * xy * sphCoefficients[4] + SpHCoeff2[1] * yz * sphCoefficients[5] +
                  SpHCoeff2[2] * (2.0f * zz - xx - yy) * sphCoefficients[6] +
                  SpHCoeff2[3] * xz * sphCoefficients[7] + SpHCoeff2[4] * (xx - yy) * sphCoefficients[8];

            if (deg > 2) {
                rad = rad + SpHCoeff3[0] * y * (3.0f * xx - yy) * sphCoefficients[9] +
                      SpHCoeff3[1] * xy * z * sphCoefficients[10] +
                      SpHCoeff3[2] * y * (4.0f * zz - xx - yy) * sphCoefficients[11] +
                      SpHCoeff3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12] +
                      SpHCoeff3[4] * x * (4.0f * zz - xx - yy) * sphCoefficients[13] +
                      SpHCoeff3[5] * z * (xx - yy) * sphCoefficients[14] +
                      SpHCoeff3[6] * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }
    rad += 0.5f;
    return clamped ? maxf3(rad, make_float3(0.0f)) : rad;
}

template <bool Atomic = false>
static inline __device__ void addSphCoeffGrd(float3* sphCoefficientsGrad, int idx, const float3& val) {
    if constexpr (Atomic) {
        atomicAdd(&sphCoefficientsGrad[idx].x, val.x);
        atomicAdd(&sphCoefficientsGrad[idx].y, val.y);
        atomicAdd(&sphCoefficientsGrad[idx].z, val.z);
    } else {
        sphCoefficientsGrad[idx] += val;
    }
}

template <bool Atomic = false>
static inline __device__ float3 radianceFromSpHBwd(int deg, const float3* sphCoefficients, const float3& rdir,
                                                   float weight, const float3& rayRadGrd, float3* sphCoefficientsGrad) {
    // radiance unclamped
    const float3 gradu = radianceFromSpH(deg, sphCoefficients, rdir, false);
    radianceFromSpHBwd<Atomic>(deg, rdir, rayRadGrd * weight, sphCoefficientsGrad, gradu);
    return make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                       gradu.y > 0.0f ? gradu.y : 0.0f,
                       gradu.z > 0.0f ? gradu.z : 0.0f);
}

template <bool Atomic = false>
static inline __device__ void radianceFromSpHBwd(int deg, const float3& rdir, const float3& rayRadGrd,
                                                 float3* sphCoefficientsGrad, const float3& gradu) {
    //
    float3 dL_dRGB = rayRadGrd;
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0);
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ---> rayRad = weight * grad = weight * explu(gsph0 * SpHCoeff0 +
    // 0.5,SHRadMinBound) with explu(x,a) = x if x > a else a*e(x-a)
    // ===> d_rayRad / d_gsph0 =   weight * SpHCoeff0
    addSphCoeffGrd(sphCoefficientsGrad, 0, SpHCoeff0 * dL_dRGB);

    if (deg > 0) {
        // const float3 sphdiru = gpos - rori;
        // const float3 sphdir = safe_normalize(sphdiru);
        const float3& sphdir = rdir;

        float x = sphdir.x;
        float y = sphdir.y;
        float z = sphdir.z;

        float dRGBdsh1 = -SpHCoeff1 * y;
        float dRGBdsh2 = SpHCoeff1 * z;
        float dRGBdsh3 = -SpHCoeff1 * x;

        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 3, dRGBdsh3 * dL_dRGB);

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SpHCoeff2[0] * xy;
            float dRGBdsh5 = SpHCoeff2[1] * yz;
            float dRGBdsh6 = SpHCoeff2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SpHCoeff2[3] * xz;
            float dRGBdsh8 = SpHCoeff2[4] * (xx - yy);

            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 8, dRGBdsh8 * dL_dRGB);

            if (deg > 2) {
                float dRGBdsh9  = SpHCoeff3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SpHCoeff3[1] * xy * z;
                float dRGBdsh11 = SpHCoeff3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SpHCoeff3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SpHCoeff3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SpHCoeff3[5] * z * (xx - yy);
                float dRGBdsh15 = SpHCoeff3[6] * x * (xx - 3.f * yy);

                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd<Atomic>(sphCoefficientsGrad, 15, dRGBdsh15 * dL_dRGB);
            }
        }
    }
}

static inline __device__ void fetchParticleDensity(
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    float3& particlePosition,
    float33& particleFactor,
    float33& particleInverseFactor,
    float& particleDensity) {
    const ParticleDensity particleData = particlesDensity[particleIdx];

    particlePosition = particleData.position;
    particleFactor[0] = particleData.factor[0];
    particleFactor[1] = particleData.factor[1];
    particleFactor[2] = particleData.factor[2];
    particleInverseFactor[0] = particleData.inverseFactor[0];
    particleInverseFactor[1] = particleData.inverseFactor[1];
    particleInverseFactor[2] = particleData.inverseFactor[2];
    particleDensity = particleData.density;
}

static inline __device__ void fetchParticleSphCoefficients(
    const int32_t particleIdx,
    const float* particlesSphCoefficients,
    float3* sphCoefficients) {
    const uint32_t particleOffset = particleIdx * PARTICLE_RADIANCE_NUM_COEFFS * 3;
#pragma unroll
    for (unsigned int i = 0; i < PARTICLE_RADIANCE_NUM_COEFFS; ++i) {
        const int offset   = i * 3;
        sphCoefficients[i] = make_float3(
            particlesSphCoefficients[particleOffset + offset + 0],
            particlesSphCoefficients[particleOffset + offset + 1],
            particlesSphCoefficients[particleOffset + offset + 2]);
    }
}

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponseGrd(float grayDist, float gres, float gresGrd) {
    /// generalized gaussian of degree b : scaling a = -4.5/3^b
    /// d_e^{a*|x|^b}/d_x^2 = a*(0.5*b)*x^{b-2}*e^{a*|x|^b}
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241 * (0.5f * 8);
        const float grayDistSq = grayDist * grayDist;
        return s * grayDistSq * grayDist * gres * gresGrd;
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185 * (0.5f * 5);
        return s * grayDist * sqrtf(grayDist) * gres * gresGrd;
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556 * (0.5f * 4);
        return s * grayDist * gres * gresGrd;
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667 * (0.5f * 3);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f * (0.5f * 1);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return gres > 0.f ? (0.5f * s * rsqrtf(grayDist)) * gresGrd : 0.f;
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return s * gres * gresGrd;
    }
    }
}

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponse(float grayDist) {
    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241f;
        const float grayDistSq = grayDist * grayDist;
        return expf(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185f;
        return expf(s * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556f;
        return expf(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667f;
        return expf(s * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f;
        return expf(s * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return expf(s * grayDist);
    }
    }
}

template <int GeneralizedGaussianDegree = 2, bool clamped>
static inline __device__ float particleScaledResponse(float grayDist, float modulatedMinResponse, float responseModulation = 1.0f) {

    const float minResponse    = fminf(modulatedMinResponse / responseModulation, 0.97f);
    const float logMinResponse = clamped ? logf(minResponse) : modulatedMinResponse;

    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        const float grayDistSq = grayDist * grayDist;
        return expf(logMinResponse * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        return expf(logMinResponse * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        return expf(logMinResponse * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        return expf(logMinResponse * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        return expf(logMinResponse * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = (1.0f - minResponse) / 3.0f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        return expf(logMinResponse * grayDist);
    }
    }
}

// Re-evaluate the (unclamped-then-MAX-clamped) Gaussian alpha of a particle at a fixed WORLD depth tStar
// along the ray, used by the differentiable OPACITY-FIELD channel for hits whose peak is behind tStar.
// grayDist(t) = AA t^2 + BB t + CC  with AA=dot(grdu,grdu), BB=2 dot(gro,grdu), CC=dot(gro,gro)
// (== squared Mahalanobis distance along the ray at world depth t; peak/min at t = -BB/2AA).
// Uses RAW density params exactly as processHitBwd so forward == backward recompute.
static inline __device__ float particleAlphaAtDepthRaw(const ParticleDensity& pd,
                                                       const float3& rayOrigin,
                                                       const float3& rayDirection,
                                                       float tStar) {
    const float3 gro = (rayOrigin - pd.position) * pd.inverseFactor;
    const float3 grdu = rayDirection * pd.inverseFactor;
    const float grayDist_ts = fmaxf(dot(grdu, grdu) * tStar * tStar + 2.f * dot(gro, grdu) * tStar + dot(gro, gro), 0.f);
    const float gres = particleResponse<GAUSSIAN_PARTICLE_KERNEL_DEGREE>(grayDist_ts);
    return fminf(GAUSSIAN_PARTICLE_MAX_ALPHA, gres * pd.density);
}

// ---- FINITE-INTERVAL ANALYTIC EXACT DEPTH ----
// SOF's quadratic opacity-level equation, solved on the finite generic-camera
// ray interval: the peak alpha is evaluated at clamp(t_line,[t_min,t_max]) and
// the crossing is solved and differentiated on that same interval.
static inline __device__ FiniteIntervalCrossingResult particleExactDepthCrossingRaw(
    const ParticleDensity& pd,
    const float3& rayOrigin,
    const float3& rayDirection,
    float transmittanceBefore,
    float minHitDistance,
    float maxHitDistance) {
    const float3 gro = (rayOrigin - pd.position) * pd.inverseFactor;
    const float3 grdu = rayDirection * pd.inverseFactor;
    const float AA = dot(grdu, grdu);
    const float BB = 2.0f * dot(gro, grdu);
    const float CC = dot(gro, gro);
    return finiteIntervalExactCrossing(
        AA, BB, CC, pd.density, transmittanceBefore,
        minHitDistance, maxHitDistance,
        0.5f, GAUSSIAN_PARTICLE_MAX_ALPHA);
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false, bool PerRayRadiance = true>
__device__ inline bool processHitFwd(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float* particlesSphCoefficients,
    const float minParticleKernelDensity,
    const float minParticleAlpha,
    const int32_t sphEvalDegree,
    float* transmittance,
    float3* radiance,
    float* depth,
    float3* normal,
    float minHitDistance = 0.0f,
    float maxHitDistance = 1.0e30f) {
    float3 particlePosition;
    float33 particleFactor;
    float33 particleInverseFactor;
    float particleDensity;

    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleFactor,
        particleInverseFactor,
        particleDensity);

    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gro     = gposc * particleInverseFactor;
    const float3 grdu    = rayDirection * particleInverseFactor;
    const float3 grd     = safe_normalize(grdu);

    const GutHalfRayClosestPoint halfRay = gutHalfRayClosestPoint(
        gro, grdu, minHitDistance, maxHitDistance);
    if (!halfRay.valid) return false;
    const float3 gcrod   = halfRay.canonicalPoint;
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(GAUSSIAN_PARTICLE_MAX_ALPHA, gres * particleDensity);

    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    if (acceptHit) {
        const float weight = galpha * (*transmittance);

        // Exact constrained ray parameter; renderer rays are unit length.
        const float hitT = halfRay.tau;

        if constexpr (PerRayRadiance) {
            // radiance from sph coefficients
            float3 sphCoefficients[PARTICLE_RADIANCE_NUM_COEFFS];
            fetchParticleSphCoefficients(
                particleIdx,
                particlesSphCoefficients,
                &sphCoefficients[0]);
            *radiance += weight * radianceFromSpH(sphEvalDegree, &sphCoefficients[0], rayDirection);
        } else {
            *radiance += weight * reinterpret_cast<const float3*>(particlesSphCoefficients)[particleIdx];
        }

        *transmittance *= (1 - galpha);
        *depth += hitT * weight;

        if (normal) {
            // grad_x ||B^-1 x|| is B^-T gcrod. For row-major float33,
            // `matrix * vector` is the transpose product.
            float3 worldNormal = particleInverseFactor * gcrod;
            if (dot(worldNormal, gposc) < 0.0f) worldNormal *= -1.0f;
            *normal += weight * safe_normalize(worldNormal);
        }
    }

    return acceptHit;
}

__device__ inline bool intersectCustomParticle(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    float3 particlePosition;
    float33 particleFactor;
    float33 particleInverseFactor;
    float particleDensity;
    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleFactor,
        particleInverseFactor,
        particleDensity);

    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gro     = gposc * particleInverseFactor;
    const float3 grdu    = rayDirection * particleInverseFactor;
    const float3 grd     = safe_normalize(grdu);

    const GutHalfRayClosestPoint closest = gutHalfRayClosestPoint(
        gro, grdu, minHitDistance, maxHitDistance);
    hitDistance = closest.tau;
    return closest.valid && dot(closest.canonicalPoint, closest.canonicalPoint)
        < maxParticleSquaredDistance;
}

__device__ inline bool intersectInstanceParticle(
    const float3& particleRayOrigin,
    const float3& particleRayDirection,
    const int32_t particleIdx,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    const GutHalfRayClosestPoint closest = gutHalfRayClosestPoint(
        particleRayOrigin, particleRayDirection, minHitDistance, maxHitDistance);
    hitDistance = closest.tau;
    return closest.valid && dot(closest.canonicalPoint, closest.canonicalPoint)
        < maxParticleSquaredDistance;
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false, bool PerRayRadiance = true>
__device__ inline void processHitBwd(
    const float3& rayOrigin,
    const float3& rayDirection,
    uint32_t particleIdx,
    const ParticleDensity& particleData,
    ParticleDensity* particleDensityGradPtr,
    const float* particleRadiancePtr,
    float* particleRadianceGradPtr,
    float minParticleKernelDensity,
    float minParticleAlpha,
    float minTransmittance,
    int32_t sphEvalDegree,
    float integratedTransmittance,
    float& transmittance,
    float transmittanceGrad,
    float3 integratedRadiance,
    float3& radiance,
    float3 radianceGrad,
    float integratedDepth,
    float& depth,
    float integratedDepthGrad,
    float integratedSumWD2 = 0.0f,
    float* sumWD2AccumPtr = nullptr,
    float integratedSumWD2Grad = 0.0f,
    float3* normalAccumPtr = nullptr,
    const float3& integratedNormal = make_float3(0.0f, 0.0f, 0.0f),
    const float3& integratedNormalGrad = make_float3(0.0f, 0.0f, 0.0f),
    float integratedSumExt = 0.0f,
    float* sumExtAccumPtr = nullptr,
    float integratedSumExtGrad = 0.0f,
    float integratedOpaField = 0.0f,
    float opaFieldGrad = 0.0f,
    float tStar = 0.0f,
    float integratedSumWinvD = 0.0f,
    float* sumWinvDAccumPtr = nullptr,
    float integratedSumWinvDGrad = 0.0f,
    float integratedSumWinvD2 = 0.0f,
    float* sumWinvD2AccumPtr = nullptr,
    float integratedSumWinvD2Grad = 0.0f,
    float crossingGrad = 0.0f,
    float crossingTStar = 0.0f,
    float minHitDistance = 0.0f,
    float maxHitDistance = 1.0e30f) {
    const float3 particlePosition = particleData.position;
    const float particleDensity = particleData.density;
    const float33& particleFactor = particleData.factor;
    const float33& particleInverseFactor = particleData.inverseFactor;

    // project ray in the gaussian
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gro     = gposc * particleInverseFactor;
    const float3 grdu    = rayDirection * particleInverseFactor;
    const float3 grd     = safe_normalize(grdu);
    const GutHalfRayClosestPoint halfRay = gutHalfRayClosestPoint(
        gro, grdu, minHitDistance, maxHitDistance);
    if (!halfRay.valid) return;
    const float3 gcrod   = halfRay.canonicalPoint;
    const float grayDist = dot(gcrod, gcrod);

    const float gres            = particleResponse<ParticleKernelDegree>(grayDist);
    const float galphaUnclamped = gres * particleDensity;
    const float galpha          = fminf(GAUSSIAN_PARTICLE_MAX_ALPHA, galphaUnclamped);

    if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha)) {

        const float3 grdd   = make_float3(0.0f);
        const float3 grds   = make_float3(0.0f);
        const float gdist   = halfRay.tau;
        const float gsqdist = gdist * gdist;

        const float weight = galpha * transmittance;

        const float nextTransmit = (1 - galpha) * transmittance;

        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        depth += weight * gdist;
        const float residualHitT =
            fmaxf((nextTransmit <= minTransmittance ? 0 : (integratedDepth - depth) / nextTransmit),
                  0);
        // ---- Differentiable 2nd-moment (E[d^2] numerator) channel for distortion loss ----
        // gdist^2 == gsqdist exactly; sumWD2 accumulates sum_i w_i*gsqdist_i just as depth accumulates sum_i w_i*gdist_i.
        const float curSumWD2 = (sumWD2AccumPtr != nullptr) ? *sumWD2AccumPtr : 0.0f;
        const float newSumWD2 = curSumWD2 + weight * gsqdist;
        if (sumWD2AccumPtr != nullptr) { *sumWD2AccumPtr = newSumWD2; }
        const float residualSumWD2 =
            fmaxf((nextTransmit <= minTransmittance ? 0.0f : (integratedSumWD2 - newSumWD2) / nextTransmit),
                  0.0f);

        // ---- Differentiable along-ray EXTENT channel (fisheye-native) ----
        // f = sigma_ray = |gscl (elementwise*) rayDirR| (along-ray Gaussian std). ray.sumExt accumulates
        // sum_i w_i * sigma_ray_i just as sumWD2 accumulates sum_i w_i * gsqdist_i (weight = galpha*T).
        // sigma_ray depends on gscl (directly) and rayDirR (=> rotation), NOT on grds -> its shape gradient
        // does NOT reuse the grds chain (unlike sumWD2); it is added directly to scale + rayDirRGrd below.
        const float3 factorDirection = particleFactor * rayDirection;
        const float sigmaRay = sqrtf(dot(factorDirection, factorDirection));
        const float curSumExt    = (sumExtAccumPtr != nullptr) ? *sumExtAccumPtr : 0.0f;
        const float newSumExt    = curSumExt + weight * sigmaRay;
        if (sumExtAccumPtr != nullptr) { *sumExtAccumPtr = newSumExt; }
        const float residualSumExt =
            fmaxf((nextTransmit <= minTransmittance ? 0.0f : (integratedSumExt - newSumExt) / nextTransmit),
                  0.0f);

        // ---- Differentiable SOF inverse-depth distortion channels: Sum w_i*(1/d_i), Sum w_i*(1/d_i^2) ----
        // d_i = gdist (per-hit metric hit distance). Guard gdist>eps (degenerate near-camera hits contribute 0),
        // matched to the forward accumulation. Mirror sumWD2's front-to-back residual for the weight/galpha path;
        // the shape gradient (through grds) is added to grdsRayHitGrd below.
        const float invDistEps  = 1e-4f;
        const bool  invDistOk   = (gdist > invDistEps);
        const float finvd       = invDistOk ? (1.0f / gdist) : 0.0f;
        const float finvd2      = invDistOk ? (1.0f / gsqdist) : 0.0f;
        const float curSumWinvD = (sumWinvDAccumPtr != nullptr) ? *sumWinvDAccumPtr : 0.0f;
        const float newSumWinvD = curSumWinvD + weight * finvd;
        if (sumWinvDAccumPtr != nullptr) { *sumWinvDAccumPtr = newSumWinvD; }
        const float residualSumWinvD =
            fmaxf((nextTransmit <= minTransmittance ? 0.0f : (integratedSumWinvD - newSumWinvD) / nextTransmit),
                  0.0f);
        const float curSumWinvD2 = (sumWinvD2AccumPtr != nullptr) ? *sumWinvD2AccumPtr : 0.0f;
        const float newSumWinvD2 = curSumWinvD2 + weight * finvd2;
        if (sumWinvD2AccumPtr != nullptr) { *sumWinvD2AccumPtr = newSumWinvD2; }
        const float residualSumWinvD2 =
            fmaxf((nextTransmit <= minTransmittance ? 0.0f : (integratedSumWinvD2 - newSumWinvD2) / nextTransmit),
                  0.0f);
        // dL/d(sigma_ray) through the direct shape path = w * dL/d(pred_extent).
        const float dL_df_ext = weight * integratedSumExtGrad;
        // factorDirection = B^T*d and sigma_ray = ||factorDirection||.
        const float3 factorDirectionGrd = (dL_df_ext / sigmaRay) * factorDirection;

        // ---- Differentiable OPACITY-FIELD channel (SOF O_N -> 0.5) ----
        // O_N = 1 - prod_i (1 - a_i), a_i = alpha at min(t_i, t*). t* (=tStar) is the DETACHED median depth.
        // Product-rule: dL/da_i = dL/dO_N * T_opa_full / (1 - a_i), with T_opa_full = 1 - O_N (read from fwd
        // output = integratedOpaField). No front-to-back residual needed (each factor appears once).
        // Before/at t* (t_i<=t*): a_i = galpha -> add dL/da_i to galphaRayHitGrd (reuses galpha->shape chain).
        // After t* (t_i>t*): a_i = dens*response(grayDist(t*)); grad flows to density + gro/grdu (direct).
        float galphaOpaFieldGrd = 0.0f;
        float3 groGrd_opa  = make_float3(0.0f, 0.0f, 0.0f);
        float3 grduGrd_opa = make_float3(0.0f, 0.0f, 0.0f);
        float dOpaFieldDens = 0.0f;
        if (opaFieldGrad != 0.0f) {
            const bool afterTStar = (tStar > 0.0f) && (gdist > tStar);
            const float AA_ut = dot(grdu, grdu);
            const float BB_ut = 2.0f * dot(gro, grdu);
            const float CC_ut = dot(gro, gro);
            const float grayDist_ts = fmaxf(AA_ut * tStar * tStar + BB_ut * tStar + CC_ut, 0.0f);
            const float gres_ts         = afterTStar ? particleResponse<ParticleKernelDegree>(grayDist_ts) : gres;
            const float a_i_unclamped = gres_ts * particleDensity;
            const float a_i            = afterTStar ? fminf(GAUSSIAN_PARTICLE_MAX_ALPHA, a_i_unclamped) : galpha;
            // Opacity-field adjoint: dO_N/da_i = prod_{j!=i}(1 - a_j) = T_opa_full/(1 - a_i) is the
            // EXACT adjoint and is MATHEMATICALLY BOUNDED in [0,1] (product of (1-a_j) each in [0,1]).
            // (i) floor 1-a_i at its PHYSICAL minimum 1-GAUSSIAN_PARTICLE_MAX_ALPHA (a_i can never exceed
            // MAX_ALPHA by construction, so this is the exact lower bound of the forward factor);
            // (ii) clamp the exclusive product to its exact analytic bound 1.0. This is the true adjoint
            // wherever the division is well-conditioned (fminf inactive) and in the near-opaque limit
            // stays at the true bound WITHOUT changing sign (T_opa_full>=0, 1-a_i>0 => dL_da keeps
            // sign(opaFieldGrad)). |dL_da| <= |opaFieldGrad|.
            const float oneMinusAi = fmaxf(1.0f - a_i, 1.0f - GAUSSIAN_PARTICLE_MAX_ALPHA);
            const float T_opa_full = 1.0f - integratedOpaField;      // full product prod_i (1 - a_i), in [0,1]
            const float exclProd   = fminf(fmaxf(T_opa_full, 0.0f) / oneMinusAi, 1.0f); // prod_{j!=i}(1-a_j) in [0,1]
            const float dL_da      = opaFieldGrad * exclProd;        // dL/da_i, |.| <= |opaFieldGrad|
            if (afterTStar) {
                // KEEP the DENSITY gradient: smooth, bounded, and the SOF-intended signal (L_opa
                // sharpens OPACITY by driving O_N->0.5 through per-particle opacity/density).
                dOpaFieldDens = alphaCapBackwardMask(a_i_unclamped, GAUSSIAN_PARTICLE_MAX_ALPHA)
                                  * dL_da * gres_ts; // exact d min(cap,dens*gres_ts) / d density
                // DETACH the after-t* GEOMETRIC re-evaluation gradient d a_i(t*)/d {pos,rot,scale}.
                // alpha@t* uses the UNNORMALIZED ray parameter grdu=(1/scale)(dir*R), so this term would
                // inject an unbounded 1/scale^2..1/scale^3 factor into the scale/rayDirR gradients, unlike
                // the main chain which passes through safe_normalize_bw(grdu,...); it is also gated by the
                // discontinuous gdist>tStar branch. SOF treats t* as a DETACHED 0.5-median target and shapes
                // geometry via the distortion/dnc/esmooth/extent losses, so L_opa's after-t* signal is
                // routed to DENSITY only (above).
                grduGrd_opa = make_float3(0.0f, 0.0f, 0.0f);
                groGrd_opa  = make_float3(0.0f, 0.0f, 0.0f);
            } else {
                galphaOpaFieldGrd = dL_da;                            // a_i == galpha (stable galpha chain)
            }
        }

        // ---- Differentiable IFT CROSSING-DEPTH channel (SOF exact 0.5-crossing t*, made differentiable) ----
        // dL/dtheta = g * dO/dtheta|_{t*}, g = crossingGrad (= -(dL/dt*)/Odot*, formed in initializeBackwardRay).
        // dO/dtheta|_{t*} = Sum_i exclProd_i * da_i/dtheta, exclProd_i = T(t*)/(1-a_i) = 0.5/(1-a_i) EXACTLY at the
        // crossing (T(t*)=0.5 by construction, bounded by 1). SAME adjoint as opafield BUT eval depth = EXACT t* and
        // geometry NOT detached: for the crossing Gaussian T_before*w*resp' cancels between dO/dtheta_k and Odot* so
        // g*(1/scale^3 geom) is BOUNDED (IFT well-conditioning); non-crossing after-t* hits are tail-suppressed. The
        // before-t* branch flows via the stable normalized galpha chain. Additive alongside the detached opafield block.
        float galphaCrossGrd = 0.0f;
        float3 groGrd_cs  = make_float3(0.0f, 0.0f, 0.0f);
        float3 grduGrd_cs = make_float3(0.0f, 0.0f, 0.0f);
        float dCrossDens  = 0.0f;
        if (crossingGrad != 0.0f && crossingTStar > 0.0f) {
            // SOF-style fixed-order/lumped-prefix IFT:
            //   F(t)=T_before*(1-alpha_k(t))-0.5.
            // Prior particles differentiate their PEAK alpha through T_before;
            // only the crossing particle is evaluated at t*. Later particles
            // do not enter F. This is not a global continuous-opacity median.
            const bool beforeThreshold = transmittance > 0.5f;
            const bool crossingParticle = beforeThreshold && nextTransmit <= 0.5f;
            const bool priorParticle = beforeThreshold && nextTransmit > 0.5f;
            if (priorParticle) {
                const float oneMinusPeak = fmaxf(
                    1.0f - galpha, 1.0f - GAUSSIAN_PARTICLE_MAX_ALPHA);
                const float prefixExclusion = fminf(0.5f / oneMinusPeak, 1.0f);
                galphaCrossGrd = crossingGrad * prefixExclusion;
            } else if (crossingParticle) {
                const float AA_cs = dot(grdu, grdu);
                const float BB_cs = 2.0f * dot(gro, grdu);
                const float CC_cs = dot(gro, gro);
                const float grayDist_cs = fmaxf(
                    AA_cs * crossingTStar * crossingTStar +
                    BB_cs * crossingTStar + CC_cs, 0.0f);
                const float gres_cs = particleResponse<ParticleKernelDegree>(grayDist_cs);
                const float alphaCrossUnclamped = gres_cs * particleDensity;
                const float alphaCross = fminf(
                    GAUSSIAN_PARTICLE_MAX_ALPHA, alphaCrossUnclamped);
                const float oneMinusCross = fmaxf(
                    1.0f - alphaCross, 1.0f - GAUSSIAN_PARTICLE_MAX_ALPHA);
                // At F=0, 0.5/(1-alpha_k(t*)) == T_before.
                const float prefixAtCross = fminf(0.5f / oneMinusCross, 1.0f);
                const float dL_da_cs = crossingGrad * prefixAtCross;
                const float capMask = alphaCapBackwardMask(
                    alphaCrossUnclamped, GAUSSIAN_PARTICLE_MAX_ALPHA);
                dCrossDens = capMask * dL_da_cs * gres_cs;
                const float dL_dgres_cs = capMask * dL_da_cs * particleDensity;
                const float dL_dgrayDist_cs = particleResponseGrd<ParticleKernelDegree>(
                    grayDist_cs, gres_cs, dL_dgres_cs);
                grduGrd_cs = dL_dgrayDist_cs *
                    (2.0f * (crossingTStar * crossingTStar * grdu +
                             crossingTStar * gro));
                groGrd_cs = dL_dgrayDist_cs *
                    (2.0f * (crossingTStar * grdu + gro));
            }
        }

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        //
        // ===> d_hitT / d_galpha = gdist * prevTrm - residualHitT * prevTrm
        //                        = (gdist - residualHitT) * prevTrm
        //
        // ---- Per-primitive normal backward (SOF render_normal; port of the Slang particleDensityHit
        // normal into the K==0 SH backward). Forward accumulates ray.normal += weight*n_i (raw; Python
        // applies F.normalize, as in SOF's train.py). ----
        float galphaNormalGrd = 0.0f;
        float3 grdNormalGrd = make_float3(0.0f);
        if (normalAccumPtr != nullptr) {
            // Exact GOF normal:
            // n = -normalize(S^-T normalize(S^-1 d)).  `matrix * vector`
            // is the transpose-matrix product for the row-major helper.
            const float3 normalUnnormalized = particleInverseFactor * grd;
            const float normalLength = sqrtf(dot(normalUnnormalized, normalUnnormalized));
            const float3 n_i = (-1.0f / normalLength) * normalUnnormalized;
            *normalAccumPtr = *normalAccumPtr + weight * n_i;
            const float3 curNormal = *normalAccumPtr;
            const float3 residualNormal = (nextTransmit <= minTransmittance)
                ? make_float3(0.0f)
                : (integratedNormal - curNormal) / nextTransmit;
            galphaNormalGrd = transmittance * dot(n_i - residualNormal, integratedNormalGrad);

            const float3 dL_dn = weight * integratedNormalGrad;
            const float3 normalUnnormalizedGrd = safe_normalize_bw(
                normalUnnormalized, -1.0f * dL_dn);
            // y=S^-T*grd: direct inverse-factor adjoint plus the grd adjoint.
            addTransposeMatrixVectorVjp(
                particleDensityGradPtr->inverseFactor, grd, normalUnnormalizedGrd);
            grdNormalGrd = normalUnnormalizedGrd * particleInverseFactor;
        }
        const float galphaRayHitGrd = (gdist - residualHitT) * transmittance * integratedDepthGrad
                                    + (gsqdist - residualSumWD2) * transmittance * integratedSumWD2Grad
                                    + 0.0f  // detach_alpha_extent=True (SOF default): extent does NOT flow into alpha/opacity grad (geometry channel dL_df_ext->scl/rayDirR still flows)
                                    + (finvd - residualSumWinvD) * transmittance * integratedSumWinvDGrad
                                    + (finvd2 - residualSumWinvD2) * transmittance * integratedSumWinvD2Grad
                                    + galphaNormalGrd
                                    + galphaOpaFieldGrd
                                    + galphaCrossGrd;
        //
        // ===> d_hitT / d_gsqdist = weight / (2*gdist)
        // ===> d_gsqdist / d_grds =  2 * grds
        const float3 grdsRayHitGrd = make_float3(0.0f);
        const float3 gsclRayHitGrd = make_float3(0.0f);
        const float3 grdRayHitGrd  = make_float3(0.0f);
        const float3 groRayHitGrd  = make_float3(0.0f);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_galpha = prevTrm * nextTrm = residualTrm
        const float residualTrm     = galpha < 0.999999f ? integratedTransmittance / (1 - galpha) : transmittance;
        const float galphaRayDnsGrd = residualTrm * -transmittanceGrad;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // compute the gradient wrt to the sph coefficients and position (through the sph view
        // direction)
        float3 grad;
        if constexpr (PerRayRadiance) {
            float3 sphCoefficients[PARTICLE_RADIANCE_NUM_COEFFS];
            fetchParticleSphCoefficients(
                particleIdx,
                particleRadiancePtr,
                &sphCoefficients[0]);
            grad = radianceFromSpHBwd<true>(sphEvalDegree, &sphCoefficients[0], rayDirection, weight, radianceGrad,
                                            (float3*)&particleRadianceGradPtr[particleIdx * PARTICLE_RADIANCE_NUM_COEFFS * 3]);
        } else {
            grad                       = reinterpret_cast<const float3*>(particleRadiancePtr)[0];
            particleRadianceGradPtr[0] = radianceGrad.x * weight;
            particleRadianceGradPtr[1] = radianceGrad.y * weight;
            particleRadianceGradPtr[2] = radianceGrad.z * weight;
        }

        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
        const float3 rayRad = weight * grad;
        radiance += rayRad;
        const float3 residualRayRad = maxf3((nextTransmit <= minTransmittance ? make_float3(0.f, 0.f, 0.f) : (integratedRadiance - radiance) / nextTransmit),
                                            make_float3(0.f, 0.f, 0.f));

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gdns = residualTrm * gres
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
        const float galphaCompositeGrd =
            galphaRayHitGrd + galphaRayDnsGrd
            + transmittance * (grad.x - residualRayRad.x) * radianceGrad.x
            + transmittance * (grad.y - residualRayRad.y) * radianceGrad.y
            + transmittance * (grad.z - residualRayRad.z) * radianceGrad.z;
        const float galphaUnclampedGrd = alphaCapBackwardMask(
            galphaUnclamped, GAUSSIAN_PARTICLE_MAX_ALPHA) * galphaCompositeGrd;
        particleDensityGradPtr->density =
            gres * galphaUnclampedGrd
            + dOpaFieldDens    // fixed-depth OPACITY-FIELD branch has its own cap mask
            + dCrossDens;      // fixed-depth IFT CROSSING branch has its own cap mask

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gres = residualTrm * gdns
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
        const float gresGrd = particleDensity * galphaUnclampedGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gres = exp(-0.0555 * grayDist * grayDist)
        // ===> d_gres / d_grayDist = -0.111 * grayDist * exp(-0.555 * grayDist * grayDist)
        //                          = -0.111 * grayDist * gres
        const float grayDistGrd = particleResponseGrd<ParticleKernelDegree>(grayDist, gres, gresGrd);

        const float tauGeometryGrad =
              weight * integratedDepthGrad
            + 2.0f * gdist * weight * integratedSumWD2Grad
            + (invDistOk ? (-weight / (gdist * gdist)) * integratedSumWinvDGrad : 0.0f)
            + (invDistOk ? (-2.0f * weight / (gdist * gdist * gdist)) * integratedSumWinvD2Grad : 0.0f);
        float3 groGrd;
        float3 halfRayGrduGrd;
        gutHalfRayClosestPointVjp(gro, grdu, halfRay, grayDistGrd,
                                  tauGeometryGrad, groGrd, halfRayGrduGrd);
        const float3 grdGrd = make_float3(0.0f);

        const float3 totalGroGrd = groGrd + groGrd_opa + groGrd_cs;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grd = safe_normalize(grdu)
        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
        const float3 grduGrd = halfRayGrduGrd + safe_normalize_bw(grdu, grdNormalGrd)
                             + grduGrd_opa + grduGrd_cs;
        const float3 metricScaleGrad = make_float3(0.0f);

        // gro = S^-1*(o-mu), grdu = S^-1*d.  The CUDA row-vector
        // spelling `x * M` evaluates M*x; these are its exact VJPs.
        const float3 gposcGrd = matmul_bw_vec(particleInverseFactor, totalGroGrd);
        particleDensityGradPtr->position = -gposcGrd;
        addMatrixVectorVjp(
            particleDensityGradPtr->inverseFactor, gposc, totalGroGrd);
        addMatrixVectorVjp(
            particleDensityGradPtr->inverseFactor, rayDirection, grduGrd);

        // factorDirection=B^T*d: row i receives d_i*outputGradient.
        addTransposeMatrixVectorVjp(
            particleDensityGradPtr->factor, rayDirection, factorDirectionGrd);

        // The quaternion/scale slots are unused packing fields in the full-shape layout.
        particleDensityGradPtr->quaternion = make_float4(0.0f);
        particleDensityGradPtr->scale = make_float3(0.0f);

        transmittance = nextTransmit;
    }
}

} // namespace threedgut
