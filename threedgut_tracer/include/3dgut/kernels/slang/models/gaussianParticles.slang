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

#include <3dgut/kernels/slang/common/transforms.slang>

namespace gaussianParticle {

struct RawParameters : IDifferentiable {
    float3 position;
    float density;
    float4 quaternion;
    float3 scale;
    float padding;
    float3x3 factor;
    float3x3 inverseFactor;
    float2 fullShapePadding;
};

struct RawParametersBuffer {
    Ptr<RawParameters, Access.Read> _dataPtr;
    Ptr<RawParameters> _gradPtr;
    bool exclusiveGradient; //< true if the gradient maybe updated without atomics
};

struct CommonParameters {
    RawParametersBuffer parametersBuffer;
};

[BackwardDifferentiable][ForceInline] RawParameters fetchParametersFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    RawParameters rawParameters = parametersBuffer._dataPtr[particleIdx];
    if (Surfel) {
    }
    return rawParameters;
}

[BackwardDerivativeOf(fetchParametersFromBuffer)][ForceInline] void fetchParametersFromBufferBwd(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    RawParameters rawParameters) {
    if (parametersBuffer.exclusiveGradient) {
        parametersBuffer._gradPtr[particleIdx].density += rawParameters.density;
        parametersBuffer._gradPtr[particleIdx].position += rawParameters.position;
        parametersBuffer._gradPtr[particleIdx].quaternion += rawParameters.quaternion;
        if (!Surfel) {
            parametersBuffer._gradPtr[particleIdx].scale += rawParameters.scale;
        } else {
            parametersBuffer._gradPtr[particleIdx].scale.xy += rawParameters.scale.xy;
        }
        parametersBuffer._gradPtr[particleIdx].factor += rawParameters.factor;
        parametersBuffer._gradPtr[particleIdx].inverseFactor += rawParameters.inverseFactor;
    } else {
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].density, rawParameters.density);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.x, rawParameters.position.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.y, rawParameters.position.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].position.z, rawParameters.position.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.x, rawParameters.quaternion.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.y, rawParameters.quaternion.y);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.z, rawParameters.quaternion.z);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].quaternion.w, rawParameters.quaternion.w);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.x, rawParameters.scale.x);
        InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.y, rawParameters.scale.y);
        if (!Surfel) {
            InterlockedAdd(parametersBuffer._gradPtr[particleIdx].scale.z, rawParameters.scale.z);
        }
        [ForceUnroll]
        for (int row = 0; row < 3; ++row) {
            [ForceUnroll]
            for (int col = 0; col < 3; ++col) {
                InterlockedAdd(parametersBuffer._gradPtr[particleIdx].factor[row][col], rawParameters.factor[row][col]);
                InterlockedAdd(parametersBuffer._gradPtr[particleIdx].inverseFactor[row][col], rawParameters.inverseFactor[row][col]);
            }
        }
    }

}

struct Parameters : IDifferentiable {
    float3 position;
    float3x3 factor;
    float3x3 inverseFactor;
    float density;
};

[BackwardDifferentiable][ForceInline] Parameters fetchParameters(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer) {
    const RawParameters rawParameters = fetchParametersFromBuffer(particleIdx, parametersBuffer);
    return {
        rawParameters.position,
        rawParameters.factor,
        rawParameters.inverseFactor,
        rawParameters.density};
}
[BackwardDifferentiable][ForceInline] void cannonicalRay(
    in float3 rayOrigin,
    in float3 rayDirection,
    in Parameters parameters,
    out float3 particleRayOrigin,
    out float3 particleRayDirection, ) {
    const float3 gposc  = (rayOrigin - parameters.position);
    particleRayOrigin = mul(parameters.inverseFactor, gposc);
    const float3 grdu = mul(parameters.inverseFactor, rayDirection);
    particleRayDirection = normalize(grdu);
}

// Exact closest point on the renderer's finite ray interval. Bounds are
// renderer state (no_diff); Slang generates the piecewise VJP through clamp.
[BackwardDifferentiable][ForceInline] void canonicalIntervalClosestPoint(
    float3 canonicalRayOrigin,
    float3 canonicalUnnormalizedRayDirection,
    no_diff float minHitDistance,
    no_diff float maxHitDistance,
    out float tauLine,
    out float tau,
    out float3 canonicalIntersection) {
    const float denominator = dot(canonicalUnnormalizedRayDirection,
                                  canonicalUnnormalizedRayDirection);
    tauLine = -dot(canonicalRayOrigin, canonicalUnnormalizedRayDirection) / denominator;
    tau = clamp(tauLine, minHitDistance, maxHitDistance);
    canonicalIntersection = canonicalRayOrigin + tau * canonicalUnnormalizedRayDirection;
}

[BackwardDifferentiable][ForceInline] float canonicalPointKernelResponse<let KernelDegree : int>(
    float3 canonicalPoint) {
    const float grayDist = dot(canonicalPoint, canonicalPoint);
    switch (KernelDegree) {
    case 8: {
        float s = -0.000685871056241;
        const float grayDistSq = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq);
    }
    case 5: {
        float s = -0.0185185185185;
        return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4: {
        float s = -0.0555555555556;
        return exp(s * grayDist * grayDist);
    }
    case 3: {
        float s = -0.166666666667;
        return exp(s * grayDist * sqrt(grayDist));
    }
    case 1: {
        float s = -1.5f;
        return exp(s * sqrt(grayDist));
    }
    case 0: {
        float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);
    }
    default: {
        float s = -0.5f;
        return exp(s * grayDist);
    }
    }
}

[BackwardDifferentiable] [ForceInline] float canonicalRayMinSquaredDistance<let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection)
{
    // NB : function is called only for intersecting rays
    float3 gcrod;
    if (Surfel) {
        // if ray z direction is null, the ray origin is on the plane z = 0
        gcrod = abs(canonicalRayDirection.z) > 1e-06 ? canonicalRayOrigin + canonicalRayDirection * -canonicalRayOrigin.z / canonicalRayDirection.z : canonicalRayOrigin;
    } else {
        gcrod = cross(canonicalRayDirection, canonicalRayOrigin);
    }
    return dot(gcrod, gcrod);
}

[BackwardDifferentiable][ForceInline] float canonicalRayMaxKernelResponse<let KernelDegree : int, let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection) {
    const float grayDist = canonicalRayMinSquaredDistance<Surfel>(canonicalRayOrigin, canonicalRayDirection);

    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (KernelDegree) {
    case 8: // Zenzizenzizenzic
    {
        /*static const*/ float s = -0.000685871056241;
        const float grayDistSq   = grayDist * grayDist;
        return exp(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        /*static const*/ float s = -0.0185185185185;
        return exp(s * grayDist * grayDist * sqrt(grayDist));
    }
    case 4: // Tesseractic
    {
        /*static const*/ float s = -0.0555555555556;
        return exp(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        /*static const*/ float s = -0.166666666667;
        return exp(s * grayDist * sqrt(grayDist));
    }
    case 1: // Laplacian
    {
        /*static const*/ float s = -1.5f;
        return exp(s * sqrt(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return max(1 + s * sqrt(grayDist), 0.f);
    }
    default: // Quadratic
    {
        /*static const*/ float s = -0.5f;
        return exp(s * grayDist);
    }
    }
}

[BackwardDifferentiable][ForceInline] float canonicalRayDistance(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale) {
    const float3 grds = scale * canonicalRayDirection * dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    return sqrt(dot(grds, grds));
}

[BackwardDifferentiable][ForceInline] float canonicalRayIntersection(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 scale,
    out float3 canonicalIntersection) {
    const float3 canonicalGrds = canonicalRayDirection * dot(canonicalRayDirection, -1 * canonicalRayOrigin);
    canonicalIntersection = canonicalRayOrigin + canonicalGrds;
    const float3 grds = scale * canonicalGrds;
    return sqrt(dot(grds, grds));
}

[BackwardDifferentiable][ForceInline] float3 canonicalRayNormal<let Surfel : bool>(
    float3 canonicalRayOrigin,
    float3 canonicalRayDirection,
    float3 canonicalIntersection,
    Parameters parameters) {

    // Exact GOF intersection-plane normal for a full covariance.  Since
    // canonicalRayDirection = normalize(S^-1 d), this is the direct full-S
    // generalization of -normalize(R diag(s^-1) canonicalRayDirection):
    // -normalize(S^-T canonicalRayDirection) = -normalize(Sigma_f^-1 d).
    return -normalize(mul(transpose(parameters.inverseFactor), canonicalRayDirection));
}

[BackwardDifferentiable][ForceInline]
bool hit(
    float3 rayOrigin,
    float3 rayDirection,
    Parameters parameters,
    out float alpha,
    inout float depth,
    out float3 canonicalIntersection,
    no_diff bool enableNormal,
    inout float3 normal) {

    float3 canonicalRayOrigin;
    float3 canonicalRayDirection;
    cannonicalRay(
        rayOrigin,
        rayDirection,
        parameters,
        canonicalRayOrigin,
        canonicalRayDirection);

    const float maxResponse = canonicalRayMaxKernelResponse<KernelDegree, Surfel>(
        canonicalRayOrigin,
        canonicalRayDirection);

    alpha = min(MaxParticleAlpha, maxResponse * parameters.density); // needed for the stability of the backward pass
    const bool acceptHit = ((maxResponse > MinParticleKernelDensity) && (alpha > MinParticleAlpha));
    if (acceptHit)
    {
        const float3 canonicalUnnormalizedDirection = mul(parameters.inverseFactor, rayDirection);
        depth = -dot(canonicalRayOrigin, canonicalUnnormalizedDirection) /
                dot(canonicalUnnormalizedDirection, canonicalUnnormalizedDirection);
        canonicalIntersection = canonicalRayOrigin + depth * canonicalUnnormalizedDirection;
        if (enableNormal)
        {
            normal = canonicalRayNormal<Surfel>(canonicalRayOrigin, canonicalRayDirection,
                                                canonicalIntersection, parameters);
        }
    }
    return acceptHit;
}

[BackwardDifferentiable][ForceInline]
bool hitInterval(
    float3 rayOrigin,
    float3 rayDirection,
    Parameters parameters,
    no_diff float minHitDistance,
    no_diff float maxHitDistance,
    out float alpha,
    out float depth,
    out float3 canonicalIntersection,
    no_diff bool enableNormal,
    out float3 normal) {
    const float3 canonicalRayOrigin =
        mul(parameters.inverseFactor, rayOrigin - parameters.position);
    const float3 canonicalRayDirection =
        mul(parameters.inverseFactor, rayDirection);
    float tauLine;
    canonicalIntervalClosestPoint(canonicalRayOrigin, canonicalRayDirection,
                                  minHitDistance, maxHitDistance,
                                  tauLine, depth, canonicalIntersection);
    const float response = canonicalPointKernelResponse<KernelDegree>(canonicalIntersection);
    alpha = min(MaxParticleAlpha, response * parameters.density);
    const bool acceptHit = maxHitDistance > minHitDistance &&
        response > MinParticleKernelDensity && alpha > MinParticleAlpha;
    if (acceptHit && enableNormal) {
        normal = canonicalRayNormal<Surfel>(canonicalRayOrigin,
                                            normalize(canonicalRayDirection),
                                            canonicalIntersection,
                                            parameters);
    }
    return acceptHit;
}

[BackwardDifferentiable][ForceInline]
float integrateHit<let backToFront : bool>(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    no_diff bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
   const float weight = backToFront ? alpha : alpha * transmittance;
   if (backToFront)
   {
       integratedDepth = lerp(integratedDepth, depth, alpha);
       if (enableNormal)
       {
           integratedNormal = lerp(integratedNormal, normal, alpha);
       }
   }
   else
   {
       integratedDepth += depth * weight;
       if (enableNormal)
       {
            integratedNormal += normal * weight;
       }
   }

   transmittance *= (1 - alpha);
   return weight;
}

[BackwardDifferentiable][ForceInline]
float processHitFromBuffer<let backToFront : bool>(
    no_diff float3 rayOrigin,
    no_diff float3 rayDirection,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    inout float transmittance,
    inout float integratedDepth,
    out float3 canonicalIntersection,
    no_diff bool enableNormal,
    inout float3 integratedNormal)
{
    float alpha = 0.0f;
    float depth;
    float3 normal;
    if (hit(rayOrigin,
            rayDirection,
            fetchParameters(particleIdx, parametersBuffer),
            alpha,
            depth,
            canonicalIntersection,
            enableNormal,
            normal))
    {
        return integrateHit<backToFront>(alpha,
                                         transmittance,
                                         depth,
                                         integratedDepth,
                                         enableNormal,
                                         normal,
                                         integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable][ForceInline]
float processHitFromBufferInterval<let backToFront : bool>(
    no_diff float3 rayOrigin,
    no_diff float3 rayDirection,
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float minHitDistance,
    no_diff float maxHitDistance,
    inout float transmittance,
    inout float integratedDepth,
    out float3 canonicalIntersection,
    no_diff bool enableNormal,
    inout float3 integratedNormal) {
    float alpha = 0.0f;
    float depth = 0.0f;
    float3 normal = float3(0.0f);
    if (hitInterval(rayOrigin, rayDirection,
                    fetchParameters(particleIdx, parametersBuffer),
                    minHitDistance, maxHitDistance,
                    alpha, depth, canonicalIntersection, enableNormal, normal)) {
        return integrateHit<backToFront>(alpha, transmittance, depth,
                                         integratedDepth, enableNormal, normal,
                                         integratedNormal);
    }
    return 0.0f;
}

[BackwardDifferentiable][ForceInline] float3x3 computeCovariance(
    in Parameters parameters) {
    return mul(parameters.factor, transpose(parameters.factor));
}

[BackwardDifferentiable] [ForceInline]
float3 incidentDirectionFromParameters(
    Parameters parameters,
    no_diff float3 sourcePosition
)
{
    return normalize(parameters.position - sourcePosition);
}

[BackwardDifferentiable][ForceInline]
float3 incidentDirectionFromBuffer(
    no_diff uint32_t particleIdx,
    no_diff RawParametersBuffer parametersBuffer,
    no_diff float3 sourcePosition
)
{
    return incidentDirectionFromParameters(
        fetchParameters(particleIdx, parametersBuffer),
        sourcePosition
    );
}
} // namespace gaussianParticle

// ------------------------------------------------------------------------------------------------------------------
// Entry points

[CudaDeviceExport]
inline gaussianParticle.Parameters particleDensityParameters(
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters)
{
    return gaussianParticle.fetchParameters(
        particleIdx,
        commonParameters.parametersBuffer);
}

[CudaDeviceExport]
inline bool particleDensityHit(
    float3 rayOrigin,
    float3 rayDirection,
    gaussianParticle.Parameters parameters,
    out float alpha,
    out float depth,
    out float3 canonicalIntersection,
    bool enableNormal,
    out float3 normal)
{
    return gaussianParticle.hit(rayOrigin,
                                rayDirection,
                                parameters,
                                alpha,
                                depth,
                                canonicalIntersection,
                                enableNormal,
                                normal);
}

[CudaDeviceExport]
inline bool particleDensityHitInterval(
    float3 rayOrigin,
    float3 rayDirection,
    gaussianParticle.Parameters parameters,
    float minHitDistance,
    float maxHitDistance,
    out float alpha,
    out float depth,
    out float3 canonicalIntersection,
    bool enableNormal,
    out float3 normal) {
    return gaussianParticle.hitInterval(rayOrigin, rayDirection, parameters,
                                        minHitDistance, maxHitDistance,
                                        alpha, depth, canonicalIntersection,
                                        enableNormal, normal);
}

[CudaDeviceExport] inline float particleDensityIntegrateHit(
    in float alpha,
    inout float transmittance,
    in float depth,
    inout float integratedDepth,
    in bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal)
{
    return gaussianParticle.integrateHit<false>(
        alpha,
        transmittance,
        depth,
        integratedDepth,
        enableNormal,
        normal,
        integratedNormal);
}

[CudaDeviceExport]
inline float particleDensityProcessHitFwdFromBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    inout float transmittance,
    inout float integratedDepth,
    out float3 canonicalIntersection,
    in bool enableNormal,
    inout float3 integratedNormal)
{
    return gaussianParticle.processHitFromBuffer<false>(
        rayOrigin,
        rayDirection,
        particleIdx,
        commonParameters.parametersBuffer,
        transmittance,
        integratedDepth,
        canonicalIntersection,
        enableNormal,
        integratedNormal);
}

[CudaDeviceExport]
inline float particleDensityProcessHitFwdFromBufferInterval(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    inout float transmittance,
    inout float integratedDepth,
    out float3 canonicalIntersection,
    bool enableNormal,
    inout float3 integratedNormal) {
    return gaussianParticle.processHitFromBufferInterval<false>(
        rayOrigin, rayDirection, particleIdx, commonParameters.parametersBuffer,
        minHitDistance, maxHitDistance, transmittance, integratedDepth,
        canonicalIntersection, enableNormal, integratedNormal);
}

[CudaDeviceExport]
void particleDensityProcessHitBwdToBuffer(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float depth,
    inout float integratedDepth,
    inout float integratedDepthGrad,
    in float3 canonicalIntersectionGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad)
{
    if (alpha > 0.0f)
    {
        const float weight = 1.0f / (1.0f - alpha);

        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff = DifferentialPair<float>(transmittance, transmittanceGrad);

        integratedDepth = (integratedDepth - depth * alpha) * weight;
        DifferentialPair<float> integratedDepthDiff = DifferentialPair<float>(integratedDepth, integratedDepthGrad);

        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal)
        {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff = DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        }
        else
        {
            integratedNormalDiff = DifferentialPair<float3>(float3(0), float3(0));
        }

        bwd_diff(gaussianParticle.processHitFromBuffer<true>)(
            rayOrigin,
            rayDirection,
            particleIdx,
            commonParameters.parametersBuffer,
            transmittanceDiff,
            integratedDepthDiff,
            canonicalIntersectionGrad,
            enableNormal,
            integratedNormalDiff,
            alphaGrad);

        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedDepthGrad = integratedDepthDiff.getDifferential();
        if (enableNormal)
        {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }
    }
}

[CudaDeviceExport]
void particleDensityProcessHitBwdToBufferInterval(
    float3 rayOrigin,
    float3 rayDirection,
    uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    in float alpha,
    in float alphaGrad,
    inout float transmittance,
    inout float transmittanceGrad,
    in float depth,
    inout float integratedDepth,
    inout float integratedDepthGrad,
    in float3 canonicalIntersectionGrad,
    bool enableNormal,
    in float3 normal,
    inout float3 integratedNormal,
    inout float3 integratedNormalGrad) {
    if (alpha > 0.0f) {
        const float weight = 1.0f / (1.0f - alpha);
        transmittance *= weight;
        DifferentialPair<float> transmittanceDiff =
            DifferentialPair<float>(transmittance, transmittanceGrad);
        integratedDepth = (integratedDepth - depth * alpha) * weight;
        DifferentialPair<float> integratedDepthDiff =
            DifferentialPair<float>(integratedDepth, integratedDepthGrad);
        DifferentialPair<float3> integratedNormalDiff;
        if (enableNormal) {
            integratedNormal = (integratedNormal - normal * alpha) * weight;
            integratedNormalDiff =
                DifferentialPair<float3>(integratedNormal, integratedNormalGrad);
        } else {
            integratedNormalDiff =
                DifferentialPair<float3>(float3(0), float3(0));
        }
        bwd_diff(gaussianParticle.processHitFromBufferInterval<true>)(
            rayOrigin, rayDirection, particleIdx,
            commonParameters.parametersBuffer,
            minHitDistance, maxHitDistance,
            transmittanceDiff, integratedDepthDiff,
            canonicalIntersectionGrad, enableNormal,
            integratedNormalDiff, alphaGrad);
        transmittanceGrad = transmittanceDiff.getDifferential();
        integratedDepthGrad = integratedDepthDiff.getDifferential();
        if (enableNormal) {
            integratedNormalGrad = integratedNormalDiff.getDifferential();
        }
    }
}

[CudaDeviceExport]
bool particleDensityHitCustom(
    float3 rayOrigin,
    float3 rayDirection,
    int32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    gaussianParticle.Parameters parameters = gaussianParticle.fetchParameters(particleIdx, commonParameters.parametersBuffer);

    const float3 canonicalRayOrigin =
        mul(parameters.inverseFactor, rayOrigin - parameters.position);
    const float3 canonicalRayDirection =
        mul(parameters.inverseFactor, rayDirection);
    float tauLine;
    float3 canonicalIntersection;
    gaussianParticle.canonicalIntervalClosestPoint(
        canonicalRayOrigin, canonicalRayDirection,
        minHitDistance, maxHitDistance,
        tauLine, hitDistance, canonicalIntersection);
    return maxHitDistance > minHitDistance &&
           dot(canonicalIntersection, canonicalIntersection) < maxParticleSquaredDistance;
}

[CudaDeviceExport]
bool particleDensityHitInstance(
    float3 canonicalRayOrigin,
    float3 canonicalUnormalizedRayDirection,
    float minHitDistance,
    float maxHitDistance,
    float maxParticleSquaredDistance,
    out float hitDistance
)
{
    float tauLine;
    float3 canonicalIntersection;
    gaussianParticle.canonicalIntervalClosestPoint(
        canonicalRayOrigin, canonicalUnormalizedRayDirection,
        minHitDistance, maxHitDistance,
        tauLine, hitDistance, canonicalIntersection);
    return maxHitDistance > minHitDistance &&
           dot(canonicalIntersection, canonicalIntersection) < maxParticleSquaredDistance;
}

[CudaDeviceExport] float3 particleDensityIncidentDirection(
    in gaussianParticle.Parameters parameters,
    in float3 sourcePosition
)
{
    return gaussianParticle.incidentDirectionFromParameters(parameters, sourcePosition);
}

[CudaDeviceExport] void particleDensityIncidentDirectionBwdToBuffer(
    in uint32_t particleIdx,
    gaussianParticle.CommonParameters commonParameters,
    in float3 sourcePosition,
    in float3 incidentDirectionGrad
)
{
    bwd_diff(gaussianParticle.incidentDirectionFromBuffer)(
        particleIdx,
        commonParameters.parametersBuffer,
        sourcePosition,
        incidentDirectionGrad
    );
}
