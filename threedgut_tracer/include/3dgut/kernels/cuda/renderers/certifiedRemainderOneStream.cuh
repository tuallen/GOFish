// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderAdmission.cuh>

namespace gofish_certified_remainder {

static inline __device__ void hardIntegrityFailure(
    uint32_t* __restrict__ integrityError) {
    if (integrityError != nullptr) atomicExch(integrityError, 1u);
}

static inline __device__ bool recordIntegrityValid(
    const SharedCountEmitRecord& record) {
    return record.certified == 1u
        && record.providerContractMarker == kProviderContractMarker
        && record.depthKeyContractMarker == kDepthKeyContractMarker
        && (record.decision.route != AdmissionRoute::ExactCone
            || record.exactConeCardinalityReserved == 1u);
}

// Shared count/emit/backward call site. This kernel must be launched over
// every stored Gaussian before any UT projection call or return. InputProvider
// is intrinsic-only: it must not consume UT margin, sigma validity,
// validProjection, conic/extent, or the native bbox/count.
// InputProvider must compute the exact
// calibrated pi(mean) enclosure, co-located J interval, matched effective
// shape, and theta proof for one particle. Count persists the sole route record;
// emit/backward never recompute or change it.
template <typename InputProvider, typename CountEllipse,
          typename CountRectangle, typename CountExactCone>
__global__ void countCertifiedRemainderOneStream(
    uint32_t numParticles,
    CameraThetaBandHessianTable table,
    InputProvider inputProvider,
    CountEllipse countEllipse,
    CountRectangle countRectangle,
    CountExactCone countExactCone,
    SharedCountEmitRecord* __restrict__ records,
    uint32_t* __restrict__ integrityError) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    AdmissionInput input{};
    uint32_t existingParticleIndex = particle;
    ProviderContractMarker providerContract{};
    const InputProviderStatus providerStatus = inputProvider(
        particle, input, existingParticleIndex, providerContract);
    SharedCountEmitRecord record{};
    record.decision.route = AdmissionRoute::ExactCone;
    record.particleIndex = existingParticleIndex;
    record.providerContractMarker = providerContract.providerContractMarker;
    record.depthKeyContractMarker = providerContract.depthKeyContractMarker;
    record.exactConeCardinalityReserved =
        providerContract.exactConeCardinalityReserved;
    record.certified = 0u;
    const bool tableInitialized = table.initializedCertified == 1u
        && table.view.header != nullptr && table.view.cameras != nullptr
        && table.view.bands != nullptr && table.camera != nullptr;
    const bool providerContractValid = providerContract.certified == 1u
        && providerContract.providerContractMarker == kProviderContractMarker
        && providerContract.depthKeyContractMarker == kDepthKeyContractMarker;
    if (!tableInitialized || !providerContractValid
        || providerStatus == InputProviderStatus::NonfiniteGaussian) {
        records[particle] = record;
        hardIntegrityFailure(integrityError);
        return;
    }
    if (providerStatus == InputProviderStatus::FinitePrepared
               && tableInitialized) {
        record = prepareSharedRecord(
            table, input, existingParticleIndex);
        record.providerContractMarker = providerContract.providerContractMarker;
        record.depthKeyContractMarker = providerContract.depthKeyContractMarker;
        record.exactConeCardinalityReserved =
            providerContract.exactConeCardinalityReserved;
    } else if (providerStatus == InputProviderStatus::FiniteExactCone) {
        record.decision.route = AdmissionRoute::ExactCone;
    }
    if (record.decision.route == AdmissionRoute::ExactCone
        && record.exactConeCardinalityReserved != 1u) {
        record.certified = 0u;
        records[particle] = record;
        hardIntegrityFailure(integrityError);
        return;
    }
    record.certified = 1u;
    records[particle] = record;
    if (record.decision.route == AdmissionRoute::CertifiedRectangle) {
        countRectangle(particle, record.decision.rectangle);
    } else if (record.decision.route == AdmissionRoute::CertifiedEllipse) {
        countEllipse(particle, record.decision.ellipse);
    } else if (record.decision.route == AdmissionRoute::ExactCone) {
        countExactCone(particle);
    }
}

template <typename EmitEllipse, typename EmitRectangle,
          typename EmitExactCone>
__global__ void emitCertifiedRemainderOneStream(
    uint32_t numParticles,
    const SharedCountEmitRecord* __restrict__ records,
    EmitEllipse emitEllipse,
    EmitRectangle emitRectangle,
    EmitExactCone emitExactCone,
    uint32_t* __restrict__ integrityError) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    const SharedCountEmitRecord record = records[particle];
    if (!recordIntegrityValid(record)) {
        hardIntegrityFailure(integrityError);
        return;
    }
    if (record.decision.route == AdmissionRoute::CertifiedRectangle) {
        emitRectangle(particle, record.particleIndex,
                      record.decision.rectangle);
    } else if (record.decision.route == AdmissionRoute::CertifiedEllipse) {
        emitEllipse(particle, record.particleIndex,
                    record.decision.ellipse);
    } else if (record.decision.route == AdmissionRoute::ExactCone) {
        emitExactCone(particle, record.particleIndex);
    }
}

template <typename BackwardEllipse, typename BackwardRectangle,
          typename BackwardExactCone>
__global__ void backwardCertifiedRemainderOneStream(
    uint32_t numParticles,
    const SharedCountEmitRecord* __restrict__ records,
    BackwardEllipse backwardEllipse,
    BackwardRectangle backwardRectangle,
    BackwardExactCone backwardExactCone,
    uint32_t* __restrict__ integrityError) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    const SharedCountEmitRecord record = records[particle];
    if (!recordIntegrityValid(record)) {
        hardIntegrityFailure(integrityError);
        return;
    }
    if (record.decision.route == AdmissionRoute::CertifiedRectangle) {
        backwardRectangle(particle, record.particleIndex,
                          record.decision.rectangle);
    } else if (record.decision.route == AdmissionRoute::CertifiedEllipse) {
        backwardEllipse(particle, record.particleIndex,
                        record.decision.ellipse);
    } else if (record.decision.route == AdmissionRoute::ExactCone) {
        backwardExactCone(particle, record.particleIndex);
    }
}

} // namespace gofish_certified_remainder

