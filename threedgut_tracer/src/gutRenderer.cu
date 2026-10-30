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

#include <math_constants.h>
#include <3dgut/renderer/renderParameters.h>
#include <3dgut/threedgut.cuh>

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/kernels/cuda/renderers/tileCapDirectCountEmit.cuh>
#include <3dgut/kernels/cuda/renderers/lazyExactConeWarp.cuh>
#include <3dgut/kernels/cuda/renderers/boundedBatchExactCone.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderAdmission.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderTableOwner.cuh>
#include <3dgut/sensors/sensors.h>

#include <tiny-cuda-nn/common_host.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <limits>
#include <vector>
#include <algorithm>
#include <3dgut/kernels/cuda/renderers/admissionPredicates.h>

using namespace tcnn;

namespace {

using namespace threedgut;

constexpr int featuresDim() {
    return model_ExternalParams::RayFeatureDim;
}

__global__ void collectSofMaxRadii2DKernel(
    uint32_t numParticles,
    const float* __restrict__ projectedRadius,
    const uint32_t* __restrict__ finalTileCount,
    float* __restrict__ output) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    output[particle] = finalTileCount[particle] > 0u
        ? projectedRadius[particle] : 0.0f;
}

constexpr uint8_t kSofRelationOutside = static_cast<uint8_t>(
    gofish_flat_cone_k_test::CameraRelation::Outside);
constexpr uint8_t kSofRelationInside = static_cast<uint8_t>(
    gofish_flat_cone_k_test::CameraRelation::Inside);
constexpr uint8_t kSofRelationUncertain = static_cast<uint8_t>(
    gofish_flat_cone_k_test::CameraRelation::Uncertain);
constexpr uint8_t kSofRelationNotEvaluated = 3u;

__global__ void collectSofScreenPruneStateKernel(
    uint32_t numParticles,
    const float* __restrict__ projectedRadius,
    const uint8_t* __restrict__ supportProvenGt20,
    const uint32_t* __restrict__ finalTileCount,
    const gofish_certified_remainder::SharedCountEmitRecord*
        __restrict__ certifiedRecords,
    const gofish_certified_remainder::ExactConeState*
        __restrict__ certifiedExactStates,
    uint8_t* __restrict__ outputProvenGt20,
    uint8_t* __restrict__ outputCameraRelation) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    uint8_t relation = kSofRelationNotEvaluated;
    uint8_t proven = 0u;
    const bool visible = finalTileCount[particle] > 0u;
    const auto& record = certifiedRecords[particle];
    if (visible && record.certified == 1u
        && record.decision.route
            == gofish_certified_remainder::AdmissionRoute::ExactCone) {
        const auto& state = certifiedExactStates[particle];
        if (state.prepared == 1u
            && state.cameraRelation <= uint32_t(kSofRelationUncertain)) {
            relation = static_cast<uint8_t>(state.cameraRelation);
            // A proven camera-inside relation is independently sufficient.
            // Otherwise use the calibrated support-diameter witness only when
            // the native conic is invalid. Uncertain without that independent
            // proof remains retained.
            proven = relation == kSofRelationInside
                || (!(projectedRadius[particle] > 0.0f)
                    && supportProvenGt20[particle] != 0u)
                ? 1u : 0u;
        }
    }
    outputProvenGt20[particle] = proven;
    outputCameraRelation[particle] = relation;
}

// identify tiles start/end indices in the sorted tile/depth keys buffer
__global__ void computeSortedTileRangeIndices(
    int numKeys,
    const uint64_t* __restrict__ sortedTileDepthKeys,
    uvec2* __restrict__ tileRangeIndices) {

    const int keyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (keyIdx >= numKeys) {
        return;
    }

    const uint32_t tileIdx = sortedTileDepthKeys[keyIdx] >> 32;
    const bool validTile   = tileIdx != GUTParameters::Tiling::InvalidTileIdx;
    if (keyIdx == 0) {
        if (validTile) {
            tileRangeIndices[tileIdx].x = keyIdx;
        }
    } else {
        const uint32_t prevKeyTileIdx = sortedTileDepthKeys[keyIdx - 1] >> 32;
        if (prevKeyTileIdx != tileIdx) {
            if (prevKeyTileIdx != GUTParameters::Tiling::InvalidTileIdx) {
                tileRangeIndices[prevKeyTileIdx].y = keyIdx;
            }
            if (validTile) {
                tileRangeIndices[tileIdx].x = keyIdx;
            }
        }
    }
    if (validTile && (keyIdx == numKeys - 1)) {
        tileRangeIndices[tileIdx].y = numKeys;
    }
}

inline uint32_t higherMsb(uint32_t n) {
    uint32_t msb  = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (n >> msb) {
        msb++;
    }
    return msb;
}

constexpr uint64_t kCertifiedCandidateMinimumHeadroom = 4096u;

struct CertifiedCandidateCapacityPlan {
    uint64_t exactCount;
    uint64_t headroomCount;
    uint64_t highWaterCount;
    std::size_t keyBytes;
    std::size_t idBytes;
    uint32_t cubItems;
};

inline bool checkedCertifiedCandidateCapacity(
    uint64_t exactCount, CertifiedCandidateCapacityPlan& output) {
    // CUB and range kernels in this codebase consume int/uint32 item counts.
    // Reject before narrowing; never wrap or saturate.
    if (exactCount > uint64_t(std::numeric_limits<int>::max())) return false;
    uint64_t headroom = exactCount / 8u;
    if (headroom < kCertifiedCandidateMinimumHeadroom)
        headroom = kCertifiedCandidateMinimumHeadroom;
    if (exactCount > std::numeric_limits<uint64_t>::max() - headroom)
        return false;
    const uint64_t highWater = exactCount + headroom;
    const uint64_t sizeMax = uint64_t(std::numeric_limits<std::size_t>::max());
    if (highWater > sizeMax / sizeof(uint64_t)
        || highWater > sizeMax / sizeof(uint32_t)) return false;
    output = CertifiedCandidateCapacityPlan{
        exactCount, headroom, highWater,
        std::size_t(highWater * sizeof(uint64_t)),
        std::size_t(highWater * sizeof(uint32_t)),
        static_cast<uint32_t>(exactCount)};
    return true;
}

__global__ void widenCertifiedCandidateCounts(
    uint32_t numParticles,
    const uint32_t* __restrict__ counts,
    uint64_t* __restrict__ wideCounts) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < numParticles) wideCounts[particle] = uint64_t(counts[particle]);
}

__global__ void finalizeCertifiedVisibility(
    uint32_t numParticles,
    const uint32_t* __restrict__ counts,
    int* __restrict__ visibility) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle < numParticles) visibility[particle] = counts[particle] ? 1 : 0;
}

__global__ void preserveExactRouteOriginalCounts(
    uint32_t numParticles,
    const gofish_certified_remainder::SharedCountEmitRecord* __restrict__ records,
    const uint32_t* __restrict__ retainedCounts,
    uint32_t* __restrict__ originalCounts) {
    const uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= numParticles) return;
    if (records[particle].decision.route
        == gofish_certified_remainder::AdmissionRoute::ExactCone) {
        // The bounded exact route already evaluates this same FlatConeCap
        // before its K predicate, so post-compaction and direct counts agree.
        originalCounts[particle] = retainedCounts[particle];
    }
}

} // namespace

struct GUTRenderer::GutRenderForwardContext {
    cudaStream_t cudaStream;

    GutRenderForwardContext(cudaStream_t iCudaStream)
        : cudaStream(iCudaStream) {
    }

    ~GutRenderForwardContext() {
        const uint64_t processQueueHandle = reinterpret_cast<uint64_t>(cudaStream);
        Logger logger;
        unsortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileDepthKeys.clear(processQueueHandle, logger);
        sortedTileRangeIndices.clear(processQueueHandle, logger);
        unsortedTileParticleIdx.clear(processQueueHandle, logger);
        sortedTileParticleIdx.clear(processQueueHandle, logger);
        sortingWorkingBuffer.clear(processQueueHandle, logger);
        particlesTilesCount.clear(processQueueHandle, logger);
        particlesTilesOffset.clear(processQueueHandle, logger);
        particlesProjectedPosition.clear(processQueueHandle, logger);
        particlesProjectedConicOpacity.clear(processQueueHandle, logger);
        particlesProjectedExtent.clear(processQueueHandle, logger);
        particlesProjectedRadius.clear(processQueueHandle, logger);
        particlesSupportProvenGt20.clear(processQueueHandle, logger);
        particlesGlobalDepth.clear(processQueueHandle, logger);
        particlesPrecomputedFeatures.clear(processQueueHandle, logger);
        particlesProjectedPositionGradient.clear(processQueueHandle, logger);
        particlesProjectedConicOpacityGradient.clear(processQueueHandle, logger);
        particlesGlobalDepthGradient.clear(processQueueHandle, logger);
        particlesPrecomputedFeaturesGradient.clear(processQueueHandle, logger);
        scanningWorkingBuffer.clear(processQueueHandle, logger);
        certifiedRemainderDeviceTable.clear(processQueueHandle, logger);
        certifiedRemainderRecords.clear(processQueueHandle, logger);
        certifiedRemainderDeviceProviderCamera.clear(processQueueHandle, logger);
        certifiedRemainderExactConeStates.clear(processQueueHandle, logger);
        boundedCapCounts.clear(processQueueHandle, logger);
        boundedChunks.clear(processQueueHandle, logger);
        boundedFlags.clear(processQueueHandle, logger);
        boundedProvisionalKeys.clear(processQueueHandle, logger);
        boundedProvisionalOwners.clear(processQueueHandle, logger);
        boundedSelectedIndices.clear(processQueueHandle, logger);
        boundedCompactKeys.clear(processQueueHandle, logger);
        boundedCompactOwners.clear(processQueueHandle, logger);
        boundedSelectedCount.clear(processQueueHandle, logger);
        boundedSelectionWorkspace.clear(processQueueHandle, logger);
        angularRangeIndexBits.clear(processQueueHandle, logger);
        certifiedRemainderIntegrityError.clear(processQueueHandle, logger);
        certifiedFinalTilesCount.clear(processQueueHandle, logger);
        tileCapRetainedTilesCount.clear(processQueueHandle, logger);
        certifiedFinalTilesCount64.clear(processQueueHandle, logger);
        certifiedFinalTilesOffset.clear(processQueueHandle, logger);
        certifiedFinalScanningWorkingBuffer.clear(processQueueHandle, logger);
        certifiedFinalUnsortedTileDepthKeys.clear(processQueueHandle, logger);
        certifiedFinalSortedTileDepthKeys.clear(processQueueHandle, logger);
        certifiedFinalUnsortedTileParticleIdx.clear(processQueueHandle, logger);
        certifiedFinalSortedTileParticleIdx.clear(processQueueHandle, logger);
        certifiedFinalSortedTileRangeIndices.clear(processQueueHandle, logger);
        certifiedFinalSortingWorkingBuffer.clear(processQueueHandle, logger);
        conservativeConeTree.clear(processQueueHandle, logger);
        conservativeCentralityFlag.clear(processQueueHandle, logger);
        particlesAddedTilesCount.clear(processQueueHandle, logger);
        particlesAddedTilesOffset.clear(processQueueHandle, logger);
        addedScanningWorkingBuffer.clear(processQueueHandle, logger);
        unsortedAddedTileDepthKeys.clear(processQueueHandle, logger);
        sortedAddedTileDepthKeys.clear(processQueueHandle, logger);
        unsortedAddedTileParticleIdx.clear(processQueueHandle, logger);
        sortedAddedTileParticleIdx.clear(processQueueHandle, logger);
        addedSortedTileRangeIndices.clear(processQueueHandle, logger);
        addedSortingWorkingBuffer.clear(processQueueHandle, logger);
        mergedTileCounts.clear(processQueueHandle, logger);
        mergedTileOffsets.clear(processQueueHandle, logger);
        mergedScanningWorkingBuffer.clear(processQueueHandle, logger);
        mergedTileRangeIndices.clear(processQueueHandle, logger);
        mergedTileDepthKeys.clear(processQueueHandle, logger);
        mergedTileParticleIdx.clear(processQueueHandle, logger);
    }

    CudaBuffer unsortedTileDepthKeys;
    CudaBuffer sortedTileDepthKeys;
    CudaBuffer sortedTileRangeIndices;
    CudaBuffer unsortedTileParticleIdx;
    CudaBuffer sortedTileParticleIdx;
    CudaBuffer sortingWorkingBuffer;

    Status updateTileSortingBuffers(const uvec2& tileGrid, int numKeys, cudaStream_t stream, const Logger& logger) {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(stream);
        const bool uptodate        = unsortedTileDepthKeys.size() >= sizeof(uint64_t) * numKeys;
        if (!uptodate) {
            CHECK_STATUS_RETURN(unsortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileDepthKeys.resize(sizeof(uint64_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(unsortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            CHECK_STATUS_RETURN(sortedTileParticleIdx.resize(sizeof(uint32_t) * numKeys, queueHandle, logger));
            size_t sortingWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceRadixSort::SortPairs(
                    nullptr,
                    sortingWorkingBufferSize,
                    static_cast<const uint64_t*>(unsortedTileDepthKeys.data()),
                    static_cast<uint64_t*>(sortedTileDepthKeys.data()),
                    static_cast<const uint32_t*>(unsortedTileParticleIdx.data()),
                    static_cast<uint32_t*>(sortedTileParticleIdx.data()),
                    numKeys,
                    0, 32 + higherMsb(tileGrid.x * tileGrid.y),
                    stream),
                logger);
            CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, queueHandle, logger));
        }
        if (tileGrid.x * tileGrid.y) {
            CHECK_STATUS_RETURN(sortedTileRangeIndices.resize(sizeof(uvec2) * tileGrid.x * tileGrid.y, queueHandle, logger));
            CUDA_CHECK_RETURN(cudaMemsetAsync(sortedTileRangeIndices.data(), 0, tileGrid.x * tileGrid.y * sizeof(uvec2), stream), logger);
        }
        return Status();
    }

    CudaBuffer particlesTilesCount;                            ///< number of intersected tiles per particles uint32_t [Nx1]
    CudaBuffer particlesTilesOffset;                           ///< cumulative sum of particle tiles count uint32_t [Nx1]
    CudaBuffer particlesProjectedPosition;                     ///< projected particles center Nx2
    CudaBuffer particlesProjectedConicOpacity;                 ///< projected particles conic and opacity Nx4
    CudaBuffer particlesProjectedExtent;                       ///< projected particles extent Nx2
    CudaBuffer particlesProjectedRadius;                       ///< SOF native-UT screen radius Nx1
    CudaBuffer particlesSupportProvenGt20;                     ///< exact-support calibrated-pixel proof uint8 Nx1
    CudaBuffer particlesGlobalDepth;                           ///< particles global depth
    CudaBuffer particlesPrecomputedFeatures;                   ///< precomputed particle features float [NxFeaturesDim]
    mutable CudaBuffer particlesProjectedPositionGradient;     ///< projected particles center Nx2
    mutable CudaBuffer particlesProjectedConicOpacityGradient; ///< projected particles conic and opacity Nx4
    mutable CudaBuffer particlesGlobalDepthGradient;           ///< particles global depth
    mutable CudaBuffer particlesPrecomputedFeaturesGradient;   ///< precomputed particle features float [NxFeaturesDim]
    CudaBuffer scanningWorkingBuffer;                          ///< working buffer to compute the cumulative sum of particles/tiles intersections number
    // Persisted across count, emit, forward rendering, and backward replay.
    gofish_certified_remainder::PackedKbTableOwner certifiedRemainderTableOwner;
    gofish_certified_remainder::CameraThetaBandHessianTable certifiedRemainderHostTable{};
    gofish_certified_remainder_provider::SelectedKbCameraDescriptor certifiedRemainderProviderCamera{};
    CudaBuffer certifiedRemainderDeviceProviderCamera;
    CudaBuffer certifiedRemainderExactConeStates;
    CudaBuffer boundedCapCounts;
    CudaBuffer boundedChunks;
    CudaBuffer boundedFlags;
    CudaBuffer boundedProvisionalKeys;
    CudaBuffer boundedProvisionalOwners;
    CudaBuffer boundedSelectedIndices;
    CudaBuffer boundedCompactKeys;
    CudaBuffer boundedCompactOwners;
    CudaBuffer boundedSelectedCount;
    CudaBuffer boundedSelectionWorkspace;
    CudaBuffer angularRangeIndexBits;
    CudaBuffer gofishSplitScratch;  // real-time renderer heavy-tile split
    CudaBuffer certifiedRemainderDeviceTable;
    CudaBuffer certifiedRemainderRecords;
    CudaBuffer certifiedRemainderIntegrityError;
    CudaBuffer certifiedFinalTilesCount;    // checked per-particle uint32
    // Retained count after the cap test, used only for allocation/scan/emit.
    // certifiedFinalTilesCount serves all pre-cap count consumers, including
    // projector backward.
    CudaBuffer tileCapRetainedTilesCount;
    CudaBuffer certifiedFinalTilesCount64;  // exact widened scan input
    CudaBuffer certifiedFinalTilesOffset;   // uint64 inclusive offsets
    CudaBuffer certifiedFinalScanningWorkingBuffer;
    CudaBuffer certifiedFinalUnsortedTileDepthKeys;
    CudaBuffer certifiedFinalSortedTileDepthKeys;
    CudaBuffer certifiedFinalUnsortedTileParticleIdx;
    CudaBuffer certifiedFinalSortedTileParticleIdx;
#if GOFISH_SOF_SCHEDULE
    CudaBuffer sofSchedIds;     // SOF emission schedule of the current forward pass
    CudaBuffer sofSchedCounts;
    uint32_t* sofSchedPtrs[2] = {nullptr, nullptr};
    std::vector<uint32_t> sofPlan;  // {rowBegin, rowEnd, firstIntersection} per tile-row batch
#endif
    CudaBuffer certifiedFinalSortedTileRangeIndices;
    CudaBuffer certifiedFinalSortingWorkingBuffer;

    inline Status updateCertifiedRemainderParticleBuffers(
        uint32_t numParticles, cudaStream_t stream, const Logger& logger) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        CHECK_STATUS_RETURN(certifiedRemainderDeviceTable.resize(
            sizeof(gofish_certified_remainder::CameraThetaBandHessianTable),
            queue, logger));
        CHECK_STATUS_RETURN(certifiedRemainderRecords.resize(
            std::size_t(numParticles)
                * sizeof(gofish_certified_remainder::SharedCountEmitRecord),
            queue, logger));
        CHECK_STATUS_RETURN(certifiedRemainderDeviceProviderCamera.resize(
            sizeof(gofish_certified_remainder_provider::SelectedKbCameraDescriptor),
            queue, logger));
        CHECK_STATUS_RETURN(certifiedRemainderExactConeStates.resize(
            std::size_t(numParticles)
                * sizeof(gofish_certified_remainder::ExactConeState),
            queue, logger));
        CHECK_STATUS_RETURN(boundedCapCounts.enlarge(
            std::size_t(numParticles) * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(boundedSelectedCount.resize(
            sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(certifiedRemainderIntegrityError.resize(
            sizeof(uint32_t), queue, logger));
        const std::size_t count32Bytes =
            std::size_t(numParticles) * sizeof(uint32_t);
        const std::size_t count64Bytes =
            std::size_t(numParticles) * sizeof(uint64_t);
        CHECK_STATUS_RETURN(certifiedFinalTilesCount.enlarge(
            count32Bytes, queue, logger));
        CHECK_STATUS_RETURN(tileCapRetainedTilesCount.enlarge(
            count32Bytes, queue, logger));
        CHECK_STATUS_RETURN(certifiedFinalTilesCount64.enlarge(
            count64Bytes, queue, logger));
        CHECK_STATUS_RETURN(certifiedFinalTilesOffset.enlarge(
            count64Bytes, queue, logger));
        size_t bytes = 0;
        CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
            nullptr, bytes,
            static_cast<const uint64_t*>(certifiedFinalTilesCount64.data()),
            static_cast<uint64_t*>(certifiedFinalTilesOffset.data()),
            numParticles, stream), logger);
        CHECK_STATUS_RETURN(certifiedFinalScanningWorkingBuffer.enlarge(
            bytes, queue, logger));
        return Status();
    }

    inline Status updateBoundedBatchBuffers(
        uint32_t slots, uint32_t chunks, cudaStream_t stream,
        const Logger& logger, uint64_t& reservationBytes) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        CHECK_STATUS_RETURN(boundedChunks.enlarge(
            std::size_t(chunks) * sizeof(gofish_bounded_exact::Chunk),
            queue, logger));
        CHECK_STATUS_RETURN(boundedFlags.enlarge(
            std::size_t(slots) * sizeof(uint8_t), queue, logger));
        CHECK_STATUS_RETURN(boundedProvisionalKeys.enlarge(
            std::size_t(slots) * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(boundedProvisionalOwners.enlarge(
            std::size_t(slots) * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(boundedSelectedIndices.enlarge(
            std::size_t(slots) * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(boundedCompactKeys.enlarge(
            std::size_t(slots) * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(boundedCompactOwners.enlarge(
            std::size_t(slots) * sizeof(uint64_t), queue, logger));
        cub::CountingInputIterator<uint32_t> input(0u);
        size_t bytes = 0u;
        CUDA_CHECK_RETURN(cub::DeviceSelect::Flagged(
            nullptr, bytes, input,
            static_cast<const uint8_t*>(boundedFlags.data()),
            static_cast<uint32_t*>(boundedSelectedIndices.data()),
            static_cast<uint32_t*>(boundedSelectedCount.data()),
            slots, stream), logger);
        CHECK_STATUS_RETURN(boundedSelectionWorkspace.enlarge(
            bytes, queue, logger));
        reservationBytes = boundedChunks.size() + boundedFlags.size()
            + boundedProvisionalKeys.size() + boundedProvisionalOwners.size()
            + boundedSelectedIndices.size() + boundedCompactKeys.size()
            + boundedCompactOwners.size() + boundedSelectedCount.size()
            + boundedSelectionWorkspace.size();
        if (reservationBytes > gofish_bounded_exact::kReservationBytes)
            return Status(ErrorCode::BadInput);
        return Status();
    }

    inline Status updateCertifiedFinalSortingBuffers(
        const uvec2& tileGrid,
        const CertifiedCandidateCapacityPlan& capacity,
        cudaStream_t stream, const Logger& logger) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        const uint64_t tileCount = uint64_t(tileGrid.x) * uint64_t(tileGrid.y);
        if (tileCount > uint64_t(std::numeric_limits<uint32_t>::max())
            || tileCount > uint64_t(std::numeric_limits<std::size_t>::max())
                               / sizeof(uvec2))
            return Status(ErrorCode::BadInput);
        const uint32_t tileCount32 = static_cast<uint32_t>(tileCount);
        CHECK_STATUS_RETURN(certifiedFinalUnsortedTileDepthKeys.enlarge(
            capacity.keyBytes, queue, logger));
        CHECK_STATUS_RETURN(certifiedFinalSortedTileDepthKeys.enlarge(
            capacity.keyBytes, queue, logger));
        CHECK_STATUS_RETURN(certifiedFinalUnsortedTileParticleIdx.enlarge(
            capacity.idBytes, queue, logger));
        CHECK_STATUS_RETURN(certifiedFinalSortedTileParticleIdx.enlarge(
            capacity.idBytes, queue, logger));
        size_t bytes = 0;
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            static_cast<const uint64_t*>(
                certifiedFinalUnsortedTileDepthKeys.data()),
            static_cast<uint64_t*>(certifiedFinalSortedTileDepthKeys.data()),
            static_cast<const uint32_t*>(
                certifiedFinalUnsortedTileParticleIdx.data()),
            static_cast<uint32_t*>(certifiedFinalSortedTileParticleIdx.data()),
            capacity.cubItems, 0,
            32 + higherMsb(tileCount32), stream), logger);
        CHECK_STATUS_RETURN(certifiedFinalSortingWorkingBuffer.enlarge(
            bytes, queue, logger));
        const std::size_t rangeBytes = std::size_t(tileCount * sizeof(uvec2));
        CHECK_STATUS_RETURN(certifiedFinalSortedTileRangeIndices.enlarge(
            rangeBytes, queue, logger));
        if (rangeBytes) CUDA_CHECK_RETURN(cudaMemsetAsync(
            certifiedFinalSortedTileRangeIndices.data(), 0, rangeBytes, stream),
            logger);
        return Status();
    }
    CudaBuffer conservativeConeTree; ///< calibrated angular cone plus certified chord radius
    CudaBuffer conservativeCentralityFlag; ///< actual ray-origin-buffer centrality certificate
#if GOFISH_INFERENCE_ONLY
    // Inference: the centrality certificate depends only on the ray-origin
    // buffer of a camera, which the dataset keeps alive; computed once per buffer.
    struct CentralityEntry {
        const void* origins;
        uint32_t pixels;
        uint32_t value;
    };
    std::vector<CentralityEntry> centralityCache;
#endif
    CudaBuffer particlesAddedTilesCount;                       ///< conservative-only tiles per particle
    CudaBuffer particlesAddedTilesOffset;
    CudaBuffer addedScanningWorkingBuffer;
    CudaBuffer unsortedAddedTileDepthKeys;
    CudaBuffer sortedAddedTileDepthKeys;
    CudaBuffer unsortedAddedTileParticleIdx;
    CudaBuffer sortedAddedTileParticleIdx;
    CudaBuffer addedSortedTileRangeIndices;
    CudaBuffer addedSortingWorkingBuffer;
    CudaBuffer mergedTileCounts;
    CudaBuffer mergedTileOffsets;
    CudaBuffer mergedScanningWorkingBuffer;
    CudaBuffer mergedTileRangeIndices;
    CudaBuffer mergedTileDepthKeys;
    CudaBuffer mergedTileParticleIdx;

    inline Status updateConservativeConeTree(uint32_t side, cudaStream_t stream, const Logger& logger) {
        const size_t bytes = static_cast<size_t>(gofish_conservative_bound::totalNodes(side))
                           * sizeof(gofish_conservative_bound::ConservativeConeNode);
        CHECK_STATUS_RETURN(conservativeConeTree.resize(bytes, reinterpret_cast<uint64_t>(stream), logger));
        CHECK_STATUS_RETURN(conservativeCentralityFlag.resize(
            sizeof(uint32_t), reinterpret_cast<uint64_t>(stream), logger));
        return Status();
    }

    inline Status updateConservativeParticleBuffers(uint32_t numParticles, cudaStream_t stream, const Logger& logger) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        CHECK_STATUS_RETURN(particlesAddedTilesCount.resize(numParticles * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(particlesAddedTilesOffset.resize(numParticles * sizeof(uint32_t), queue, logger));
        size_t bytes = 0;
        CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
            nullptr, bytes,
            static_cast<const uint32_t*>(particlesAddedTilesCount.data()),
            static_cast<uint32_t*>(particlesAddedTilesOffset.data()), numParticles, stream), logger);
        CHECK_STATUS_RETURN(addedScanningWorkingBuffer.resize(bytes, queue, logger));
        return Status();
    }

    inline Status updateAddedSortingBuffers(const uvec2& tileGrid, uint32_t count,
                                            cudaStream_t stream, const Logger& logger) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        CHECK_STATUS_RETURN(unsortedAddedTileDepthKeys.resize(count * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(sortedAddedTileDepthKeys.resize(count * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(unsortedAddedTileParticleIdx.resize(count * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(sortedAddedTileParticleIdx.resize(count * sizeof(uint32_t), queue, logger));
        size_t bytes = 0;
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            static_cast<const uint64_t*>(unsortedAddedTileDepthKeys.data()),
            static_cast<uint64_t*>(sortedAddedTileDepthKeys.data()),
            static_cast<const uint32_t*>(unsortedAddedTileParticleIdx.data()),
            static_cast<uint32_t*>(sortedAddedTileParticleIdx.data()), count,
            0, 32 + higherMsb(tileGrid.x * tileGrid.y), stream), logger);
        CHECK_STATUS_RETURN(addedSortingWorkingBuffer.resize(bytes, queue, logger));
        CHECK_STATUS_RETURN(addedSortedTileRangeIndices.resize(
            tileGrid.x * tileGrid.y * sizeof(uvec2), queue, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(addedSortedTileRangeIndices.data(), 0,
            tileGrid.x * tileGrid.y * sizeof(uvec2), stream), logger);
        return Status();
    }

    inline Status updateMergedBuffers(const uvec2& tileGrid, uint32_t count,
                                      cudaStream_t stream, const Logger& logger) {
        const uint64_t queue = reinterpret_cast<uint64_t>(stream);
        const uint32_t tiles = tileGrid.x * tileGrid.y;
        CHECK_STATUS_RETURN(mergedTileCounts.resize(tiles * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(mergedTileOffsets.resize(tiles * sizeof(uint32_t), queue, logger));
        CHECK_STATUS_RETURN(mergedTileRangeIndices.resize(tiles * sizeof(uvec2), queue, logger));
        CHECK_STATUS_RETURN(mergedTileDepthKeys.resize(count * sizeof(uint64_t), queue, logger));
        CHECK_STATUS_RETURN(mergedTileParticleIdx.resize(count * sizeof(uint32_t), queue, logger));
        size_t bytes = 0;
        CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
            nullptr, bytes,
            static_cast<const uint32_t*>(mergedTileCounts.data()),
            static_cast<uint32_t*>(mergedTileOffsets.data()), tiles, stream), logger);
        CHECK_STATUS_RETURN(mergedScanningWorkingBuffer.resize(bytes, queue, logger));
        return Status();
    }

    inline Status updateParticlesWorkingBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) {
        const bool uptodate = particlesTilesCount.size() >= numParticles * sizeof(uint32_t);
        if (!uptodate) {
            const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);
            CHECK_STATUS_RETURN(particlesTilesCount.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesTilesOffset.resize(numParticles * sizeof(uint32_t), queueHandle, logger));
            size_t scanningWorkingBufferSize = 0;
            CUDA_CHECK_RETURN(
                cub::DeviceScan::InclusiveSum(
                    nullptr,
                    scanningWorkingBufferSize,
                    static_cast<const uint32_t*>(particlesTilesCount.data()),
                    static_cast<uint32_t*>(particlesTilesOffset.data()),
                    numParticles,
                    cudaStream),
                logger);
            CHECK_STATUS_RETURN(scanningWorkingBuffer.resize(scanningWorkingBufferSize, queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedPosition.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedConicOpacity.resize(numParticles * sizeof(vec4), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedExtent.resize(numParticles * sizeof(vec2), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesProjectedRadius.resize(numParticles * sizeof(float), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesSupportProvenGt20.resize(numParticles * sizeof(uint8_t), queueHandle, logger));
            CHECK_STATUS_RETURN(particlesGlobalDepth.resize(numParticles * sizeof(float), queueHandle, logger));
        }
        CHECK_STATUS_RETURN(updateCertifiedRemainderParticleBuffers(
            numParticles, cudaStream, logger));

        return Status();
    }

    inline Status updateParticlesProjectionGradientBuffers(int numParticles, cudaStream_t cudaStream, const Logger& logger) const {
        const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);

        CHECK_STATUS_RETURN(particlesProjectedPositionGradient.enlarge(numParticles * sizeof(vec2), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedPositionGradient.data(), 0, numParticles * sizeof(vec2), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesProjectedConicOpacityGradient.resize(numParticles * sizeof(vec4), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesProjectedConicOpacityGradient.data(), 0, numParticles * sizeof(vec4), cudaStream), logger);

        CHECK_STATUS_RETURN(particlesGlobalDepthGradient.resize(numParticles * sizeof(float), queueHandle, logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesGlobalDepthGradient.data(), 0, numParticles * sizeof(float), cudaStream), logger);

        return Status();
    }

    inline Status
    updateParticlesFeaturesBuffer(int featuresSize, cudaStream_t cudaStream, const Logger& logger) {
        CHECK_STATUS_RETURN(particlesPrecomputedFeatures.enlarge(featuresSize * sizeof(float), reinterpret_cast<uint64_t>(cudaStream), logger));
        return Status();
    }

    inline Status updateParticlesFeaturesGradientBuffer(uint32_t featuresSize, cudaStream_t cudaStream, const Logger& logger) const {
        const size_t newSize = featuresSize * sizeof(float);
        CHECK_STATUS_RETURN(particlesPrecomputedFeaturesGradient.enlarge(newSize, reinterpret_cast<uint64_t>(cudaStream), logger));
        CUDA_CHECK_RETURN(cudaMemsetAsync(particlesPrecomputedFeaturesGradient.data(), 0, newSize, cudaStream), logger);
        return Status();
    }
};

threedgut::GUTRenderer::GUTRenderer(const nlohmann::json& config, const Logger& logger)
    : m_logger(logger) {
}

threedgut::GUTRenderer::~GUTRenderer() {
}

void threedgut::GUTRenderer::setPhysicalTileMask(
    const uint8_t* mask, uint32_t count) {
    m_physicalTileMaskCudaPtr = mask;
    m_physicalTileMaskCount = count;
}

threedgut::Status threedgut::GUTRenderer::renderForward(const RenderParameters& params,
                                                        const vec3* sensorRayOriginCudaPtr,
                                                        const vec3* sensorRayDirectionCudaPtr,
                                                        float* worldHitCountCudaPtr,
                                                        float* worldHitDistanceCudaPtr,
                                                        float* worldNormalCudaPtr,
                                                        float* worldDepthVarCudaPtr,
                                                        float* worldQuantDepthCudaPtr,
                                                        float* worldExtentCudaPtr,
                                                        float* worldOpacityFieldCudaPtr,
                                                        float* worldInvDCudaPtr,
                                                        float* worldInvD2CudaPtr,
                                                        TFeatureDensityElem* featureDensityCudaPtr,
                                                        int* particlesVisibilityCudaPtr,
                                                        Parameters& parameters,
                                                        int cudaDeviceIndex,
                                                        cudaStream_t cudaStream) {

    if (!gofish_admission::globalShutterAccepted(static_cast<int>(params.sensorModel.shutterType), static_cast<int>(threedgut::TSensorModel::GlobalShutter))) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] unconditional support union requires global shutter.");
    }

    if (!m_forwardContext) {
        m_forwardContext = std::make_unique<GutRenderForwardContext>(cudaStream);
    }
    CUDA_CHECK_RETURN(
        m_forwardContext->certifiedRemainderTableOwner.initialize(cudaStream),
        m_logger);
    if (!m_forwardContext->certifiedRemainderTableOwner.bind(
            params.sensorModel,
            static_cast<uint32_t>(params.resolution.x),
            static_cast<uint32_t>(params.resolution.y),
            &m_forwardContext->certifiedRemainderHostTable,
            &m_forwardContext->certifiedRemainderProviderCamera)) {
        const auto& sensor = params.sensorModel;
        if (sensor.modelType == threedgut::TSensorModel::OpenCVFisheyeModel) {
            const auto& p = sensor.ocvFisheyeParams;
            RETURN_ERROR(m_logger, ErrorCode::BadInput,
                         "[GUTRenderer] camera OPENCV_FISHEYE %dx%d fx %a fy %a cx %a cy %a k %a %a %a %a max_angle %a "
                         "is missing from the complete-admission table; add it with: python -m "
                         "gofish.tools.build_calibration_table --camera=OPENCV_FISHEYE,%d,%d,%a,%a,%a,%a,%a,%a,%a,%a,%a",
                         params.resolution.x, params.resolution.y, p.focalLength.x, p.focalLength.y,
                         p.principalPoint.x, p.principalPoint.y, p.radialCoeffs[0], p.radialCoeffs[1],
                         p.radialCoeffs[2], p.radialCoeffs[3], p.maxAngle, params.resolution.x, params.resolution.y,
                         p.focalLength.x, p.focalLength.y, p.principalPoint.x, p.principalPoint.y, p.radialCoeffs[0],
                         p.radialCoeffs[1], p.radialCoeffs[2], p.radialCoeffs[3], p.maxAngle);
        }
        if (sensor.modelType == threedgut::TSensorModel::OpenCVPinholeModel) {
            const auto& p = sensor.ocvPinholeParams;
            RETURN_ERROR(m_logger, ErrorCode::BadInput,
                         "[GUTRenderer] camera PINHOLE %dx%d fx %a fy %a cx %a cy %a is missing from the "
                         "complete-admission table; a distorted pinhole camera must first be undistorted with COLMAP "
                         "(colmap image_undistorter), a distortion-free one added with: python -m "
                         "gofish.tools.build_calibration_table --camera=PINHOLE,%d,%d,%a,%a,%a,%a",
                         params.resolution.x, params.resolution.y, p.focalLength.x, p.focalLength.y,
                         p.principalPoint.x, p.principalPoint.y, params.resolution.x, params.resolution.y,
                         p.focalLength.x, p.focalLength.y, p.principalPoint.x, p.principalPoint.y);
        }
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] only OPENCV_FISHEYE and PINHOLE cameras are supported; undistort other camera "
                     "models with COLMAP (colmap image_undistorter).");
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    if (m_physicalTileMaskCudaPtr == nullptr
        || m_physicalTileMaskCount != tileGrid.x * tileGrid.y) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] registered physical tile mask differs.");
    }
    const uint32_t numParticles = parameters.values.numParticles;
    struct BoundedHostRun {
        std::vector<uint64_t> keys;
        std::vector<uint64_t> owners;
    };
    std::vector<BoundedHostRun> boundedHostRuns;
    uint32_t boundedSingleDeviceCount = 0u;
    const uint32_t certifiedGlobalShutter =
        gofish_admission::globalShutterAccepted(
            static_cast<int>(params.sensorModel.shutterType),
            static_cast<int>(threedgut::TSensorModel::GlobalShutter))
        ? 1u : 0u;
    uint32_t certifiedCentralGlobalShutter = 0u;
    m_lastNumParticles = numParticles;

    // Sensor pose at mid-exposure (one central pose per frame).
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    if (numParticles == 0u) {
        // No count/scan/emit storage exists for an empty certified stream.
        return Status();
    }

    // The conservative angular support proof is for one central camera center.
    // Fail closed rather than silently approximating rolling shutter.
    for (int component = 0; component < 7; ++component) {
        if (params.sensorState.startPose[component] != params.sensorState.endPose[component]) {
            RETURN_ERROR(m_logger, ErrorCode::BadInput,
                         "[GUTRenderer] conservative opacity bound requires exact global-shutter poses.");
        }
    }
    uint32_t conservativeConeSide = 1;
    uint32_t conservativeConeLevels = 0;
    const uint32_t maximumGridSide = max(tileGrid.x, tileGrid.y);
    while (conservativeConeSide < maximumGridSide) {
        conservativeConeSide <<= 1;
        ++conservativeConeLevels;
    }
    if (3 * conservativeConeLevels + 1 > gofish_conservative_bound::kTraversalStack
        || conservativeConeSide > 4096u) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] calibrated tile quadtree exceeds the certified traversal stack.");
    }
    CHECK_STATUS_RETURN(m_forwardContext->updateConservativeConeTree(
        conservativeConeSide, cudaStream, m_logger));
    const uint32_t conservativeNumPixels =
        static_cast<uint32_t>(params.resolution.x * params.resolution.y);
    uint32_t verifiedConservativeCentrality = 0u;
#if GOFISH_INFERENCE_ONLY
    bool centralityCached = false;
    for (const auto& entry : m_forwardContext->centralityCache) {
        if (entry.origins == sensorRayOriginCudaPtr && entry.pixels == conservativeNumPixels) {
            verifiedConservativeCentrality = entry.value;
            centralityCached = true;
            break;
        }
    }
    if (centralityCached) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            m_forwardContext->conservativeCentralityFlag.data(),
            &verifiedConservativeCentrality, sizeof(uint32_t),
            cudaMemcpyHostToDevice, cudaStream), m_logger);
    } else
#endif
    {
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->conservativeCentralityFlag.data(), 0xff,
        sizeof(uint32_t), cudaStream), m_logger);
    gofish_conservative_bound::verifyCentralRayOrigins<<<
        div_round_up(conservativeNumPixels, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            conservativeNumPixels,
            reinterpret_cast<const tcnn::vec3*>(sensorRayOriginCudaPtr),
            static_cast<uint32_t*>(
                m_forwardContext->conservativeCentralityFlag.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        &verifiedConservativeCentrality,
        m_forwardContext->conservativeCentralityFlag.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
#if GOFISH_INFERENCE_ONLY
    m_forwardContext->centralityCache.push_back(
        {sensorRayOriginCudaPtr, conservativeNumPixels, verifiedConservativeCentrality});
#endif
    }
    certifiedCentralGlobalShutter =
        (certifiedGlobalShutter != 0u
         && verifiedConservativeCentrality != 0u) ? 1u : 0u;
    gofish_conservative_bound::buildLeafCones<<<
        conservativeConeSide * conservativeConeSide,
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            conservativeConeSide, tileGrid, params.resolution,
            sensorRayDirectionCudaPtr, sensorPoseToMat(sensorPoseInv),
            static_cast<gofish_conservative_bound::ConservativeConeNode*>(
                m_forwardContext->conservativeConeTree.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    for (uint32_t level = 1; level <= conservativeConeLevels; ++level) {
        const uint32_t parentSide = conservativeConeSide >> level;
        const uint32_t parentCount = parentSide * parentSide;
        gofish_conservative_bound::buildParentCones<<<
            div_round_up(parentCount, GUTParameters::Tiling::BlockSize),
            GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
                conservativeConeSide, level,
                static_cast<gofish_conservative_bound::ConservativeConeNode*>(
                    m_forwardContext->conservativeConeTree.data()));
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }
    const uint32_t angularRangeWords =
        gofish_angular_range_once::wordsFor(tileGrid.x * tileGrid.y);
    const uint64_t angularRangeBytes =
        gofish_angular_range_once::bytesFor(tileGrid.x * tileGrid.y);
    if (angularRangeWords == 0u
        || angularRangeWords > gofish_angular_range_once::kMaxWords
        || angularRangeBytes > uint64_t(std::numeric_limits<std::size_t>::max())) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] angular range index capacity invalid.");
    }
    CHECK_STATUS_RETURN(m_forwardContext->angularRangeIndexBits.resize(
        std::size_t(angularRangeBytes),
        reinterpret_cast<uint64_t>(cudaStream), m_logger));
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->angularRangeIndexBits.data(), 0,
        std::size_t(angularRangeBytes), cudaStream), m_logger);
    gofish_angular_range_once::build<<<
        div_round_up(tileGrid.x * tileGrid.y,
                     GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            gofish_physical_candidate::PhysicalTileView{
                m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                tileGrid.x, tileGrid.y},
            static_cast<const gofish_conservative_bound::ConservativeConeNode*>(
                m_forwardContext->conservativeConeTree.data()),
            conservativeConeSide,
            static_cast<uint32_t*>(
                m_forwardContext->angularRangeIndexBits.data()),
            angularRangeWords);
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    const gofish_angular_range_once::View angularRangeView{
        static_cast<const uint32_t*>(
            m_forwardContext->angularRangeIndexBits.data()),
        angularRangeWords, tileGrid.x * tileGrid.y};

    CHECK_STATUS_RETURN(m_forwardContext->updateParticlesWorkingBuffers(numParticles, cudaStream, m_logger));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        m_forwardContext->certifiedRemainderDeviceTable.data(),
        &m_forwardContext->certifiedRemainderHostTable,
        sizeof(gofish_certified_remainder::CameraThetaBandHessianTable),
        cudaMemcpyHostToDevice, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        m_forwardContext->certifiedRemainderDeviceProviderCamera.data(),
        &m_forwardContext->certifiedRemainderProviderCamera,
        sizeof(gofish_certified_remainder_provider::SelectedKbCameraDescriptor),
        cudaMemcpyHostToDevice, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->certifiedRemainderIntegrityError.data(), 0,
        sizeof(uint32_t), cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->certifiedFinalTilesCount.data(), 0,
        std::size_t(numParticles) * sizeof(uint32_t), cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->tileCapRetainedTilesCount.data(), 0,
        std::size_t(numParticles) * sizeof(uint32_t), cudaStream), m_logger);
    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(m_forwardContext->updateParticlesFeaturesBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

#if !GOFISH_FAST_UT_ADMISSION  // fast admission: no exact-cone states
    gofish_certified_remainder::prepareExactConeStates<
        TGUTProjector, typename TGUTProjector::TParticles><<<
        div_round_up(numParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            numParticles,
            sensorPoseInverse(sensorPose).slice<0, 3>(),
            certifiedCentralGlobalShutter,
            {parameters.m_dptrParametersBuffer},
            static_cast<gofish_certified_remainder::ExactConeState*>(
                m_forwardContext->certifiedRemainderExactConeStates.data()),
            static_cast<uint32_t*>(
                m_forwardContext->certifiedRemainderIntegrityError.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
#endif
    {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::project"};
        ::projectOnTiles<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            gofish_physical_candidate::PhysicalTileView{
                m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                tileGrid.x, tileGrid.y},
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose) /** tcnn::mat4(params.objectToWorldTransform)*/,
            params.sensorState,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const gofish_conservative_bound::ConservativeConeNode*)
                m_forwardContext->conservativeConeTree.data(),
            conservativeConeSide,
            conservativeConeLevels,
            (const uint32_t*)m_forwardContext->conservativeCentralityFlag.data(),
            (const gofish_certified_remainder::CameraThetaBandHessianTable*)
                m_forwardContext->certifiedRemainderDeviceTable.data(),
            (gofish_certified_remainder::SharedCountEmitRecord*)
                m_forwardContext->certifiedRemainderRecords.data(),
            (const gofish_certified_remainder_provider::SelectedKbCameraDescriptor*)
                m_forwardContext->certifiedRemainderDeviceProviderCamera.data(),
            (gofish_certified_remainder::ExactConeState*)
                m_forwardContext->certifiedRemainderExactConeStates.data(),
            (uint32_t*)m_forwardContext->certifiedRemainderIntegrityError.data(),
            (uint32_t*)m_forwardContext->certifiedFinalTilesCount.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (float*)m_forwardContext->particlesProjectedRadius.data(),
            (uint8_t*)m_forwardContext->particlesSupportProvenGt20.data(),
            (float*)m_forwardContext->particlesGlobalDepth.data(),
            (float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            particlesVisibilityCudaPtr,
            parameters.m_dptrParametersBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

#if !GOFISH_FAST_UT_ADMISSION  // fast admission: no tile-cap / bounded exact refinement
    gofish_tile_cap_direct::count<TGUTProjector><<<
        div_round_up(numParticles, gofish_tile_cap_direct::kWarpsPerBlock),
        gofish_tile_cap_direct::kThreadsPerBlock, 0, cudaStream>>>(
            numParticles, tileGrid,
            gofish_physical_candidate::PhysicalTileView{
                m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                tileGrid.x, tileGrid.y},
            m_forwardContext->certifiedRemainderProviderCamera
                .physicalSampleOffsetLower,
            m_forwardContext->certifiedRemainderProviderCamera
                .physicalSampleOffsetUpper,
            conservativeConeSide,
            static_cast<const gofish_conservative_bound::ConservativeConeNode*>(
                m_forwardContext->conservativeConeTree.data()),
            static_cast<const gofish_certified_remainder::ExactConeState*>(
                m_forwardContext->certifiedRemainderExactConeStates.data()),
            static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                m_forwardContext->certifiedRemainderRecords.data()),
            static_cast<uint32_t*>(
                m_forwardContext->tileCapRetainedTilesCount.data()),
            static_cast<uint32_t*>(
                m_forwardContext->certifiedRemainderIntegrityError.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);

    CUDA_CHECK_RETURN(cudaMemsetAsync(
        m_forwardContext->boundedCapCounts.data(), 0,
        std::size_t(numParticles) * sizeof(uint32_t), cudaStream), m_logger);
    gofish_bounded_exact::prepareAndCapCount<
        TGUTProjector, typename TGUTProjector::TParticles><<<
        div_round_up(numParticles, gofish_bounded_exact::kWarpsPerBlock),
        gofish_bounded_exact::kThreadsPerBlock, 0, cudaStream>>>(
            numParticles, tileGrid,
            gofish_physical_candidate::PhysicalTileView{
                m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                tileGrid.x, tileGrid.y},
            static_cast<const gofish_conservative_bound::ConservativeConeNode*>(
                m_forwardContext->conservativeConeTree.data()),
            conservativeConeSide, conservativeConeLevels,
            angularRangeView,
            sensorPoseInverse(sensorPose).slice<0, 3>(),
            certifiedCentralGlobalShutter,
            static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                m_forwardContext->certifiedRemainderRecords.data()),
            {parameters.m_dptrParametersBuffer},
            static_cast<gofish_certified_remainder::ExactConeState*>(
                m_forwardContext->certifiedRemainderExactConeStates.data()),
            static_cast<uint32_t*>(m_forwardContext->boundedCapCounts.data()),
            static_cast<uint32_t*>(
                m_forwardContext->certifiedRemainderIntegrityError.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    uint32_t boundedIntegrity = 0u;
#if GOFISH_INFERENCE_ONLY
    // One synchronization for the integrity word and the cap counts.
    std::vector<uint32_t> boundedCapCounts(numParticles, 0u);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&boundedIntegrity,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(boundedCapCounts.data(),
        m_forwardContext->boundedCapCounts.data(),
        std::size_t(numParticles) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
    if (boundedIntegrity != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] bounded cap-count integrity failed.");
    }
#else
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&boundedIntegrity,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
    if (boundedIntegrity != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] bounded cap-count integrity failed.");
    }
    std::vector<uint32_t> boundedCapCounts(numParticles, 0u);
    CUDA_CHECK_RETURN(cudaMemcpy(boundedCapCounts.data(),
        m_forwardContext->boundedCapCounts.data(),
        std::size_t(numParticles) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost), m_logger);
#endif

    constexpr uint32_t boundedSlotCapacity = uint32_t(
        (gofish_bounded_exact::kReservationBytes
         - gofish_bounded_exact::kFixedWorkspaceBytes) / 40ull);
    std::vector<std::vector<gofish_bounded_exact::Chunk>> boundedBatches;
    std::vector<gofish_bounded_exact::Chunk> boundedCurrent;
    uint32_t boundedUsed = 0u;
    auto boundedClose = [&]() {
        if (!boundedCurrent.empty()) {
            boundedBatches.push_back(std::move(boundedCurrent));
            boundedCurrent.clear();
            boundedUsed = 0u;
        }
    };
    const uint32_t boundedPhysicalTiles = tileGrid.x * tileGrid.y;
    for (uint32_t particle = 0u; particle < numParticles; ++particle) {
        const uint32_t upper = boundedCapCounts[particle];
        if (upper == 0u) continue;
        if (upper <= boundedSlotCapacity) {
            if (boundedUsed > boundedSlotCapacity - upper) boundedClose();
            boundedCurrent.push_back(gofish_bounded_exact::Chunk{
                particle, 0u, boundedPhysicalTiles, boundedUsed, upper,
                uint32_t(gofish_bounded_exact::kModeCapCompact)});
            boundedUsed += upper;
        } else {
            boundedClose();
            for (uint32_t begin = 0u; begin < boundedPhysicalTiles;) {
                const uint32_t length = std::min(
                    boundedSlotCapacity, boundedPhysicalTiles - begin);
                boundedBatches.push_back({gofish_bounded_exact::Chunk{
                    particle, begin, begin + length, 0u, length,
                    uint32_t(gofish_bounded_exact::kModePhysicalRange)}});
                begin += length;
            }
        }
    }
    boundedClose();

    std::vector<uint32_t> boundedRankCarry(numParticles, 0u);
    gofish_lazy_exact_cone::ExactDepthKey<
        TGUTProjector, typename TGUTProjector::TParticles> boundedDepthKey{
            params.resolution,
            reinterpret_cast<const tcnn::vec3*>(sensorRayOriginCudaPtr),
            reinterpret_cast<const tcnn::vec3*>(sensorRayDirectionCudaPtr),
            sensorPoseToMat(sensorPoseInv),
            {parameters.m_dptrParametersBuffer}};
    for (const auto& batch : boundedBatches) {
        uint64_t slotSum64 = 0u;
        for (const auto& chunk : batch)
            slotSum64 = std::max(slotSum64,
                uint64_t(chunk.slotBegin) + uint64_t(chunk.reservation));
        if (slotSum64 > boundedSlotCapacity
            || slotSum64 > uint64_t(std::numeric_limits<uint32_t>::max())) {
            RETURN_ERROR(m_logger, ErrorCode::BadInput,
                         "[GUTRenderer] bounded batch exceeds reservation.");
        }
        const uint32_t slots = uint32_t(slotSum64);
        uint64_t reservationBytes = 0u;
        CHECK_STATUS_RETURN(m_forwardContext->updateBoundedBatchBuffers(
            slots, uint32_t(batch.size()), cudaStream, m_logger,
            reservationBytes));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            m_forwardContext->boundedChunks.data(), batch.data(),
            batch.size() * sizeof(gofish_bounded_exact::Chunk),
            cudaMemcpyHostToDevice, cudaStream), m_logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            m_forwardContext->boundedSelectedCount.data(), 0,
            sizeof(uint32_t), cudaStream), m_logger);
        gofish_bounded_exact::evaluateOnce<
            TGUTProjector, typename TGUTProjector::TParticles><<<
            div_round_up(uint32_t(batch.size()),
                gofish_bounded_exact::kWarpsPerBlock),
            gofish_bounded_exact::kThreadsPerBlock, 0, cudaStream>>>(
                static_cast<const gofish_bounded_exact::Chunk*>(
                    m_forwardContext->boundedChunks.data()),
                uint32_t(batch.size()), numParticles, slots, tileGrid,
                gofish_physical_candidate::PhysicalTileView{
                    m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                    tileGrid.x, tileGrid.y},
                static_cast<const gofish_conservative_bound::ConservativeConeNode*>(
                    m_forwardContext->conservativeConeTree.data()),
                conservativeConeSide, angularRangeView,
                static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                    m_forwardContext->certifiedRemainderRecords.data()),
                static_cast<const gofish_certified_remainder::ExactConeState*>(
                    m_forwardContext->certifiedRemainderExactConeStates.data()),
                static_cast<uint8_t*>(m_forwardContext->boundedFlags.data()),
                static_cast<uint64_t*>(
                    m_forwardContext->boundedProvisionalKeys.data()),
                static_cast<uint64_t*>(
                    m_forwardContext->boundedProvisionalOwners.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->tileCapRetainedTilesCount.data()),
                boundedDepthKey,
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedRemainderIntegrityError.data()));
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
        boundedIntegrity = 0u;
#if !GOFISH_INFERENCE_ONLY
        CUDA_CHECK_RETURN(cudaMemcpyAsync(&boundedIntegrity,
            m_forwardContext->certifiedRemainderIntegrityError.data(),
            sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
        if (boundedIntegrity != 0u) {
            RETURN_ERROR(m_logger, ErrorCode::Runtime,
                         "[GUTRenderer] bounded evaluate integrity failed.");
        }
#endif
        cub::CountingInputIterator<uint32_t> boundedInput(0u);
        size_t selectionBytes =
            m_forwardContext->boundedSelectionWorkspace.size();
        CUDA_CHECK_RETURN(cub::DeviceSelect::Flagged(
            m_forwardContext->boundedSelectionWorkspace.data(),
            selectionBytes, boundedInput,
            static_cast<const uint8_t*>(m_forwardContext->boundedFlags.data()),
            static_cast<uint32_t*>(
                m_forwardContext->boundedSelectedIndices.data()),
            static_cast<uint32_t*>(
                m_forwardContext->boundedSelectedCount.data()),
            slots, cudaStream), m_logger);
        uint32_t selected = 0u;
#if GOFISH_INFERENCE_ONLY
        // Integrity word of the evaluation and the selected count, one synchronization.
        CUDA_CHECK_RETURN(cudaMemcpyAsync(&boundedIntegrity,
            m_forwardContext->certifiedRemainderIntegrityError.data(),
            sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
#endif
        CUDA_CHECK_RETURN(cudaMemcpyAsync(&selected,
            m_forwardContext->boundedSelectedCount.data(), sizeof(uint32_t),
            cudaMemcpyDeviceToHost, cudaStream), m_logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
#if GOFISH_INFERENCE_ONLY
        if (boundedIntegrity != 0u) {
            RETURN_ERROR(m_logger, ErrorCode::Runtime,
                         "[GUTRenderer] bounded evaluate integrity failed.");
        }
#endif
        if (selected > slots) {
            RETURN_ERROR(m_logger, ErrorCode::Runtime,
                         "[GUTRenderer] bounded select count exceeds slots.");
        }
        if (selected) {
            gofish_bounded_exact::gatherSelected<<<
                div_round_up(selected, GUTParameters::Tiling::BlockSize),
                GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
                    selected,
                    static_cast<const uint32_t*>(
                        m_forwardContext->boundedSelectedIndices.data()),
                    static_cast<const uint64_t*>(
                        m_forwardContext->boundedProvisionalKeys.data()),
                    static_cast<const uint64_t*>(
                        m_forwardContext->boundedProvisionalOwners.data()),
                    static_cast<uint64_t*>(
                        m_forwardContext->boundedCompactKeys.data()),
                    static_cast<uint64_t*>(
                        m_forwardContext->boundedCompactOwners.data()));
            CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
        }
        if (boundedBatches.size() == 1u) {
            boundedSingleDeviceCount = selected;
            continue;
        }
        BoundedHostRun run;
        run.keys.resize(selected);
        run.owners.resize(selected);
        if (selected) {
            CUDA_CHECK_RETURN(cudaMemcpyAsync(run.keys.data(),
                m_forwardContext->boundedCompactKeys.data(),
                std::size_t(selected) * sizeof(uint64_t),
                cudaMemcpyDeviceToHost, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyAsync(run.owners.data(),
                m_forwardContext->boundedCompactOwners.data(),
                std::size_t(selected) * sizeof(uint64_t),
                cudaMemcpyDeviceToHost, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
            uint32_t previous = std::numeric_limits<uint32_t>::max();
            uint32_t localCount = 0u;
            for (uint64_t& owner : run.owners) {
                const uint32_t particle = uint32_t(owner >> 32);
                if (particle != previous) {
                    if (previous != std::numeric_limits<uint32_t>::max())
                        boundedRankCarry[previous] += localCount;
                    previous = particle;
                    localCount = 0u;
                }
                const uint32_t rank = uint32_t(owner);
                owner = (uint64_t(particle) << 32)
                    | uint64_t(boundedRankCarry[particle] + rank);
                ++localCount;
            }
            if (previous != std::numeric_limits<uint32_t>::max())
                boundedRankCarry[previous] += localCount;
        }
        boundedHostRuns.push_back(std::move(run));
    }
#endif

#if !GOFISH_FAST_UT_ADMISSION  // fast admission: counts are final
    preserveExactRouteOriginalCounts<<<
        div_round_up(numParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            numParticles,
            static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                m_forwardContext->certifiedRemainderRecords.data()),
            static_cast<const uint32_t*>(
                m_forwardContext->tileCapRetainedTilesCount.data()),
            static_cast<uint32_t*>(
                m_forwardContext->certifiedFinalTilesCount.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
#endif

    // A failed count may leave only the pre-zeroed row value.  Observe the
    // integrity status before scanning or trusting any terminal total.
    uint32_t certifiedIntegrityError = 0u;
#if !GOFISH_INFERENCE_ONLY
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&certifiedIntegrityError,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
    if (certifiedIntegrityError != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] certified-remainder count integrity failed.");
    }
#endif

    deviceLaunchesLogger.push("render::prepare-expand");

    // One inclusive scan owns the final one-stream cardinality.
    finalizeCertifiedVisibility<<<
        div_round_up(numParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            numParticles,
            static_cast<const uint32_t*>(
                m_forwardContext->certifiedFinalTilesCount.data()),
            particlesVisibilityCudaPtr);
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    widenCertifiedCandidateCounts<<<
        div_round_up(numParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            numParticles,
            static_cast<const uint32_t*>(
#if GOFISH_FAST_UT_ADMISSION
                m_forwardContext->certifiedFinalTilesCount.data()),
#else
                m_forwardContext->tileCapRetainedTilesCount.data()),
#endif
            static_cast<uint64_t*>(
                m_forwardContext->certifiedFinalTilesCount64.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    size_t certifiedFinalScanBytes =
        m_forwardContext->certifiedFinalScanningWorkingBuffer.size();
    CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
        m_forwardContext->certifiedFinalScanningWorkingBuffer.data(),
        certifiedFinalScanBytes,
        static_cast<const uint64_t*>(
            m_forwardContext->certifiedFinalTilesCount64.data()),
        static_cast<uint64_t*>(
            m_forwardContext->certifiedFinalTilesOffset.data()),
        numParticles, cudaStream), m_logger);

    // Fetch the exact 64-bit scan tail, then cross every narrower
    // allocation/CUB ABI only through checked conversion.
    uint32_t numParticleTileIntersections = 0;
    uint64_t exactCertifiedCandidateCount = 0u;
#if GOFISH_INFERENCE_ONLY
    // The count-stage integrity word travels with the scan tail (one synchronization).
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&certifiedIntegrityError,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
#endif
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&exactCertifiedCandidateCount,
        static_cast<uint64_t*>(
            m_forwardContext->certifiedFinalTilesOffset.data()) + numParticles - 1,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
#if GOFISH_INFERENCE_ONLY
    if (certifiedIntegrityError != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] certified-remainder count integrity failed.");
    }
#endif
    CertifiedCandidateCapacityPlan certifiedCapacity{};
    if (!checkedCertifiedCandidateCapacity(
            exactCertifiedCandidateCount, certifiedCapacity)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] certified candidate capacity overflow.");
    }
    numParticleTileIntersections = certifiedCapacity.cubItems;

    if (numParticleTileIntersections == 0) {
        std::vector<uint32_t> originalCounts(numParticles, 0u);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            originalCounts.data(),
            m_forwardContext->certifiedFinalTilesCount.data(),
            std::size_t(numParticles) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, cudaStream), m_logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
        const bool originalNonempty = std::any_of(
            originalCounts.begin(), originalCounts.end(),
            [](uint32_t value) { return value != 0u; });
        if (!originalNonempty) return Status();
        // Every original record was removed: continue with zero ranges so
        // render/finalizeRay still executes.
    }

    // Allocate only the final one-stream key/ID/range family when enabled.
    CHECK_STATUS_RETURN(m_forwardContext->updateCertifiedFinalSortingBuffers(
        tileGrid, certifiedCapacity, cudaStream, m_logger));

    deviceLaunchesLogger.pop("render::prepare-expand");

    {
        const auto expandProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::expand"};
        ::expandTileProjections<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            gofish_physical_candidate::PhysicalTileView{
                m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                tileGrid.x, tileGrid.y},
            numParticles,
            params.resolution,
            params.sensorModel,
            params.sensorState,
            (const gofish_certified_remainder::SharedCountEmitRecord*)
                m_forwardContext->certifiedRemainderRecords.data(),
            (const gofish_certified_remainder_provider::SelectedKbCameraDescriptor*)
                m_forwardContext->certifiedRemainderDeviceProviderCamera.data(),
            (const gofish_certified_remainder::ExactConeState*)
                m_forwardContext->certifiedRemainderExactConeStates.data(),
            (uint32_t*)m_forwardContext->certifiedRemainderIntegrityError.data(),
            (const uint64_t*)m_forwardContext->certifiedFinalTilesOffset.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const tcnn::vec2*)m_forwardContext->particlesProjectedExtent.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            parameters.m_dptrParametersBuffer,
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const gofish_conservative_bound::ConservativeConeNode*)
                m_forwardContext->conservativeConeTree.data(),
            conservativeConeSide,
            conservativeConeLevels,
            (const uint32_t*)m_forwardContext->conservativeCentralityFlag.data(),
            (uint64_t*)m_forwardContext->certifiedFinalUnsortedTileDepthKeys.data(),
            (uint32_t*)m_forwardContext->certifiedFinalUnsortedTileParticleIdx.data());
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }
#if !GOFISH_FAST_UT_ADMISSION  // fast admission: keys are emitted by expand
    gofish_tile_cap_direct::emit<
        TGUTProjector, decltype(boundedDepthKey)><<<
            div_round_up(
                numParticles, gofish_tile_cap_direct::kWarpsPerBlock),
            gofish_tile_cap_direct::kThreadsPerBlock, 0, cudaStream>>>(
                numParticles, tileGrid,
                gofish_physical_candidate::PhysicalTileView{
                    m_physicalTileMaskCudaPtr, m_physicalTileMaskCount,
                    tileGrid.x, tileGrid.y},
                m_forwardContext->certifiedRemainderProviderCamera
                    .physicalSampleOffsetLower,
                m_forwardContext->certifiedRemainderProviderCamera
                    .physicalSampleOffsetUpper,
                conservativeConeSide,
                static_cast<const gofish_conservative_bound::ConservativeConeNode*>(
                    m_forwardContext->conservativeConeTree.data()),
                static_cast<const gofish_certified_remainder::ExactConeState*>(
                    m_forwardContext->certifiedRemainderExactConeStates.data()),
                static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                    m_forwardContext->certifiedRemainderRecords.data()),
                static_cast<const uint64_t*>(
                    m_forwardContext->certifiedFinalTilesOffset.data()),
                uint64_t(numParticleTileIntersections),
                static_cast<uint64_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileDepthKeys.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileParticleIdx.data()),
                boundedDepthKey,
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedRemainderIntegrityError.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
#endif
    if (boundedSingleDeviceCount) {
        gofish_bounded_exact::scatterCompacted<<<
            div_round_up(boundedSingleDeviceCount,
                GUTParameters::Tiling::BlockSize),
            GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
                boundedSingleDeviceCount,
                static_cast<const uint64_t*>(
                    m_forwardContext->boundedCompactKeys.data()),
                static_cast<const uint64_t*>(
                    m_forwardContext->boundedCompactOwners.data()),
                static_cast<const uint64_t*>(
                    m_forwardContext->certifiedFinalTilesOffset.data()),
                uint64_t(numParticleTileIntersections),
                static_cast<uint64_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileDepthKeys.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileParticleIdx.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedRemainderIntegrityError.data()));
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }
    for (const auto& run : boundedHostRuns) {
        const uint32_t count = uint32_t(run.keys.size());
        if (run.owners.size() != run.keys.size()) {
            RETURN_ERROR(m_logger, ErrorCode::Runtime,
                         "[GUTRenderer] bounded compact run differs.");
        }
        if (!count) continue;
        uint64_t reservationBytes = 0u;
        CHECK_STATUS_RETURN(m_forwardContext->updateBoundedBatchBuffers(
            count, 1u, cudaStream, m_logger, reservationBytes));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            m_forwardContext->boundedCompactKeys.data(), run.keys.data(),
            std::size_t(count) * sizeof(uint64_t),
            cudaMemcpyHostToDevice, cudaStream), m_logger);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            m_forwardContext->boundedCompactOwners.data(), run.owners.data(),
            std::size_t(count) * sizeof(uint64_t),
            cudaMemcpyHostToDevice, cudaStream), m_logger);
        gofish_bounded_exact::scatterCompacted<<<
            div_round_up(count, GUTParameters::Tiling::BlockSize),
            GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
                count,
                static_cast<const uint64_t*>(
                    m_forwardContext->boundedCompactKeys.data()),
                static_cast<const uint64_t*>(
                    m_forwardContext->boundedCompactOwners.data()),
                static_cast<const uint64_t*>(
                    m_forwardContext->certifiedFinalTilesOffset.data()),
                uint64_t(numParticleTileIntersections),
                static_cast<uint64_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileDepthKeys.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedFinalUnsortedTileParticleIdx.data()),
                static_cast<uint32_t*>(
                    m_forwardContext->certifiedRemainderIntegrityError.data()));
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

#if !GOFISH_INFERENCE_ONLY
    certifiedIntegrityError = 0u;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&certifiedIntegrityError,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    cudaStreamSynchronize(cudaStream);
    if (certifiedIntegrityError != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] certified-remainder emit integrity failed.");
    }
#endif

    deviceLaunchesLogger.push("render::sort");

    size_t certifiedFinalSortingBytes =
        m_forwardContext->certifiedFinalSortingWorkingBuffer.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
        m_forwardContext->certifiedFinalSortingWorkingBuffer.data(),
        certifiedFinalSortingBytes,
        static_cast<const uint64_t*>(
            m_forwardContext->certifiedFinalUnsortedTileDepthKeys.data()),
        static_cast<uint64_t*>(
            m_forwardContext->certifiedFinalSortedTileDepthKeys.data()),
        static_cast<const uint32_t*>(
            m_forwardContext->certifiedFinalUnsortedTileParticleIdx.data()),
        static_cast<uint32_t*>(
            m_forwardContext->certifiedFinalSortedTileParticleIdx.data()),
        numParticleTileIntersections,
        0, 32 + higherMsb(tileGrid.x * tileGrid.y), cudaStream), m_logger);
    linear_kernel(computeSortedTileRangeIndices, 0, cudaStream,
        numParticleTileIntersections,
        static_cast<const uint64_t*>(
            m_forwardContext->certifiedFinalSortedTileDepthKeys.data()),
        static_cast<uvec2*>(
            m_forwardContext->certifiedFinalSortedTileRangeIndices.data()));
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);

    deviceLaunchesLogger.pop("render::sort");

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render::render"};

#if FINE_GRAINED_LOAD_BALANCING
        constexpr uint32_t VirtualTilesPerTile = GUTParameters::Tiling::VirtualTilesPerTile;
        constexpr uint32_t ThreadsPerBlock     = GUTParameters::Tiling::FineGrainedThreadsPerBlock;
        const uint32_t virtual_tiles_total     = tileGrid.x * tileGrid.y * VirtualTilesPerTile;
        ::renderBalanced<<<virtual_tiles_total, ThreadsPerBlock, 0, cudaStream>>>( // ThreadsPerBlock = FineGrainedWarpsPerBlock * WarpSize
            params,
            (const tcnn::uvec2*)
                m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            (const uint32_t*)
                m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            worldHitCountCudaPtr,
            worldHitDistanceCudaPtr,
            featureDensityCudaPtr,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer,
            tcnn::uvec2{tileGrid.x, tileGrid.y});
#else
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
        {
            const std::size_t splitBytes = std::size_t(GOFISH_FAST_TILE_SPLIT) * std::size_t(params.resolution.x)
                * std::size_t(params.resolution.y) * std::size_t(RAY_FEATURE_DIM + 2) * sizeof(float);
            if (m_forwardContext->gofishSplitScratch.size() < splitBytes)
                CHECK_STATUS_RETURN(m_forwardContext->gofishSplitScratch.resize(splitBytes, reinterpret_cast<uint64_t>(cudaStream), m_logger));
            float* splitPtr = static_cast<float*>(m_forwardContext->gofishSplitScratch.data());
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gofishSplitScratch, &splitPtr, sizeof(float*), 0,
                                                      cudaMemcpyHostToDevice, cudaStream), m_logger);
        }
        constexpr uint32_t gofishSplitZ = uint32_t(GOFISH_FAST_TILE_SPLIT);
#else
        constexpr uint32_t gofishSplitZ = 1u;
#endif
#if GOFISH_SOF_SCHEDULE
        {   // SOF emission schedule, consumed by render and renderBackward.  The tile
            // rows are split into batches whose schedule fits the free device memory;
            // a single batch is the common case.
            const uint64_t q = reinterpret_cast<uint64_t>(cudaStream);
            const uint32_t nTiles = tileGrid.x * tileGrid.y;
            std::vector<tcnn::uvec2> ranges(nTiles);
            CUDA_CHECK_RETURN(cudaMemcpyAsync(ranges.data(), m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
                nTiles * sizeof(tcnn::uvec2), cudaMemcpyDeviceToHost, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
            std::vector<uint64_t> rowLo(tileGrid.y, ~0ull), rowHi(tileGrid.y, 0ull);
            for (uint32_t t = 0; t < nTiles; ++t) {
                const uint32_t r = t / tileGrid.x;
                if (ranges[t].y > ranges[t].x) {
                    rowLo[r] = std::min<uint64_t>(rowLo[r], ranges[t].x);
                    rowHi[r] = std::max<uint64_t>(rowHi[r], ranges[t].y);
                }
            }
            size_t freeB = 0, totalB = 0;
            CUDA_CHECK_RETURN(cudaMemGetInfo(&freeB, &totalB), m_logger);
            const size_t marginB = size_t(6) << 30;  // keep 6 GiB free for densification growth
            const size_t haveIds = m_forwardContext->sofSchedIds.size() / sizeof(uint32_t);
            const size_t capIds = std::min<size_t>(size_t(3) << 29,  // at most 6 GiB of schedule
                haveIds + (freeB > marginB ? (freeB - marginB) / sizeof(uint32_t) : 0));
            const size_t fullIds = static_cast<size_t>(numParticleTileIntersections) * 64u;
            auto& plan = m_forwardContext->sofPlan;
            plan.clear();
            size_t needIds = 1;
            if (fullIds <= capIds) {
                plan = {0u, tileGrid.y, 0u};
                needIds = std::max<size_t>(fullIds, 1);
            } else {
                for (uint32_t r0 = 0; r0 < tileGrid.y;) {
                    uint64_t lo = ~0ull, hi = 0;
                    uint32_t r = r0;
                    for (; r < tileGrid.y; ++r) {
                        uint64_t nlo = lo, nhi = hi;
                        if (rowHi[r] > 0) { nlo = std::min(lo, rowLo[r]); nhi = std::max(hi, rowHi[r]); }
                        if (r > r0 && nhi > 0 && (nhi - nlo) * 64u > capIds) break;  // one row always forms a batch
                        lo = nlo; hi = nhi;
                    }
                    plan.push_back(r0); plan.push_back(r); plan.push_back(hi > 0 ? static_cast<uint32_t>(lo) : 0u);
                    if (hi > 0) needIds = std::max<size_t>(needIds, (hi - lo) * 64u);
                    r0 = r;
                }
            }
            CHECK_STATUS_RETURN(m_forwardContext->sofSchedIds.enlarge(sizeof(uint32_t) * needIds, q, m_logger));
            CHECK_STATUS_RETURN(m_forwardContext->sofSchedCounts.enlarge(sizeof(uint32_t) * 64u * nTiles, q, m_logger));
            m_forwardContext->sofSchedPtrs[0] = static_cast<uint32_t*>(m_forwardContext->sofSchedIds.data());
            m_forwardContext->sofSchedPtrs[1] = static_cast<uint32_t*>(m_forwardContext->sofSchedCounts.data());
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofSchedIds, &m_forwardContext->sofSchedPtrs[0], sizeof(uint32_t*), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofSchedCounts, &m_forwardContext->sofSchedPtrs[1], sizeof(uint32_t*), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
        }
        for (size_t batch = 0; batch + 2 < m_forwardContext->sofPlan.size(); batch += 3) {
            const uint32_t* bp = &m_forwardContext->sofPlan[batch];
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofRowBegin, bp + 0, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofRowEnd, bp + 1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofSchedBase, bp + 2, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            ::sofMid8Schedule<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
                params, m_physicalTileMaskCudaPtr,
                (const tcnn::uvec2*)m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
                (const uint32_t*)m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
                (const tcnn::vec3*)sensorRayOriginCudaPtr, (const tcnn::vec3*)sensorRayDirectionCudaPtr,
                sensorPoseToMat(sensorPoseInv), parameters.m_dptrParametersBuffer);
            CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
#endif
        ::render<<<dim3{tileGrid.x, tileGrid.y, gofishSplitZ}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params, // threedgut::RenderParameters params
            m_physicalTileMaskCudaPtr,
            (const tcnn::uvec2*)
                m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            (const uint32_t*)
                m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            worldHitCountCudaPtr,
            worldHitDistanceCudaPtr,
            featureDensityCudaPtr,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            worldNormalCudaPtr,
            worldDepthVarCudaPtr,
            worldQuantDepthCudaPtr,
            worldExtentCudaPtr,
            worldOpacityFieldCudaPtr,
            worldInvDCudaPtr,
            worldInvD2CudaPtr,
            parameters.m_dptrParametersBuffer);
#if GOFISH_SOF_SCHEDULE
            CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
        }  // tile-row batches
#endif
#if GOFISH_INFERENCE_ONLY && GOFISH_FAST_TILE_SPLIT > 1
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
        ::gofishSplitCombine<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params, m_physicalTileMaskCudaPtr,
            (const tcnn::uvec2*)m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            tileGrid.x, worldHitDistanceCudaPtr, featureDensityCudaPtr);
#endif
#endif
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    if (worldOpacityFieldCudaPtr != nullptr
        && worldQuantDepthCudaPtr != nullptr) {
        const auto opacityReplayProfile = DeviceLaunchesLogger::ScopePush{
            deviceLaunchesLogger, "render::sof-exact-crossing-opacity"};
        ::sofExactCrossingOpacityForward<<<
            dim3{tileGrid.x, tileGrid.y, 1u},
            dim3{GUTParameters::Tiling::BlockX,
                 GUTParameters::Tiling::BlockY, 1u},
            0, cudaStream>>>(
            params,
            m_physicalTileMaskCudaPtr,
            (const tcnn::uvec2*)
                m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            (const uint32_t*)
                m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const float*)worldQuantDepthCudaPtr,
            (float*)worldOpacityFieldCudaPtr,
            parameters.m_dptrParametersBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

#if GOFISH_INFERENCE_ONLY
    // Emit-stage integrity, checked once the frame is queued (fails the call
    // exactly as the reference does, after rather than before the render).
    certifiedIntegrityError = 0u;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&certifiedIntegrityError,
        m_forwardContext->certifiedRemainderIntegrityError.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, cudaStream), m_logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), m_logger);
    if (certifiedIntegrityError != 0u) {
        RETURN_ERROR(m_logger, ErrorCode::Runtime,
                     "[GUTRenderer] certified-remainder emit integrity failed.");
    }
#endif
    return Status();
}

threedgut::Status threedgut::GUTRenderer::collectSofMaxRadii2D(
    float* outputCudaPtr, cudaStream_t cudaStream) const {
    if (!m_forwardContext || outputCudaPtr == nullptr) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] SOF radius collection requires a completed forward pass and output.");
    }
    if (m_lastNumParticles == 0u) return Status();
    collectSofMaxRadii2DKernel<<<
        div_round_up(m_lastNumParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            m_lastNumParticles,
            static_cast<const float*>(
                m_forwardContext->particlesProjectedRadius.data()),
            static_cast<const uint32_t*>(
                m_forwardContext->certifiedFinalTilesCount.data()),
            outputCudaPtr);
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    return Status();
}

threedgut::Status threedgut::GUTRenderer::collectSofScreenPruneState(
    uint8_t* provenGt20CudaPtr, uint8_t* cameraRelationCudaPtr,
    cudaStream_t cudaStream) const {
    if (!m_forwardContext || provenGt20CudaPtr == nullptr
        || cameraRelationCudaPtr == nullptr) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] SOF prune-state collection requires a completed forward pass and outputs.");
    }
    if (m_lastNumParticles == 0u) return Status();
    collectSofScreenPruneStateKernel<<<
        div_round_up(m_lastNumParticles, GUTParameters::Tiling::BlockSize),
        GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            m_lastNumParticles,
            static_cast<const float*>(
                m_forwardContext->particlesProjectedRadius.data()),
            static_cast<const uint8_t*>(
                m_forwardContext->particlesSupportProvenGt20.data()),
            static_cast<const uint32_t*>(
                m_forwardContext->certifiedFinalTilesCount.data()),
            static_cast<const gofish_certified_remainder::SharedCountEmitRecord*>(
                m_forwardContext->certifiedRemainderRecords.data()),
            static_cast<const gofish_certified_remainder::ExactConeState*>(
                m_forwardContext->certifiedRemainderExactConeStates.data()),
            provenGt20CudaPtr, cameraRelationCudaPtr);
    CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    return Status();
}

threedgut::Status threedgut::GUTRenderer::renderBackward(const RenderParameters& params,
                                                         const vec3* sensorRayOriginCudaPtr,
                                                         const vec3* sensorRayDirectionCudaPtr,
                                                         const float* worldHitDistanceCudaPtr,
                                                         const float* worldHitDistanceGradientCudaPtr,
                                                         const float* worldDepthVarCudaPtr,
                                                         const float* worldDepthVarGradientCudaPtr,
                                                         const float* worldExtentCudaPtr,
                                                         const float* worldExtentGradientCudaPtr,
                                                         const float* worldOpacityFieldCudaPtr,
                                                         const float* worldOpacityFieldGradientCudaPtr,
                                                         const float* worldQuantDepthCudaPtr,
                                                         const float* worldQuantDepthGradientCudaPtr,
                                                         const TFeatureDensityElem* featureDensityCudaPtr,
                                                         const float* featureDensityGradientCudaPtr,
                                                         vec3* worldRayOriginGradientCudaPtr,    // unused
                                                         vec3* worldRayDirectionGradientCudaPtr, // unused
                                                         float* particlesPositionAbsGradientCudaPtr,
                                                         const float* worldNormalCudaPtr,
                                                         const float* worldNormalGradientCudaPtr,
                                                         const float* worldInvDCudaPtr,
                                                         const float* worldInvDGradientCudaPtr,
                                                         const float* worldInvD2CudaPtr,
                                                         const float* worldInvD2GradientCudaPtr,
                                                         Parameters& parameters,
                                                         int cudaDeviceIndex,
                                                         cudaStream_t cudaStream) {

    if (!gofish_admission::globalShutterAccepted(static_cast<int>(params.sensorModel.shutterType), static_cast<int>(threedgut::TSensorModel::GlobalShutter))) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] unconditional support union requires global shutter.");
    }

    if (!m_forwardContext || (m_forwardContext->cudaStream != cudaStream)) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] cannot render backward, invalid forward context on device %d, cudaStream = %p.",
                     cudaDeviceIndex, cudaStream);
    }

    DeviceLaunchesLogger deviceLaunchesLogger(m_logger, cudaDeviceIndex, reinterpret_cast<uint64_t>(cudaStream));
    deviceLaunchesLogger.push("render-backward");

    const uvec2 tileGrid{
        div_round_up<uint32_t>(params.resolution.x, GUTParameters::Tiling::BlockX),
        div_round_up<uint32_t>(params.resolution.y, GUTParameters::Tiling::BlockY),
    };
    if (m_physicalTileMaskCudaPtr == nullptr
        || m_physicalTileMaskCount != tileGrid.x * tileGrid.y) {
        RETURN_ERROR(m_logger, ErrorCode::BadInput,
                     "[GUTRenderer] registered physical tile mask differs.");
    }
    const uint32_t numParticles = parameters.values.numParticles;

    // Sensor pose at mid-exposure (one central pose per frame).
    const TSensorPose sensorPose    = interpolatedSensorPose(params.sensorState.startPose, params.sensorState.endPose, 0.5f); // Transform from world to sensor space
    const TSensorPose sensorPoseInv = sensorPoseInverse(sensorPose);                                                          // Transform from sensor to world space

    if (numParticles == 0) {
        return Status();
    }

    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesFeaturesGradientBuffer(numParticles * featuresDim(), cudaStream, m_logger));
    }

    if (/*m_settings.renderMode == Settings::Splat*/ TGUTProjectorParams::BackwardProjection) {
        CHECK_STATUS_RETURN(
            m_forwardContext->updateParticlesProjectionGradientBuffers(numParticles, cudaStream, m_logger));
    }

    {
        const auto renderProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::render"};
#if GOFISH_SOF_SCHEDULE
        // One batch reuses the forward schedule; several batches recompute it per batch.
        const size_t numBatches = m_forwardContext->sofPlan.size() / 3;
        for (size_t batch = 0; batch < numBatches; ++batch) {
        if (numBatches > 1) {
            const uint32_t* bp = &m_forwardContext->sofPlan[3 * batch];
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofRowBegin, bp + 0, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofRowEnd, bp + 1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync(gSofSchedBase, bp + 2, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, cudaStream), m_logger);
            ::sofMid8Schedule<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
                params, m_physicalTileMaskCudaPtr,
                (const tcnn::uvec2*)m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
                (const uint32_t*)m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
                (const tcnn::vec3*)sensorRayOriginCudaPtr, (const tcnn::vec3*)sensorRayDirectionCudaPtr,
                sensorPoseToMat(sensorPoseInv), parameters.m_dptrParametersBuffer);
            CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
        }
#endif
        ::renderBackward<<<dim3{tileGrid.x, tileGrid.y, 1u}, dim3{GUTParameters::Tiling::BlockX, GUTParameters::Tiling::BlockY, 1u}, 0, cudaStream>>>(
            params,
            m_physicalTileMaskCudaPtr,
            (const tcnn::uvec2*)
                m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            (const uint32_t*)
                m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const float*)worldHitDistanceCudaPtr,         //
            (const float*)worldHitDistanceGradientCudaPtr, // unused
            (const float*)worldDepthVarCudaPtr,
            (const float*)worldDepthVarGradientCudaPtr,
            (const float*)worldExtentCudaPtr,
            (const float*)worldExtentGradientCudaPtr,
            (const float*)worldOpacityFieldCudaPtr,
            (const float*)worldOpacityFieldGradientCudaPtr,
            (const float*)worldQuantDepthCudaPtr,
            (const float*)worldQuantDepthGradientCudaPtr,
            featureDensityCudaPtr,
            featureDensityGradientCudaPtr,
            (tcnn::vec3*)worldRayOriginGradientCudaPtr,    // unused
            (tcnn::vec3*)worldRayDirectionGradientCudaPtr, // unused
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPosition.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacity.data(),
            (const float*)m_forwardContext->particlesGlobalDepth.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            parameters.m_dptrParametersBuffer,
            (tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer,
            particlesPositionAbsGradientCudaPtr,
            worldNormalCudaPtr,
            worldNormalGradientCudaPtr,
            (const float*)worldInvDCudaPtr,
            (const float*)worldInvDGradientCudaPtr,
            (const float*)worldInvD2CudaPtr,
            (const float*)worldInvD2GradientCudaPtr);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
#if GOFISH_SOF_SCHEDULE
        }  // tile-row batches
#endif
    }

    if (worldOpacityFieldCudaPtr != nullptr
        && worldOpacityFieldGradientCudaPtr != nullptr
        && worldQuantDepthCudaPtr != nullptr) {
        const auto opacityReplayProfile = DeviceLaunchesLogger::ScopePush{
            deviceLaunchesLogger,
            "render-backward::sof-exact-crossing-opacity"};
        ::sofExactCrossingOpacityBackward<<<
            dim3{tileGrid.x, tileGrid.y, 1u},
            dim3{GUTParameters::Tiling::BlockX,
                 GUTParameters::Tiling::BlockY, 1u},
            0, cudaStream>>>(
            params,
            m_physicalTileMaskCudaPtr,
            (const tcnn::uvec2*)
                m_forwardContext->certifiedFinalSortedTileRangeIndices.data(),
            (const uint32_t*)
                m_forwardContext->certifiedFinalSortedTileParticleIdx.data(),
            (const tcnn::vec3*)sensorRayOriginCudaPtr,
            (const tcnn::vec3*)sensorRayDirectionCudaPtr,
            sensorPoseToMat(sensorPoseInv),
            (const float*)worldQuantDepthCudaPtr,
            (const float*)worldOpacityFieldCudaPtr,
            (const float*)worldOpacityFieldGradientCudaPtr,
            parameters.m_dptrParametersBuffer,
            parameters.m_dptrGradientsBuffer,
            particlesPositionAbsGradientCudaPtr);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    if (!/*m_settings.perRayFeatures*/ TGUTRendererParams::PerRayParticleFeatures) {
        const auto projectProfile = DeviceLaunchesLogger::ScopePush{deviceLaunchesLogger, "render-backward::project"};
        ::projectBackward<<<div_round_up(numParticles, GUTParameters::Tiling::BlockSize), GUTParameters::Tiling::BlockSize, 0, cudaStream>>>(
            tileGrid,
            numParticles,
            params.resolution,
            params.sensorModel,
            // NOTE : the sensor world position is an approximated position for preprocessing gaussian colors
            /*sensorWorldPosition=*/sensorPoseInverse(sensorPose).slice<0, 3>(),
            // NOTE : this sensor to world transform is used to estimate the Z-depth of the particles
            sensorPoseToMat(sensorPose),
            (const gofish_certified_remainder::SharedCountEmitRecord*)
                m_forwardContext->certifiedRemainderRecords.data(),
            (uint32_t*)m_forwardContext->certifiedRemainderIntegrityError.data(),
            (const uint32_t*)m_forwardContext->certifiedFinalTilesCount.data(),
            parameters.m_dptrParametersBuffer,
            (const tcnn::vec2*)m_forwardContext->particlesProjectedPositionGradient.data(),
            (const tcnn::vec4*)m_forwardContext->particlesProjectedConicOpacityGradient.data(),
            (const float*)m_forwardContext->particlesGlobalDepthGradient.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeatures.data(),
            (const float*)m_forwardContext->particlesPrecomputedFeaturesGradient.data(),
            parameters.m_dptrGradientsBuffer);
        CUDA_CHECK_STREAM_RETURN(cudaStream, m_logger);
    }

    return Status();
}
