// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <3dgut/renderer/gutRenderer.h>
#include <3dgut/utils/logger.h>

#include <3dgut/kernels/cuda/common/opacityField.cuh>
#include <3dgut/kernels/cuda/common/bisectionTable.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace threedgut::fast_sof_persistent_camera_table {
namespace {

namespace field = threedgut::fast_sof_tnt_full_shape;
namespace detail = threedgut::fast_sof_tnt_full_shape::cooperative_detail;

struct Camera {
    RenderParameters params;
    const tcnn::vec3* rayOrigins = nullptr;
    const tcnn::vec3* rayDirections = nullptr;
    const std::uint8_t* physicalMask = nullptr;
    tcnn::mat4x3 worldToSensor;
    tcnn::mat4x3 sensorToWorld;
    tcnn::uvec2 tileGrid = tcnn::uvec2::zero();
    std::uint32_t tileCount = 0;
    std::uint32_t intersections = 0;
};

__device__ __forceinline__ void recordRoundFailure(
    std::uint32_t* failure, std::uint32_t code,
    std::uint32_t camera, std::uint32_t index) {
    if (atomicCAS(failure, 0u, code) == 0u) {
        failure[1] = camera;
        failure[2] = index;
    }
}

__global__ void checkPreparedCount(
    const std::uint64_t* offsets, std::uint32_t particles,
    std::uint32_t expected, std::uint32_t camera,
    std::uint32_t* failure) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const std::uint64_t actual = offsets[particles - 1];
    if (actual != expected) {
        recordRoundFailure(failure, 1u, camera,
                           actual > UINT32_MAX ? UINT32_MAX
                                               : static_cast<std::uint32_t>(actual));
    }
}

__global__ void checkDeviceFlag(
    const std::uint32_t* flag, const std::uint32_t* first,
    std::uint32_t code, std::uint32_t camera, std::uint32_t* failure) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *flag != 0u) {
        recordRoundFailure(failure, code, camera, *first);
    }
}

struct BoundedKeyEmitter {
    std::uint64_t* keys;
    std::uint32_t* ids;
    std::uint64_t capacity;
    std::uint32_t camera;
    std::uint32_t* failure;

    __device__ __forceinline__ void operator()(
        std::uint64_t cursor, std::uint32_t tile, std::uint32_t particle) const {
        if (cursor >= capacity) {
            recordRoundFailure(
                failure, 5u, camera,
                cursor > UINT32_MAX ? UINT32_MAX
                                    : static_cast<std::uint32_t>(cursor));
            return;
        }
        keys[cursor] = detail::tileIdKey(tile, particle);
        ids[cursor] = particle;
    }
};

__global__ void expandParticlesExpectedCount(
    tcnn::uvec2 tileGrid,
    std::uint32_t particleCount,
    const std::uint64_t* tileOffsets,
    const std::uint32_t* nativeTileCounts,
    const tcnn::vec2* projectedCenters,
    const tcnn::vec2* projectedExtents,
    const std::uint8_t* projectedValidity,
    const tcnn::vec4* conicOpacities,
    const float* maxConicPowers,
    const float4* supportCaps,
    detail::direct_support::CentralPhysicalDomain physicalDomain,
    tcnn::mat4x3 worldToSensor,
    std::uint32_t* certificationFailed,
    std::uint32_t* firstErrorIndex,
    std::uint64_t* keys,
    std::uint32_t* ids,
    std::uint32_t expectedCount,
    std::uint32_t camera,
    std::uint32_t* failure) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particleCount) return;
    // The sealed count is checked before any candidate write. A mismatch
    // leaves the deterministically-cleared candidate arena untouched.
    const std::uint64_t actualCount = tileOffsets[particleCount - 1u];
    if (actualCount != static_cast<std::uint64_t>(expectedCount)) {
        if (particle == 0u) {
            recordRoundFailure(
                failure, 1u, camera,
                actualCount > UINT32_MAX ? UINT32_MAX
                                         : static_cast<std::uint32_t>(actualCount));
        }
        return;
    }
    const std::uint64_t cursor =
        particle == 0u ? 0u : tileOffsets[particle - 1u];
    const std::uint64_t end = tileOffsets[particle];
    if (cursor > end || end > expectedCount) {
        recordRoundFailure(
            failure, 5u, camera,
            end > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(end));
        return;
    }
    if (cursor == end) return;
    const tcnn::vec2 center = projectedCenters[particle];
    const tcnn::vec2 extent = projectedExtents[particle];
    const tcnn::vec4 conic = conicOpacities[particle];
    const float maxPower = maxConicPowers[particle];
    const bool validProjection = projectedValidity[particle] != 0u;
    const TGUTProjector::BoundingBox2D nativeBox = validProjection
        ? TGUTProjector::computeTileSpaceBBox(tileGrid, center, extent)
        : TGUTProjector::BoundingBox2D{{0, 0}, {0, 0}};
    const float4 cap = supportCaps[particle];
    const detail::direct_support::SupportBoxResult support =
        detail::directSupportBox(physicalDomain, tileGrid, cap, worldToSensor);
    const detail::DirectNativeAdmission<TGUTProjector> admission{
        validProjection, nativeBox, conic, center, maxPower};
    detail::direct_support::TileBox directNativeBox{};
    if (!detail::checkedDirectTileBox(nativeBox, directNativeBox)) {
        detail::recordIndexedError(certificationFailed, firstErrorIndex, particle);
        return;
    }
    const detail::direct_support::EnumerationResult enumeration =
        detail::direct_support::enumerateContinuousPhysicalSupport(
            detail::direct_support::TileGrid{tileGrid.x, tileGrid.y}, particle,
            validProjection, directNativeBox, support,
            detail::direct_support::kEmit, cursor, end,
            nativeTileCounts[particle], admission,
            BoundedKeyEmitter{keys, ids, expectedCount, camera, failure});
    if (enumeration.failure != detail::direct_support::kEnumerationOk) {
        detail::recordIndexedError(certificationFailed, firstErrorIndex, particle);
    }
}

__global__ void fillQueryIds(std::uint32_t count, std::uint32_t* ids) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) ids[index] = index;
}

__global__ void preprocessActivePoints(
    RenderParameters params,
    const tcnn::vec3* sensorRayOrigins,
    tcnn::mat4x3 sensorToWorld,
    const std::uint8_t* physicalMask,
    const tcnn::vec3* queryPoints,
    const std::uint32_t* compactIds,
    const std::uint32_t* compactCount,
    std::uint32_t queryCount,
    std::uint64_t* keys,
    std::uint32_t* ids,
    tcnn::vec3* origins,
    tcnn::vec3* directions,
    float* ranges,
    std::uint8_t* valid,
    tcnn::uvec2 tileGrid) {
    const std::uint32_t compact = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact >= queryCount || compact >= *compactCount) return;
    const std::uint32_t query = compactIds[compact];

    tcnn::vec2 projected;
    if (!projectPointWithShutter<GAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS>(
            queryPoints[query], params.resolution, params.sensorModel,
            params.sensorState, 0.0f, projected)) return;
    const int x = static_cast<int>(floorf(projected.x));
    const int y = static_cast<int>(floorf(projected.y));
    if (x < 0 || y < 0 || x >= params.resolution.x || y >= params.resolution.y) return;
    const std::uint32_t pixel = static_cast<std::uint32_t>(x) +
        static_cast<std::uint32_t>(params.resolution.x) * static_cast<std::uint32_t>(y);
    if (physicalMask != nullptr && physicalMask[pixel] == 0u) return;

    const tcnn::vec3 origin = sensorToWorld *
        tcnn::vec4(sensorRayOrigins[pixel], 1.0f);
    const tcnn::vec3 delta = queryPoints[query] - origin;
    const float range = length(delta);
    if (!(range > 0.0f) || !isfinite(range)) return;
    const std::uint32_t gridX =
        (static_cast<std::uint32_t>(params.resolution.x) +
         GUTParameters::Tiling::BlockX - 1u) / GUTParameters::Tiling::BlockX;
    const std::uint32_t tileX = static_cast<std::uint32_t>(x) /
        GUTParameters::Tiling::BlockX;
    const std::uint32_t tileY = static_cast<std::uint32_t>(y) /
        GUTParameters::Tiling::BlockY;
    if (tileX >= tileGrid.x || tileY >= tileGrid.y) return;
    keys[compact] = detail::tileIdKey(tileX + gridX * tileY, query);
    ids[compact] = query;
    origins[query] = origin;
    directions[query] = delta / range;
    ranges[query] = range;
    valid[query] = 1u;
}

// This kernel invokes the same force-inlined response+alpha helper as the
// opacity field, so -use_fast_math contraction is identical in both.
__global__ void integratePreparedPoints(
    const tcnn::uvec2* candidateTileRanges,
    const std::uint32_t* candidateIds,
    const std::uint32_t* sortedPointIds,
    const field::PointBlock* pointBlocks,
    const std::uint32_t* totalPointBlocks,
    const tcnn::vec3* pointOrigins,
    const tcnn::vec3* pointDirections,
    const float* pointRanges,
    const std::uint8_t* pointValid,
    const float* particleMinimumRanges,
    float* cameraOpacity,
    std::uint8_t* cameraObserved,
    MemoryHandles handles) {
    if (blockIdx.x >= *totalPointBlocks) return;
    __shared__ std::uint32_t sharedIds[256];
    __shared__ float sharedLower[256];
    const field::PointBlock block = pointBlocks[blockIdx.x];
    const bool laneActive = threadIdx.x < block.count;
    const std::uint32_t query = laneActive
        ? sortedPointIds[block.sortedBegin + threadIdx.x] : 0u;
    const tcnn::uvec2 candidates = candidateTileRanges[block.tile];
    float transmittance = 1.0f;
    bool strictlyAbove = false;
    model_::Particles particles;
    particles.initializeDensity(handles);
    for (std::uint32_t batch = candidates.x; batch < candidates.y; batch += 256u) {
        const std::uint32_t load = batch + threadIdx.x;
        if (load < candidates.y) {
            const std::uint32_t id = candidateIds[load];
            sharedIds[threadIdx.x] = id;
            sharedLower[threadIdx.x] = particleMinimumRanges[id];
        }
        __syncthreads();
        const std::uint32_t count = min(256u, candidates.y - batch);
        if (laneActive && pointValid[query] != 0u && !strictlyAbove) {
            const float queryRange = pointRanges[query];
            for (std::uint32_t slot = 0; slot < count; ++slot) {
                const float lower = sharedLower[slot];
                if (lower >= 0.0f && lower > queryRange) continue;
                field::fixed8ResponseAlphaUpdate(
                    particles, sharedIds[slot], pointOrigins[query],
                    pointDirections[query], queryRange, transmittance);
                if (1.0f - transmittance > 0.5f) {
                    strictlyAbove = true;
                    break;
                }
            }
        }
        __syncthreads();
    }
    if (laneActive && pointValid[query] != 0u) {
        cameraOpacity[query] = strictlyAbove ? 1.0f : 1.0f - transmittance;
        cameraObserved[query] = 1u;
    }
}

__global__ void classifyCamera(
    std::uint32_t queryCount,
    std::uint32_t camera,
    const float* cameraOpacity,
    const std::uint8_t* cameraObserved,
    std::uint8_t* active,
    std::uint8_t* observed,
    std::uint8_t* exactEqual,
    std::uint32_t* counters) {
    const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= queryCount || active[query] == 0u) return;
    std::uint32_t* row = counters + camera * 3u;
    if (cameraObserved[query] != 0u) {
        atomicAdd(row + 1u, 1u);
        observed[query] = 1u;
        const float opacity = cameraOpacity[query];
        if (opacity < 0.5f) {
            active[query] = 0u;
            exactEqual[query] = 0u;
            atomicAdd(row + 2u, 1u);
            return;
        }
        if (opacity == 0.5f) exactEqual[query] = 1u;
    }
    atomicAdd(row, 1u);
}

__global__ void checkPointBlockBound(
    const std::uint32_t* totalPointBlocks,
    std::uint32_t bound,
    std::uint32_t camera,
    std::uint32_t* failure) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *totalPointBlocks > bound) {
        recordRoundFailure(failure, 4u, camera, *totalPointBlocks);
    }
}

__global__ void fillBlocksBounded(
    std::uint32_t tileCount,
    const tcnn::uvec2* ranges,
    const std::uint32_t* counts,
    const std::uint32_t* offsets,
    field::PointBlock* blocks,
    std::uint32_t bound,
    std::uint32_t camera,
    std::uint32_t* failure) {
    const std::uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tileCount) return;
    const std::uint32_t begin = ranges[tile].x;
    const std::uint32_t points = ranges[tile].y - begin;
    for (std::uint32_t block = 0; block < counts[tile]; ++block) {
        const std::uint32_t output = offsets[tile] + block;
        if (output >= bound) {
            recordRoundFailure(failure, 4u, camera, output);
            continue;
        }
        const std::uint32_t local = block * 256u;
        blocks[output] = field::PointBlock{
            tile, begin + local, min(256u, points - local)};
    }
}

__global__ void finalizeDecisionVectors(
    std::uint32_t queryCount,
    std::uint8_t* active,
    const std::uint8_t* observed,
    std::uint8_t* exactEqual) {
    const std::uint32_t query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= queryCount) return;
    // Tie-aware sign: above = observed && no completed camera below;
    // exact = above && at least one completed camera exactly equal.
    const std::uint8_t above =
        (active[query] != 0u && observed[query] != 0u) ? 1u : 0u;
    active[query] = above;
    exactEqual[query] =
        (above != 0u && exactEqual[query] != 0u) ? 1u : 0u;
}

} // namespace

struct Table {
    std::uint32_t particles = 0;
    std::uint32_t queries = 0;
    const float* sigmaMax = nullptr;
    std::uint64_t* parameterBuffer = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    bool sealed = false;
    field::CooperativeWorkspace workspace;
    std::vector<Camera> cameras;
    std::uint32_t maxIntersections = 0;
    std::uint32_t maxTiles = 0;
    std::uint32_t maxPointBlocks = 0;
};

Table* createTable(
    std::uint32_t particleCount,
    std::uint32_t queryCountValue,
    const float* particleSigmaMax,
    std::uint64_t* parameterBuffer,
    int device,
    cudaStream_t stream,
    const Logger&) {
    // CUDA stream value 0 is the valid default stream, so a null stream handle
    // is accepted.
    if (particleCount == 0u || queryCountValue == 0u || device < 0 ||
        particleSigmaMax == nullptr || parameterBuffer == nullptr) {
        return nullptr;
    }
    Table* table = new (std::nothrow) Table();
    if (table == nullptr) return nullptr;
    table->particles = particleCount;
    table->queries = queryCountValue;
    table->sigmaMax = particleSigmaMax;
    table->parameterBuffer = parameterBuffer;
    table->device = device;
    table->stream = stream;
    return table;
}

Status appendCamera(
    Table& table,
    const RenderParameters& params,
    const tcnn::vec3* rayOrigins,
    const tcnn::vec3* rayDirections,
    const std::uint8_t* physicalMask,
    tcnn::mat4x3 worldToSensor,
    tcnn::mat4x3 sensorToWorld,
    CameraStageTelemetry& telemetry,
    const Logger& logger) {
    if (table.sealed || rayOrigins == nullptr || rayDirections == nullptr) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent camera table append received invalid state");
    }
    tcnn::uvec2 tileGrid = tcnn::uvec2::zero();
    std::uint32_t tiles = 0u;
    detail::direct_support::CentralPhysicalDomain domain{};
    if (!field::directFieldGeometry(params, tileGrid, tiles) ||
        !detail::makeDirectPhysicalDomain(params, domain)) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent camera table physical domain differs");
    }
    CHECK_STATUS_RETURN(table.workspace.reserveFixed(
        table.particles, table.queries, tiles, table.stream, logger));
    CUDA_CHECK_RETURN(cudaMemsetAsync(table.workspace.certificationFailed.data(), 0,
        sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        table.workspace.certificationFirstErrorIndex.data(), 0xff,
        sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(table.workspace.boundedParticleReasonMask.data(), 0,
        sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(table.workspace.boundedParticleFirst.data(), 0xff,
        sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(table.workspace.boundedParticleCount.data(), 0,
        sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(table.workspace.cameraInputInvalid.data(), 0,
        sizeof(std::uint32_t), table.stream), logger);

    const std::uint32_t pixelCount = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(params.resolution.x) * params.resolution.y);
    detail::validateCameraRays<<<
        (pixelCount + 255u) / 256u, 256u, 0, table.stream>>>(
        pixelCount, rayOrigins, rayDirections,
        static_cast<std::uint32_t*>(table.workspace.cameraInputInvalid.data()),
        static_cast<std::uint32_t*>(
            table.workspace.certificationFirstErrorIndex.data()));

    const tcnn::vec3 cameraWorld = sensorToWorld *
        tcnn::vec4(tcnn::vec3::zero(), 1.0f);
    const std::uint32_t particleBlocks =
        (table.particles + field::kThreads - 1u) / field::kThreads;
    detail::preprocessParticles<<<particleBlocks, field::kThreads, 0, table.stream>>>(
        tileGrid, table.particles, params, cameraWorld, worldToSensor, domain,
        table.sigmaMax,
        static_cast<std::uint64_t*>(table.workspace.particleTileCounts.data()),
        static_cast<std::uint32_t*>(table.workspace.particleNativeTileCounts.data()),
        static_cast<tcnn::vec2*>(table.workspace.particleProjectedCenter.data()),
        static_cast<tcnn::vec2*>(table.workspace.particleProjectedExtent.data()),
        static_cast<std::uint8_t*>(table.workspace.particleValidProjection.data()),
        static_cast<tcnn::vec4*>(table.workspace.particleConicOpacity.data()),
        static_cast<float*>(table.workspace.particleMaxConicPower.data()),
        static_cast<float4*>(table.workspace.particleSupportCap.data()),
        static_cast<float*>(table.workspace.particleMinimumRange.data()),
        static_cast<std::uint32_t*>(table.workspace.boundedParticleReasonMask.data()),
        static_cast<std::uint32_t*>(table.workspace.boundedParticleFirst.data()),
        static_cast<std::uint32_t*>(table.workspace.boundedParticleCount.data()),
        static_cast<std::uint32_t*>(table.workspace.certificationFailed.data()),
        static_cast<std::uint32_t*>(table.workspace.certificationFirstErrorIndex.data()),
        MemoryHandles{table.parameterBuffer});
    CUDA_CHECK_STREAM_RETURN(table.stream, logger);
    size_t bytes = table.workspace.candidateScanTemp.size();
    CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
        table.workspace.candidateScanTemp.data(), bytes,
        static_cast<const std::uint64_t*>(table.workspace.particleTileCounts.data()),
        static_cast<std::uint64_t*>(table.workspace.particleTileOffsets.data()),
        static_cast<int>(table.particles), table.stream), logger);

    std::uint64_t intersections64 = 0;
    std::uint32_t failed = 0, first = UINT32_MAX, cameraInvalid = 0;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        &intersections64,
        static_cast<const std::uint64_t*>(table.workspace.particleTileOffsets.data()) +
            table.particles - 1u,
        sizeof(intersections64), cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&failed,
        table.workspace.certificationFailed.data(), sizeof(failed),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&first,
        table.workspace.certificationFirstErrorIndex.data(), sizeof(first),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&cameraInvalid,
        table.workspace.cameraInputInvalid.data(), sizeof(cameraInvalid),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&telemetry.boundedReasonMask,
        table.workspace.boundedParticleReasonMask.data(), sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&telemetry.boundedFirst,
        table.workspace.boundedParticleFirst.data(), sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&telemetry.boundedCount,
        table.workspace.boundedParticleCount.data(), sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(table.stream), logger);
    if (cameraInvalid != 0u || failed != 0u || intersections64 == 0u ||
        intersections64 > INT_MAX) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent camera table candidate census failed closed");
    }
    telemetry.intersections = static_cast<std::uint32_t>(intersections64);
    CHECK_STATUS_RETURN(table.workspace.reserveCandidates(
        telemetry.intersections, tiles, table.stream, logger));
    detail::expandParticles<<<particleBlocks, field::kThreads, 0, table.stream>>>(
        tileGrid, table.particles,
        static_cast<const std::uint64_t*>(table.workspace.particleTileOffsets.data()),
        static_cast<const std::uint32_t*>(table.workspace.particleNativeTileCounts.data()),
        static_cast<const tcnn::vec2*>(table.workspace.particleProjectedCenter.data()),
        static_cast<const tcnn::vec2*>(table.workspace.particleProjectedExtent.data()),
        static_cast<const std::uint8_t*>(table.workspace.particleValidProjection.data()),
        static_cast<const tcnn::vec4*>(table.workspace.particleConicOpacity.data()),
        static_cast<const float*>(table.workspace.particleMaxConicPower.data()),
        static_cast<const float4*>(table.workspace.particleSupportCap.data()),
        domain, worldToSensor,
        static_cast<std::uint32_t*>(table.workspace.certificationFailed.data()),
        static_cast<std::uint32_t*>(table.workspace.certificationFirstErrorIndex.data()),
        static_cast<std::uint64_t*>(table.workspace.candidateKeysUnsorted.data()),
        static_cast<std::uint32_t*>(table.workspace.candidateIdsUnsorted.data()));
    bytes = table.workspace.candidateSortTemp.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
        table.workspace.candidateSortTemp.data(), bytes,
        static_cast<const std::uint64_t*>(table.workspace.candidateKeysUnsorted.data()),
        static_cast<std::uint64_t*>(table.workspace.candidateKeysSorted.data()),
        static_cast<const std::uint32_t*>(table.workspace.candidateIdsUnsorted.data()),
        static_cast<std::uint32_t*>(table.workspace.candidateIdsSorted.data()),
        static_cast<int>(telemetry.intersections), 0,
        32 + detail::bitWidth(tiles), table.stream), logger);
    detail::validateSortedCandidateKeys<<<
        (telemetry.intersections + 255u) / 256u, 256u, 0, table.stream>>>(
        static_cast<const std::uint64_t*>(table.workspace.candidateKeysSorted.data()),
        static_cast<const std::uint32_t*>(table.workspace.candidateIdsSorted.data()),
        telemetry.intersections, tiles,
        static_cast<std::uint32_t*>(table.workspace.certificationFailed.data()),
        static_cast<std::uint32_t*>(table.workspace.certificationFirstErrorIndex.data()));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&failed,
        table.workspace.certificationFailed.data(), sizeof(failed),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&first,
        table.workspace.certificationFirstErrorIndex.data(), sizeof(first),
        cudaMemcpyDeviceToHost, table.stream), logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(table.stream), logger);
    if (failed != 0u) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent camera table expansion/sort prepass failed closed");
    }
    table.maxIntersections = std::max(table.maxIntersections, telemetry.intersections);
    table.maxTiles = std::max(table.maxTiles, tiles);
    table.cameras.push_back(Camera{
        params, rayOrigins, nullptr, physicalMask, worldToSensor,
        sensorToWorld, tileGrid, tiles, telemetry.intersections});
    (void)first;
    return Status();
}

Status sealTable(Table& table, const Logger& logger) {
    if (table.sealed || table.cameras.empty() || table.maxIntersections == 0u) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent camera table cannot seal empty/duplicate state");
    }
    CHECK_STATUS_RETURN(table.workspace.reserveFixed(
        table.particles, table.queries, table.maxTiles, table.stream, logger));
    CHECK_STATUS_RETURN(table.workspace.reserveCandidates(
        table.maxIntersections, table.maxTiles, table.stream, logger));
    // If n_t is the number of active points in tile t, PointBlock count is
    // sum_t ceil(n_t/256).  Since sum_t n_t <= Q and there are at most T
    // tiles, this is <= ceil(Q/256)+T.  It is also <= Q because every nonempty
    // block owns at least one distinct point.  Compute in uint64, then take the
    // tighter bound; the final value is therefore in [1,Q] and cannot overflow
    // uint32 because Q is already a validated uint32/INT_MAX-bounded count.
    const std::uint64_t queryCeil =
        (static_cast<std::uint64_t>(table.queries) + 255u) / 256u;
    const std::uint64_t looseBound =
        queryCeil + static_cast<std::uint64_t>(table.maxTiles);
    const std::uint64_t tightBound = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(table.queries), looseBound);
    if (tightBound == 0u || tightBound > UINT32_MAX ||
        tightBound > std::numeric_limits<size_t>::max() /
            sizeof(field::PointBlock)) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent point-block bound overflow");
    }
    table.maxPointBlocks = static_cast<std::uint32_t>(tightBound);
    CHECK_STATUS_RETURN(table.workspace.pointBlocks.resize(
        static_cast<size_t>(table.maxPointBlocks) * sizeof(field::PointBlock),
        reinterpret_cast<std::uint64_t>(table.stream), logger));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(table.stream), logger);
    table.sealed = true;
    return Status();
}

Status enqueueRound(
    Table& table,
    const tcnn::vec3* queryPoints,
    std::uint8_t* active,
    std::uint8_t* observed,
    std::uint8_t* exactEqual,
    std::uint32_t* cameraCounters,
    std::uint32_t* deviceFailure,
    std::uint8_t* packedRoundOutput,
    size_t packedRoundBytes,
    const Logger& logger) {
    if (!table.sealed || queryPoints == nullptr || active == nullptr ||
        observed == nullptr || exactEqual == nullptr || cameraCounters == nullptr ||
        deviceFailure == nullptr || packedRoundOutput == nullptr) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent fixed8 round received invalid state");
    }
    const size_t vectorBytes = static_cast<size_t>(table.queries);
    if (vectorBytes > (std::numeric_limits<size_t>::max() - 3u) / 3u) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent packed vector byte overflow");
    }
    const size_t reportOffset = (3u * vectorBytes + 3u) & ~size_t{3u};
    if (table.cameras.size() >
        (std::numeric_limits<size_t>::max() / sizeof(std::uint32_t) - 3u) / 3u) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent packed report word overflow");
    }
    const size_t reportBytes =
        (3u + table.cameras.size() * 3u) * sizeof(std::uint32_t);
    if (reportOffset > std::numeric_limits<size_t>::max() - reportBytes ||
        packedRoundBytes != reportOffset + reportBytes) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "persistent packed round byte layout differs");
    }
    CUDA_CHECK_RETURN(cudaMemsetAsync(active, 1, table.queries, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(observed, 0, table.queries, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(exactEqual, 0, table.queries, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(cameraCounters, 0,
        table.cameras.size() * 3u * sizeof(std::uint32_t), table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(deviceFailure, 0,
        3u * sizeof(std::uint32_t), table.stream), logger);

    const std::uint32_t particleBlocks =
        (table.particles + field::kThreads - 1u) / field::kThreads;
    const std::uint32_t queryGrid = (table.queries + 255u) / 256u;
    fillQueryIds<<<queryGrid, 256u, 0, table.stream>>>(
        table.queries,
        static_cast<std::uint32_t*>(table.workspace.activeIdsInput.data()));
    for (std::uint32_t cameraOrdinal = 0;
         cameraOrdinal < table.cameras.size(); ++cameraOrdinal) {
        const Camera& camera = table.cameras[cameraOrdinal];
        auto& workspace = table.workspace;
        detail::direct_support::CentralPhysicalDomain domain{};
        if (!detail::makeDirectPhysicalDomain(camera.params, domain)) {
            RETURN_ERROR(logger, ErrorCode::BadInput,
                         "sealed camera physical domain changed");
        }
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.candidateTileRanges.data(), 0,
            sizeof(tcnn::uvec2) * camera.tileCount, table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.pointTileRanges.data(), 0,
            sizeof(tcnn::uvec2) * camera.tileCount, table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.pointKeysUnsorted.data(), 0xff,
            sizeof(std::uint64_t) * table.queries, table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.pointIdsUnsorted.data(), 0,
            sizeof(std::uint32_t) * table.queries, table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.pointValid.data(), 0,
            sizeof(std::uint8_t) * table.queries, table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.certificationFailed.data(), 0,
            sizeof(std::uint32_t), table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            workspace.certificationFirstErrorIndex.data(), 0xff,
            sizeof(std::uint32_t), table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleReasonMask.data(), 0,
            sizeof(std::uint32_t), table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleFirst.data(), 0xff,
            sizeof(std::uint32_t), table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleCount.data(), 0,
            sizeof(std::uint32_t), table.stream), logger);

        const tcnn::vec3 cameraWorld = camera.sensorToWorld *
            tcnn::vec4(tcnn::vec3::zero(), 1.0f);
        detail::preprocessParticles<<<particleBlocks, field::kThreads, 0, table.stream>>>(
            camera.tileGrid, table.particles, camera.params, cameraWorld,
            camera.worldToSensor, domain, table.sigmaMax,
            static_cast<std::uint64_t*>(workspace.particleTileCounts.data()),
            static_cast<std::uint32_t*>(workspace.particleNativeTileCounts.data()),
            static_cast<tcnn::vec2*>(workspace.particleProjectedCenter.data()),
            static_cast<tcnn::vec2*>(workspace.particleProjectedExtent.data()),
            static_cast<std::uint8_t*>(workspace.particleValidProjection.data()),
            static_cast<tcnn::vec4*>(workspace.particleConicOpacity.data()),
            static_cast<float*>(workspace.particleMaxConicPower.data()),
            static_cast<float4*>(workspace.particleSupportCap.data()),
            static_cast<float*>(workspace.particleMinimumRange.data()),
            static_cast<std::uint32_t*>(workspace.boundedParticleReasonMask.data()),
            static_cast<std::uint32_t*>(workspace.boundedParticleFirst.data()),
            static_cast<std::uint32_t*>(workspace.boundedParticleCount.data()),
            static_cast<std::uint32_t*>(workspace.certificationFailed.data()),
            static_cast<std::uint32_t*>(workspace.certificationFirstErrorIndex.data()),
            MemoryHandles{table.parameterBuffer});
        size_t bytes = workspace.candidateScanTemp.size();
        CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
            workspace.candidateScanTemp.data(), bytes,
            static_cast<const std::uint64_t*>(workspace.particleTileCounts.data()),
            static_cast<std::uint64_t*>(workspace.particleTileOffsets.data()),
            static_cast<int>(table.particles), table.stream), logger);
        checkPreparedCount<<<1, 1, 0, table.stream>>>(
            static_cast<const std::uint64_t*>(workspace.particleTileOffsets.data()),
            table.particles, camera.intersections, cameraOrdinal, deviceFailure);
        checkDeviceFlag<<<1, 1, 0, table.stream>>>(
            static_cast<const std::uint32_t*>(workspace.certificationFailed.data()),
            static_cast<const std::uint32_t*>(workspace.certificationFirstErrorIndex.data()),
            2u, cameraOrdinal, deviceFailure);

        // A count mismatch must not expose stale candidates or write beyond
        // the sealed arena. Clear exactly the sealed range, then let every
        // expansion write re-check that range. Sorting the sealed count is
        // deterministic even on the failing path, which is rejected at the
        // single end-round status read.
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            workspace.candidateKeysUnsorted.data(), 0xff,
            static_cast<size_t>(camera.intersections) * sizeof(std::uint64_t),
            table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            workspace.candidateIdsUnsorted.data(), 0xff,
            static_cast<size_t>(camera.intersections) * sizeof(std::uint32_t),
            table.stream), logger);
        expandParticlesExpectedCount<<<
            particleBlocks, field::kThreads, 0, table.stream>>>(
            camera.tileGrid, table.particles,
            static_cast<const std::uint64_t*>(workspace.particleTileOffsets.data()),
            static_cast<const std::uint32_t*>(workspace.particleNativeTileCounts.data()),
            static_cast<const tcnn::vec2*>(workspace.particleProjectedCenter.data()),
            static_cast<const tcnn::vec2*>(workspace.particleProjectedExtent.data()),
            static_cast<const std::uint8_t*>(workspace.particleValidProjection.data()),
            static_cast<const tcnn::vec4*>(workspace.particleConicOpacity.data()),
            static_cast<const float*>(workspace.particleMaxConicPower.data()),
            static_cast<const float4*>(workspace.particleSupportCap.data()),
            domain, camera.worldToSensor,
            static_cast<std::uint32_t*>(workspace.certificationFailed.data()),
            static_cast<std::uint32_t*>(workspace.certificationFirstErrorIndex.data()),
            static_cast<std::uint64_t*>(workspace.candidateKeysUnsorted.data()),
            static_cast<std::uint32_t*>(workspace.candidateIdsUnsorted.data()),
            camera.intersections, cameraOrdinal, deviceFailure);
        bytes = workspace.candidateSortTemp.size();
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            workspace.candidateSortTemp.data(), bytes,
            static_cast<const std::uint64_t*>(workspace.candidateKeysUnsorted.data()),
            static_cast<std::uint64_t*>(workspace.candidateKeysSorted.data()),
            static_cast<const std::uint32_t*>(workspace.candidateIdsUnsorted.data()),
            static_cast<std::uint32_t*>(workspace.candidateIdsSorted.data()),
            static_cast<int>(camera.intersections), 0,
            32 + detail::bitWidth(camera.tileCount), table.stream), logger);
        detail::validateSortedCandidateKeys<<<
            (camera.intersections + 255u) / 256u, 256u, 0, table.stream>>>(
            static_cast<const std::uint64_t*>(workspace.candidateKeysSorted.data()),
            static_cast<const std::uint32_t*>(workspace.candidateIdsSorted.data()),
            camera.intersections, camera.tileCount,
            static_cast<std::uint32_t*>(workspace.certificationFailed.data()),
            static_cast<std::uint32_t*>(workspace.certificationFirstErrorIndex.data()));
        checkDeviceFlag<<<1, 1, 0, table.stream>>>(
            static_cast<const std::uint32_t*>(workspace.certificationFailed.data()),
            static_cast<const std::uint32_t*>(workspace.certificationFirstErrorIndex.data()),
            3u, cameraOrdinal, deviceFailure);
        detail::identifyRanges<<<
            (camera.intersections + 255u) / 256u, 256u, 0, table.stream>>>(
            static_cast<const std::uint64_t*>(workspace.candidateKeysSorted.data()),
            camera.intersections, camera.tileCount,
            static_cast<tcnn::uvec2*>(workspace.candidateTileRanges.data()));

        bytes = workspace.activeSelectTemp.size();
        CUDA_CHECK_RETURN(cub::DeviceSelect::Flagged(
            workspace.activeSelectTemp.data(), bytes,
            static_cast<const std::uint32_t*>(workspace.activeIdsInput.data()),
            active,
            static_cast<std::uint32_t*>(workspace.activeIdsCompacted.data()),
            static_cast<std::uint32_t*>(workspace.activeCount.data()),
            static_cast<int>(table.queries), table.stream), logger);
        preprocessActivePoints<<<queryGrid, 256u, 0, table.stream>>>(
            camera.params, camera.rayOrigins, camera.sensorToWorld,
            camera.physicalMask, queryPoints,
            static_cast<const std::uint32_t*>(workspace.activeIdsCompacted.data()),
            static_cast<const std::uint32_t*>(workspace.activeCount.data()),
            table.queries,
            static_cast<std::uint64_t*>(workspace.pointKeysUnsorted.data()),
            static_cast<std::uint32_t*>(workspace.pointIdsUnsorted.data()),
            static_cast<tcnn::vec3*>(workspace.pointOrigins.data()),
            static_cast<tcnn::vec3*>(workspace.pointDirections.data()),
            static_cast<float*>(workspace.pointRanges.data()),
            static_cast<std::uint8_t*>(workspace.pointValid.data()), camera.tileGrid);
        bytes = workspace.pointSortTemp.size();
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
            workspace.pointSortTemp.data(), bytes,
            static_cast<const std::uint64_t*>(workspace.pointKeysUnsorted.data()),
            static_cast<std::uint64_t*>(workspace.pointKeysSorted.data()),
            static_cast<const std::uint32_t*>(workspace.pointIdsUnsorted.data()),
            static_cast<std::uint32_t*>(workspace.pointIdsSorted.data()),
            static_cast<int>(table.queries), 0,
            32 + detail::bitWidth(camera.tileCount), table.stream), logger);
        detail::identifyRanges<<<queryGrid, 256u, 0, table.stream>>>(
            static_cast<const std::uint64_t*>(workspace.pointKeysSorted.data()),
            table.queries, camera.tileCount,
            static_cast<tcnn::uvec2*>(workspace.pointTileRanges.data()));
        const std::uint32_t tileBlocks = (camera.tileCount + 255u) / 256u;
        detail::blocksPerTile<<<tileBlocks, 256u, 0, table.stream>>>(
            camera.tileCount,
            static_cast<const tcnn::uvec2*>(workspace.pointTileRanges.data()),
            static_cast<std::uint32_t*>(workspace.pointBlocksPerTile.data()));
        bytes = workspace.pointScanTemp.size();
        CUDA_CHECK_RETURN(cub::DeviceScan::ExclusiveSum(
            workspace.pointScanTemp.data(), bytes,
            static_cast<const std::uint32_t*>(workspace.pointBlocksPerTile.data()),
            static_cast<std::uint32_t*>(workspace.pointBlockOffsets.data()),
            static_cast<int>(camera.tileCount), table.stream), logger);
        detail::totalBlocks<<<1, 1, 0, table.stream>>>(
            camera.tileCount,
            static_cast<const std::uint32_t*>(workspace.pointBlocksPerTile.data()),
            static_cast<const std::uint32_t*>(workspace.pointBlockOffsets.data()),
            static_cast<std::uint32_t*>(workspace.totalPointBlocks.data()));
        checkPointBlockBound<<<1, 1, 0, table.stream>>>(
            static_cast<const std::uint32_t*>(workspace.totalPointBlocks.data()),
            table.maxPointBlocks, cameraOrdinal, deviceFailure);
        fillBlocksBounded<<<tileBlocks, 256u, 0, table.stream>>>(
            camera.tileCount,
            static_cast<const tcnn::uvec2*>(workspace.pointTileRanges.data()),
            static_cast<const std::uint32_t*>(workspace.pointBlocksPerTile.data()),
            static_cast<const std::uint32_t*>(workspace.pointBlockOffsets.data()),
            static_cast<field::PointBlock*>(workspace.pointBlocks.data()),
            table.maxPointBlocks, cameraOrdinal, deviceFailure);

        // Full-Q scratch avoids every per-camera allocation and host compaction.
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            workspace.cameraOpacityScratch.data(), 0,
            table.queries * sizeof(float), table.stream), logger);
        CUDA_CHECK_RETURN(cudaMemsetAsync(
            workspace.cameraObservedScratch.data(), 0,
            table.queries * sizeof(std::uint8_t), table.stream), logger);
        // The block count is device-produced. Launch the proven static bound;
        // every excess block exits before touching pointBlocks.
        integratePreparedPoints<<<table.maxPointBlocks, 256u, 0, table.stream>>>(
            static_cast<const tcnn::uvec2*>(workspace.candidateTileRanges.data()),
            static_cast<const std::uint32_t*>(workspace.candidateIdsSorted.data()),
            static_cast<const std::uint32_t*>(workspace.pointIdsSorted.data()),
            static_cast<const field::PointBlock*>(workspace.pointBlocks.data()),
            static_cast<const std::uint32_t*>(workspace.totalPointBlocks.data()),
            static_cast<const tcnn::vec3*>(workspace.pointOrigins.data()),
            static_cast<const tcnn::vec3*>(workspace.pointDirections.data()),
            static_cast<const float*>(workspace.pointRanges.data()),
            static_cast<const std::uint8_t*>(workspace.pointValid.data()),
            static_cast<const float*>(workspace.particleMinimumRange.data()),
            static_cast<float*>(workspace.cameraOpacityScratch.data()),
            static_cast<std::uint8_t*>(workspace.cameraObservedScratch.data()),
            MemoryHandles{table.parameterBuffer});
        classifyCamera<<<queryGrid, 256u, 0, table.stream>>>(
            table.queries, cameraOrdinal,
            static_cast<const float*>(workspace.cameraOpacityScratch.data()),
            static_cast<const std::uint8_t*>(workspace.cameraObservedScratch.data()),
            active, observed, exactEqual, cameraCounters);
    }
    finalizeDecisionVectors<<<queryGrid, 256u, 0, table.stream>>>(
        table.queries, active, observed, exactEqual);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        packedRoundOutput, 0, packedRoundBytes, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        packedRoundOutput, active, vectorBytes,
        cudaMemcpyDeviceToDevice, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        packedRoundOutput + vectorBytes, observed, vectorBytes,
        cudaMemcpyDeviceToDevice, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        packedRoundOutput + 2u * vectorBytes, exactEqual, vectorBytes,
        cudaMemcpyDeviceToDevice, table.stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        packedRoundOutput + reportOffset, deviceFailure, reportBytes,
        cudaMemcpyDeviceToDevice, table.stream), logger);
    CUDA_CHECK_RETURN(cudaGetLastError(), logger);
    return Status();
}

std::uint32_t cameraCount(const Table& table) {
    return static_cast<std::uint32_t>(table.cameras.size());
}

std::uint32_t queryCount(const Table& table) { return table.queries; }

std::uint32_t pointBlockBound(const Table& table) {
    return table.maxPointBlocks;
}

DestroyResult destroyTable(Table* table) noexcept {
    DestroyResult result;
    if (table == nullptr) {
        result.freed = true;
        return result;
    }
    const int ownerDevice = table->device;
    int priorDevice = -1;
    result.getDevice = cudaGetDevice(&priorDevice);
    if (result.getDevice != cudaSuccess) return result;
    result.setDevice = cudaSetDevice(ownerDevice);
    if (result.setDevice != cudaSuccess) return result;
    result.synchronize = cudaStreamSynchronize(table->stream);
    if (result.synchronize == cudaSuccess) {
        // CudaBuffer destruction is reached only after the exact owning stream
        // completed successfully on its owning device.
        delete table;
        result.freed = true;
    }
    if (priorDevice != ownerDevice) {
        result.restoreDevice = cudaSetDevice(priorDevice);
    }
    return result;
}

} // namespace threedgut::fast_sof_persistent_camera_table
