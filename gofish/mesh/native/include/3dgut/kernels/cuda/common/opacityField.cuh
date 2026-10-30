// Portions derived from SOF and GOF (https://github.com/r4dl/SOF, https://github.com/autonomousvision/gaussian-opacity-fields),
// distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
// research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md
// Matched camera-min opacity field at query points (mesh extraction).
//
// Particles are read as packed full-shape ParticleDensity records (B, B^-1).
// For each camera, the point opacity is 1 - prod(1 - alpha_i) over the
// Gaussians hit on the closed segment [0, t_query] from the camera center to
// the query point. A particle is skipped only when a conservative sphere from
// sigma_max(B) and the packed B columns proves that its support starts beyond
// t_query; invalid support data never skips a particle.
#pragma once

#include <3dgut/threedgut.cuh>

#include <3dgut/kernels/cuda/models/gutHalfRayResponse.cuh>
#include <3dgut/kernels/cuda/sensors/cameraProjections.cuh>

#include "continuousPhysicalSupportEnumerator.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <math_constants.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>




namespace threedgut::fast_sof_tnt_full_shape {

static_assert(sizeof(threedgut::ParticleDensity) == 32 * sizeof(float),
              "ParticleDensity ABI must be float32 [N,32]");
static_assert(offsetof(threedgut::ParticleDensity, position) == 0 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, density) == 3 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, quaternion) == 4 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, scale) == 8 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, padding) == 11 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, factor) == 12 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, inverseFactor) == 21 * sizeof(float));
static_assert(offsetof(threedgut::ParticleDensity, fullShapePadding) == 30 * sizeof(float));

constexpr uint32_t kThreads = 128;
constexpr float kTau = 0.5f;

enum CooperativeFault : uint32_t {
    kFaultNone = 0,
    kFaultGeometry = 1u << 0,
    kFaultCountOverflow = 1u << 1,
    kFaultCandidateAllocation = 1u << 2,
    kFaultExpansionMismatch = 1u << 3,
    kFaultCandidateSort = 1u << 4,
    kFaultInvalidCap = 1u << 5,
    kFaultZeroIntersections = 1u << 6,
};


__device__ __forceinline__ float canonicalSupportRadius(float opacity) {
    if (!(GAUSSIAN_PARTICLE_MAX_ALPHA > GAUSSIAN_PARTICLE_MIN_ALPHA) ||
        !(opacity > GAUSSIAN_PARTICLE_MIN_ALPHA) || !isfinite(opacity)) {
        return -1.0f;
    }
    const float threshold = fmaxf(
        GAUSSIAN_PARTICLE_MIN_KERNEL_DENSITY,
        GAUSSIAN_PARTICLE_MIN_ALPHA / opacity);
    if (!(threshold < 1.0f)) return -1.0f;
    const float safe = fmaxf(threshold, FLT_MIN);
#if GAUSSIAN_PARTICLE_KERNEL_DEGREE == 0
    return fmaxf(0.0f, (1.0f - safe) / 0.329630334487f);
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 1
    return -logf(safe) / 1.5f;
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 2
    return sqrtf(-logf(safe) / 0.5f);
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 3
    return powf(-logf(safe) / 0.166666666667f, 1.0f / 3.0f);
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 4
    return powf(-logf(safe) / 0.0555555555556f, 0.25f);
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 5
    return powf(-logf(safe) / 0.0185185185185f, 0.2f);
#elif GAUSSIAN_PARTICLE_KERNEL_DEGREE == 8
    return powf(-logf(safe) / 0.000685871056241f, 0.125f);
#else
#error "full-shape field does not know this particle kernel degree"
#endif
}

__device__ __forceinline__ double packedFactorFrobeniusUpper(
    const threedgut::ParticleDensity& particle) {
    // factor is standard C-row-major B. The Frobenius norm, computed from all
    // B columns, upper-bounds sigma_max(B). Every product, sum, and square root
    // is explicitly rounded upward in double precision.
    const float* values = reinterpret_cast<const float*>(&particle.factor[0]);
    double sum = 0.0;
#pragma unroll
    for (int entry = 0; entry < 9; ++entry) {
        const double value = fabs((double)values[entry]);
        sum = __dadd_ru(sum, __dmul_ru(value, value));
    }
    if (!(sum > 0.0) || !isfinite(sum)) return -1.0;
    return __dsqrt_ru(sum);
}

__device__ __forceinline__ double representedFloatAbsDifferenceLower(
    float lhs, float rhs) {
    const double lo = __dsub_rd((double)lhs, (double)rhs);
    const double hi = __dsub_ru((double)lhs, (double)rhs);
    if (lo <= 0.0 && hi >= 0.0) return 0.0;
    return fmin(fabs(lo), fabs(hi));
}

__device__ __forceinline__ float conservativeMinimumUnitRange(
    const threedgut::ParticleDensity& particle,
    float artifactSigmaMax,
    const tcnn::vec3& cameraOrigin) {
    const double columnUpper = packedFactorFrobeniusUpper(particle);
    if (!(artifactSigmaMax > 0.0f) || !isfinite(artifactSigmaMax) ||
        !(columnUpper > 0.0) || !isfinite(columnUpper)) {
        return -1.0f;
    }
    const double sigmaUpper = fmax((double)artifactSigmaMax, columnUpper);
    // Degree-2 response >0.0113 has canonical radius <3.0. The 3.33 radius is
    // a strict, density-independent support enclosure shared with the angular
    // candidate union.
    const double supportRadiusUpper = __dmul_ru((double)3.33f, sigmaUpper);
    // Directed subtraction intervals cover every represented-float magnitude
    // and cancellation case. Square only the certified absolute lower bound.
    const double dxLower = representedFloatAbsDifferenceLower(
        particle.position.x, cameraOrigin.x);
    const double dyLower = representedFloatAbsDifferenceLower(
        particle.position.y, cameraOrigin.y);
    const double dzLower = representedFloatAbsDifferenceLower(
        particle.position.z, cameraOrigin.z);
    double distanceSquaredLower = __dmul_rd(dxLower, dxLower);
    distanceSquaredLower = __dadd_rd(
        distanceSquaredLower, __dmul_rd(dyLower, dyLower));
    distanceSquaredLower = __dadd_rd(
        distanceSquaredLower, __dmul_rd(dzLower, dzLower));
    const double distanceLower = __dsqrt_rd(distanceSquaredLower);
    if (!isfinite(supportRadiusUpper) || !isfinite(distanceLower)) return -1.0f;
    if (!(distanceLower > supportRadiusUpper)) return 0.0f;
    const double lower = __dsub_rd(distanceLower, supportRadiusUpper);
    if (!(lower >= 0.0) || !isfinite(lower)) return -1.0f;
    return __double2float_rd(lower);
}

__device__ __forceinline__ bool queryHit(
    model_::Particles& particles,
    uint32_t particleIdx,
    const tcnn::vec3& origin,
    const tcnn::vec3& direction,
    float queryRange,
    float& alpha) {
    const auto fetched = particles.fetchDensityParameters(particleIdx);
    float peakRange = -1.0f;
    float3 canonical = make_float3(0.0f);
    // The full-shape evaluation clamps the quadratic closest point to
    // the closed physical interval [0,t_query] and evaluates B_inverse there.
    return particles.densityHitInterval(
        origin, direction, fetched, alpha, peakRange, canonical,
        0.0f, queryRange);
}

// Response plus alpha product shared by the opacity field and the bisection
// rounds. One force-inlined body keeps -use_fast_math FMA contraction
// identical across kernels.
__device__ __forceinline__ bool fixed8ResponseAlphaUpdate(
    model_::Particles& particles,
    uint32_t particleIdx,
    const tcnn::vec3& origin,
    const tcnn::vec3& direction,
    float queryRange,
    float& transmittance) {
    float alpha = 0.0f;
    if (!queryHit(particles, particleIdx, origin, direction, queryRange, alpha)) {
        return false;
    }
    transmittance *= 1.0f - alpha;
    return true;
}

// -------------------------------------------------------------------------
// Cooperative scheduler. Candidates follow the nonlinear native-UT union
// angular-cap rule. Candidate keys are [tile|particle_id], never range or
// depth, so every query multiplies accepted alphas in ascending particle ID.

struct PointBlock {
    uint32_t tile;
    uint32_t sortedBegin;
    uint32_t count;
};

struct CooperativeWorkspace {
    CudaBuffer particleTileCounts;
    CudaBuffer particleTileOffsets;
    CudaBuffer particleNativeTileCounts;
    CudaBuffer particleProjectedCenter;
    CudaBuffer particleProjectedExtent;
    CudaBuffer particleValidProjection;
    CudaBuffer particleConicOpacity;
    CudaBuffer particleMaxConicPower;
    CudaBuffer particleSupportCap;
    CudaBuffer particleMinimumRange;
    CudaBuffer boundedParticleReasonMask;
    CudaBuffer boundedParticleFirst;
    CudaBuffer boundedParticleCount;
    CudaBuffer candidateKeysUnsorted;
    CudaBuffer candidateKeysSorted;
    CudaBuffer candidateIdsUnsorted;
    CudaBuffer candidateIdsSorted;
    CudaBuffer candidateTileRanges;
    CudaBuffer candidateScanTemp;
    CudaBuffer candidateSortTemp;
    CudaBuffer certificationFailed;
    CudaBuffer certificationFirstErrorIndex;
    CudaBuffer cameraInputInvalid;

    CudaBuffer pointKeysUnsorted;
    CudaBuffer pointKeysSorted;
    CudaBuffer pointIdsUnsorted;
    CudaBuffer pointIdsSorted;
    CudaBuffer pointTileRanges;
    CudaBuffer pointBlocksPerTile;
    CudaBuffer pointBlockOffsets;
    CudaBuffer pointBlocks;
    CudaBuffer pointSortTemp;
    CudaBuffer pointScanTemp;
    CudaBuffer totalPointBlocks;
    CudaBuffer pointOrigins;
    CudaBuffer pointDirections;
    CudaBuffer pointRanges;
    CudaBuffer pointValid;
    // Persistent-table-only fixed-capacity state.
    CudaBuffer activeIdsInput;
    CudaBuffer activeIdsCompacted;
    CudaBuffer activeCount;
    CudaBuffer activeSelectTemp;
    CudaBuffer cameraOpacityScratch;
    CudaBuffer cameraObservedScratch;

    Status reserveFixed(uint32_t particles, uint32_t points, uint32_t tiles,
                        cudaStream_t stream, const Logger& logger);
    Status reserveCandidates(uint32_t intersections, uint32_t tiles,
                             cudaStream_t stream, const Logger& logger);
    void clear(cudaStream_t stream, const Logger& logger);
};

namespace cooperative_detail {

// Slots returned through hostFallbackTelemetry:
// [0] camera-global cause, [1] bounded-particle reason mask,
// [2] first bounded particle, [3] bounded-particle count.
enum GlobalFallbackCause : uint32_t {
    kGlobalFallbackNone = 0u,
    kGlobalFallbackInjectedGeometry = 1u,
    kGlobalFallbackConeGeometry = 2u,
    kGlobalFallbackFixedWorkspace = 3u,
    kGlobalFallbackInvalidRayDirection = 4u,
    kGlobalFallbackCountScan = 5u,
    kGlobalFallbackInjectedInvalidCap = 6u,
    kGlobalFallbackExpansion = 7u,
    kGlobalFallbackIntersectionOverflow = 8u,
    kGlobalFallbackZeroIntersections = 9u,
    kGlobalFallbackInjectedCandidateAllocation = 10u,
    kGlobalFallbackCandidateWorkspace = 11u,
    kGlobalFallbackInjectedCandidateSort = 12u,
    kGlobalFallbackCandidateSort = 13u,
    kGlobalFallbackSortedValidation = 14u,
    kGlobalFallbackCauseCount = 15u,
};

enum BoundedParticleReason : uint32_t {
    kBoundedParticleNone = 0u,
    kBoundedParticleInvalidLowerRange = 1u << 0,
    kBoundedParticleMissingSupportCap = 1u << 1,
    kBoundedParticleOutsideCapDomain = 1u << 2,
};

inline void recordFirstError(
    uint32_t* telemetry, uint32_t cause) {
    if (telemetry[0] == kGlobalFallbackNone) {
        telemetry[0] = cause;
        telemetry[2] = UINT32_MAX;
    }
}

__device__ __forceinline__ void recordIndexedError(
    uint32_t* error, uint32_t* firstIndex, uint32_t index) {
    atomicExch(error, 1u);
    atomicMin(firstIndex, index);
}

constexpr uint64_t kInvalidKey = UINT64_MAX;

inline uint32_t bitWidth(uint32_t value) {
    uint32_t bits = 0;
    while (value != 0) { ++bits; value >>= 1; }
    return bits == 0 ? 1 : bits;
}

__host__ __device__ __forceinline__ uint64_t tileIdKey(uint32_t tile, uint32_t id) {
    return (static_cast<uint64_t>(tile) << 32) | static_cast<uint64_t>(id);
}

template <typename T>
Status resize(CudaBuffer& buffer, size_t count, cudaStream_t stream,
              const Logger& logger) {
    return buffer.resize(sizeof(T) * count, reinterpret_cast<uint64_t>(stream), logger);
}

namespace direct_support = gofish_continuous_support;

inline bool makeDirectPhysicalDomain(
    const RenderParameters& params,
    direct_support::CentralPhysicalDomain& output) {
    if (params.sensorModel.shutterType != TSensorModel::GlobalShutter ||
        params.resolution.x <= 0 || params.resolution.y <= 0) {
        return false;
    }
    output = direct_support::CentralPhysicalDomain{};
    if (params.sensorModel.modelType == TSensorModel::OpenCVFisheyeModel) {
        const auto& camera = params.sensorModel.ocvFisheyeParams;
        output.kind = direct_support::kOpenCvFisheyeProjection;
        auto& domain = output.fisheye;
        domain.fx = camera.focalLength.x;
        domain.fy = camera.focalLength.y;
        domain.cx = camera.principalPoint.x;
        domain.cy = camera.principalPoint.y;
        for (int index = 0; index < 4; ++index) {
            domain.radial[index] = camera.radialCoeffs[index];
        }
        domain.modelMaxAngle = camera.maxAngle;
        domain.width = static_cast<uint32_t>(params.resolution.x);
        domain.height = static_cast<uint32_t>(params.resolution.y);
        domain.tileWidth = GUTParameters::Tiling::BlockX;
        domain.tileHeight = GUTParameters::Tiling::BlockY;
        return direct_support::completePhysicalDomain(domain);
    }
    if (params.sensorModel.modelType == TSensorModel::OpenCVPinholeModel) {
        const auto& camera = params.sensorModel.ocvPinholeParams;
        for (int index = 0; index < 6; ++index) {
            if (camera.radialCoeffs[index] != 0.0f) return false;
        }
        for (int index = 0; index < 2; ++index) {
            if (camera.tangentialCoeffs[index] != 0.0f) return false;
        }
        for (int index = 0; index < 4; ++index) {
            if (camera.thinPrismCoeffs[index] != 0.0f) return false;
        }
        output.kind = direct_support::kUndistortedPinholeProjection;
        auto& domain = output.pinhole;
        domain.fx = camera.focalLength.x;
        domain.fy = camera.focalLength.y;
        domain.cx = camera.principalPoint.x;
        domain.cy = camera.principalPoint.y;
        domain.width = static_cast<uint32_t>(params.resolution.x);
        domain.height = static_cast<uint32_t>(params.resolution.y);
        domain.tileWidth = GUTParameters::Tiling::BlockX;
        domain.tileHeight = GUTParameters::Tiling::BlockY;
        return domain.fx > 0.0 && domain.fy > 0.0 &&
            direct_support::finiteValue(domain.fx) &&
            direct_support::finiteValue(domain.fy) &&
            direct_support::finiteValue(domain.cx) &&
            direct_support::finiteValue(domain.cy);
    }
    return false;
}

__device__ __forceinline__ bool makeIntrinsicSupportCap(
    float opacity,
    const threedgut::ParticleDensity& particle,
    const tcnn::vec3& relative,
    float4& cap) {
    if (opacity < TGUTProjector::AlphaThreshold) {
        cap = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
        return true;
    }
    const float canonicalRadius = nextafterf(
        canonicalSupportRadius(opacity), CUDART_INF_F);
    const double factorRadius = packedFactorFrobeniusUpper(particle);
    const double rx = relative.x;
    const double ry = relative.y;
    const double rz = relative.z;
    double distanceSquaredLower = __dmul_rd(rx, rx);
    distanceSquaredLower = __dadd_rd(
        distanceSquaredLower, __dmul_rd(ry, ry));
    distanceSquaredLower = __dadd_rd(
        distanceSquaredLower, __dmul_rd(rz, rz));
    const double distanceLower = __dsqrt_rd(distanceSquaredLower);
    const double distance = sqrt(rx*rx + ry*ry + rz*rz);
    if (!(canonicalRadius > 0.0f) || !(factorRadius > 0.0) ||
        !(distanceLower > 0.0) || !(distance > 0.0) || !isfinite(canonicalRadius) ||
        !isfinite(factorRadius) || !isfinite(distance)) {
        return false;
    }
    const double radius = __dmul_ru((double)canonicalRadius, factorRadius);
    double angle = direct_support::kPi;
    if (radius < distanceLower) {
        angle = asin(fmin(1.0, __ddiv_ru(radius, distanceLower)));
        angle = nextafter(angle, CUDART_INF);
    }
    const double centerX = rx / distance;
    const double centerY = ry / distance;
    const double centerZ = rz / distance;
    const float storedX = static_cast<float>(centerX);
    const float storedY = static_cast<float>(centerY);
    const float storedZ = static_cast<float>(centerZ);
    const double chord = sqrt(
        (storedX-centerX)*(storedX-centerX) +
        (storedY-centerY)*(storedY-centerY) +
        (storedZ-centerZ)*(storedZ-centerZ));
    angle = fmin(direct_support::kPi,
                 __dadd_ru(angle, __dmul_ru(0.5*direct_support::kPi, chord)));
    cap = make_float4(storedX, storedY, storedZ,
        nextafterf(static_cast<float>(angle), CUDART_INF_F));
    return isfinite(cap.x) && isfinite(cap.y) && isfinite(cap.z) &&
        isfinite(cap.w);
}

__device__ __forceinline__ direct_support::SupportBoxResult directSupportBox(
    const direct_support::CentralPhysicalDomain& domain,
    tcnn::uvec2 tileGrid,
    const float4 cap,
    tcnn::mat4x3 worldToSensor) {
    if (cap.w < 0.0f) {
        return direct_support::SupportBoxResult{
            direct_support::TileBox{0, 0, 0, 0}, direct_support::kEnumerationOk};
    }
    const tcnn::vec3 sensor = worldToSensor * tcnn::vec4(
        tcnn::vec3{cap.x, cap.y, cap.z}, 0.0f);
    const direct_support::Mat3 identity{{
        direct_support::Vec3{1.0, 0.0, 0.0},
        direct_support::Vec3{0.0, 1.0, 0.0},
        direct_support::Vec3{0.0, 0.0, 1.0}}};
    return direct_support::continuousSupportTileBox(
        domain, direct_support::TileGrid{tileGrid.x, tileGrid.y},
        direct_support::Vec3{sensor.x, sensor.y, sensor.z}, cap.w, identity);
}

template <typename TProjector>
struct DirectNativeAdmission {
    bool valid;
    typename TProjector::BoundingBox2D box;
    tcnn::vec4 conic;
    tcnn::vec2 center;
    float maxPower;

    __device__ __forceinline__ bool operator()(int x, int y) const {
        return valid && TProjector::nativeTileAdmissionBase(
            x, y, true, box, conic, center, maxPower);
    }
};

struct DirectNullEmitter {
    __device__ __forceinline__ void operator()(uint64_t, uint32_t, uint32_t) const {}
};

struct DirectKeyEmitter {
    uint64_t* keys;
    uint32_t* ids;
    __device__ __forceinline__ void operator()(
        uint64_t cursor, uint32_t tile, uint32_t particle) const {
        keys[cursor] = tileIdKey(tile, particle);
        ids[cursor] = particle;
    }
};

template <typename TBox>
__device__ __forceinline__ bool checkedDirectTileBox(
    const TBox& source, direct_support::TileBox& output) {
    if (source.min.x > static_cast<uint32_t>(INT_MAX) ||
        source.min.y > static_cast<uint32_t>(INT_MAX) ||
        source.max.x > static_cast<uint32_t>(INT_MAX) ||
        source.max.y > static_cast<uint32_t>(INT_MAX)) {
        return false;
    }
    output = direct_support::TileBox{
        static_cast<int>(source.min.x), static_cast<int>(source.min.y),
        static_cast<int>(source.max.x), static_cast<int>(source.max.y)};
    return true;
}

__global__ void preprocessParticles(
    tcnn::uvec2 tileGrid,
    uint32_t particleCount,
    RenderParameters params,
    tcnn::vec3 sensorWorldPosition,
    tcnn::mat4x3 worldToSensor,
    direct_support::CentralPhysicalDomain physicalDomain,
    const float* particleSigmaMax,
    uint64_t* tileCounts,
    uint32_t* nativeTileCounts,
    tcnn::vec2* projectedCenters,
    tcnn::vec2* projectedExtents,
    uint8_t* projectedValidity,
    tcnn::vec4* conicOpacities,
    float* maxConicPowers,
    float4* supportCaps,
    float* minimumRanges,
    uint32_t* boundedReasonMask,
    uint32_t* boundedFirst,
    uint32_t* boundedCount,
    uint32_t* certificationFailed,
    uint32_t* firstErrorIndex,
    MemoryHandles handles) {
    const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= particleCount) return;
    tileCounts[particleIdx] = 0;
    nativeTileCounts[particleIdx] = 0;
    supportCaps[particleIdx] = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
    minimumRanges[particleIdx] = -1.0f;
    uint32_t boundedReason = kBoundedParticleNone;

    model_::Particles particles;
    particles.initializeDensity(handles);
    const auto density = particles.fetchDensityParameters(particleIdx);
    const auto raw = particles.fetchDensityRawParameters(particleIdx);
    const float lower = conservativeMinimumUnitRange(
        raw, particleSigmaMax[particleIdx], sensorWorldPosition);
    if (!(lower >= 0.0f) || !isfinite(lower)) {
        // An invalid lower bound disables the range skip (exact).
        boundedReason |= kBoundedParticleInvalidLowerRange;
    } else {
        minimumRanges[particleIdx] = lower;
    }

    tcnn::vec3 sensorRay;
    float projectedOpacity = 0.0f;
    tcnn::vec2 center = tcnn::vec2::zero();
    tcnn::vec3 covariance = tcnn::vec3::zero();
    bool supportCapEligible = false;
    bool validProjection = TGUTProjector::unscentedParticleProjection(
        params.resolution, params.sensorModel, sensorWorldPosition,
        worldToSensor, params.sensorState, particles, density, sensorRay,
        projectedOpacity, center, covariance, supportCapEligible);

    tcnn::vec2 extent = tcnn::vec2::zero();
    tcnn::vec4 conicOpacity = tcnn::vec4::zero();
    float maxPower = 0.0f;
    if (validProjection) {
        validProjection = TGUTProjector::computeProjectedExtentConicOpacity(
            covariance, projectedOpacity, extent, conicOpacity, maxPower);
    }
    // The final projector decision drives candidate expansion.
    projectedValidity[particleIdx] = validProjection ? 1u : 0u;
    TGUTProjector::BoundingBox2D nativeBox{{0, 0}, {0, 0}};
    if (validProjection) {
        nativeBox = TGUTProjector::computeTileSpaceBBox(tileGrid, center, extent);
    }

    (void)supportCapEligible;
    const tcnn::vec3 relative = particles.position(density) - sensorWorldPosition;
    float4 cap = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
    if (!makeIntrinsicSupportCap(particles.opacity(density), raw, relative, cap)) {
        recordIndexedError(certificationFailed, firstErrorIndex, particleIdx);
    }
    supportCaps[particleIdx] = cap;
    const direct_support::SupportBoxResult support = directSupportBox(
        physicalDomain, tileGrid, cap, worldToSensor);
    const DirectNativeAdmission<TGUTProjector> admission{
        validProjection, nativeBox, conicOpacity, center, maxPower};
    direct_support::TileBox directNativeBox{};
    if (!checkedDirectTileBox(nativeBox, directNativeBox)) {
        recordIndexedError(certificationFailed, firstErrorIndex, particleIdx);
        return;
    }
    const direct_support::EnumerationResult enumeration =
        direct_support::enumerateContinuousPhysicalSupport(
            direct_support::TileGrid{tileGrid.x, tileGrid.y}, particleIdx,
            validProjection, directNativeBox,
            support, direct_support::kCountOnly,
            0u, 0u, 0u, admission, DirectNullEmitter{});
    if (enumeration.failure != direct_support::kEnumerationOk) {
        recordIndexedError(certificationFailed, firstErrorIndex, particleIdx);
    }
    projectedCenters[particleIdx] = center;
    projectedExtents[particleIdx] = validProjection ? extent : tcnn::vec2::zero();
    conicOpacities[particleIdx] = validProjection ? conicOpacity : tcnn::vec4::zero();
    maxConicPowers[particleIdx] = validProjection ? maxPower : 0.0f;
    nativeTileCounts[particleIdx] = enumeration.nativeCount;
    tileCounts[particleIdx] = enumeration.totalCount;
    if (boundedReason != kBoundedParticleNone) {
        atomicOr(boundedReasonMask, boundedReason);
        atomicMin(boundedFirst, particleIdx);
        atomicAdd(boundedCount, 1u);
    }
}

__global__ void expandParticles(
    tcnn::uvec2 tileGrid,
    uint32_t particleCount,
    const uint64_t* tileOffsets,
    const uint32_t* nativeTileCounts,
    const tcnn::vec2* projectedCenters,
    const tcnn::vec2* projectedExtents,
    const uint8_t* projectedValidity,
    const tcnn::vec4* conicOpacities,
    const float* maxConicPowers,
    const float4* supportCaps,
    direct_support::CentralPhysicalDomain physicalDomain,
    tcnn::mat4x3 worldToSensor,
    uint32_t* certificationFailed,
    uint32_t* firstErrorIndex,
    uint64_t* keys,
    uint32_t* ids) {
    const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= particleCount) return;
    uint64_t cursor = particleIdx == 0 ? 0 : tileOffsets[particleIdx - 1];
    const uint64_t end = tileOffsets[particleIdx];
    if (cursor == end) return;
    const tcnn::vec2 center = projectedCenters[particleIdx];
    const tcnn::vec2 extent = projectedExtents[particleIdx];
    const tcnn::vec4 conic = conicOpacities[particleIdx];
    const float maxPower = maxConicPowers[particleIdx];
    const bool validProjection = projectedValidity[particleIdx] != 0u;
    const TGUTProjector::BoundingBox2D nativeBox = validProjection
        ? TGUTProjector::computeTileSpaceBBox(tileGrid, center, extent)
        : TGUTProjector::BoundingBox2D{{0, 0}, {0, 0}};
    const float4 cap = supportCaps[particleIdx];
    const direct_support::SupportBoxResult support = directSupportBox(
        physicalDomain, tileGrid, cap, worldToSensor);
    const DirectNativeAdmission<TGUTProjector> admission{
        validProjection, nativeBox, conic, center, maxPower};
    direct_support::TileBox directNativeBox{};
    if (!checkedDirectTileBox(nativeBox, directNativeBox)) {
        recordIndexedError(certificationFailed, firstErrorIndex, particleIdx);
        return;
    }
    const direct_support::EnumerationResult enumeration =
        direct_support::enumerateContinuousPhysicalSupport(
            direct_support::TileGrid{tileGrid.x, tileGrid.y}, particleIdx,
            validProjection, directNativeBox,
            support, direct_support::kEmit,
            cursor, end, nativeTileCounts[particleIdx], admission,
            DirectKeyEmitter{keys, ids});
    if (enumeration.failure != direct_support::kEnumerationOk) {
        recordIndexedError(certificationFailed, firstErrorIndex, particleIdx);
    }
}

__global__ void identifyRanges(const uint64_t* keys, uint32_t count,
                               uint32_t tileCount, tcnn::uvec2* ranges) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t tile = static_cast<uint32_t>(keys[index] >> 32);
    if (tile >= tileCount) return;
    const uint32_t previous = index == 0 ? UINT32_MAX :
        static_cast<uint32_t>(keys[index - 1] >> 32);
    const uint32_t next = index + 1 == count ? UINT32_MAX :
        static_cast<uint32_t>(keys[index + 1] >> 32);
    if (tile != previous) ranges[tile].x = index;
    if (tile != next) ranges[tile].y = index + 1;
}

__global__ void validateSortedCandidateKeys(
    const uint64_t* keys, const uint32_t* ids, uint32_t count,
    uint32_t tileCount, uint32_t* certificationFailed,
    uint32_t* firstErrorIndex) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint64_t key = keys[index];
    const uint32_t tile = static_cast<uint32_t>(key >> 32);
    const uint32_t id = static_cast<uint32_t>(key);
    if (key == kInvalidKey || tile >= tileCount || id == UINT32_MAX ||
        ids[index] != id || (index > 0 && key <= keys[index - 1])) {
        recordIndexedError(certificationFailed, firstErrorIndex, index);
    }
}

__global__ void validateCameraRays(
    uint32_t pixelCount,
    const tcnn::vec3* sensorRayOrigins,
    const tcnn::vec3* sensorRayDirections,
    uint32_t* invalidMask,
    uint32_t* firstErrorIndex) {
    const uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixelCount) return;
    const tcnn::vec3 origin = sensorRayOrigins[pixel];
    const tcnn::vec3 direction = sensorRayDirections[pixel];
    if (!isfinite(origin.x) || !isfinite(origin.y) || !isfinite(origin.z) ||
        origin.x != 0.0f || origin.y != 0.0f || origin.z != 0.0f) {
        atomicOr(invalidMask, 1u);
        atomicMin(firstErrorIndex, pixel);
    }
    const float normSquared = dot(direction, direction);
    if (!isfinite(direction.x) || !isfinite(direction.y) ||
        !isfinite(direction.z) || !isfinite(normSquared) ||
        fabsf(normSquared - 1.0f) > 2.0e-5f) {
        atomicOr(invalidMask, 2u);
        atomicMin(firstErrorIndex, pixel);
    }
}

__global__ void preprocessPoints(
    RenderParameters params,
    const tcnn::vec3* sensorRayOrigins,
    tcnn::mat4x3 sensorToWorld,
    const uint8_t* physicalMask,
    const tcnn::vec3* queryPoints,
    uint32_t queryCount,
    uint64_t* keys,
    uint32_t* ids,
    tcnn::vec3* origins,
    tcnn::vec3* directions,
    float* ranges,
    uint8_t* valid,
    tcnn::uvec2 tileGrid) {
    const uint32_t queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIdx >= queryCount) return;
    keys[queryIdx] = kInvalidKey;
    ids[queryIdx] = queryIdx;
    valid[queryIdx] = 0;
    tcnn::vec2 projected;
    if (!projectPointWithShutter<GAUSSIAN_N_ROLLING_SHUTTER_ITERATIONS>(
            queryPoints[queryIdx], params.resolution, params.sensorModel,
            params.sensorState, 0.0f, projected)) return;
    const int x = static_cast<int>(floorf(projected.x));
    const int y = static_cast<int>(floorf(projected.y));
    if (x < 0 || y < 0 || x >= params.resolution.x || y >= params.resolution.y) return;
    const uint32_t pixel = static_cast<uint32_t>(x) +
        static_cast<uint32_t>(params.resolution.x) * static_cast<uint32_t>(y);
    if (physicalMask != nullptr && physicalMask[pixel] == 0) return;
    const tcnn::vec3 origin = sensorToWorld * tcnn::vec4(sensorRayOrigins[pixel], 1.0f);
    const tcnn::vec3 delta = queryPoints[queryIdx] - origin;
    const float range = length(delta);
    if (!(range > 0.0f) || !isfinite(range)) return;
    const uint32_t gridX = (static_cast<uint32_t>(params.resolution.x) +
        GUTParameters::Tiling::BlockX - 1) / GUTParameters::Tiling::BlockX;
    const uint32_t tile = static_cast<uint32_t>(x) / GUTParameters::Tiling::BlockX +
        gridX * (static_cast<uint32_t>(y) / GUTParameters::Tiling::BlockY);
    const tcnn::vec3 unitDirection = delta / range;
    const uint32_t tileX = static_cast<uint32_t>(x) / GUTParameters::Tiling::BlockX;
    const uint32_t tileY = static_cast<uint32_t>(y) / GUTParameters::Tiling::BlockY;
    if (tileX >= tileGrid.x || tileY >= tileGrid.y) return;
    keys[queryIdx] = tileIdKey(tile, queryIdx);
    origins[queryIdx] = origin;
    directions[queryIdx] = unitDirection;
    ranges[queryIdx] = range;
    valid[queryIdx] = 1;
}

__global__ void blocksPerTile(uint32_t tileCount, const tcnn::uvec2* ranges,
                             uint32_t* counts) {
    const uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tileCount) return;
    const uint32_t points = ranges[tile].y - ranges[tile].x;
    counts[tile] = (points + 255u) / 256u;
}

__global__ void totalBlocks(uint32_t tileCount, const uint32_t* counts,
                            const uint32_t* offsets, uint32_t* total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *total = tileCount == 0 ? 0 : offsets[tileCount - 1] + counts[tileCount - 1];
}

__global__ void fillBlocks(uint32_t tileCount, const tcnn::uvec2* ranges,
                           const uint32_t* counts, const uint32_t* offsets,
                           PointBlock* blocks) {
    const uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tileCount) return;
    const uint32_t begin = ranges[tile].x;
    const uint32_t points = ranges[tile].y - begin;
    for (uint32_t block = 0; block < counts[tile]; ++block) {
        const uint32_t local = block * 256u;
        blocks[offsets[tile] + block] = PointBlock{
            tile, begin + local, min(256u, points - local)};
    }
}

__global__ void integratePoints(
    const tcnn::uvec2* candidateTileRanges,
    const uint32_t* candidateIds,
    const uint32_t* sortedPointIds,
    const PointBlock* pointBlocks,
    const tcnn::vec3* pointOrigins,
    const tcnn::vec3* pointDirections,
    const float* pointRanges,
    const uint8_t* pointValid,
    const float* particleMinimumRanges,
    float* crossViewMinimum,
    uint8_t* crossViewObserved,
    MemoryHandles handles) {
    __shared__ uint32_t sharedIds[256];
    __shared__ float sharedLower[256];
    const PointBlock block = pointBlocks[blockIdx.x];
    const bool active = threadIdx.x < block.count;
    const uint32_t queryIdx = active
        ? sortedPointIds[block.sortedBegin + threadIdx.x] : 0;
    const tcnn::uvec2 candidateRange = candidateTileRanges[block.tile];
    float transmittance = 1.0f;
    model_::Particles particles;
    particles.initializeDensity(handles);
    for (uint32_t batch = candidateRange.x; batch < candidateRange.y; batch += 256u) {
        const uint32_t load = batch + threadIdx.x;
        if (load < candidateRange.y) {
            const uint32_t id = candidateIds[load];
            sharedIds[threadIdx.x] = id;
            sharedLower[threadIdx.x] = particleMinimumRanges[id];
        }
        __syncthreads();
        const uint32_t count = min(256u, candidateRange.y - batch);
        if (active && pointValid[queryIdx]) {
            const float queryRange = pointRanges[queryIdx];
            for (uint32_t slot = 0; slot < count; ++slot) {
                const float lower = sharedLower[slot];
                if (lower >= 0.0f && lower > queryRange) continue;
                if (fixed8ResponseAlphaUpdate(
                        particles, sharedIds[slot], pointOrigins[queryIdx],
                        pointDirections[queryIdx], queryRange, transmittance)) {
                }
            }
        }
        __syncthreads();
    }
    if (active && pointValid[queryIdx]) {
        const float opacity = 1.0f - transmittance;
        if (crossViewObserved[queryIdx]) {
            crossViewMinimum[queryIdx] = fminf(crossViewMinimum[queryIdx], opacity);
        } else {
            crossViewMinimum[queryIdx] = opacity;
            crossViewObserved[queryIdx] = 1;
        }
    }
}

// Threshold-only ordered integration. Work may stop only after the cumulative
// float32 opacity is strictly greater than tau. Exact equality must continue
// through every remaining ordered hit so the equality side channel is exact.
__global__ void integratePointsThresholdTieAware(
    const tcnn::uvec2* candidateTileRanges,
    const uint32_t* candidateIds,
    const uint32_t* sortedPointIds,
    const PointBlock* pointBlocks,
    const tcnn::vec3* pointOrigins,
    const tcnn::vec3* pointDirections,
    const float* pointRanges,
    const uint8_t* pointValid,
    const float* particleMinimumRanges,
    float* cameraOpacity,
    uint8_t* cameraObserved,
    MemoryHandles handles) {
    __shared__ uint32_t sharedIds[256];
    __shared__ float sharedLower[256];
    const PointBlock block = pointBlocks[blockIdx.x];
    const bool active = threadIdx.x < block.count;
    const uint32_t queryIdx = active
        ? sortedPointIds[block.sortedBegin + threadIdx.x] : 0;
    const tcnn::uvec2 candidateRange = candidateTileRanges[block.tile];
    float transmittance = 1.0f;
    bool strictlyAbove = false;
    model_::Particles particles;
    particles.initializeDensity(handles);
    for (uint32_t batch = candidateRange.x; batch < candidateRange.y; batch += 256u) {
        const uint32_t load = batch + threadIdx.x;
        if (load < candidateRange.y) {
            const uint32_t id = candidateIds[load];
            sharedIds[threadIdx.x] = id;
            sharedLower[threadIdx.x] = particleMinimumRanges[id];
        }
        __syncthreads();
        const uint32_t count = min(256u, candidateRange.y - batch);
        if (active && pointValid[queryIdx] && !strictlyAbove) {
            const float queryRange = pointRanges[queryIdx];
            for (uint32_t slot = 0; slot < count; ++slot) {
                const float lower = sharedLower[slot];
                if (lower >= 0.0f && lower > queryRange) continue;
                if (fixed8ResponseAlphaUpdate(
                        particles, sharedIds[slot], pointOrigins[queryIdx],
                        pointDirections[queryIdx], queryRange, transmittance)) {
                    if (1.0f - transmittance > 0.5f) {
                        strictlyAbove = true;
                        break;
                    }
                }
            }
        }
        // Both barriers remain unconditional for every thread and batch.
        __syncthreads();
    }
    if (active && pointValid[queryIdx]) {
        // Any value strictly above tau is classification-equivalent after the
        // proof fires. Below/equal paths retain the exact ordered float result.
        cameraOpacity[queryIdx] = strictlyAbove
            ? 1.0f : 1.0f - transmittance;
        cameraObserved[queryIdx] = 1u;
    }
}
} // namespace cooperative_detail

inline Status CooperativeWorkspace::reserveFixed(
    uint32_t particles, uint32_t points, uint32_t tiles,
    cudaStream_t stream, const Logger& logger) {
    using namespace cooperative_detail;
    if (particles == 0 || points == 0 || tiles == 0 ||
        particles > static_cast<uint32_t>(INT_MAX) ||
        points > static_cast<uint32_t>(INT_MAX) ||
        tiles > static_cast<uint32_t>(INT_MAX)) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "cooperative field sizes exceed CUB bounds");
    }
    CHECK_STATUS_RETURN(resize<uint64_t>(particleTileCounts, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint64_t>(particleTileOffsets, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(particleNativeTileCounts, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::vec2>(particleProjectedCenter, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::vec2>(particleProjectedExtent, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint8_t>(particleValidProjection, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::vec4>(particleConicOpacity, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<float>(particleMaxConicPower, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<float4>(particleSupportCap, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<float>(particleMinimumRange, particles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(boundedParticleReasonMask, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(boundedParticleFirst, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(boundedParticleCount, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::uvec2>(candidateTileRanges, tiles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(certificationFailed, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(certificationFirstErrorIndex, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(cameraInputInvalid, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<uint64_t>(pointKeysUnsorted, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint64_t>(pointKeysSorted, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(pointIdsUnsorted, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(pointIdsSorted, points, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::uvec2>(pointTileRanges, tiles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(pointBlocksPerTile, tiles, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(pointBlockOffsets, tiles, stream, logger));
    CHECK_STATUS_RETURN(resize<PointBlock>(pointBlocks, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(totalPointBlocks, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::vec3>(pointOrigins, points, stream, logger));
    CHECK_STATUS_RETURN(resize<tcnn::vec3>(pointDirections, points, stream, logger));
    CHECK_STATUS_RETURN(resize<float>(pointRanges, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint8_t>(pointValid, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(activeIdsInput, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(activeIdsCompacted, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(activeCount, 1, stream, logger));
    CHECK_STATUS_RETURN(resize<float>(cameraOpacityScratch, points, stream, logger));
    CHECK_STATUS_RETURN(resize<uint8_t>(cameraObservedScratch, points, stream, logger));
    size_t bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceSelect::Flagged(
        nullptr, bytes,
        static_cast<const uint32_t*>(activeIdsInput.data()),
        static_cast<const uint8_t*>(pointValid.data()),
        static_cast<uint32_t*>(activeIdsCompacted.data()),
        static_cast<uint32_t*>(activeCount.data()),
        static_cast<int>(points), stream), logger);
    CHECK_STATUS_RETURN(activeSelectTemp.resize(
        bytes, reinterpret_cast<uint64_t>(stream), logger));
    bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceScan::InclusiveSum(
        nullptr, bytes, static_cast<const uint64_t*>(particleTileCounts.data()),
        static_cast<uint64_t*>(particleTileOffsets.data()),
        static_cast<int>(particles), stream), logger);
    CHECK_STATUS_RETURN(candidateScanTemp.resize(bytes, reinterpret_cast<uint64_t>(stream), logger));
    bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, static_cast<const uint64_t*>(pointKeysUnsorted.data()),
        static_cast<uint64_t*>(pointKeysSorted.data()),
        static_cast<const uint32_t*>(pointIdsUnsorted.data()),
        static_cast<uint32_t*>(pointIdsSorted.data()), static_cast<int>(points),
        0, 32 + bitWidth(tiles), stream), logger);
    CHECK_STATUS_RETURN(pointSortTemp.resize(bytes, reinterpret_cast<uint64_t>(stream), logger));
    bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, static_cast<const uint32_t*>(pointBlocksPerTile.data()),
        static_cast<uint32_t*>(pointBlockOffsets.data()), static_cast<int>(tiles), stream), logger);
    return pointScanTemp.resize(bytes, reinterpret_cast<uint64_t>(stream), logger);
}

inline Status CooperativeWorkspace::reserveCandidates(
    uint32_t intersections, uint32_t tiles, cudaStream_t stream,
    const Logger& logger) {
    using namespace cooperative_detail;
    if (intersections == 0) return Status();
    if (intersections > static_cast<uint32_t>(INT_MAX)) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "candidate intersections exceed CUB bounds");
    }
    CHECK_STATUS_RETURN(resize<uint64_t>(candidateKeysUnsorted, intersections, stream, logger));
    CHECK_STATUS_RETURN(resize<uint64_t>(candidateKeysSorted, intersections, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(candidateIdsUnsorted, intersections, stream, logger));
    CHECK_STATUS_RETURN(resize<uint32_t>(candidateIdsSorted, intersections, stream, logger));
    size_t bytes = 0;
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, static_cast<const uint64_t*>(candidateKeysUnsorted.data()),
        static_cast<uint64_t*>(candidateKeysSorted.data()),
        static_cast<const uint32_t*>(candidateIdsUnsorted.data()),
        static_cast<uint32_t*>(candidateIdsSorted.data()),
        static_cast<int>(intersections), 0, 32 + bitWidth(tiles), stream), logger);
    return candidateSortTemp.resize(bytes, reinterpret_cast<uint64_t>(stream), logger);
}

inline void CooperativeWorkspace::clear(cudaStream_t stream, const Logger& logger) {
    CudaBuffer* buffers[] = {
        &particleTileCounts, &particleTileOffsets, &particleNativeTileCounts,
        &particleProjectedCenter, &particleProjectedExtent,
        &particleValidProjection, &particleConicOpacity,
        &particleMaxConicPower, &particleSupportCap, &particleMinimumRange,
        &boundedParticleReasonMask,
        &boundedParticleFirst, &boundedParticleCount,
        &candidateKeysUnsorted, &candidateKeysSorted, &candidateIdsUnsorted,
        &candidateIdsSorted, &candidateTileRanges, &candidateScanTemp,
        &candidateSortTemp, &certificationFailed,
        &certificationFirstErrorIndex,
        &cameraInputInvalid,
        &pointKeysUnsorted, &pointKeysSorted, &pointIdsUnsorted, &pointIdsSorted,
        &pointTileRanges, &pointBlocksPerTile, &pointBlockOffsets, &pointBlocks,
        &pointSortTemp, &pointScanTemp, &totalPointBlocks, &pointOrigins,
        &pointDirections, &pointRanges, &pointValid,
        &activeIdsInput, &activeIdsCompacted, &activeCount, &activeSelectTemp,
        &cameraOpacityScratch, &cameraObservedScratch};
    for (CudaBuffer* buffer : buffers) {
        buffer->clear(reinterpret_cast<uint64_t>(stream), logger);
    }
}


inline bool directFieldGeometry(
    const RenderParameters& params, tcnn::uvec2& tileGrid,
    uint32_t& tileCount) {
    using namespace cooperative_detail;
    if (params.resolution.x <= 0 || params.resolution.y <= 0) return false;
    const uint64_t gridX =
        (static_cast<uint64_t>(params.resolution.x) +
         GUTParameters::Tiling::BlockX - 1) / GUTParameters::Tiling::BlockX;
    const uint64_t gridY =
        (static_cast<uint64_t>(params.resolution.y) +
         GUTParameters::Tiling::BlockY - 1) / GUTParameters::Tiling::BlockY;
    if (gridX == 0 || gridY == 0 || gridX > UINT32_MAX || gridY > UINT32_MAX) return false;
    const uint64_t tiles64 = gridX * gridY;
    if (tiles64 > static_cast<uint64_t>(INT_MAX)) return false;
    const uint64_t pixels64 = static_cast<uint64_t>(params.resolution.x) *
        static_cast<uint64_t>(params.resolution.y);
    if (pixels64 == 0 || pixels64 > UINT32_MAX) return false;
    tileGrid = tcnn::uvec2{static_cast<uint32_t>(gridX), static_cast<uint32_t>(gridY)};
    tileCount = static_cast<uint32_t>(tiles64);
    return true;
}

inline Status evaluateCooperative(
    CooperativeWorkspace& workspace,
    const RenderParameters& params,
    const tcnn::vec3* sensorRayOrigins,
    const tcnn::vec3* sensorRayDirections,
    const uint8_t* physicalMask,
    const tcnn::vec3* queryPoints,
    uint32_t queryCount,
    uint32_t particleCount,
    const float* particleSigmaMax,
    float* crossViewMinimum,
    uint8_t* crossViewObserved,
    uint32_t* fallbackCameraCounter,
    uint32_t* hostSelectiveFallbackQueryCount,
    uint32_t* hostSelectiveFallbackFirstQuery,
    uint32_t* hostFallbackTelemetry,
    bool thresholdTieAware,
    uint64_t* parameterBuffer,
    tcnn::mat4x3 worldToSensor,
    tcnn::mat4x3 sensorToWorld,
    cudaStream_t stream,
    const Logger& logger
    ) {
    using namespace cooperative_detail;
    (void)fallbackCameraCounter;
    if (hostSelectiveFallbackQueryCount == nullptr ||
        hostSelectiveFallbackFirstQuery == nullptr ||
        hostFallbackTelemetry == nullptr) {
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "selective fallback host telemetry is null");
    }
    *hostSelectiveFallbackQueryCount = 0;
    *hostSelectiveFallbackFirstQuery = UINT32_MAX;
    hostFallbackTelemetry[0] = kGlobalFallbackNone;
    hostFallbackTelemetry[1] = kBoundedParticleNone;
    hostFallbackTelemetry[2] = UINT32_MAX;
    hostFallbackTelemetry[3] = 0u;
    constexpr uint32_t faultMask = kFaultNone;
    tcnn::uvec2 tileGrid = tcnn::uvec2::zero();
    uint32_t tileCount = 0;
    direct_support::CentralPhysicalDomain physicalDomain{};
    if ((faultMask & kFaultGeometry) != 0) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackInjectedGeometry);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    if (!directFieldGeometry(params, tileGrid, tileCount) ||
        !makeDirectPhysicalDomain(params, physicalDomain)) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackConeGeometry);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "direct physical field domain is unsupported");
    }
    const Status fixedStatus = workspace.reserveFixed(
        particleCount, queryCount, tileCount, stream, logger);
    if (!fixedStatus) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackFixedWorkspace);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.candidateTileRanges.data(), 0,
        sizeof(tcnn::uvec2) * tileCount, stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.pointTileRanges.data(), 0,
        sizeof(tcnn::uvec2) * tileCount, stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.certificationFailed.data(), 0,
        sizeof(uint32_t), stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.cameraInputInvalid.data(), 0,
        sizeof(uint32_t), stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(
        workspace.certificationFirstErrorIndex.data(), 0xff,
        sizeof(uint32_t), stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleReasonMask.data(), 0,
        sizeof(uint32_t), stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleFirst.data(), 0xff,
        sizeof(uint32_t), stream), logger);
    CUDA_CHECK_RETURN(cudaMemsetAsync(workspace.boundedParticleCount.data(), 0,
        sizeof(uint32_t), stream), logger);

    const uint32_t pixelCount = static_cast<uint32_t>(
        static_cast<uint64_t>(params.resolution.x) * params.resolution.y);
    validateCameraRays<<<(pixelCount + 255u) / 256u, 256u, 0, stream>>>(
        pixelCount, sensorRayOrigins, sensorRayDirections,
        static_cast<uint32_t*>(workspace.cameraInputInvalid.data()),
        static_cast<uint32_t*>(
            workspace.certificationFirstErrorIndex.data()));
    uint32_t cameraInvalid = 0;
    uint32_t cameraFirstErrorIndex = UINT32_MAX;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        &cameraInvalid, workspace.cameraInputInvalid.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        &cameraFirstErrorIndex,
        workspace.certificationFirstErrorIndex.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), logger);
    if ((cameraInvalid & 1u) != 0) {
        recordFirstError(
            hostFallbackTelemetry, kGlobalFallbackInvalidRayDirection);
        hostFallbackTelemetry[2] = cameraFirstErrorIndex;
        RETURN_ERROR(logger, ErrorCode::BadInput,
                     "camera sensor-space origins are not exact finite zero");
    }
    if ((cameraInvalid & 2u) != 0) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackInvalidRayDirection);
        hostFallbackTelemetry[2] = cameraFirstErrorIndex;
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }

    const tcnn::vec3 sensorWorldPosition = sensorToWorld *
        tcnn::vec4(tcnn::vec3::zero(), 1.0f);
    const uint32_t particleBlocks = (particleCount + kThreads - 1) / kThreads;
    preprocessParticles<<<particleBlocks, kThreads, 0, stream>>>(
        tileGrid, particleCount, params, sensorWorldPosition, worldToSensor,
        physicalDomain, particleSigmaMax,
        static_cast<uint64_t*>(workspace.particleTileCounts.data()),
        static_cast<uint32_t*>(workspace.particleNativeTileCounts.data()),
        static_cast<tcnn::vec2*>(workspace.particleProjectedCenter.data()),
        static_cast<tcnn::vec2*>(workspace.particleProjectedExtent.data()),
        static_cast<uint8_t*>(workspace.particleValidProjection.data()),
        static_cast<tcnn::vec4*>(workspace.particleConicOpacity.data()),
        static_cast<float*>(workspace.particleMaxConicPower.data()),
        static_cast<float4*>(workspace.particleSupportCap.data()),
        static_cast<float*>(workspace.particleMinimumRange.data()),
        static_cast<uint32_t*>(workspace.boundedParticleReasonMask.data()),
        static_cast<uint32_t*>(workspace.boundedParticleFirst.data()),
        static_cast<uint32_t*>(workspace.boundedParticleCount.data()),
        static_cast<uint32_t*>(workspace.certificationFailed.data()),
        static_cast<uint32_t*>(workspace.certificationFirstErrorIndex.data()),
        MemoryHandles{parameterBuffer});
    CUDA_CHECK_STREAM_RETURN(stream, logger);
    size_t bytes = workspace.candidateScanTemp.size();
    const cudaError_t countScanError = cub::DeviceScan::InclusiveSum(
        workspace.candidateScanTemp.data(), bytes,
        static_cast<const uint64_t*>(workspace.particleTileCounts.data()),
        static_cast<uint64_t*>(workspace.particleTileOffsets.data()),
        static_cast<int>(particleCount), stream);
    if (countScanError != cudaSuccess) {
        (void)cudaGetLastError();
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackCountScan);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    uint64_t intersections64 = 0;
    uint32_t failed = 0;
    uint32_t preprocessingFirstErrorIndex = UINT32_MAX;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&intersections64,
        static_cast<const uint64_t*>(workspace.particleTileOffsets.data()) + particleCount - 1,
        sizeof(uint64_t), cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&failed, workspace.certificationFailed.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        &preprocessingFirstErrorIndex,
        workspace.certificationFirstErrorIndex.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hostFallbackTelemetry + 1,
        workspace.boundedParticleReasonMask.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hostFallbackTelemetry + 2,
        workspace.boundedParticleFirst.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hostFallbackTelemetry + 3,
        workspace.boundedParticleCount.data(), sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), logger);
    if ((faultMask & kFaultInvalidCap) != 0) {
        failed = 1;
    }
    if (failed != 0) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackInjectedInvalidCap);
        hostFallbackTelemetry[2] = preprocessingFirstErrorIndex;
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    if ((faultMask & kFaultCountOverflow) != 0) {
        intersections64 = static_cast<uint64_t>(INT_MAX) + 1u;
    }
    if ((faultMask & kFaultZeroIntersections) != 0) {
    }
    if (intersections64 > static_cast<uint64_t>(INT_MAX)) {
        recordFirstError(hostFallbackTelemetry,
                             kGlobalFallbackIntersectionOverflow);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    if (intersections64 == 0 ||
        (faultMask & kFaultZeroIntersections) != 0) {
        recordFirstError(hostFallbackTelemetry,
                             kGlobalFallbackZeroIntersections);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    const uint32_t intersections = static_cast<uint32_t>(intersections64);
    if ((faultMask & kFaultCandidateAllocation) != 0) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackInjectedCandidateAllocation);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    const Status candidateStatus = workspace.reserveCandidates(
        intersections, tileCount, stream, logger);
    if (!candidateStatus) {
        recordFirstError(hostFallbackTelemetry, kGlobalFallbackCandidateWorkspace);
        RETURN_ERROR(
            logger, ErrorCode::BadInput,
            "cooperative field invariant failed closed");
    }
    {
        expandParticles<<<particleBlocks, kThreads, 0, stream>>>(
            tileGrid, particleCount,
            static_cast<const uint64_t*>(workspace.particleTileOffsets.data()),
            static_cast<const uint32_t*>(workspace.particleNativeTileCounts.data()),
            static_cast<const tcnn::vec2*>(workspace.particleProjectedCenter.data()),
            static_cast<const tcnn::vec2*>(workspace.particleProjectedExtent.data()),
            static_cast<const uint8_t*>(workspace.particleValidProjection.data()),
            static_cast<const tcnn::vec4*>(workspace.particleConicOpacity.data()),
            static_cast<const float*>(workspace.particleMaxConicPower.data()),
            static_cast<const float4*>(workspace.particleSupportCap.data()),
            physicalDomain, worldToSensor,
            static_cast<uint32_t*>(workspace.certificationFailed.data()),
            static_cast<uint32_t*>(workspace.certificationFirstErrorIndex.data()),
            static_cast<uint64_t*>(workspace.candidateKeysUnsorted.data()),
            static_cast<uint32_t*>(workspace.candidateIdsUnsorted.data()));
        CUDA_CHECK_STREAM_RETURN(stream, logger);
        uint32_t expansionFailed = 0;
        uint32_t expansionFirstErrorIndex = UINT32_MAX;
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            &expansionFailed, workspace.certificationFailed.data(),
            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), logger);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(
            &expansionFirstErrorIndex,
            workspace.certificationFirstErrorIndex.data(),
            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), logger);
        failed |= expansionFailed;
        if ((faultMask & kFaultExpansionMismatch) != 0) {
            failed = 1;
        }
        if (failed != 0) {
            recordFirstError(
                hostFallbackTelemetry, kGlobalFallbackExpansion);
            hostFallbackTelemetry[2] = expansionFirstErrorIndex;
            RETURN_ERROR(
                logger, ErrorCode::BadInput,
                "cooperative particle expansion cardinality mismatch");
        }
        {
            bytes = workspace.candidateSortTemp.size();
            if ((faultMask & kFaultCandidateSort) != 0) {
                recordFirstError(hostFallbackTelemetry, kGlobalFallbackInjectedCandidateSort);
                RETURN_ERROR(
                    logger, ErrorCode::BadInput,
                    "cooperative field invariant failed closed");
            }
            const cudaError_t candidateSortError = cub::DeviceRadixSort::SortPairs(
                workspace.candidateSortTemp.data(), bytes,
                static_cast<const uint64_t*>(workspace.candidateKeysUnsorted.data()),
                static_cast<uint64_t*>(workspace.candidateKeysSorted.data()),
                static_cast<const uint32_t*>(workspace.candidateIdsUnsorted.data()),
                static_cast<uint32_t*>(workspace.candidateIdsSorted.data()),
                static_cast<int>(intersections), 0, 32 + bitWidth(tileCount), stream);
            if (candidateSortError != cudaSuccess) {
                (void)cudaGetLastError();
                recordFirstError(hostFallbackTelemetry, kGlobalFallbackCandidateSort);
                RETURN_ERROR(
                    logger, ErrorCode::BadInput,
                    "cooperative field invariant failed closed");
            }
            CUDA_CHECK_RETURN(cudaMemsetAsync(
                workspace.certificationFailed.data(), 0, sizeof(uint32_t), stream),
                logger);
            CUDA_CHECK_RETURN(cudaMemsetAsync(
                workspace.certificationFirstErrorIndex.data(), 0xff,
                sizeof(uint32_t), stream), logger);
            validateSortedCandidateKeys<<<
                (intersections + 255u) / 256u, 256u, 0, stream>>>(
                    static_cast<const uint64_t*>(workspace.candidateKeysSorted.data()),
                    static_cast<const uint32_t*>(workspace.candidateIdsSorted.data()),
                    intersections, tileCount,
                    static_cast<uint32_t*>(workspace.certificationFailed.data()),
                    static_cast<uint32_t*>(
                        workspace.certificationFirstErrorIndex.data()));
            uint32_t sortedFailed = 0;
            uint32_t sortedFirstErrorIndex = UINT32_MAX;
            CUDA_CHECK_RETURN(cudaMemcpyAsync(
                &sortedFailed, workspace.certificationFailed.data(), sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream), logger);
            CUDA_CHECK_RETURN(cudaMemcpyAsync(
                &sortedFirstErrorIndex,
                workspace.certificationFirstErrorIndex.data(), sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream), logger);
            CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), logger);
            if (sortedFailed != 0) {
                recordFirstError(hostFallbackTelemetry, kGlobalFallbackSortedValidation);
                hostFallbackTelemetry[2] = sortedFirstErrorIndex;
                RETURN_ERROR(
                    logger, ErrorCode::BadInput,
                    "cooperative field invariant failed closed");
            }
            identifyRanges<<<(intersections + 255u) / 256u, 256u, 0, stream>>>(
                static_cast<const uint64_t*>(workspace.candidateKeysSorted.data()),
                intersections, tileCount,
                static_cast<tcnn::uvec2*>(workspace.candidateTileRanges.data()));
        }
    }

    const uint32_t pointGrid = (queryCount + 255u) / 256u;
    preprocessPoints<<<pointGrid, 256u, 0, stream>>>(
        params, sensorRayOrigins, sensorToWorld, physicalMask, queryPoints,
        queryCount, static_cast<uint64_t*>(workspace.pointKeysUnsorted.data()),
        static_cast<uint32_t*>(workspace.pointIdsUnsorted.data()),
        static_cast<tcnn::vec3*>(workspace.pointOrigins.data()),
        static_cast<tcnn::vec3*>(workspace.pointDirections.data()),
        static_cast<float*>(workspace.pointRanges.data()),
        static_cast<uint8_t*>(workspace.pointValid.data()),
        tileGrid);
    bytes = workspace.pointSortTemp.size();
    CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(
        workspace.pointSortTemp.data(), bytes,
        static_cast<const uint64_t*>(workspace.pointKeysUnsorted.data()),
        static_cast<uint64_t*>(workspace.pointKeysSorted.data()),
        static_cast<const uint32_t*>(workspace.pointIdsUnsorted.data()),
        static_cast<uint32_t*>(workspace.pointIdsSorted.data()),
        static_cast<int>(queryCount), 0, 32 + bitWidth(tileCount), stream), logger);
    identifyRanges<<<pointGrid, 256u, 0, stream>>>(
        static_cast<const uint64_t*>(workspace.pointKeysSorted.data()), queryCount,
        tileCount, static_cast<tcnn::uvec2*>(workspace.pointTileRanges.data()));
    const uint32_t tileBlocks = (tileCount + 255u) / 256u;
    blocksPerTile<<<tileBlocks, 256u, 0, stream>>>(
        tileCount, static_cast<const tcnn::uvec2*>(workspace.pointTileRanges.data()),
        static_cast<uint32_t*>(workspace.pointBlocksPerTile.data()));
    bytes = workspace.pointScanTemp.size();
    CUDA_CHECK_RETURN(cub::DeviceScan::ExclusiveSum(
        workspace.pointScanTemp.data(), bytes,
        static_cast<const uint32_t*>(workspace.pointBlocksPerTile.data()),
        static_cast<uint32_t*>(workspace.pointBlockOffsets.data()),
        static_cast<int>(tileCount), stream), logger);
    totalBlocks<<<1, 1, 0, stream>>>(
        tileCount, static_cast<const uint32_t*>(workspace.pointBlocksPerTile.data()),
        static_cast<const uint32_t*>(workspace.pointBlockOffsets.data()),
        static_cast<uint32_t*>(workspace.totalPointBlocks.data()));
    fillBlocks<<<tileBlocks, 256u, 0, stream>>>(
        tileCount, static_cast<const tcnn::uvec2*>(workspace.pointTileRanges.data()),
        static_cast<const uint32_t*>(workspace.pointBlocksPerTile.data()),
        static_cast<const uint32_t*>(workspace.pointBlockOffsets.data()),
        static_cast<PointBlock*>(workspace.pointBlocks.data()));
    uint32_t hostBlocks = 0;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&hostBlocks, workspace.totalPointBlocks.data(),
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), logger);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(stream), logger);
    if (hostBlocks > 0) {
        if (thresholdTieAware) {
            integratePointsThresholdTieAware<<<hostBlocks, 256u, 0, stream>>>(
                static_cast<const tcnn::uvec2*>(workspace.candidateTileRanges.data()),
                intersections == 0 ? nullptr :
                    static_cast<const uint32_t*>(workspace.candidateIdsSorted.data()),
                static_cast<const uint32_t*>(workspace.pointIdsSorted.data()),
                static_cast<const PointBlock*>(workspace.pointBlocks.data()),
                static_cast<const tcnn::vec3*>(workspace.pointOrigins.data()),
                static_cast<const tcnn::vec3*>(workspace.pointDirections.data()),
                static_cast<const float*>(workspace.pointRanges.data()),
                static_cast<const uint8_t*>(workspace.pointValid.data()),
                static_cast<const float*>(workspace.particleMinimumRange.data()),
                crossViewMinimum, crossViewObserved,
                MemoryHandles{parameterBuffer});
        } else {
            integratePoints<<<hostBlocks, 256u, 0, stream>>>(
                static_cast<const tcnn::uvec2*>(workspace.candidateTileRanges.data()),
                intersections == 0 ? nullptr :
                    static_cast<const uint32_t*>(workspace.candidateIdsSorted.data()),
                static_cast<const uint32_t*>(workspace.pointIdsSorted.data()),
                static_cast<const PointBlock*>(workspace.pointBlocks.data()),
                static_cast<const tcnn::vec3*>(workspace.pointOrigins.data()),
                static_cast<const tcnn::vec3*>(workspace.pointDirections.data()),
                static_cast<const float*>(workspace.pointRanges.data()),
                static_cast<const uint8_t*>(workspace.pointValid.data()),
                static_cast<const float*>(workspace.particleMinimumRange.data()),
                crossViewMinimum, crossViewObserved,
                MemoryHandles{parameterBuffer});
        }
    }
    CUDA_CHECK_STREAM_RETURN(stream, logger);
    return Status();
}

} // namespace threedgut::fast_sof_tnt_full_shape
