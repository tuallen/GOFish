// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedFp32Rejection.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedSupportCap.cuh>
#include <3dgut/kernels/cuda/renderers/conservativeOpacityBound.cuh>

#include <cuda_runtime.h>
#include <cstdint>

namespace gofish_flat_cone_k_test {

using gofish_whitened_cone::FiniteEllipsoidConeCache;

constexpr double kSupportSquared = 9.0;

enum class CameraRelation : uint32_t {
    Outside = 0,
    Inside = 1,
    Uncertain = 2,
};

enum class TileDecision : uint32_t {
    Retain = 0,
    ProvenRejectFp32 = 1,
    ProvenRejectFp64 = 2,
};

struct TileRayCone {
    uint32_t tile;
    float3 center;
    float chordRadius;
    float directionMaxAbs;
    uint32_t directionOperandsCertified;
};

struct Input {
    FiniteEllipsoidConeCache fp64;
    gofish_fp32_certificate::Cache fp32;
    float rendererDensity;
    float alphaThreshold;
    uint32_t angularCapCertified;
};

struct Summary {
    uint32_t enumerated;
    uint32_t retained;
    uint32_t rejectedFp32;
    uint32_t rejectedFp64;
    uint32_t admittedAll;
};

// Inputs. physicalTileNonempty is the
// tile-wise OR reduction of the same immutable physical pixel mask consumed by
// rendering. The tree is the exact ConservativeConeNode tree built
// from the consumed ray table; level-zero records are never recomputed.
struct FlatConeLeafTree {
    const gofish_conservative_bound::ConservativeConeNode* tree;
    const uint8_t* physicalTileNonempty;
    uint32_t paddedSide;
    tcnn::uvec2 tileGrid;
    uint32_t treeLeavesCertified;
    uint32_t physicalMaskCertified;
};

struct FlatConeCap {
    float3 relative;
    double worldRadiusUpper;
    uint32_t allTiles;
    uint32_t certified;
};

// Deterministic tile-wise OR of the exact physical pixel mask (one block per
// active tile), supplied through FlatConeLeafTree.
__global__ void buildPhysicalTileNonempty(
    tcnn::uvec2 tileGrid, tcnn::ivec2 resolution,
    const uint8_t* __restrict__ physicalPixelMask,
    uint8_t* __restrict__ physicalTileNonempty) {
    const uint32_t tile = blockIdx.x;
    if (tile >= tileGrid.x * tileGrid.y) return;
    __shared__ uint32_t any;
    if (threadIdx.x == 0) any = 0u;
    __syncthreads();
    const uint32_t tileX = tile % tileGrid.x;
    const uint32_t tileY = tile / tileGrid.x;
    constexpr uint32_t blockX = threedgut::GUTParameters::Tiling::BlockX;
    constexpr uint32_t blockY = threedgut::GUTParameters::Tiling::BlockY;
    for (uint32_t lane = threadIdx.x; lane < blockX * blockY;
         lane += blockDim.x) {
        const uint32_t x = tileX * blockX + lane % blockX;
        const uint32_t y = tileY * blockY + lane / blockX;
        if (x < uint32_t(resolution.x) && y < uint32_t(resolution.y)
            && physicalPixelMask[y * uint32_t(resolution.x) + x] != 0u) {
            atomicExch(&any, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) physicalTileNonempty[tile] = uint8_t(any != 0u);
}

static inline __device__ Input makeInput(
    const FiniteEllipsoidConeCache& cache,
    float rendererDensity,
    float alphaThreshold,
    bool angularCapCertified) {
    return Input{
        cache,
        gofish_fp32_certificate::makeCache(cache),
        rendererDensity,
        alphaThreshold,
        angularCapCertified ? 1u : 0u,
    };
}

// Replays the certifiedSupportCap factor/radius construction; flat
// enumeration uses the equivalent cosine-free ball inequality below, not
// acos/cos overlap.
static inline __device__ FlatConeCap makeFlatConeCap(
    float opacity, float alphaThreshold, float3 relative,
    const float factor[9]) {
    FlatConeCap result{relative, 0.0, 1u, 0u};
    float3 sourceCenter = make_float3(0.0f, 0.0f, 0.0f);
    float sourceHalfAngle = 0.0f;
    const bool sourceAccepted =
        gofish_certified_support_cap::certifiedSupportCap(
            opacity, alphaThreshold, relative, factor,
            sourceCenter, sourceHalfAngle);
    // A false cap result is the opacity-empty branch. This flat core is
    // only a nonempty fallback, so retain all rather than claiming emptiness.
    if (!sourceAccepted) return result;

    double rowSum[3] = {0.0, 0.0, 0.0};
    double colSum[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int row = 0; row < 3; ++row) {
#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const double value = fabs(double(factor[3 * row + col]));
            if (!isfinite(value)) return result;
            rowSum[row] = __dadd_ru(rowSum[row], value);
            colSum[col] = __dadd_ru(colSum[col], value);
        }
    }
    const double normOneUpper = fmax(fmax(colSum[0], colSum[1]), colSum[2]);
    const double normInfUpper = fmax(fmax(rowSum[0], rowSum[1]), rowSum[2]);
    const double inducedNormUpper = __dmul_ru(
        0.5, __dadd_ru(normOneUpper, normInfUpper));
    result.worldRadiusUpper = __dmul_ru(4.0, inducedNormUpper);
    if (!isfinite(result.worldRadiusUpper)) return result;
    const double distanceLower = fmax(
        fmax(fabs(double(relative.x)), fabs(double(relative.y))),
        fabs(double(relative.z)));
    result.allTiles = !(distanceLower > result.worldRadiusUpper) ? 1u : 0u;
    result.certified = 1u;
    (void)sourceCenter;
    (void)sourceHalfAngle;
    return result;
}

static inline __device__ TileRayCone leafCone(
    const FlatConeLeafTree& leaves, uint32_t x, uint32_t y) {
    const uint32_t index = gofish_conservative_bound::levelOffset(
        leaves.paddedSide, 0u) + y * leaves.paddedSide + x;
    const auto node = leaves.tree[index];
    const float4 angular = gofish_conservative_bound::angularConeOf(node);
    return TileRayCone{
        y * leaves.tileGrid.x + x,
        make_float3(angular.x, angular.y, angular.z),
        node.chordRadius,
        node.directionMaxAbs,
        node.directionOperandsCertified,
    };
}

static inline __device__ bool leafTreeInputsValid(
    const FlatConeLeafTree& leaves) {
    return leaves.tree != nullptr && leaves.physicalTileNonempty != nullptr
        && leaves.treeLeavesCertified == 1u
        && leaves.physicalMaskCertified == 1u
        && leaves.paddedSide > 0u
        && leaves.tileGrid.x > 0u && leaves.tileGrid.y > 0u
        && leaves.tileGrid.x <= leaves.paddedSide
        && leaves.tileGrid.y <= leaves.paddedSide;
}

// True means enumerate/test the tile. False is a certified outer-ball miss.
static inline __device__ bool capMayOverlap(
    const FlatConeCap& cap, const TileRayCone& cone) {
    if (cap.certified != 1u || cap.allTiles != 0u
        || cone.directionOperandsCertified == 0u
        || !(cone.chordRadius >= 0.0f)) return true;
    const double m[3] = {
        double(cap.relative.x), double(cap.relative.y),
        double(cap.relative.z)};
    double normSquaredLower = 0.0;
    double normSquaredUpper = 0.0;
    double dotUpper = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        normSquaredLower = __dadd_rd(
            normSquaredLower, __dmul_rd(m[axis], m[axis]));
        normSquaredUpper = __dadd_ru(
            normSquaredUpper, __dmul_ru(m[axis], m[axis]));
        const double center = axis == 0 ? double(cone.center.x)
            : (axis == 1 ? double(cone.center.y) : double(cone.center.z));
        dotUpper = __dadd_ru(dotUpper, __dmul_ru(m[axis], center));
    }
    const double radiusSquaredUpper = __dmul_ru(
        cap.worldRadiusUpper, cap.worldRadiusUpper);
    const double separationSquaredLower = __dsub_rd(
        normSquaredLower, radiusSquaredUpper);
    if (!(separationSquaredLower > 0.0)
        || !isfinite(separationSquaredLower)) return true;
    const double thresholdLower = __dsqrt_rd(separationSquaredLower);
    const double normUpper = __dsqrt_ru(normSquaredUpper);
    dotUpper = __dadd_ru(
        dotUpper, __dmul_ru(double(cone.chordRadius), normUpper));
    return !(isfinite(dotUpper) && isfinite(thresholdLower)
             && dotUpper < thresholdLower);
}

// The camera is inside only when the complete outward upper norm is <=9.
// An interval straddling 9 is uncertain and therefore admits all physical
// tiles; uncertainty can never become a rejection.
static inline __device__ CameraRelation cameraRelation(
    const FiniteEllipsoidConeCache& cache) {
    if (!cache.certified) return CameraRelation::Uncertain;
    const double lower = gofish_whitened_cone::normSquaredLower(cache.a);
    const double upper = gofish_whitened_cone::normSquaredUpper(cache.a);
    if (!isfinite(lower) || !isfinite(upper)) return CameraRelation::Uncertain;
    if (upper <= kSupportSquared) return CameraRelation::Inside;
    if (lower > kSupportSquared) return CameraRelation::Outside;
    return CameraRelation::Uncertain;
}

// For P=Sigma^{-1}, w=mu-o, p=Pw and kappa=w^T P w-r^2, a ray direction d
// intersects support only if d^T p>0 and d^T K d>=0, where
// K=p p^T-kappa P. The called predicates implement outward chord-ball upper
// bounds for those two expressions. FP32 may only prove the same rejection;
// every inconclusive result executes the directed-FP64 predicate.
static inline __device__ TileDecision testTile(
    const Input& input, const TileRayCone& cone) {
    if (cone.directionOperandsCertified == 0u
        || !(cone.chordRadius >= 0.0f)
        || !(cone.directionMaxAbs > 0.0f)
        || !isfinite(cone.center.x) || !isfinite(cone.center.y)
        || !isfinite(cone.center.z) || !isfinite(cone.chordRadius)
        || !isfinite(cone.directionMaxAbs)) {
        return TileDecision::Retain;
    }
    if (gofish_fp32_certificate::ellipticProvesSameRejection(
            input.fp32, cone.center, cone.chordRadius,
            cone.directionMaxAbs, true)) {
        return TileDecision::ProvenRejectFp32;
    }
    if (gofish_elliptic_leaf::certifiedNoEllipticNodeIntersection(
            input.fp64, cone.center, cone.chordRadius,
            cone.directionMaxAbs, true,
            input.rendererDensity, input.alphaThreshold)) {
        return TileDecision::ProvenRejectFp64;
    }
    return TileDecision::Retain;
}

template <typename EnumerateCapTiles, typename EnumerateAllPhysicalTiles,
          typename EmitTile>
static inline __device__ Summary enumerate(
    const Input& input,
    EnumerateCapTiles enumerateCapTiles,
    EnumerateAllPhysicalTiles enumerateAllPhysicalTiles,
    EmitTile emitTile) {
    Summary summary{};
    const CameraRelation relation = cameraRelation(input.fp64);
    if (relation != CameraRelation::Outside
        || input.angularCapCertified != 1u) {
        // Camera-inside and every uncertain input retain all physical
        // tiles. The all-physical enumerator owns mask/domain clipping.
        enumerateAllPhysicalTiles([&](const TileRayCone& cone) {
            ++summary.enumerated;
            ++summary.retained;
            emitTile(cone.tile);
        });
        summary.admittedAll = 1u;
        return summary;
    }

    // The cap enumerator must be a certified superset of every tile-ray cone
    // containing a support-intersecting ray. K can only remove a tile after a
    // proof over its complete directed chord cone.
    enumerateCapTiles([&](const TileRayCone& cone) {
        ++summary.enumerated;
        const TileDecision decision = testTile(input, cone);
        if (decision == TileDecision::ProvenRejectFp32) {
            ++summary.rejectedFp32;
        } else if (decision == TileDecision::ProvenRejectFp64) {
            ++summary.rejectedFp64;
        } else {
            ++summary.retained;
            emitTile(cone.tile);
        }
    });
    return summary;
}

// Flat-cone adapter. Returns false only when its inputs are absent; the caller
// then keeps the hierarchy route. A true return is a complete flat result over
// the physical tile set.
template <typename EmitTile>
static inline __device__ bool enumerateFlatCone(
    const Input& input, const FlatConeCap& cap,
    const FlatConeLeafTree& leaves, EmitTile emitTile,
    Summary& summary) {
    if (!leafTreeInputsValid(leaves)) return false;
    auto allPhysical = [&](auto visitor) {
        for (uint32_t y = 0; y < leaves.tileGrid.y; ++y) {
            for (uint32_t x = 0; x < leaves.tileGrid.x; ++x) {
                const uint32_t tile = y * leaves.tileGrid.x + x;
                if (leaves.physicalTileNonempty[tile] == 0u) continue;
                visitor(leafCone(leaves, x, y));
            }
        }
    };
    auto capTiles = [&](auto visitor) {
        allPhysical([&](const TileRayCone& cone) {
            if (capMayOverlap(cap, cone)) visitor(cone);
        });
    };
    Input bound = input;
    bound.angularCapCertified = cap.certified;
    summary = enumerate(bound, capTiles, allPhysical, emitTile);
    return true;
}

} // namespace gofish_flat_cone_k_test

