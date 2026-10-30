// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/renderer/gutRendererParameters.h>
#include <tiny-cuda-nn/common_device.h>
#include <3dgut/kernels/cuda/renderers/certifiedWhitenedConeIntersection.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedEllipticLeafIntersection.cuh>

namespace gofish_conservative_bound {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFloatEpsilon = 1.1920928955078125e-7f;
constexpr float kFloatMax = 3.4028234663852886e+38f;
constexpr float kGamma3 = (3.0f * kFloatEpsilon) / (1.0f - 3.0f * kFloatEpsilon);
constexpr float kConsumedDirectionRoundPad = 32.0f * kFloatEpsilon;
constexpr uint32_t kTraversalStack = 64;

struct ConservativeConeNode {
    float4 angularCone;
    // Directed-up Euclidean chord radius: every actual descendant direction d
    // satisfies ||d-angularCone.xyz|| <= chordRadius.
    float chordRadius;
    float directionMaxAbs;
    uint32_t directionOperandsCertified;
};

static inline __device__ float4 angularConeOf(const ConservativeConeNode node) {
    return node.angularCone;
}

static inline __device__ ConservativeConeNode makeConeNode(
    const float4 angularCone, float chordRadius,
    float directionMaxAbs = 0.0f,
    uint32_t directionOperandsCertified = 0u) {
    return ConservativeConeNode{
        angularCone, chordRadius, directionMaxAbs,
        directionOperandsCertified};
}

static inline __host__ __device__ uint32_t levelOffset(uint32_t side, uint32_t level) {
    uint32_t offset = 0;
    for (uint32_t value = 0; value < level; ++value) {
        offset += side * side;
        side >>= 1;
    }
    return offset;
}

static inline __host__ __device__ uint32_t totalNodes(uint32_t side) {
    uint32_t total = 0;
    while (side) {
        total += side * side;
        side >>= 1;
    }
    return total;
}

static inline __device__ float3 normalizeSafe(const float3 value) {
    const float norm = sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
    return (norm > 0.0f && isfinite(norm))
        ? make_float3(value.x / norm, value.y / norm, value.z / norm)
        : make_float3(0.0f, 0.0f, 0.0f);
}

static inline __device__ float dot3(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __device__ bool floatOperandCertified(float value) {
    const float magnitude = fabsf(value);
    return isfinite(value)
        && !(magnitude > 0.0f && magnitude < 1.1754943508222875e-38f);
}

static inline __device__ float euclideanChordUpper(const float3 a, const float3 b) {
    const double ax[3] = {(double)a.x, (double)a.y, (double)a.z};
    const double bx[3] = {(double)b.x, (double)b.y, (double)b.z};
    double squaredUpper = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const double differenceLower = __dsub_rd(ax[axis], bx[axis]);
        const double differenceUpper = __dsub_ru(ax[axis], bx[axis]);
        const double absoluteUpper = fmax(fabs(differenceLower), fabs(differenceUpper));
        squaredUpper = __dadd_ru(
            squaredUpper, __dmul_ru(absoluteUpper, absoluteUpper));
    }
    const double chordUpper = __dsqrt_ru(squaredUpper);
    return nextafterf(__double2float_ru(chordUpper), kFloatMax);
}

static inline __device__ bool overlap(const float3 gaussianCenter,
                                      float gaussianHalfAngle,
                                      const ConservativeConeNode node) {
    const float4 angularCone = angularConeOf(node);
    if (!(angularCone.w >= 0.0f)) return false;
    const float total = fminf(kPi, gaussianHalfAngle + angularCone.w);
    if (total >= kPi) return true;
    // gamma_3 is the exact first-order forward bound for the three-product
    // dot.  Relaxing by this amount is an IEEE certificate, not a margin.
    return dot3(gaussianCenter, make_float3(
        angularCone.x, angularCone.y, angularCone.z)) + kGamma3
        >= cosf(total);
}

// The finite-ellipsoid proof has one camera origin.  Verify the actual
// per-pixel origin buffer consumed by initializeRay, rather than inferring
// centrality from a camera-model enum.  Central camera contracts store exact
// sensor-space zero origins; any different/nonfinite value disables pruning.
__global__ void verifyCentralRayOrigins(
    uint32_t numPixels,
    const tcnn::vec3* __restrict__ sensorRayOrigin,
    uint32_t* __restrict__ centralityFlag) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numPixels) return;
    const tcnn::vec3 origin = sensorRayOrigin[index];
    if (!isfinite(origin.x) || !isfinite(origin.y) || !isfinite(origin.z)
        || origin.x != 0.0f || origin.y != 0.0f || origin.z != 0.0f) {
        atomicAnd(centralityFlag, 0u);
    }
}

// One shared traversal predicate is used by count and emit.  The whitened
// finite-ellipsoid certificate is evaluated at every node, including leaves; it
// can omit only nodes containing no positive-ray support intersection.
static inline __device__ bool traversalNodeAdmitted(
    const float3 gaussianCenter,
    float gaussianHalfAngle,
    const ConservativeConeNode node,
    uint32_t level,
    const gofish_whitened_cone::FiniteEllipsoidConeCache& cache
    , float density, float alphaThreshold
    ) {
    // Exact-leaf mode does not use the Gaussian angular-cap reject. The
    // whitened certificate prunes hierarchy nodes; the chord-elliptic
    // certificate additionally rejects any certified hierarchy node. Only
    // level-zero nodes can emit tiles in the traversal loops.
    (void)gaussianCenter;
    (void)gaussianHalfAngle;
    const float4 angularCone = angularConeOf(node);
    if (gofish_whitened_cone::certifiedNoPositiveRayIntersection(
            cache,
            make_float3(angularCone.x, angularCone.y, angularCone.z),
            node.chordRadius,
            node.directionMaxAbs,
            node.directionOperandsCertified != 0u)) return false;
    (void)level;
    if (gofish_elliptic_leaf::certifiedNoEllipticNodeIntersection(
            cache,
            make_float3(angularCone.x, angularCone.y, angularCone.z),
            node.chordRadius,
            node.directionMaxAbs,
            node.directionOperandsCertified != 0u,
            density,
            alphaThreshold)) return false;
    return true;
}

__global__ void buildLeafCones(uint32_t paddedSide,
                               tcnn::uvec2 tileGrid,
                               tcnn::ivec2 resolution,
                               const tcnn::vec3* __restrict__ sensorRayDirection,
                               tcnn::mat4x3 sensorToWorld,
                               ConservativeConeNode* __restrict__ tree) {
    constexpr uint32_t blockX = threedgut::GUTParameters::Tiling::BlockX;
    constexpr uint32_t blockY = threedgut::GUTParameters::Tiling::BlockY;
    constexpr uint32_t count = blockX * blockY;
    __shared__ float sx[count], sy[count], sz[count], reduction[count];
    __shared__ uint32_t operandCertified[count];
    __shared__ float3 center;
    __shared__ float angularRadius;
    __shared__ float leafChordRadius;
    const uint32_t leaf = blockIdx.x;
    const uint32_t tileX = leaf % paddedSide;
    const uint32_t tileY = leaf / paddedSide;
    const uint32_t lane = threadIdx.x;
    bool valid = tileX < tileGrid.x && tileY < tileGrid.y;
    const uint32_t pixelX = tileX * blockX + (lane % blockX);
    const uint32_t pixelY = tileY * blockY + (lane / blockX);
    valid = valid && pixelX < static_cast<uint32_t>(resolution.x)
                  && pixelY < static_cast<uint32_t>(resolution.y);
    float3 direction = make_float3(0.0f, 0.0f, 0.0f);
    float3 consumedDirection = make_float3(0.0f, 0.0f, 0.0f);
    bool directionOperandsCertified = true;
    if (valid) {
        const tcnn::vec3 sensor = sensorRayDirection[pixelY * resolution.x + pixelX];
        directionOperandsCertified =
            floatOperandCertified(sensor.x)
            && floatOperandCertified(sensor.y)
            && floatOperandCertified(sensor.z);
#pragma unroll
        for (int row = 0; row < 3; ++row) {
#pragma unroll
            for (int col = 0; col < 3; ++col) {
                directionOperandsCertified = directionOperandsCertified
                    && floatOperandCertified(sensorToWorld[col][row]);
            }
        }
        const tcnn::vec3 world = tcnn::mat3(sensorToWorld) * sensor;
        consumedDirection = make_float3(world.x, world.y, world.z);
        directionOperandsCertified = directionOperandsCertified
            && floatOperandCertified(consumedDirection.x)
            && floatOperandCertified(consumedDirection.y)
            && floatOperandCertified(consumedDirection.z);
        direction = normalizeSafe(consumedDirection);
        valid = isfinite(direction.x) && isfinite(direction.y) && isfinite(direction.z)
             && dot3(direction, direction) > 0.0f;
    }
    sx[lane] = valid ? direction.x : 0.0f;
    sy[lane] = valid ? direction.y : 0.0f;
    sz[lane] = valid ? direction.z : 0.0f;
    reduction[lane] = valid ? 1.0f : 0.0f;
    operandCertified[lane] = valid && !directionOperandsCertified ? 0u : 1u;
    __syncthreads();
    for (uint32_t stride = count / 2; stride; stride >>= 1) {
        if (lane < stride) {
            sx[lane] += sx[lane + stride];
            sy[lane] += sy[lane + stride];
            sz[lane] += sz[lane + stride];
            reduction[lane] += reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) center = normalizeSafe(make_float3(sx[0], sy[0], sz[0]));
    __syncthreads();
    reduction[lane] = valid ? dot3(direction, center) : 1.0f;
    __syncthreads();
    for (uint32_t stride = count / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] = fminf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) {
        if (reduction[0] >= 1.0f && dot3(center, center) == 0.0f) {
            angularRadius = -1.0f;
        } else {
            const float certifiedDot = fmaxf(-1.0f, fminf(1.0f, reduction[0] - kGamma3));
            angularRadius = nextafterf(acosf(certifiedDot), kFloatMax);
        }
    }
    __syncthreads();
    reduction[lane] = valid
        ? __fadd_ru(euclideanChordUpper(consumedDirection, center),
                    kConsumedDirectionRoundPad)
        : 0.0f;
    __syncthreads();
    for (uint32_t stride = count / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) leafChordRadius = nextafterf(reduction[0], kFloatMax);
    __syncthreads();
    reduction[lane] = valid
        ? fmaxf(fmaxf(fabsf(consumedDirection.x), fabsf(consumedDirection.y)),
                fabsf(consumedDirection.z))
        : 0.0f;
    __syncthreads();
    for (uint32_t stride = count / 2; stride; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            operandCertified[lane] &= operandCertified[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        tree[leaf] = angularRadius >= 0.0f
            ? makeConeNode(
                make_float4(center.x, center.y, center.z, angularRadius),
                leafChordRadius,
                nextafterf(reduction[0], kFloatMax),
                operandCertified[0])
            : makeConeNode(make_float4(0.0f, 0.0f, 0.0f, -1.0f), -1.0f);
    }
}

__global__ void buildParentCones(uint32_t paddedSide,
                                 uint32_t level,
                                 ConservativeConeNode* __restrict__ tree) {
    const uint32_t parentSide = paddedSide >> level;
    const uint32_t parentIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (parentIndex >= parentSide * parentSide) return;
    const uint32_t childSide = parentSide << 1;
    const uint32_t childOffset = levelOffset(paddedSide, level - 1);
    const uint32_t parentOffset = levelOffset(paddedSide, level);
    const uint32_t px = parentIndex % parentSide;
    const uint32_t py = parentIndex / parentSide;
    ConservativeConeNode child[4];
    uint32_t valid = 0;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t iy = 0; iy < 2; ++iy) {
        for (uint32_t ix = 0; ix < 2; ++ix) {
            const uint32_t slot = iy * 2 + ix;
            child[slot] = tree[childOffset + (2 * py + iy) * childSide + (2 * px + ix)];
            const float4 childCone = angularConeOf(child[slot]);
            if (childCone.w >= 0.0f) {
                sum.x += childCone.x;
                sum.y += childCone.y;
                sum.z += childCone.z;
                ++valid;
            }
        }
    }
    if (!valid) {
        tree[parentOffset + parentIndex] = makeConeNode(
            make_float4(0.0f, 0.0f, 0.0f, -1.0f), -1.0f);
        return;
    }
    float3 center = normalizeSafe(sum);
    if (dot3(center, center) == 0.0f) {
        for (uint32_t slot = 0; slot < 4; ++slot) {
            const float4 childCone = angularConeOf(child[slot]);
            if (childCone.w >= 0.0f) {
                center = make_float3(
                    childCone.x, childCone.y, childCone.z);
                break;
            }
        }
    }
    float angle = 0.0f;
    float chordRadius = 0.0f;
    float directionMaxAbs = 0.0f;
    uint32_t directionOperandsCertified = 1u;
    for (uint32_t slot = 0; slot < 4; ++slot) {
        const float4 childCone = angularConeOf(child[slot]);
        if (childCone.w < 0.0f) continue;
        const float3 childCenter = make_float3(
            childCone.x, childCone.y, childCone.z);
        const float dot = fmaxf(-1.0f, fminf(1.0f,
            dot3(center, childCenter) - kGamma3));
        angle = fmaxf(angle, acosf(dot) + childCone.w);
        chordRadius = fmaxf(
            chordRadius,
            __fadd_ru(euclideanChordUpper(center, childCenter), child[slot].chordRadius));
        directionMaxAbs = fmaxf(directionMaxAbs, child[slot].directionMaxAbs);
        directionOperandsCertified &= child[slot].directionOperandsCertified;
    }
    angle = nextafterf(fminf(kPi, angle), kFloatMax);
    tree[parentOffset + parentIndex] = makeConeNode(
        make_float4(center.x, center.y, center.z, angle),
        nextafterf(chordRadius, kFloatMax),
        nextafterf(directionMaxAbs, kFloatMax),
        directionOperandsCertified
    );
}

} // namespace gofish_conservative_bound

