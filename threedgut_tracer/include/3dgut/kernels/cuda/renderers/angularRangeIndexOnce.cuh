// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/flatConeKTest.cuh>
#include <3dgut/kernels/cuda/renderers/physicalCandidateTileFilter.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace gofish_angular_range_once {

// Coarse cells affect only over-enumeration; the exact cap and K tests decide
// admission. Ten-degree cells keep the index small.
constexpr uint32_t kThetaBins = 18u;
constexpr uint32_t kPhiBins = 36u;
constexpr uint32_t kCellCount = kThetaBins * kPhiBins;
constexpr uint32_t kMaxWords = 128u; // <= 4096 raster tiles
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
// Rounding allowance of the consumed ray directions. This is
// a numerical proof allowance, not a tunable geometric margin.
constexpr double kAngularArithmeticPad =
    2.0 * double(gofish_conservative_bound::kConsumedDirectionRoundPad);
constexpr double kDirectedDotPad = 64.0 * DBL_EPSILON;

struct View {
    const uint32_t* bits;
    uint32_t words;
    uint32_t tiles;
};

struct BinRange {
    int theta0, theta1;
    int phi0a, phi1a;
    int phi0b, phi1b;
    uint32_t phiRanges;
    uint32_t valid;
    uint32_t allCells;
};

static inline __host__ __device__ uint32_t wordsFor(uint32_t tiles) {
    return (tiles + 31u) >> 5u;
}

static inline __host__ __device__ uint64_t bytesFor(uint32_t tiles) {
    return uint64_t(kCellCount) * uint64_t(wordsFor(tiles))
        * sizeof(uint32_t);
}

static inline __device__ double clampUnit(double value) {
    return fmin(1.0, fmax(-1.0, value));
}

static inline __device__ double wrapPhi(double value) {
    while (value < 0.0) value += kTwoPi;
    while (value >= kTwoPi) value -= kTwoPi;
    return value;
}

static inline __device__ int thetaBin(double value) {
    const int result = int(floor(value * (double(kThetaBins) / kPi)));
    return max(0, min(int(kThetaBins) - 1, result));
}

static inline __device__ int phiBin(double value) {
    const int result = int(floor(wrapPhi(value)
        * (double(kPhiBins) / kTwoPi)));
    return max(0, min(int(kPhiBins) - 1, result));
}

// Outward angular bounding box of a spherical disk. Branch-cut arcs split;
// any pole or inconclusive operation promotes to all phi / all cells.
static inline __device__ BinRange diskRange(float3 rawAxis, double halfUpper) {
    BinRange out{};
    const double x = double(rawAxis.x);
    const double y = double(rawAxis.y);
    const double z = double(rawAxis.z);
    double n2 = 0.0;
    n2 = __dadd_rd(n2, __dmul_rd(x, x));
    n2 = __dadd_rd(n2, __dmul_rd(y, y));
    n2 = __dadd_rd(n2, __dmul_rd(z, z));
    if (!(n2 > 0.0) || !isfinite(n2) || !(halfUpper >= 0.0)
        || !isfinite(halfUpper)) {
        out.valid = 1u;
        out.allCells = 1u;
        return out;
    }
    const double normLower = __dsqrt_rd(n2);
    const double normRn = sqrt(x*x + y*y + z*z);
    if (!(normLower > 0.0) || !(normRn > 0.0) || !isfinite(normRn)) {
        out.valid = 1u;
        out.allCells = 1u;
        return out;
    }
    const double ax = x / normRn;
    const double ay = y / normRn;
    const double az = z / normRn;
    halfUpper = __dadd_ru(halfUpper, kAngularArithmeticPad);
    if (!(halfUpper < kPi)) {
        out.valid = 1u;
        out.allCells = 1u;
        return out;
    }
    const double theta = acos(clampUnit(az));
    if (!isfinite(theta)) {
        out.valid = 1u;
        out.allCells = 1u;
        return out;
    }
    const double lo = nextafter(fmax(0.0, theta - halfUpper), -INFINITY);
    const double hi = nextafter(fmin(kPi, theta + halfUpper), INFINITY);
    out.theta0 = thetaBin(lo);
    out.theta1 = thetaBin(hi);
    if (halfUpper >= fmin(theta, kPi - theta)
        || fabs(sin(theta)) <= kAngularArithmeticPad) {
        out.phi0a = 0;
        out.phi1a = int(kPhiBins) - 1;
        out.phiRanges = 1u;
        out.valid = 1u;
        return out;
    }
    const double ratio = fmin(1.0, fmax(0.0,
        __ddiv_ru(sin(halfUpper), sin(theta))));
    double delta = asin(ratio);
    delta = nextafter(delta + kAngularArithmeticPad, INFINITY);
    const double phi = wrapPhi(atan2(ay, ax));
    const double a = phi - delta;
    const double b = phi + delta;
    if (a < 0.0) {
        out.phi0a = 0;
        out.phi1a = phiBin(nextafter(b, INFINITY));
        out.phi0b = phiBin(nextafter(a + kTwoPi, -INFINITY));
        out.phi1b = int(kPhiBins) - 1;
        out.phiRanges = 2u;
    } else if (b >= kTwoPi) {
        out.phi0a = phiBin(nextafter(a, -INFINITY));
        out.phi1a = int(kPhiBins) - 1;
        out.phi0b = 0;
        out.phi1b = phiBin(nextafter(b - kTwoPi, INFINITY));
        out.phiRanges = 2u;
    } else {
        out.phi0a = phiBin(nextafter(a, -INFINITY));
        out.phi1a = phiBin(nextafter(b, INFINITY));
        out.phiRanges = 1u;
    }
    out.valid = 1u;
    return out;
}

static inline __device__ void markRange(
    const BinRange& range, uint32_t word, uint32_t bit,
    uint32_t words, uint32_t* bits) {
    if (range.allCells != 0u) {
        for (uint32_t cell = 0u; cell < kCellCount; ++cell)
            atomicOr(bits + uint64_t(cell) * words + word, bit);
        return;
    }
    for (int theta = range.theta0; theta <= range.theta1; ++theta) {
        for (int phi = range.phi0a; phi <= range.phi1a; ++phi)
            atomicOr(bits + (uint64_t(theta) * kPhiBins
                + uint32_t(phi)) * words + word, bit);
        if (range.phiRanges == 2u) {
            for (int phi = range.phi0b; phi <= range.phi1b; ++phi)
                atomicOr(bits + (uint64_t(theta) * kPhiBins
                    + uint32_t(phi)) * words + word, bit);
        }
    }
}

// Every leaf was built from the exact calibrated pixel-centre directions
// consumed by rendering. chordRadius is a directed-up ball containing those
// consumed directions. We additionally absorb normalization and the exact
// directed-dot arithmetic used by FlatConeCap::capMayOverlap. For a unit cap
// axis a, capMayOverlap implies angle(a, normalized(center)) <= alpha+delta,
// where cos(delta)=1-rho. Thus this tile disk contains every cap axis that
// can survive the cap predicate.
__global__ void build(
    tcnn::uvec2 tileGrid,
    gofish_physical_candidate::PhysicalTileView physicalTiles,
    const gofish_conservative_bound::ConservativeConeNode* tree,
    uint32_t side, uint32_t* bits, uint32_t words) {
    const uint32_t tile = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tiles = tileGrid.x * tileGrid.y;
    if (tile >= tiles || !physicalTiles.retainTile(tile)) return;
    const uint32_t y = tile / tileGrid.x;
    const uint32_t x = tile - y * tileGrid.x;
    const auto leaf = tree[gofish_conservative_bound::levelOffset(side, 0u)
        + y * side + x];
    const float4 angular = gofish_conservative_bound::angularConeOf(leaf);
    BinRange range{};
    if (leaf.directionOperandsCertified != 1u
        || !(leaf.chordRadius >= 0.0f)
        || !isfinite(angular.x) || !isfinite(angular.y)
        || !isfinite(angular.z)) {
        range.valid = 1u;
        range.allCells = 1u;
    } else {
        const double cx = double(angular.x);
        const double cy = double(angular.y);
        const double cz = double(angular.z);
        double n2 = 0.0;
        n2 = __dadd_ru(n2, __dmul_ru(cx, cx));
        n2 = __dadd_ru(n2, __dmul_ru(cy, cy));
        n2 = __dadd_ru(n2, __dmul_ru(cz, cz));
        const double n = __dsqrt_ru(n2);
        double centerChord = INFINITY;
        if (n > 0.0 && isfinite(n)) {
            const double nx = cx / n;
            const double ny = cy / n;
            const double nz = cz / n;
            double d2 = 0.0;
            d2 = __dadd_ru(d2, __dmul_ru(cx-nx, cx-nx));
            d2 = __dadd_ru(d2, __dmul_ru(cy-ny, cy-ny));
            d2 = __dadd_ru(d2, __dmul_ru(cz-nz, cz-nz));
            centerChord = __dsqrt_ru(d2);
        }
        double rho = __dadd_ru(double(leaf.chordRadius), centerChord);
        rho = __dadd_ru(rho, kDirectedDotPad);
        if (!(rho >= 0.0) || !isfinite(rho) || rho >= 2.0) {
            range.valid = 1u;
            range.allCells = 1u;
        } else {
            double delta = acos(clampUnit(1.0 - rho));
            delta = nextafter(delta, INFINITY);
            range = diskRange(make_float3(angular.x, angular.y, angular.z),
                              delta);
        }
    }
    markRange(range, tile >> 5u, 1u << (tile & 31u), words, bits);
}

// Same directed separation threshold as FlatConeCap.
static inline __device__ bool sourceRange(
    const gofish_flat_cone_k_test::FlatConeCap& cap, BinRange& range) {
    if (cap.certified != 1u || cap.allTiles != 0u
        || !(cap.worldRadiusUpper >= 0.0)
        || !isfinite(cap.worldRadiusUpper)) return false;
    const double m[3] = {double(cap.relative.x), double(cap.relative.y),
                         double(cap.relative.z)};
    double normLower2 = 0.0;
    double normUpper2 = 0.0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        normLower2 = __dadd_rd(normLower2, __dmul_rd(m[axis], m[axis]));
        normUpper2 = __dadd_ru(normUpper2, __dmul_ru(m[axis], m[axis]));
    }
    const double radiusUpper2 = __dmul_ru(
        cap.worldRadiusUpper, cap.worldRadiusUpper);
    const double separationLower2 = __dsub_rd(normLower2, radiusUpper2);
    if (!(separationLower2 > 0.0) || !isfinite(separationLower2))
        return false;
    const double thresholdLower = __dsqrt_rd(separationLower2);
    const double normUpper = __dsqrt_ru(normUpper2);
    if (!(normUpper > 0.0) || !isfinite(normUpper)) return false;
    double cosineLower = __ddiv_rd(thresholdLower, normUpper);
    cosineLower = nextafter(clampUnit(cosineLower), -INFINITY);
    double alphaUpper = acos(cosineLower);
    alphaUpper = nextafter(alphaUpper, INFINITY);
    range = diskRange(cap.relative, alphaUpper);
    return range.valid != 0u && range.allCells == 0u;
}

static inline __device__ bool buildQueryMask(
    const View index,
    const gofish_certified_remainder::ExactConeState& state,
    uint32_t lane, uint32_t* sharedWords) {
    if (index.bits == nullptr || index.words == 0u
        || index.words > kMaxWords) return false;
    BinRange range{};
    if (!sourceRange(state.flatCap, range)) return false;
    for (uint32_t word = lane; word < index.words; word += 32u)
        sharedWords[word] = 0u;
    __syncwarp();
    for (int theta = range.theta0; theta <= range.theta1; ++theta) {
        for (int phi = range.phi0a; phi <= range.phi1a; ++phi) {
            const uint32_t cell = uint32_t(theta) * kPhiBins + uint32_t(phi);
            for (uint32_t word = lane; word < index.words; word += 32u)
                sharedWords[word] |= index.bits[uint64_t(cell)
                    * index.words + word];
        }
        if (range.phiRanges == 2u) {
            for (int phi = range.phi0b; phi <= range.phi1b; ++phi) {
                const uint32_t cell = uint32_t(theta) * kPhiBins
                    + uint32_t(phi);
                for (uint32_t word = lane; word < index.words; word += 32u)
                    sharedWords[word] |= index.bits[uint64_t(cell)
                        * index.words + word];
            }
        }
    }
    __syncwarp();
    return true;
}

static inline __device__ bool contains(const uint32_t* words, uint32_t tile) {
    return ((words[tile >> 5u] >> (tile & 31u)) & 1u) != 0u;
}

} // namespace gofish_angular_range_once

