// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

// Direct complete support enumeration for a monotone OpenCV-fisheye physical
// domain, computed from analytic calibration bounds rather than per-pixel
// cones or angular padding.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define GOFISH_SUPPORT_HD __host__ __device__ __forceinline__
#else
#define GOFISH_SUPPORT_HD inline
#endif

namespace gofish_continuous_support {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Vec2 {
    double x, y;
};

struct Vec3 {
    double x, y, z;
};

// Rows of the world-to-sensor rotation.
struct Mat3 {
    Vec3 row[3];
};

struct TileGrid {
    uint32_t x, y;
};

struct TileBox {
    int minX, minY, maxX, maxY;  // half-open

    GOFISH_SUPPORT_HD bool empty() const {
        return minX >= maxX || minY >= maxY;
    }
};

// The two chart-scale bounds are calibration-derived once on the host.
// For every physically visible theta,
//   radialScaleLower <= r(theta)/sin(theta) <= radialScaleUpper.
struct OpenCvFisheyePhysicalDomain {
    double fx, fy, cx, cy;
    double radial[4];
    double modelMaxAngle;
    double physicalThetaUpper;
    double radialScaleLower;
    double radialScaleUpper;
    uint32_t width, height;
    uint32_t tileWidth, tileHeight;
};


enum CentralProjectionKind : uint32_t {
    kOpenCvFisheyeProjection = 0u,
    kUndistortedPinholeProjection = 1u,
};

struct UndistortedPinholePhysicalDomain {
    double fx, fy, cx, cy;
    uint32_t width, height;
    uint32_t tileWidth, tileHeight;
};

struct CentralPhysicalDomain {
    uint32_t kind;
    OpenCvFisheyePhysicalDomain fisheye;
    UndistortedPinholePhysicalDomain pinhole;
};

enum EnumerationMode : uint32_t {
    kCountOnly = 0u,
    kEmit = 1u,
};

enum EnumerationFailure : uint32_t {
    kEnumerationOk = 0u,
    kInvalidDomain = 1u,
    kInvalidSupport = 2u,
    kInvalidEmitRange = 3u,
    kNativeCapacity = 4u,
    kNativeCountMismatch = 5u,
    kSupportCapacity = 6u,
    kTotalCountMismatch = 7u,
};

struct EnumerationResult {
    uint64_t totalCount;
    uint32_t nativeCount;
    uint32_t failure;
};

struct SupportBoxResult {
    TileBox box;
    uint32_t failure;
};

GOFISH_SUPPORT_HD bool tileBoxInsideGrid(const TileBox& box, TileGrid grid) {
    return box.minX >= 0 && box.minY >= 0 &&
        box.minX <= box.maxX && box.minY <= box.maxY &&
        static_cast<uint32_t>(box.maxX) <= grid.x &&
        static_cast<uint32_t>(box.maxY) <= grid.y;
}

GOFISH_SUPPORT_HD bool finiteValue(double value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

GOFISH_SUPPORT_HD double clampValue(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

GOFISH_SUPPORT_HD double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

GOFISH_SUPPORT_HD Vec3 normalize(const Vec3& value) {
    const double n2 = dot(value, value);
    if (!(n2 > 0.0) || !finiteValue(n2)) return Vec3{0.0, 0.0, 0.0};
    const double inverse = 1.0 / ::sqrt(n2);
    return Vec3{value.x * inverse, value.y * inverse, value.z * inverse};
}

GOFISH_SUPPORT_HD Vec3 transform(const Mat3& matrix, const Vec3& value) {
    return Vec3{
        dot(matrix.row[0], value),
        dot(matrix.row[1], value),
        dot(matrix.row[2], value),
    };
}

GOFISH_SUPPORT_HD void productInterval(
    double a0, double a1, double b0, double b1,
    double& outputLo, double& outputHi) {
    const double values[4] = {a0*b0, a0*b1, a1*b0, a1*b1};
    outputLo = values[0];
    outputHi = values[0];
    for (int index = 1; index < 4; ++index) {
        outputLo = values[index] < outputLo ? values[index] : outputLo;
        outputHi = values[index] > outputHi ? values[index] : outputHi;
    }
}

GOFISH_SUPPORT_HD double chooseSmall(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    double value = 1.0;
    for (int index = 1; index <= k; ++index) {
        value *= static_cast<double>(n - index + 1) / static_cast<double>(index);
    }
    return value;
}

GOFISH_SUPPORT_HD double powerSmall(double base, int exponent) {
    double value = 1.0;
    for (int index = 0; index < exponent; ++index) value *= base;
    return value;
}

// Bound P(y)=1+k1*y+...+k4*y^4 on an arbitrary interval by converting
// P(y0+(y1-y0)t) to degree-four Bernstein form.  This is a constant-size
// device calculation, not a sampled approximation.
GOFISH_SUPPORT_HD bool localRadialPolynomialBounds(
    const OpenCvFisheyePhysicalDomain& domain,
    double y0,
    double y1,
    double& lower,
    double& upper) {
    if (!(y0 >= 0.0) || !(y1 >= y0)) return false;
    const double source[5] = {
        1.0, domain.radial[0], domain.radial[1],
        domain.radial[2], domain.radial[3]};
    const double width = y1 - y0;
    double shiftedPower[5]{};
    for (int degree = 0; degree <= 4; ++degree) {
        for (int power = 0; power <= degree; ++power) {
            shiftedPower[power] += source[degree] * chooseSmall(degree, power) *
                powerSmall(y0, degree - power) * powerSmall(width, power);
        }
    }
    double bernstein[5]{};
    for (int index = 0; index <= 4; ++index) {
        for (int power = 0; power <= index; ++power) {
            bernstein[index] += shiftedPower[power] *
                chooseSmall(index, power) / chooseSmall(4, power);
        }
    }
    lower = bernstein[0];
    upper = bernstein[0];
    for (int index = 1; index <= 4; ++index) {
        lower = bernstein[index] < lower ? bernstein[index] : lower;
        upper = bernstein[index] > upper ? bernstein[index] : upper;
    }
    lower = ::nextafter(lower, -std::numeric_limits<double>::infinity());
    upper = ::nextafter(upper,  std::numeric_limits<double>::infinity());
    return lower > 0.0 && upper >= lower && finiteValue(upper);
}

GOFISH_SUPPORT_HD double sineLowerZeroToPi(double angle) {
    if (!(angle >= 0.0) || !(angle < kPi)) return 0.0;
    const double value = angle <= 0.5 * kPi ? angle : kPi - angle;
    if (value == 0.0) return 0.0;
    const double square = value * value;
    const double result = value * (
        1.0 - square / 6.0
        + square * square / 120.0
        - square * square * square / 5040.0);
    return ::nextafter(result, -std::numeric_limits<double>::infinity());
}

GOFISH_SUPPORT_HD bool localRadialScaleBounds(
    const OpenCvFisheyePhysicalDomain& domain,
    double thetaLo,
    double thetaHi,
    double& lower,
    double& upper) {
    if (!(thetaLo >= 0.0) || !(thetaHi >= thetaLo) ||
        !(thetaHi <= domain.physicalThetaUpper) || !(thetaHi < kPi)) {
        return false;
    }
    double polynomialLo = 0.0;
    double polynomialHi = 0.0;
    if (!localRadialPolynomialBounds(
            domain, thetaLo*thetaLo, thetaHi*thetaHi,
            polynomialLo, polynomialHi)) {
        return false;
    }
    lower = polynomialLo;  // theta/sin(theta) >= 1.
    double angularScale = 1.0;
    if (thetaHi > 0.0) {
        const double sineLower = sineLowerZeroToPi(thetaHi);
        if (!(sineLower > 0.0)) return false;
        angularScale = ::nextafter(
            thetaHi / sineLower, std::numeric_limits<double>::infinity());
    }
    upper = ::nextafter(
        polynomialHi * angularScale,
        std::numeric_limits<double>::infinity());
    return lower > 0.0 && upper >= lower && finiteValue(upper);
}

// A lower sine bound on [0, pi/2].  The alternating Taylor series has
// decreasing terms on this interval and the final retained term is negative.
inline long double sinLowerFirstQuadrant(long double value) {
    const long double square = value * value;
    const long double result = value * (
        1.0L - square / 6.0L
        + square * square / 120.0L
        - square * square * square / 5040.0L);
    return std::nextafter(result, -std::numeric_limits<long double>::infinity());
}

inline long double radialPolynomial(
    const OpenCvFisheyePhysicalDomain& domain, long double theta) {
    const long double y = theta * theta;
    return 1.0L + y * (
        static_cast<long double>(domain.radial[0]) + y * (
        static_cast<long double>(domain.radial[1]) + y * (
        static_cast<long double>(domain.radial[2]) + y *
        static_cast<long double>(domain.radial[3]))));
}

inline long double radialDistance(
    const OpenCvFisheyePhysicalDomain& domain, long double theta) {
    return theta * radialPolynomial(domain, theta);
}

// Degree-four power-to-Bernstein conversion.  A polynomial over [0,Y]
// lies inside the convex hull of these five coefficients, so this gives
// direct global bounds without roots, samples, or per-camera tuning.
inline void radialPolynomialBernsteinBounds(
    const OpenCvFisheyePhysicalDomain& domain,
    long double thetaUpper,
    long double& lower,
    long double& upper) {
    const long double y = thetaUpper * thetaUpper;
    const long double power[5] = {
        1.0L,
        static_cast<long double>(domain.radial[0]) * y,
        static_cast<long double>(domain.radial[1]) * y * y,
        static_cast<long double>(domain.radial[2]) * y * y * y,
        static_cast<long double>(domain.radial[3]) * y * y * y * y,
    };
    static constexpr long double choose4[5] = {1.0L, 4.0L, 6.0L, 4.0L, 1.0L};
    long double bernstein[5]{};
    for (int i = 0; i <= 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            // C(i,j) for i<=4.
            long double choose = 1.0L;
            for (int value = 1; value <= j; ++value) {
                choose *= static_cast<long double>(i - value + 1) /
                          static_cast<long double>(value);
            }
            bernstein[i] += power[j] * choose / choose4[j];
        }
    }
    lower = bernstein[0];
    upper = bernstein[0];
    for (int index = 1; index < 5; ++index) {
        lower = std::min(lower, bernstein[index]);
        upper = std::max(upper, bernstein[index]);
    }
    lower = std::nextafter(lower, -std::numeric_limits<long double>::infinity());
    upper = std::nextafter(upper, std::numeric_limits<long double>::infinity());
}

// Host-side calibration reduction.  The raster is a conservative superset of
// the exact physical mask.  Bisection only intersects the monotone camera
// branch with that raster; it is not a support approximation.
inline bool completePhysicalDomain(OpenCvFisheyePhysicalDomain& domain) {
    if (!(domain.fx > 0.0) || !(domain.fy > 0.0) ||
        !(domain.modelMaxAngle > 0.0) ||
        domain.width == 0 || domain.height == 0 ||
        domain.tileWidth == 0 || domain.tileHeight == 0) {
        return false;
    }
    long double rasterRadius = 0.0L;
    for (double u : {0.0, static_cast<double>(domain.width)}) {
        for (double v : {0.0, static_cast<double>(domain.height)}) {
            const long double x = (static_cast<long double>(u) - domain.cx) / domain.fx;
            const long double y = (static_cast<long double>(v) - domain.cy) / domain.fy;
            rasterRadius = std::max(rasterRadius, std::sqrt(x*x + y*y));
        }
    }
    long double lo = 0.0L;
    long double hi = std::min(
        static_cast<long double>(domain.modelMaxAngle),
        static_cast<long double>(kPi));
    if (radialDistance(domain, hi) > rasterRadius) {
        for (int iteration = 0; iteration < 96; ++iteration) {
            const long double middle = (lo + hi) * 0.5L;
            if (radialDistance(domain, middle) < rasterRadius) lo = middle;
            else hi = middle;
        }
    }
    const long double thetaUpper = std::nextafter(
        hi, std::numeric_limits<long double>::infinity());
    // A physical raster that actually reaches the antipode has a singular
    // finite-image chart and is outside this direct finite-box method.
    if (!(thetaUpper > 0.0L) || !(thetaUpper < static_cast<long double>(kPi))) {
        return false;
    }

    long double polynomialLower = 0.0L;
    long double polynomialUpper = 0.0L;
    radialPolynomialBernsteinBounds(
        domain, thetaUpper, polynomialLower, polynomialUpper);
    if (!(polynomialLower > 0.0L) || !(polynomialUpper >= polynomialLower)) {
        return false;
    }

    // theta/sin(theta) is at least one.  On [0,pi/2] it is at most pi/2
    // from sin(theta)>=2theta/pi.  Above pi/2, the raster radial bound and
    // sin(theta)>=sin(thetaUpper) give the second direct upper bound.
    long double scaleUpper = (static_cast<long double>(kPi) * 0.5L) * polynomialUpper;
    if (thetaUpper > static_cast<long double>(kPi) * 0.5L) {
        const long double reflected = static_cast<long double>(kPi) - thetaUpper;
        const long double sineLower = sinLowerFirstQuadrant(reflected);
        if (!(sineLower > 0.0L)) return false;
        scaleUpper = std::max(scaleUpper, rasterRadius / sineLower);
    }
    domain.physicalThetaUpper = std::nextafter(
        static_cast<double>(thetaUpper), std::numeric_limits<double>::infinity());
    domain.radialScaleLower = std::nextafter(
        static_cast<double>(polynomialLower), -std::numeric_limits<double>::infinity());
    domain.radialScaleUpper = std::nextafter(
        static_cast<double>(scaleUpper), std::numeric_limits<double>::infinity());
    return domain.radialScaleLower > 0.0 &&
           domain.radialScaleUpper >= domain.radialScaleLower &&
           std::isfinite(domain.radialScaleUpper);
}

GOFISH_SUPPORT_HD SupportBoxResult continuousSupportTileBox(
    const OpenCvFisheyePhysicalDomain& domain,
    TileGrid grid,
    const Vec3& supportCenterWorld,
    double supportHalfAngle,
    const Mat3& worldToSensor) {
    const TileBox empty{0, 0, 0, 0};
    if (!(domain.fx > 0.0) || !(domain.fy > 0.0) ||
        !(domain.physicalThetaUpper > 0.0) ||
        !(domain.physicalThetaUpper < kPi) || grid.x == 0 || grid.y == 0) {
        return SupportBoxResult{empty, kInvalidDomain};
    }
    if (grid.x > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        grid.y > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return SupportBoxResult{empty, kInvalidDomain};
    }
    if (!(supportHalfAngle >= 0.0) || !finiteValue(supportHalfAngle)) {
        return SupportBoxResult{empty, kInvalidSupport};
    }
    if (supportHalfAngle >= kPi) {
        return SupportBoxResult{
            TileBox{0, 0, static_cast<int>(grid.x), static_cast<int>(grid.y)},
            kEnumerationOk};
    }
    const Vec3 center = normalize(transform(worldToSensor, supportCenterWorld));
    if (!(dot(center, center) > 0.0)) {
        return SupportBoxResult{empty, kInvalidSupport};
    }

    const double centerTheta = ::acos(clampValue(center.z, -1.0, 1.0));
    const double thetaLo = centerTheta > supportHalfAngle
        ? centerTheta - supportHalfAngle : 0.0;
    if (thetaLo > domain.physicalThetaUpper) {
        return SupportBoxResult{empty, kEnumerationOk};
    }
    const double thetaHi = centerTheta + supportHalfAngle < domain.physicalThetaUpper
        ? centerTheta + supportHalfAngle : domain.physicalThetaUpper;
    double scaleLo = 0.0;
    double scaleHi = 0.0;
    if (!localRadialScaleBounds(domain, thetaLo, thetaHi, scaleLo, scaleHi)) {
        return SupportBoxResult{empty, kInvalidDomain};
    }

    // For angular distance delta<=alpha,
    // ||direction-center||=2 sin(delta/2)<=delta<=alpha.  Therefore every
    // component is inside center_j +/- alpha.  This is the sole cap-to-box
    // relaxation and follows directly from spherical chord length.
    const double radius = supportHalfAngle;
    const double componentLo[2] = {
        clampValue(center.x - radius, -1.0, 1.0),
        clampValue(center.y - radius, -1.0, 1.0),
    };
    const double componentHi[2] = {
        clampValue(center.x + radius, -1.0, 1.0),
        clampValue(center.y + radius, -1.0, 1.0),
    };
    double normalizedLo[2], normalizedHi[2];
    for (int axis = 0; axis < 2; ++axis) {
        productInterval(
            scaleLo, scaleHi,
            componentLo[axis], componentHi[axis],
            normalizedLo[axis], normalizedHi[axis]);
    }
    double u0 = domain.cx + domain.fx * normalizedLo[0];
    double u1 = domain.cx + domain.fx * normalizedHi[0];
    double v0 = domain.cy + domain.fy * normalizedLo[1];
    double v1 = domain.cy + domain.fy * normalizedHi[1];
    u0 = ::nextafter(u0, -std::numeric_limits<double>::infinity());
    v0 = ::nextafter(v0, -std::numeric_limits<double>::infinity());
    u1 = ::nextafter(u1,  std::numeric_limits<double>::infinity());
    v1 = ::nextafter(v1,  std::numeric_limits<double>::infinity());
    if (u1 < 0.0 || v1 < 0.0 ||
        u0 >= static_cast<double>(domain.width) ||
        v0 >= static_cast<double>(domain.height)) {
        return SupportBoxResult{empty, kEnumerationOk};
    }
    u0 = clampValue(u0, 0.0, static_cast<double>(domain.width));
    u1 = clampValue(u1, 0.0, static_cast<double>(domain.width));
    v0 = clampValue(v0, 0.0, static_cast<double>(domain.height));
    v1 = clampValue(v1, 0.0, static_cast<double>(domain.height));
    const int minX = static_cast<int>(::floor(u0 / domain.tileWidth));
    const int minY = static_cast<int>(::floor(v0 / domain.tileHeight));
    const int maxX = static_cast<int>(::floor(u1 / domain.tileWidth)) + 1;
    const int maxY = static_cast<int>(::floor(v1 / domain.tileHeight)) + 1;
    return SupportBoxResult{
        TileBox{
            minX < 0 ? 0 : minX,
            minY < 0 ? 0 : minY,
            maxX > static_cast<int>(grid.x) ? static_cast<int>(grid.x) : maxX,
            maxY > static_cast<int>(grid.y) ? static_cast<int>(grid.y) : maxY},
        kEnumerationOk};
}


GOFISH_SUPPORT_HD SupportBoxResult continuousSupportTileBox(
    const UndistortedPinholePhysicalDomain& domain,
    TileGrid grid,
    const Vec3& supportCenterWorld,
    double supportHalfAngle,
    const Mat3& worldToSensor) {
    const TileBox empty{0, 0, 0, 0};
    if (!(domain.fx > 0.0) || !(domain.fy > 0.0) ||
        domain.width == 0 || domain.height == 0 ||
        domain.tileWidth == 0 || domain.tileHeight == 0 ||
        grid.x == 0 || grid.y == 0 ||
        grid.x != (domain.width + domain.tileWidth - 1u) / domain.tileWidth ||
        grid.y != (domain.height + domain.tileHeight - 1u) / domain.tileHeight ||
        grid.x > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        grid.y > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return SupportBoxResult{empty, kInvalidDomain};
    }
    if (!(supportHalfAngle >= 0.0) || !finiteValue(supportHalfAngle)) {
        return SupportBoxResult{empty, kInvalidSupport};
    }
    const TileBox full{0, 0, static_cast<int>(grid.x), static_cast<int>(grid.y)};
    if (supportHalfAngle >= kPi) {
        return SupportBoxResult{full, kEnumerationOk};
    }
    const Vec3 center = normalize(transform(worldToSensor, supportCenterWorld));
    if (!(dot(center, center) > 0.0)) {
        return SupportBoxResult{empty, kInvalidSupport};
    }

    // Every direction d in an angular cap of radius alpha obeys
    // |d_j-c_j| <= ||d-c|| = 2 sin(delta/2) <= delta <= alpha.
    // If that component bound reaches the focal plane, the finite raster is
    // the smallest coefficient-free complete image-space bound. Otherwise,
    // interval division by the strictly positive z lower bound gives a direct
    // conservative bound for the exact pinhole ratios x/z and y/z.
    const double radius = supportHalfAngle;
    const double zLo = center.z - radius;
    const double zHi = center.z + radius < 1.0 ? center.z + radius : 1.0;
    if (!(zHi > 0.0)) {
        return SupportBoxResult{empty, kEnumerationOk};
    }
    if (!(zLo > 0.0)) {
        return SupportBoxResult{full, kEnumerationOk};
    }
    const double reciprocalLo = 1.0 / zHi;
    const double reciprocalHi = 1.0 / zLo;
    double normalizedLo[2], normalizedHi[2];
    const double componentLo[2] = {
        clampValue(center.x - radius, -1.0, 1.0),
        clampValue(center.y - radius, -1.0, 1.0)};
    const double componentHi[2] = {
        clampValue(center.x + radius, -1.0, 1.0),
        clampValue(center.y + radius, -1.0, 1.0)};
    for (int axis = 0; axis < 2; ++axis) {
        productInterval(
            componentLo[axis], componentHi[axis], reciprocalLo, reciprocalHi,
            normalizedLo[axis], normalizedHi[axis]);
    }
    double u0 = ::nextafter(
        domain.cx + domain.fx * normalizedLo[0],
        -std::numeric_limits<double>::infinity());
    double u1 = ::nextafter(
        domain.cx + domain.fx * normalizedHi[0],
         std::numeric_limits<double>::infinity());
    double v0 = ::nextafter(
        domain.cy + domain.fy * normalizedLo[1],
        -std::numeric_limits<double>::infinity());
    double v1 = ::nextafter(
        domain.cy + domain.fy * normalizedHi[1],
         std::numeric_limits<double>::infinity());
    if (u1 < 0.0 || v1 < 0.0 ||
        u0 >= static_cast<double>(domain.width) ||
        v0 >= static_cast<double>(domain.height)) {
        return SupportBoxResult{empty, kEnumerationOk};
    }
    u0 = clampValue(u0, 0.0, static_cast<double>(domain.width));
    u1 = clampValue(u1, 0.0, static_cast<double>(domain.width));
    v0 = clampValue(v0, 0.0, static_cast<double>(domain.height));
    v1 = clampValue(v1, 0.0, static_cast<double>(domain.height));
    const int minX = static_cast<int>(::floor(u0 / domain.tileWidth));
    const int minY = static_cast<int>(::floor(v0 / domain.tileHeight));
    const int maxX = static_cast<int>(::floor(u1 / domain.tileWidth)) + 1;
    const int maxY = static_cast<int>(::floor(v1 / domain.tileHeight)) + 1;
    return SupportBoxResult{
        TileBox{
            minX < 0 ? 0 : minX,
            minY < 0 ? 0 : minY,
            maxX > static_cast<int>(grid.x) ? static_cast<int>(grid.x) : maxX,
            maxY > static_cast<int>(grid.y) ? static_cast<int>(grid.y) : maxY},
        kEnumerationOk};
}

GOFISH_SUPPORT_HD SupportBoxResult continuousSupportTileBox(
    const CentralPhysicalDomain& domain,
    TileGrid grid,
    const Vec3& supportCenterWorld,
    double supportHalfAngle,
    const Mat3& worldToSensor) {
    if (domain.kind == kOpenCvFisheyeProjection) {
        return continuousSupportTileBox(
            domain.fisheye, grid, supportCenterWorld, supportHalfAngle,
            worldToSensor);
    }
    if (domain.kind == kUndistortedPinholeProjection) {
        return continuousSupportTileBox(
            domain.pinhole, grid, supportCenterWorld, supportHalfAngle,
            worldToSensor);
    }
    return SupportBoxResult{TileBox{0, 0, 0, 0}, kInvalidDomain};
}

// NativeAdmission(x,y) is the strict native UT predicate.  Emit is
// called only in emit mode as emit(cursor,tileId,particleIdx).  The exact same
// loops own count and emit, so cardinality cannot drift.
template <class NativeAdmission, class Emit>
GOFISH_SUPPORT_HD EnumerationResult enumerateContinuousPhysicalSupport(
    TileGrid grid,
    uint32_t particleIdx,
    bool validNativeProjection,
    TileBox nativeBox,
    const SupportBoxResult& support,
    EnumerationMode mode,
    uint64_t begin,
    uint64_t end,
    uint32_t expectedNativeCount,
    NativeAdmission nativeAdmission,
    Emit emit) {
    EnumerationResult result{0u, 0u, kEnumerationOk};
    if (support.failure != kEnumerationOk) {
        result.failure = support.failure;
        return result;
    }
    if (!tileBoxInsideGrid(support.box, grid) ||
        (validNativeProjection && !tileBoxInsideGrid(nativeBox, grid))) {
        result.failure = kInvalidDomain;
        return result;
    }
    if (mode == kEmit && end < begin) {
        result.failure = kInvalidEmitRange;
        return result;
    }
    uint64_t cursor = begin;
    if (validNativeProjection) {
        for (int y = nativeBox.minY; y < nativeBox.maxY; ++y) {
            for (int x = nativeBox.minX; x < nativeBox.maxX; ++x) {
                if (!nativeAdmission(x, y)) continue;
                if (mode == kEmit) {
                    if (cursor >= end) {
                        result.failure = kNativeCapacity;
                        result.totalCount = cursor - begin;
                        return result;
                    }
                    emit(cursor, static_cast<uint32_t>(y) * grid.x +
                        static_cast<uint32_t>(x), particleIdx);
                }
                ++cursor;
                ++result.nativeCount;
            }
        }
    }
    if (mode == kEmit && result.nativeCount != expectedNativeCount) {
        result.failure = kNativeCountMismatch;
        result.totalCount = cursor - begin;
        return result;
    }
    for (int y = support.box.minY; y < support.box.maxY; ++y) {
        for (int x = support.box.minX; x < support.box.maxX; ++x) {
            if (validNativeProjection && nativeAdmission(x, y)) continue;
            if (mode == kEmit) {
                if (cursor >= end) {
                    result.failure = kSupportCapacity;
                    result.totalCount = cursor - begin;
                    return result;
                }
                emit(cursor, static_cast<uint32_t>(y) * grid.x +
                    static_cast<uint32_t>(x), particleIdx);
            }
            ++cursor;
        }
    }
    result.totalCount = cursor - begin;
    if (mode == kEmit && cursor != end) result.failure = kTotalCountMismatch;
    return result;
}

}  // namespace gofish_continuous_support

#undef GOFISH_SUPPORT_HD
