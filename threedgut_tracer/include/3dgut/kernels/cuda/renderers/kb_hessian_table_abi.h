// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define GOFISH_KB_HD __host__ __device__
#else
#define GOFISH_KB_HD
#endif

namespace gofish { namespace kb_hessian_abi {

static constexpr std::uint32_t kVersion = 2u;
static constexpr std::uint32_t kEndianTag = 0x01020304u;
static constexpr std::uint32_t kModelOpenCvFisheyeKb4 = 1u;
static constexpr std::uint32_t kModelIdealPinhole = 2u;
static constexpr std::uint32_t kRequiredFlags = 0x3fu;
static constexpr std::uint32_t kHeaderBytes = 160u;
static constexpr std::uint32_t kCameraBytes = 128u;
static constexpr std::uint32_t kBandBytes = 24u;

struct alignas(16) Header {
    std::uint8_t magic[8];
    std::uint32_t abi_version;
    std::uint32_t endian_tag;
    std::uint32_t header_bytes;
    std::uint32_t camera_stride;
    std::uint32_t band_stride;
    std::uint32_t camera_count;
    std::uint32_t band_count;
    std::uint32_t flags;
    std::uint64_t cameras_offset;
    std::uint64_t bands_offset;
    std::uint64_t payload_bytes;
    std::uint8_t table_json_sha256[32];
    std::uint8_t camera_domain_source_sha256[32];
    std::uint8_t derived_domains_sha256[32];
};

struct alignas(16) CameraDescriptor {
    std::uint8_t camera_identity_sha256[32];
    std::uint8_t active_canonical_domain_sha256[32];
    std::uint8_t upstream_domain_specification_sha256[32];
    std::uint32_t first_band;
    std::uint32_t band_count;
    std::uint32_t model_id;
    std::uint32_t native_width;
    std::uint32_t native_height;
    std::uint32_t reserved;
    double theta_max_radians;
};

struct alignas(8) Band {
    double theta_lo_radians;
    double theta_hi_radians;
    double hbar_native_pixels;
};

struct TableView {
    const Header* header;
    const CameraDescriptor* cameras;
    const Band* bands;
};

GOFISH_KB_HD inline TableView view(const void* aligned_base) {
    const auto* bytes = static_cast<const std::uint8_t*>(aligned_base);
    const auto* header = reinterpret_cast<const Header*>(bytes);
    return TableView{
        header,
        reinterpret_cast<const CameraDescriptor*>(bytes + header->cameras_offset),
        reinterpret_cast<const Band*>(bytes + header->bands_offset)
    };
}

GOFISH_KB_HD inline bool finiteDouble(double value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// The table is certified in native-calibration pixels. If the active chart is
// an affine full-raster resize, its output Hessian is scaled by
// diag(W_active/W_native, H_active/H_native); translation has zero Hessian.
// Each ratio and the final product are rounded upward.
GOFISH_KB_HD inline double hbarForRaster(
    double hbar_native_pixels,
    std::uint32_t native_width, std::uint32_t native_height,
    std::uint32_t active_width, std::uint32_t active_height) {
#if defined(__CUDA_ARCH__)
    const double x_scale_upper = __ddiv_ru(double(active_width), double(native_width));
    const double y_scale_upper = __ddiv_ru(double(active_height), double(native_height));
    return __dmul_ru(hbar_native_pixels,
                     x_scale_upper > y_scale_upper ? x_scale_upper : y_scale_upper);
#else
    const double x_scale_upper = std::nextafter(
        double(active_width) / double(native_width),
        std::numeric_limits<double>::infinity());
    const double y_scale_upper = std::nextafter(
        double(active_height) / double(native_height),
        std::numeric_limits<double>::infinity());
    return std::nextafter(
        hbar_native_pixels * (x_scale_upper > y_scale_upper
                                  ? x_scale_upper : y_scale_upper),
        std::numeric_limits<double>::infinity());
#endif
}

struct AggregatedBands {
    double hbar_native_pixels;
    std::uint32_t first_global_band;
    std::uint32_t last_global_band;
};

// Aggregate every closed band intersecting [theta_lower, theta_upper]. This
// deliberately includes both adjacent records at an exact shared endpoint.
// The maximum of certified upper bounds remains a certified upper bound, so a
// numerical table boundary never forces exact fallback.
GOFISH_KB_HD inline bool aggregateIntersectedBands(
    const Band* all_bands, const CameraDescriptor& camera,
    double theta_lower, double theta_upper, AggregatedBands* result) {
    if (all_bands == nullptr || result == nullptr || camera.band_count == 0u
        || !finiteDouble(theta_lower) || !finiteDouble(theta_upper)
        || theta_lower < 0.0 || theta_lower > theta_upper) return false;
    bool selected = false;
    double hbar = 0.0;
    double prior_hi = 0.0;
    std::uint32_t first = 0u;
    std::uint32_t last = 0u;
    for (std::uint32_t local = 0; local < camera.band_count; ++local) {
        const std::uint32_t global = camera.first_band + local;
        const Band& band = all_bands[global];
        if (!finiteDouble(band.theta_lo_radians)
            || !finiteDouble(band.theta_hi_radians)
            || !finiteDouble(band.hbar_native_pixels)
            || !(band.theta_lo_radians < band.theta_hi_radians)
            || !(band.hbar_native_pixels > 0.0)
            || (local > 0u && band.theta_lo_radians != prior_hi)) return false;
        prior_hi = band.theta_hi_radians;
        if (band.theta_hi_radians < theta_lower) continue;
        if (band.theta_lo_radians > theta_upper) break;
        if (!selected) {
            if (band.theta_lo_radians > theta_lower) return false;
            first = global;
            hbar = band.hbar_native_pixels;
            selected = true;
        } else if (band.hbar_native_pixels > hbar) {
            hbar = band.hbar_native_pixels;
        }
        last = global;
    }
    if (!selected || all_bands[last].theta_hi_radians < theta_upper) return false;
    result->hbar_native_pixels = hbar;
    result->first_global_band = first;
    result->last_global_band = last;
    return true;
}

// Preconditions: range > 0 and its binary64 square is finite and nonzero.
// Hbar is already an outward pixel-space upper bound. Round r^2 downward and
// the quotient upward so the returned pixel/range^2 value remains an upper.
GOFISH_KB_HD inline double upperFromRange(double hbar_pixels, double range) {
#if defined(__CUDA_ARCH__)
    const double range2_lower = __dmul_rd(range, range);
    return __ddiv_ru(hbar_pixels, range2_lower);
#else
    const double range2_lower = std::nextafter(
        range * range, -std::numeric_limits<double>::infinity());
    return std::nextafter(
        hbar_pixels / range2_lower,
        std::numeric_limits<double>::infinity());
#endif
}

static_assert(sizeof(Header) == kHeaderBytes, "KB ABI header layout changed");
static_assert(sizeof(CameraDescriptor) == kCameraBytes, "KB ABI camera layout changed");
static_assert(sizeof(Band) == kBandBytes, "KB ABI band layout changed");
static_assert(offsetof(Header, cameras_offset) == 40, "KB ABI header offset changed");
static_assert(offsetof(Header, table_json_sha256) == 64, "KB ABI hash offset changed");
static_assert(offsetof(CameraDescriptor, first_band) == 96, "KB ABI camera offset changed");
static_assert(offsetof(CameraDescriptor, theta_max_radians) == 120,
              "KB ABI theta offset changed");
static_assert(offsetof(Band, hbar_native_pixels) == 16, "KB ABI Hbar offset changed");
static_assert(std::is_standard_layout<Header>::value, "KB ABI header must be standard layout");
static_assert(std::is_standard_layout<CameraDescriptor>::value,
              "KB ABI camera must be standard layout");
static_assert(std::is_standard_layout<Band>::value, "KB ABI band must be standard layout");

}}  // namespace gofish::kb_hessian_abi

#undef GOFISH_KB_HD
