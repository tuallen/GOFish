// Portions derived from SOF (https://github.com/r4dl/SOF),
// distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
// research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md
// Exact SOF coarse tile key: positive-ray maximum-response depth.
#pragma once

#include <3dgut/kernels/cuda/renderers/sofDepthFallback.cuh>

namespace gofish_sof_sort {

// Return the positive-ray maximum-response parameter for
//   exp(-0.5 ||B^{-1}(o + t d - mu)||^2).
//
// B^{-1} is stored row-major in float33. In this CUDA codebase,
// `vector * matrix` evaluates the mathematical matrix-vector product, so the
// two products below are exactly B^{-1}(o-mu) and B^{-1}d.  The operation is
// key-only and has no alpha/support admission test.
template <typename RawParameters>
static inline __device__ float maximumResponseDepthBInverse(
    const RawParameters& raw,
    const tcnn::vec3& rayOrigin,
    const tcnn::vec3& rayDirection,
    float fallback) {
    const float3 localOrigin = make_float3(
        rayOrigin.x - raw.position.x,
        rayOrigin.y - raw.position.y,
        rayOrigin.z - raw.position.z);
    const float3 direction = make_float3(
        rayDirection.x, rayDirection.y, rayDirection.z);
    const float3 canonicalOrigin = localOrigin * raw.inverseFactor;
    const float3 canonicalDirection = direction * raw.inverseFactor;
    const float denominator = dot(canonicalDirection, canonicalDirection);
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        return takeDepthFallback(fallback, 0.0f);
    }
    const float unconstrained =
        -dot(canonicalOrigin, canonicalDirection) / denominator;
    if (!isfinite(unconstrained)) {
        return takeDepthFallback(fallback, 0.0f);
    }
    return fmaxf(0.0f, unconstrained);
}

} // namespace gofish_sof_sort
