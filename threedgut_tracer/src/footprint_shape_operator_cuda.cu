// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <memory>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Projected pixel-footprint variance gamma (px^2); set at JIT build time from
// render.gofish.pixel_footprint_variance (0 when the factorization is disabled).
#ifndef GOFISH_PIXEL_FOOTPRINT_VARIANCE
#error "GOFISH_PIXEL_FOOTPRINT_VARIANCE must be defined"
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kMatrixOrder = 3;
constexpr int kMaxSolverBatch = 16384;
constexpr float kResolutionUlps = 16.0f;

enum StatusBits : int {
    kInvalidScale = 1 << 0,
    kInvalidQuaternion = 1 << 1,
    kInvalidDensity = 1 << 2,
    kInvalidQ = 1 << 3,
    kSolverFailure = 1 << 4,
    kNotStrictSpd = 1 << 5,
    kUnresolvedFloat32 = 1 << 6,
    kInvalidOutput = 1 << 7,
    kInvalidGradient = 1 << 8,
    kInvalidSylvesterDenominator = 1 << 9,
};

// Stable identifiers for every kInvalidOutput site, reported in error messages; they are not part of
// the aggregate status ABI.
enum InvalidOutputSubtype : int {
    kNoInvalidOutput = 0,
    kFloat32FactorInverse = 1,
    kFloat32ProjectedM2 = 2,
    kFloat32FinalOutput = 3,
    kEscalatedSpectralFactorInverse = 4,
    kEscalatedProjectedM2 = 5,
    kProjectedOnlyProjectedM2 = 6,
    kEscalatedNarrowedOutput = 7,
};

__device__ __forceinline__ void record_invalid_output(
    int64_t row,
    InvalidOutputSubtype subtype,
    int* failure_subtype,
    unsigned long long* first_failure_row,
    int* status) {
    failure_subtype[row] = static_cast<int>(subtype);
    atomicMin(first_failure_row, static_cast<unsigned long long>(row));
    atomicOr(status, kInvalidOutput);
}

void check_cusolver(cusolverStatus_t status, const char* expression) {
    TORCH_CHECK(
        status == CUSOLVER_STATUS_SUCCESS,
        expression,
        " failed with cuSOLVER status ",
        static_cast<int>(status));
}

#define GOFISH_CUSOLVER_CHECK(expression) \
    check_cusolver((expression), #expression)

class SyevjParameters final {
public:
    SyevjParameters() {
        GOFISH_CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&parameters_));
        GOFISH_CUSOLVER_CHECK(
            cusolverDnXsyevjSetMaxSweeps(parameters_, 100));
        GOFISH_CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(parameters_, 1));
    }

    SyevjParameters(const SyevjParameters&) = delete;
    SyevjParameters& operator=(const SyevjParameters&) = delete;

    ~SyevjParameters() {
        if (parameters_ != nullptr) {
            cusolverDnDestroySyevjInfo(parameters_);
        }
    }

    syevjInfo_t get() const { return parameters_; }

private:
    syevjInfo_t parameters_ = nullptr;
};

class SolverHandle final {
public:
    SolverHandle() {
        GOFISH_CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    }

    SolverHandle(const SolverHandle&) = delete;
    SolverHandle& operator=(const SolverHandle&) = delete;

    ~SolverHandle() {
        if (handle_ != nullptr) {
            cusolverDnDestroy(handle_);
        }
    }

    cusolverDnHandle_t get() const { return handle_; }

private:
    cusolverDnHandle_t handle_ = nullptr;
};

// Inference renderers keep one cuSOLVER handle and syevj descriptor alive
// across frames instead of creating them per call.  The handle carries no
// numerical state, so factors are bit-identical; the default (per-call
// handles) is the training behaviour.
bool g_persistent_solver = false;
std::unique_ptr<SolverHandle> g_solver;
std::unique_ptr<SyevjParameters> g_syevj;
int g_solver_device = -1;

__device__ __forceinline__ int offset(int row, int column) {
    return row * 3 + column;
}

__device__ __forceinline__ bool finite9(const float* value) {
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        if (!isfinite(value[index])) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ double q_value(
    const double* q6, int row, int column) {
    if (row > column) {
        const int temporary = row;
        row = column;
        column = temporary;
    }
    if (row == 0) {
        return q6[column];
    }
    if (row == 1) {
        return q6[column + 2];
    }
    return q6[5];
}

__device__ __forceinline__ void quaternion_rotation(
    const float* quaternion, float* rotation) {
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    rotation[offset(0, 0)] = 1.0f - 2.0f * (y * y + z * z);
    rotation[offset(0, 1)] = 2.0f * (x * y - z * w);
    rotation[offset(0, 2)] = 2.0f * (x * z + y * w);
    rotation[offset(1, 0)] = 2.0f * (x * y + z * w);
    rotation[offset(1, 1)] = 1.0f - 2.0f * (x * x + z * z);
    rotation[offset(1, 2)] = 2.0f * (y * z - x * w);
    rotation[offset(2, 0)] = 2.0f * (x * z - y * w);
    rotation[offset(2, 1)] = 2.0f * (y * z + x * w);
    rotation[offset(2, 2)] = 1.0f - 2.0f * (x * x + y * y);
}

__global__ void assemble_covariance_kernel(
    const float* scales,
    const float* quaternion,
    const float* density,
    const double* q6,
    const bool* observed,
    int64_t count,
    float* covariance,
    float* eigensystem,
    float* rotation,
    int* status) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    const float* scales_row = scales + row * 3;
    const float* quaternion_row = quaternion + row * 4;
    const double* q_row = q6 + row * 6;
    float* rotation_row = rotation + row * 9;
    const bool row_observed = observed[row];

    if (!(isfinite(scales_row[0]) && scales_row[0] > 0.0f &&
          isfinite(scales_row[1]) && scales_row[1] > 0.0f &&
          isfinite(scales_row[2]) && scales_row[2] > 0.0f)) {
        atomicOr(status, kInvalidScale);
    }
    const float norm = sqrtf(
        quaternion_row[0] * quaternion_row[0] +
        quaternion_row[1] * quaternion_row[1] +
        quaternion_row[2] * quaternion_row[2] +
        quaternion_row[3] * quaternion_row[3]);
    if (!(isfinite(norm) && norm > 0.0f && fabsf(norm - 1.0f) < 1.0e-3f)) {
        atomicOr(status, kInvalidQuaternion);
    }
    if (!isfinite(density[row])) {
        atomicOr(status, kInvalidDensity);
    }
    if (row_observed) {
#pragma unroll
        for (int index = 0; index < 6; ++index) {
            if (!isfinite(q_row[index])) {
                atomicOr(status, kInvalidQ);
            }
        }
    }

    quaternion_rotation(quaternion_row, rotation_row);
    float* covariance_row = covariance + row * 9;
    float* eigensystem_row = eigensystem + row * 9;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = i; j < 3; ++j) {
            float narrowed;
            if (row_observed) {
                double value = q_value(q_row, i, j);
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const double scale = static_cast<double>(scales_row[k]);
                    value += static_cast<double>(rotation_row[offset(i, k)]) *
                        static_cast<double>(rotation_row[offset(j, k)]) *
                        scale * scale;
                }
                narrowed = static_cast<float>(value);
            } else {
                // No-sensor path in float32.  The double path exists only
                // to carry an observed float64 Q.
                float value = 0.0f;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const float scale = scales_row[k];
                    value += rotation_row[offset(i, k)] *
                        rotation_row[offset(j, k)] * scale * scale;
                }
                narrowed = value;
            }
            covariance_row[offset(i, j)] = narrowed;
            covariance_row[offset(j, i)] = narrowed;
            // Symmetry makes the row-major bytes a valid column-major input.
            eigensystem_row[offset(i, j)] = narrowed;
            eigensystem_row[offset(j, i)] = narrowed;
        }
    }
}

__device__ __forceinline__ float eigenvector_value(
    const float* column_major_vectors, int row, int column) {
    return column_major_vectors[row + column * 3];
}

__device__ __forceinline__ float determinant3(const float* matrix) {
    return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

__device__ __forceinline__ bool inverse3(
    const float* matrix, float* inverse) {
    const float determinant = determinant3(matrix);
    if (!isfinite(determinant) || determinant == 0.0f) {
        return false;
    }
    const float reciprocal = 1.0f / determinant;
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * reciprocal;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * reciprocal;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * reciprocal;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * reciprocal;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * reciprocal;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * reciprocal;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * reciprocal;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * reciprocal;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * reciprocal;
    return finite9(inverse);
}

__device__ __forceinline__ bool cholesky_logdet3(
    const float* covariance, float* logdet) {
    const float l00 = sqrtf(covariance[offset(0, 0)]);
    if (!(isfinite(l00) && l00 > 0.0f)) {
        return false;
    }
    const float l10 = covariance[offset(1, 0)] / l00;
    const float l20 = covariance[offset(2, 0)] / l00;
    const float pivot11 = covariance[offset(1, 1)] - l10 * l10;
    if (!(isfinite(pivot11) && pivot11 > 0.0f)) {
        return false;
    }
    const float l11 = sqrtf(pivot11);
    const float l21 =
        (covariance[offset(2, 1)] - l20 * l10) / l11;
    const float pivot22 =
        covariance[offset(2, 2)] - l20 * l20 - l21 * l21;
    if (!(isfinite(pivot22) && pivot22 > 0.0f)) {
        return false;
    }
    const float l22 = sqrtf(pivot22);
    *logdet = 2.0f * (logf(l00) + logf(l11) + logf(l22));
    return isfinite(*logdet);
}

template <typename T>
struct Projected2DCompensationTerms {
    T inv0_xx, inv0_xy, inv0_yy;
    T inv1_xx, inv1_xy, inv1_yy;
    T logdet0, logdet1;
    T coefficient;
};

template <typename T>
__device__ __forceinline__ T projected_abs(T value) {
    return value < T(0) ? -value : value;
}

template <typename T>
__device__ __forceinline__ T projected_max3(T a, T b, T c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

// Scaled two-by-two Cholesky.  It returns both logdet and inverse without ever
// forming a determinant, ratio, floor, clamp, or epsilon-adjusted matrix.
template <typename T>
__device__ __forceinline__ bool projected_spd2_logdet_inverse(
    T xx, T xy, T yy, T resolution_epsilon,
    T* logdet, T* inv_xx, T* inv_xy, T* inv_yy) {
    const T scale = projected_max3(
        projected_abs(xx), projected_abs(xy), projected_abs(yy));
    if (!(isfinite(scale) && scale > T(0))) return false;
    const T a = xx / scale;
    const T b = xy / scale;
    const T c = yy / scale;
    if (!(isfinite(a) && isfinite(b) && isfinite(c) && a > T(0))) return false;
    const T l00 = sqrt(a);
    const T l10 = b / l00;
    const T pivot = c - l10 * l10;
    if (!(isfinite(l00) && isfinite(l10) && isfinite(pivot) && pivot > T(0))) {
        return false;
    }
    if (resolution_epsilon > T(0)) {
        const T threshold = T(kResolutionUlps) * resolution_epsilon;
        if (!(a > threshold && pivot > threshold)) return false;
    }
    *logdet = T(2) * log(scale) + log(a) + log(pivot);
    const T inverse_pivot = T(1) / pivot;
    const T inverse_a = T(1) / a;
    *inv_yy = inverse_pivot / scale;
    *inv_xy = (-b * inverse_a * inverse_pivot) / scale;
    *inv_xx = (inverse_a + b * b * inverse_a * inverse_a * inverse_pivot) / scale;
    return isfinite(*logdet) && isfinite(*inv_xx) && isfinite(*inv_xy) &&
        isfinite(*inv_yy);
}

template <typename T>
__device__ __forceinline__ bool projected_2d_terms(
    const T* scales,
    const T* rotation,
    const T* covariance,
    const T* jacobian,
    T resolution_epsilon,
    Projected2DCompensationTerms<T>* output) {
    T projected_factor[6];
#pragma unroll
    for (int pixel = 0; pixel < 2; ++pixel) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            T value = T(0);
#pragma unroll
            for (int world = 0; world < 3; ++world) {
                value += jacobian[pixel * 3 + world] *
                    rotation[offset(world, axis)] * scales[axis];
            }
            projected_factor[pixel * 3 + axis] = value;
        }
    }
    T s0_xx = T(0), s0_xy = T(0), s0_yy = T(0);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const T x = projected_factor[axis];
        const T y = projected_factor[3 + axis];
        s0_xx += x * x;
        s0_xy += x * y;
        s0_yy += y * y;
    }

    T cj[6];
#pragma unroll
    for (int pixel = 0; pixel < 2; ++pixel) {
#pragma unroll
        for (int world = 0; world < 3; ++world) {
            T value = T(0);
#pragma unroll
            for (int source = 0; source < 3; ++source) {
                value += jacobian[pixel * 3 + source] *
                    covariance[offset(source, world)];
            }
            cj[pixel * 3 + world] = value;
        }
    }
    T s1_xx = T(0), s1_xy = T(0), s1_yy = T(0);
#pragma unroll
    for (int world = 0; world < 3; ++world) {
        s1_xx += cj[world] * jacobian[world];
        s1_xy += cj[world] * jacobian[3 + world];
        s1_yy += cj[3 + world] * jacobian[3 + world];
    }
    if (!projected_spd2_logdet_inverse(
            s0_xx, s0_xy, s0_yy, resolution_epsilon,
            &output->logdet0, &output->inv0_xx, &output->inv0_xy,
            &output->inv0_yy) ||
        !projected_spd2_logdet_inverse(
            s1_xx, s1_xy, s1_yy, resolution_epsilon,
            &output->logdet1, &output->inv1_xx, &output->inv1_xy,
            &output->inv1_yy)) {
        return false;
    }
    if (!(output->logdet1 >= output->logdet0)) return false;
    output->coefficient = exp(T(0.5) * (output->logdet0 - output->logdet1));
    return isfinite(output->coefficient) && output->coefficient > T(0) &&
        output->coefficient <= T(1);
}

// Evaluate a projected 2x3 factor F without reconstructing and factorizing
// F F^T.  Cauchy--Binet gives det(F F^T) as a sum of squared 2x2 minors,
// which is positive by construction for a rank-two factor and avoids the
// cancellation in xx*yy-xy*xy and Cholesky pivot subtraction.
template <typename T>
__device__ __forceinline__ bool projected_factor_terms(
    const T* factor,
    T* xx,
    T* xy,
    T* yy,
    T* determinant,
    T* logdet,
    T* inv_xx,
    T* inv_xy,
    T* inv_yy) {
    *xx = T(0);
    *xy = T(0);
    *yy = T(0);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const T x = factor[axis];
        const T y = factor[3 + axis];
        if (!(isfinite(x) && isfinite(y))) return false;
        *xx += x * x;
        *xy += x * y;
        *yy += y * y;
    }
    T squared_minors = T(0);
#pragma unroll
    for (int first = 0; first < 3; ++first) {
#pragma unroll
        for (int second = first + 1; second < 3; ++second) {
            const T minor = factor[first] * factor[3 + second] -
                factor[second] * factor[3 + first];
            squared_minors += minor * minor;
        }
    }
    *determinant = squared_minors;
    if (!(isfinite(*xx) && isfinite(*xy) && isfinite(*yy) &&
          isfinite(*determinant) && *determinant > T(0))) {
        return false;
    }
    *logdet = log(*determinant);
    *inv_xx = *yy / *determinant;
    *inv_xy = -*xy / *determinant;
    *inv_yy = *xx / *determinant;
    return isfinite(*logdet) && isfinite(*inv_xx) && isfinite(*inv_xy) &&
        isfinite(*inv_yy);
}

template <typename T>
__device__ __forceinline__ bool
projected_2d_terms_from_intrinsic_factor_and_projected_q(
    const T* scales,
    const T* rotation,
    const T* projected_q,
    const T* jacobian,
    Projected2DCompensationTerms<T>* output) {
    T intrinsic_factor[6];
#pragma unroll
    for (int pixel = 0; pixel < 2; ++pixel) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            T intrinsic = T(0);
#pragma unroll
            for (int world = 0; world < 3; ++world) {
                intrinsic += jacobian[pixel * 3 + world] *
                    rotation[offset(world, axis)] * scales[axis];
            }
            intrinsic_factor[pixel * 3 + axis] = intrinsic;
        }
    }
    T xx0, xy0, yy0, det0;
    if (!projected_factor_terms(
            intrinsic_factor, &xx0, &xy0, &yy0, &det0,
            &output->logdet0, &output->inv0_xx, &output->inv0_xy,
            &output->inv0_yy)) {
        return false;
    }
    const T s1_xx = xx0 + projected_q[0];
    const T s1_xy = xy0 + projected_q[1];
    const T s1_yy = yy0 + projected_q[2];
    const T determinant_q = projected_q[0] * projected_q[2] -
        projected_q[1] * projected_q[1];
    const T mixed_determinant = xx0 * projected_q[2] +
        yy0 * projected_q[0] - T(2) * xy0 * projected_q[1];
    const T determinant1 = det0 + mixed_determinant + determinant_q;
    if (!(isfinite(s1_xx) && isfinite(s1_xy) && isfinite(s1_yy) &&
          isfinite(determinant1) && determinant1 > T(0))) {
        return false;
    }
    output->logdet1 = log(determinant1);
    output->inv1_xx = s1_yy / determinant1;
    output->inv1_xy = -s1_xy / determinant1;
    output->inv1_yy = s1_xx / determinant1;
    if (!(isfinite(output->logdet1) && isfinite(output->inv1_xx) &&
          isfinite(output->inv1_xy) && isfinite(output->inv1_yy))) return false;
    if (!(output->logdet1 >= output->logdet0)) return false;
    output->coefficient = exp(T(0.5) *
        (output->logdet0 - output->logdet1));
    return isfinite(output->coefficient) && output->coefficient > T(0) &&
        output->coefficient <= T(1);
}

template <typename T>
struct AnalyticPCompensationState {
    T factor[6];
    T projected_q[3];
    T minor[3];
    T determinant0;
    T determinant1;
    T logdet0;
    T logdet1;
    T s1_inv_xx;
    T s1_inv_xy;
    T s1_inv_yy;
    T coefficient;
};

template <typename T>
__device__ __forceinline__ bool robust_analytic_p_compensation_state(
    const T* scales,
    const T* rotation,
    const T* jacobian,
    AnalyticPCompensationState<T>* state) {
    // The same float32 J bytes define Q and enter this operator, so
    // J Q J^T is analytically gamma * I (the pixel footprint).  Never
    // re-form that cancellation-prone product numerically.
#pragma unroll
    for (int pixel = 0; pixel < 2; ++pixel) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            T value = T(0);
#pragma unroll
            for (int world = 0; world < 3; ++world) {
                value += jacobian[pixel * 3 + world] *
                    rotation[offset(world, axis)] * scales[axis];
            }
            state->factor[pixel * 3 + axis] = value;
            if (!isfinite(value)) return false;
        }
    }
    const T* f0 = state->factor;
    const T* f1 = state->factor + 3;
    state->minor[0] = f0[1] * f1[2] - f0[2] * f1[1];
    state->minor[1] = f0[2] * f1[0] - f0[0] * f1[2];
    state->minor[2] = f0[0] * f1[1] - f0[1] * f1[0];
    state->determinant0 = state->minor[0] * state->minor[0] +
        state->minor[1] * state->minor[1] +
        state->minor[2] * state->minor[2];
    if (!(isfinite(state->determinant0) && state->determinant0 > T(0))) {
        return false;
    }
    state->projected_q[0] = T(GOFISH_PIXEL_FOOTPRINT_VARIANCE);
    state->projected_q[1] = T(0);
    state->projected_q[2] = T(GOFISH_PIXEL_FOOTPRINT_VARIANCE);
    const T s0_xx = f0[0] * f0[0] + f0[1] * f0[1] + f0[2] * f0[2];
    const T s0_xy = f0[0] * f1[0] + f0[1] * f1[1] + f0[2] * f1[2];
    const T s0_yy = f1[0] * f1[0] + f1[1] * f1[1] + f1[2] * f1[2];
    const T s1_xx = s0_xx + state->projected_q[0];
    const T s1_xy = s0_xy + state->projected_q[1];
    const T s1_yy = s0_yy + state->projected_q[2];
    const T determinant_q = state->projected_q[0] * state->projected_q[2] -
        state->projected_q[1] * state->projected_q[1];
    const T mixed = s0_xx * state->projected_q[2] +
        s0_yy * state->projected_q[0] -
        T(2) * s0_xy * state->projected_q[1];
    state->determinant1 = state->determinant0 + mixed + determinant_q;
    if (!(isfinite(s1_xx) && isfinite(s1_xy) && isfinite(s1_yy) &&
          isfinite(state->determinant1) && state->determinant1 > T(0))) {
        return false;
    }
    state->logdet0 = log(state->determinant0);
    state->logdet1 = log(state->determinant1);
    state->s1_inv_xx = s1_yy / state->determinant1;
    state->s1_inv_xy = -s1_xy / state->determinant1;
    state->s1_inv_yy = s1_xx / state->determinant1;
    state->coefficient = exp(T(0.5) *
        (state->logdet0 - state->logdet1));
    return isfinite(state->logdet0) && isfinite(state->logdet1) &&
        isfinite(state->s1_inv_xx) && isfinite(state->s1_inv_xy) &&
        isfinite(state->s1_inv_yy) && isfinite(state->coefficient) &&
        state->logdet1 >= state->logdet0 && state->coefficient > T(0) &&
        state->coefficient <= T(1);
}

template <typename T>
__device__ __forceinline__ bool add_analytic_p_compensation_vjp(
    const AnalyticPCompensationState<T>& state,
    const T* jacobian,
    T gamma,
    T* a_adjoint) {
    if (!isfinite(gamma)) return false;
    const T inverse_det0 = T(1) / state.determinant0;
    T g0[6];
    // s0^{-1} F = [f1 x m; m x f0] / |m|^2.
    const T* f0 = state.factor;
    const T* f1 = state.factor + 3;
    g0[0] = (f1[1] * state.minor[2] - f1[2] * state.minor[1]) * inverse_det0;
    g0[1] = (f1[2] * state.minor[0] - f1[0] * state.minor[2]) * inverse_det0;
    g0[2] = (f1[0] * state.minor[1] - f1[1] * state.minor[0]) * inverse_det0;
    g0[3] = (state.minor[1] * f0[2] - state.minor[2] * f0[1]) * inverse_det0;
    g0[4] = (state.minor[2] * f0[0] - state.minor[0] * f0[2]) * inverse_det0;
    g0[5] = (state.minor[0] * f0[1] - state.minor[1] * f0[0]) * inverse_det0;
    T p_g0[6];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        p_g0[axis] = state.projected_q[0] * g0[axis] +
            state.projected_q[1] * g0[3 + axis];
        p_g0[3 + axis] = state.projected_q[1] * g0[axis] +
            state.projected_q[2] * g0[3 + axis];
    }
    T delta_factor[6];
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        delta_factor[axis] = state.s1_inv_xx * p_g0[axis] +
            state.s1_inv_xy * p_g0[3 + axis];
        delta_factor[3 + axis] = state.s1_inv_xy * p_g0[axis] +
            state.s1_inv_yy * p_g0[3 + axis];
    }
#pragma unroll
    for (int world = 0; world < 3; ++world) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            const T value = gamma * (
                jacobian[world] * delta_factor[axis] +
                jacobian[3 + world] * delta_factor[3 + axis]);
            if (!isfinite(value)) return false;
            a_adjoint[offset(world, axis)] += value;
        }
    }
    return true;
}

__global__ void classify_float32_rows_kernel(
    const bool* observed,
    const float* eigenvalues,
    const int* solver_info,
    int64_t count,
    bool* escalated) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    if (!observed[row]) {
        escalated[row] = false;
        return;
    }
    if (solver_info[row] != 0) {
        escalated[row] = true;
        return;
    }
    const float* values = eigenvalues + row * 3;
    const float spectral = fmaxf(fabsf(values[2]), FLT_MIN);
    const float resolution = kResolutionUlps * FLT_EPSILON * spectral;
    escalated[row] =
        !isfinite(values[0]) ||
        !isfinite(values[1]) || !isfinite(values[2]) ||
        values[0] <= resolution;
}

__global__ void finish_full_shape_kernel(
    const float* scales,
    const float* density,
    const bool* observed,
    const bool* escalated,
    const float* covariance,
    const float* eigenvectors,
    const float* eigenvalues,
    const float* rotation,
    int64_t count,
    bool mass_preserving,
    float* root,
    float* factor,
    float* inverse_factor,
    float* compensation,
    float* filtered_density,
    float* roots_output,
    int* failure_subtype,
    unsigned long long* first_failure_row,
    int* status) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    if (observed[row] && escalated[row]) {
        return;
    }
    const float* scales_row = scales + row * 3;
    const float* rotation_row = rotation + row * 9;
    float* root_row = root + row * 9;
    float* factor_row = factor + row * 9;
    float* inverse_row = inverse_factor + row * 9;

    if (!observed[row]) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                root_row[offset(i, j)] = 0.0f;
                factor_row[offset(i, j)] =
                    rotation_row[offset(i, j)] * scales_row[j];
                inverse_row[offset(i, j)] =
                    rotation_row[offset(j, i)] / scales_row[i];
            }
            roots_output[row * 3 + i] = 0.0f;
        }
        compensation[row] = 1.0f;
        filtered_density[row] = density[row];
        return;
    }

    if (escalated[row]) {
        return;
    }
    const float* values = eigenvalues + row * 3;
    if (!(isfinite(values[0]) && isfinite(values[1]) &&
          isfinite(values[2]) && values[0] > 0.0f &&
          values[1] > 0.0f && values[2] > 0.0f)) {
        atomicOr(status, kNotStrictSpd);
        return;
    }
    const float spectral = fmaxf(fabsf(values[2]), FLT_MIN);
    const float resolution = kResolutionUlps * FLT_EPSILON * spectral;
    if (values[0] <= resolution) {
        atomicOr(status, kUnresolvedFloat32);
        return;
    }
    const float local_roots[3] = {
        sqrtf(values[0]), sqrtf(values[1]), sqrtf(values[2])};
    const float* vectors = eigenvectors + row * 9;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        roots_output[row * 3 + i] = local_roots[i];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float value = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += eigenvector_value(vectors, i, k) * local_roots[k] *
                    eigenvector_value(vectors, j, k);
            }
            root_row[offset(i, j)] = value;
        }
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float value = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += root_row[offset(i, k)] *
                    rotation_row[offset(k, j)];
            }
            factor_row[offset(i, j)] = value;
        }
    }
    if (!inverse3(factor_row, inverse_row)) {
        record_invalid_output(
            row, kFloat32FactorInverse, failure_subtype,
            first_failure_row, status);
        return;
    }
    if (!(finite9(root_row) && finite9(factor_row) &&
          finite9(inverse_row))) {
        record_invalid_output(
            row, kFloat32FinalOutput, failure_subtype,
            first_failure_row, status);
        return;
    }
    // The shared all-observed double kernel overwrites these placeholders.
    compensation[row] = 1.0f;
    filtered_density[row] = density[row];
}

__device__ __forceinline__ double eigenvector_value64(
    const double* column_major_vectors, int row, int column) {
    return column_major_vectors[row + column * 3];
}

__device__ __forceinline__ void quaternion_rotation64(
    const float* quaternion, double* rotation) {
    const double w = static_cast<double>(quaternion[0]);
    const double x = static_cast<double>(quaternion[1]);
    const double y = static_cast<double>(quaternion[2]);
    const double z = static_cast<double>(quaternion[3]);
    rotation[offset(0, 0)] = 1.0 - 2.0 * (y * y + z * z);
    rotation[offset(0, 1)] = 2.0 * (x * y - z * w);
    rotation[offset(0, 2)] = 2.0 * (x * z + y * w);
    rotation[offset(1, 0)] = 2.0 * (x * y + z * w);
    rotation[offset(1, 1)] = 1.0 - 2.0 * (x * x + z * z);
    rotation[offset(1, 2)] = 2.0 * (y * z - x * w);
    rotation[offset(2, 0)] = 2.0 * (x * z - y * w);
    rotation[offset(2, 1)] = 2.0 * (y * z + x * w);
    rotation[offset(2, 2)] = 1.0 - 2.0 * (x * x + y * y);
}

__device__ __forceinline__ bool inverse_rotation3_64(
    const double* rotation, double* inverse) {
    const double determinant =
        rotation[0] * (rotation[4] * rotation[8] -
                       rotation[5] * rotation[7]) -
        rotation[1] * (rotation[3] * rotation[8] -
                       rotation[5] * rotation[6]) +
        rotation[2] * (rotation[3] * rotation[7] -
                       rotation[4] * rotation[6]);
    if (!isfinite(determinant) || determinant == 0.0) return false;
    const double reciprocal = 1.0 / determinant;
    inverse[0] = (rotation[4] * rotation[8] - rotation[5] * rotation[7]) * reciprocal;
    inverse[1] = (rotation[2] * rotation[7] - rotation[1] * rotation[8]) * reciprocal;
    inverse[2] = (rotation[1] * rotation[5] - rotation[2] * rotation[4]) * reciprocal;
    inverse[3] = (rotation[5] * rotation[6] - rotation[3] * rotation[8]) * reciprocal;
    inverse[4] = (rotation[0] * rotation[8] - rotation[2] * rotation[6]) * reciprocal;
    inverse[5] = (rotation[2] * rotation[3] - rotation[0] * rotation[5]) * reciprocal;
    inverse[6] = (rotation[3] * rotation[7] - rotation[4] * rotation[6]) * reciprocal;
    inverse[7] = (rotation[1] * rotation[6] - rotation[0] * rotation[7]) * reciprocal;
    inverse[8] = (rotation[0] * rotation[4] - rotation[1] * rotation[3]) * reciprocal;
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        if (!isfinite(inverse[index])) return false;
    }
    return true;
}

// B = U diag(roots) U^T R, so B^{-1} is formed directly as
// R^{-1} U diag(1/roots) U^T.  R^{-1} is evaluated by its exact double
// adjugate/determinant rather than assuming a float32-normalized quaternion
// produced a perfectly orthogonal polynomial rotation.
__device__ __forceinline__ bool spectral_inverse_factor64(
    const double* eigenvectors,
    const double* roots,
    const double* rotation,
    double* inverse) {
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        if (!(isfinite(roots[axis]) && roots[axis] > 0.0)) return false;
    }
    double rotation_inverse[9];
    if (!inverse_rotation3_64(rotation, rotation_inverse)) return false;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int world = 0; world < 3; ++world) {
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    value += rotation_inverse[offset(i, world)] *
                        eigenvector_value64(eigenvectors, world, axis) *
                        (1.0 / roots[axis]) *
                        eigenvector_value64(eigenvectors, j, axis);
                }
            }
            inverse[offset(i, j)] = value;
            if (!isfinite(value)) return false;
        }
    }
    return true;
}

__global__ void gather_unsafe_q_kernel(
    const int64_t* unsafe_indices,
    int64_t unsafe_count,
    const float* quaternion,
    const double* q6,
    double* q_eigenvectors,
    double* rotation64) {
    const int64_t compact =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact >= unsafe_count) {
        return;
    }
    const int64_t row = unsafe_indices[compact];
    const double* q_row = q6 + row * 6;
    double* q_matrix = q_eigenvectors + compact * 9;
    double* rotation_row = rotation64 + compact * 9;
    quaternion_rotation64(quaternion + row * 4, rotation_row);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = i; j < 3; ++j) {
            const double value = q_value(q_row, i, j);
            q_matrix[offset(i, j)] = value;
            q_matrix[offset(j, i)] = value;
        }
    }
}

__global__ void build_unsafe_covariance_kernel(
    const int64_t* unsafe_indices,
    int64_t unsafe_count,
    const float* scales,
    const double* q6,
    const double* q_eigenvalues,
    const int* q_solver_info,
    const double* rotation64,
    double* covariance64,
    double* covariance_eigenvectors64,
    double* projected_q64,
    bool* q_roundoff_projected,
    int* status) {
    const int64_t compact =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact >= unsafe_count) {
        return;
    }
    const int64_t row = unsafe_indices[compact];
    const double* q_row = q6 + row * 6;
    const double* q_values = q_eigenvalues + compact * 3;
    const bool q_valid = q_solver_info[compact] == 0 &&
        isfinite(q_values[0]) && isfinite(q_values[1]) &&
        isfinite(q_values[2]);
    if (!q_valid) {
        atomicOr(status, kInvalidQ);
    }
    const double q_spectral = q_valid
        ? fmax(fabs(q_values[2]), DBL_MIN)
        : DBL_MIN;
    const double q_resolution =
        static_cast<double>(kResolutionUlps) *
        static_cast<double>(FLT_EPSILON) * q_spectral;
    if (q_valid && q_values[0] < -q_resolution) {
        atomicOr(status, kInvalidQ);
    }
    // The ABI slot is kept, but Q is never projected: exact q64 defines the shape.
    q_roundoff_projected[row] = false;
    double q_exact[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            q_exact[offset(i, j)] = q_valid ? q_value(q_row, i, j) : 0.0;
        }
    }
    double* projected_q_row = projected_q64 + compact * 3;
    projected_q_row[0] = GOFISH_PIXEL_FOOTPRINT_VARIANCE;
    projected_q_row[1] = 0.0;
    projected_q_row[2] = GOFISH_PIXEL_FOOTPRINT_VARIANCE;
    const float* scales_row = scales + row * 3;
    const double* rotation_row = rotation64 + compact * 9;
    double* covariance_row = covariance64 + compact * 9;
    double* eigenvectors_row = covariance_eigenvectors64 + compact * 9;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = i; j < 3; ++j) {
            double value = q_exact[offset(i, j)];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const double scale = static_cast<double>(scales_row[k]);
                value += rotation_row[offset(i, k)] *
                    rotation_row[offset(j, k)] * scale * scale;
            }
            covariance_row[offset(i, j)] = value;
            covariance_row[offset(j, i)] = value;
            eigenvectors_row[offset(i, j)] = value;
            eigenvectors_row[offset(j, i)] = value;
        }
    }
}

__global__ void finish_unsafe_full_shape_kernel(
    const int64_t* unsafe_indices,
    int64_t unsafe_count,
    const float* scales,
    const float* density,
    const double* covariance64,
    const double* covariance_eigenvectors64,
    const double* covariance_eigenvalues64,
    const int* covariance_solver_info,
    const double* rotation64,
    const float* jacobian,
    bool mass_preserving,
    float* covariance,
    float* root,
    float* factor,
    float* inverse_factor,
    float* compensation,
    float* filtered_density,
    float* roots,
    double* root64,
    double* inverse_factor64,
    double* compensation64,
    double* roots64,
    int* failure_subtype,
    unsigned long long* first_failure_row,
    int* status) {
    const int64_t compact =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact >= unsafe_count) {
        return;
    }
    const int64_t row = unsafe_indices[compact];
    const double* values = covariance_eigenvalues64 + compact * 3;
    if (covariance_solver_info[compact] != 0 ||
        !(isfinite(values[0]) && isfinite(values[1]) &&
          isfinite(values[2]) && values[0] > 0.0 &&
          values[1] > 0.0 && values[2] > 0.0)) {
        atomicOr(status, kNotStrictSpd);
        return;
    }
    const double local_roots[3] = {
        sqrt(values[0]), sqrt(values[1]), sqrt(values[2])};
    bool narrowed_finite = true;
    const double* vectors = covariance_eigenvectors64 + compact * 9;
    double* root64_row = root64 + compact * 9;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        roots64[compact * 3 + i] = local_roots[i];
        const float narrowed_root = static_cast<float>(local_roots[i]);
        roots[row * 3 + i] = narrowed_root;
        narrowed_finite = narrowed_finite && isfinite(narrowed_root);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += eigenvector_value64(vectors, i, k) *
                    local_roots[k] * eigenvector_value64(vectors, j, k);
            }
            root64_row[offset(i, j)] = value;
            const float narrowed_value = static_cast<float>(value);
            root[row * 9 + offset(i, j)] = narrowed_value;
            narrowed_finite = narrowed_finite && isfinite(narrowed_value);
        }
    }
    const double* rotation_row = rotation64 + compact * 9;
    double factor64[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += root64_row[offset(i, k)] *
                    rotation_row[offset(k, j)];
            }
            factor64[offset(i, j)] = value;
            const float narrowed_value = static_cast<float>(value);
            factor[row * 9 + offset(i, j)] = narrowed_value;
            narrowed_finite = narrowed_finite && isfinite(narrowed_value);
        }
    }
    double* inverse64_row = inverse_factor64 + compact * 9;
    if (!spectral_inverse_factor64(
            vectors, local_roots, rotation_row, inverse64_row)) {
        record_invalid_output(
            row, kEscalatedSpectralFactorInverse, failure_subtype,
            first_failure_row, status);
        return;
    }
    const double* covariance64_row = covariance64 + compact * 9;
    const float* scales_row = scales + row * 3;
    const double scales64[3] = {
        static_cast<double>(scales_row[0]),
        static_cast<double>(scales_row[1]),
        static_cast<double>(scales_row[2])};
    double jacobian64[6];
#pragma unroll
    for (int index = 0; index < 6; ++index) {
        jacobian64[index] = static_cast<double>(jacobian[row * 6 + index]);
    }
    AnalyticPCompensationState<double> compensation_state;
    if (mass_preserving && !robust_analytic_p_compensation_state(
            scales64, rotation_row, jacobian64, &compensation_state)) {
        record_invalid_output(
            row, kEscalatedProjectedM2, failure_subtype,
            first_failure_row, status);
        return;
    }
    const double coefficient = mass_preserving
        ? compensation_state.coefficient : 1.0;
    compensation64[compact] = coefficient;
    const float narrowed_coefficient = static_cast<float>(coefficient);
    const float narrowed_filtered = density[row] * narrowed_coefficient;
    narrowed_finite = narrowed_finite && isfinite(narrowed_coefficient) &&
        isfinite(narrowed_filtered);
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        const float narrowed_covariance =
            static_cast<float>(covariance64_row[index]);
        const float narrowed_inverse =
            static_cast<float>(inverse64_row[index]);
        covariance[row * 9 + index] = narrowed_covariance;
        inverse_factor[row * 9 + index] = narrowed_inverse;
        narrowed_finite = narrowed_finite && isfinite(narrowed_covariance) &&
            isfinite(narrowed_inverse);
    }
    if (!narrowed_finite) {
        record_invalid_output(
            row, kEscalatedNarrowedOutput, failure_subtype,
            first_failure_row, status);
        return;
    }
    compensation[row] = narrowed_coefficient;
    filtered_density[row] = narrowed_filtered;
}

__device__ __forceinline__ void promote_projected_state64(
    const float* scales,
    const float* quaternion,
    const float* jacobian,
    double* scales64,
    double* rotation64,
    double* jacobian64) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        scales64[i] = static_cast<double>(scales[i]);
    }
    quaternion_rotation64(quaternion, rotation64);
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        jacobian64[i] = static_cast<double>(jacobian[i]);
    }
}

__global__ void finish_all_observed_compensation_kernel(
    const float* scales,
    const float* quaternion,
    const float* density,
    const float* jacobian,
    const bool* observed,
    const bool* escalated,
    int64_t count,
    float* compensation,
    float* filtered_density,
    int* failure_subtype,
    unsigned long long* first_failure_row,
    int* status) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count || !observed[row] || escalated[row]) return;
    double scales64[3], rotation64[9], jacobian64[6];
    promote_projected_state64(
        scales + row * 3, quaternion + row * 4, jacobian + row * 6,
        scales64, rotation64, jacobian64);
    AnalyticPCompensationState<double> state;
    if (!robust_analytic_p_compensation_state(
            scales64, rotation64, jacobian64, &state)) {
        record_invalid_output(
            row, kFloat32ProjectedM2, failure_subtype,
            first_failure_row, status);
        return;
    }
    const float coefficient = static_cast<float>(state.coefficient);
    const float filtered = density[row] * coefficient;
    if (!(isfinite(coefficient) && isfinite(filtered))) {
        record_invalid_output(
            row, kFloat32FinalOutput, failure_subtype,
            first_failure_row, status);
        return;
    }
    compensation[row] = coefficient;
    filtered_density[row] = filtered;
}

__device__ __forceinline__ void load9(
    const float* source, float* target) {
#pragma unroll
    for (int index = 0; index < 9; ++index) {
        target[index] = source[index];
    }
}

__device__ __forceinline__ void symmetrize(
    const float* source, float* target) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            target[offset(i, j)] = 0.5f *
                (source[offset(i, j)] + source[offset(j, i)]);
        }
    }
}

__device__ __forceinline__ void multiply(
    const float* left, const float* right, float* output) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float value = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += left[offset(i, k)] * right[offset(k, j)];
            }
            output[offset(i, j)] = value;
        }
    }
}

__device__ __forceinline__ void quaternion_rotation_vjp(
    const float* quaternion,
    const float* rotation_adjoint,
    float* quaternion_adjoint) {
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    const float* g = rotation_adjoint;
    quaternion_adjoint[0] = 2.0f * (
        z * (g[offset(1, 0)] - g[offset(0, 1)]) +
        y * (g[offset(0, 2)] - g[offset(2, 0)]) +
        x * (g[offset(2, 1)] - g[offset(1, 2)]));
    quaternion_adjoint[1] = 2.0f * (
        y * (g[offset(0, 1)] + g[offset(1, 0)]) +
        z * (g[offset(0, 2)] + g[offset(2, 0)]) +
        w * (g[offset(2, 1)] - g[offset(1, 2)]) -
        2.0f * x * (g[offset(1, 1)] + g[offset(2, 2)]));
    quaternion_adjoint[2] = 2.0f * (
        x * (g[offset(0, 1)] + g[offset(1, 0)]) +
        w * (g[offset(0, 2)] - g[offset(2, 0)]) +
        z * (g[offset(1, 2)] + g[offset(2, 1)]) -
        2.0f * y * (g[offset(0, 0)] + g[offset(2, 2)]));
    quaternion_adjoint[3] = 2.0f * (
        w * (g[offset(1, 0)] - g[offset(0, 1)]) +
        x * (g[offset(0, 2)] + g[offset(2, 0)]) +
        y * (g[offset(1, 2)] + g[offset(2, 1)]) -
        2.0f * z * (g[offset(0, 0)] + g[offset(1, 1)]));
}

__global__ void full_shape_backward_kernel(
    const float* scales,
    const float* quaternion,
    const float* density,
    const float* jacobian,
    const bool* observed,
    const bool* escalated,
    const float* covariance,
    const float* inverse_factor,
    const float* compensation,
    const float* root,
    const float* rotation,
    const float* eigenvectors,
    const float* roots,
    const float* grad_covariance,
    const float* grad_factor,
    const float* grad_inverse_factor,
    const float* grad_compensation,
    const float* grad_filtered_density,
    int64_t count,
    bool mass_preserving,
    float* grad_scales,
    float* grad_quaternion,
    float* grad_density,
    int* status) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    if (observed[row] && escalated[row]) {
        return;
    }
    const float* scales_row = scales + row * 3;
    const float* quaternion_row = quaternion + row * 4;
    const float* rotation_row = rotation + row * 9;
    const float* covariance_row = covariance + row * 9;
    const float* inverse_row = inverse_factor + row * 9;
    const float* root_row = root + row * 9;
    const float* vectors = eigenvectors + row * 9;
    const float* roots_row = roots + row * 3;
    const float* gb_input = grad_factor + row * 9;
    const float* gk_input = grad_inverse_factor + row * 9;
    float gc[9];
    symmetrize(grad_covariance + row * 9, gc);
    float gb[9];
    float gk[9];
    load9(gb_input, gb);
    load9(gk_input, gk);
    float rotation_adjoint[9] = {0.0f};
    float scale_adjoint[3] = {0.0f, 0.0f, 0.0f};
    const float grad_filtered = grad_filtered_density[row];

    if (!observed[row]) {
        grad_density[row] = grad_filtered;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                rotation_adjoint[offset(i, j)] =
                    gb[offset(i, j)] * scales_row[j] +
                    gk[offset(j, i)] / scales_row[j];
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float value = 0.0f;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                value += gb[offset(i, j)] * rotation_row[offset(i, j)] -
                    gk[offset(j, i)] * rotation_row[offset(i, j)] /
                        (scales_row[j] * scales_row[j]);
            }
            scale_adjoint[j] = value;
        }
    } else {
        const float coefficient = compensation[row];
        grad_density[row] = mass_preserving
            ? coefficient * grad_filtered
            : grad_filtered;

        // K=B^-1: B_bar_total=B_bar-K^T K_bar K^T.
        float inverse_transpose[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                inverse_transpose[offset(i, j)] = inverse_row[offset(j, i)];
            }
        }
        float temporary[9];
        float inverse_pullback[9];
        multiply(inverse_transpose, gk, temporary);
        multiply(temporary, inverse_transpose, inverse_pullback);
#pragma unroll
        for (int index = 0; index < 9; ++index) {
            gb[index] -= inverse_pullback[index];
        }

        // B=H R.
        float root_adjoint_raw[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float h_value = 0.0f;
                float r_value = 0.0f;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    h_value += gb[offset(i, k)] * rotation_row[offset(j, k)];
                    r_value += root_row[offset(k, i)] * gb[offset(k, j)];
                }
                root_adjoint_raw[offset(i, j)] = h_value;
                rotation_adjoint[offset(i, j)] = r_value;
            }
        }
        float root_adjoint[9];
        symmetrize(root_adjoint_raw, root_adjoint);

        // H X + X H = sym(H_bar). In the retained cuSOLVER basis,
        // M_ij=(U^T sym(H_bar) U)_ij/(r_i+r_j). This denominator is
        // strictly positive and remains well-defined at repeated eigenvalues.
        float local[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float numerator = 0.0f;
#pragma unroll
                for (int a = 0; a < 3; ++a) {
#pragma unroll
                    for (int b = 0; b < 3; ++b) {
                        numerator += eigenvector_value(vectors, a, i) *
                            root_adjoint[offset(a, b)] *
                            eigenvector_value(vectors, b, j);
                    }
                }
                const float denominator = roots_row[i] + roots_row[j];
                if (!(isfinite(denominator) && denominator > 0.0f)) {
                    atomicOr(status, kInvalidSylvesterDenominator);
                    return;
                }
                local[offset(i, j)] = numerator / denominator;
            }
        }
        float root_pullback[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float value = 0.0f;
#pragma unroll
                for (int a = 0; a < 3; ++a) {
#pragma unroll
                    for (int b = 0; b < 3; ++b) {
                        value += eigenvector_value(vectors, i, a) *
                            local[offset(a, b)] *
                            eigenvector_value(vectors, j, b);
                    }
                }
                gc[offset(i, j)] += value;
            }
        }

        // C=A A^T+Q, A=R D and Q is detached.
        float a_matrix[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                a_matrix[offset(i, j)] =
                    rotation_row[offset(i, j)] * scales_row[j];
            }
        }
        float a_adjoint[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float value = 0.0f;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    value += (gc[offset(i, k)] + gc[offset(k, i)]) *
                        a_matrix[offset(k, j)];
                }
                a_adjoint[offset(i, j)] = value;
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float value = 0.0f;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                value += a_adjoint[offset(i, j)] *
                    rotation_row[offset(i, j)];
                rotation_adjoint[offset(i, j)] +=
                    a_adjoint[offset(i, j)] * scales_row[j];
            }
            scale_adjoint[j] = value;
        }
    }

    // The direct invalid branch also exposes covariance in the full contract.
    if (!observed[row]) {
        float a_matrix[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                a_matrix[offset(i, j)] =
                    rotation_row[offset(i, j)] * scales_row[j];
            }
        }
        float a_adjoint[9];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float value = 0.0f;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    value += (gc[offset(i, k)] + gc[offset(k, i)]) *
                        a_matrix[offset(k, j)];
                }
                a_adjoint[offset(i, j)] = value;
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                scale_adjoint[j] += a_adjoint[offset(i, j)] *
                    rotation_row[offset(i, j)];
                rotation_adjoint[offset(i, j)] +=
                    a_adjoint[offset(i, j)] * scales_row[j];
            }
        }
    }

    grad_scales[row * 3] = scale_adjoint[0];
    grad_scales[row * 3 + 1] = scale_adjoint[1];
    grad_scales[row * 3 + 2] = scale_adjoint[2];
    quaternion_rotation_vjp(
        quaternion_row, rotation_adjoint, grad_quaternion + row * 4);
    if (!(isfinite(grad_scales[row * 3]) &&
          isfinite(grad_scales[row * 3 + 1]) &&
          isfinite(grad_scales[row * 3 + 2]) &&
          isfinite(grad_density[row]) &&
          isfinite(grad_quaternion[row * 4]) &&
          isfinite(grad_quaternion[row * 4 + 1]) &&
          isfinite(grad_quaternion[row * 4 + 2]) &&
          isfinite(grad_quaternion[row * 4 + 3]))) {
        atomicOr(status, kInvalidGradient);
    }
}

__device__ __forceinline__ void quaternion_rotation_vjp64(
    const float* quaternion,
    const double* rotation_adjoint,
    float* quaternion_adjoint) {
    const double w = static_cast<double>(quaternion[0]);
    const double x = static_cast<double>(quaternion[1]);
    const double y = static_cast<double>(quaternion[2]);
    const double z = static_cast<double>(quaternion[3]);
    const double* g = rotation_adjoint;
    quaternion_adjoint[0] = static_cast<float>(2.0 * (
        z * (g[offset(1, 0)] - g[offset(0, 1)]) +
        y * (g[offset(0, 2)] - g[offset(2, 0)]) +
        x * (g[offset(2, 1)] - g[offset(1, 2)])));
    quaternion_adjoint[1] = static_cast<float>(2.0 * (
        y * (g[offset(0, 1)] + g[offset(1, 0)]) +
        z * (g[offset(0, 2)] + g[offset(2, 0)]) +
        w * (g[offset(2, 1)] - g[offset(1, 2)]) -
        2.0 * x * (g[offset(1, 1)] + g[offset(2, 2)])));
    quaternion_adjoint[2] = static_cast<float>(2.0 * (
        x * (g[offset(0, 1)] + g[offset(1, 0)]) +
        w * (g[offset(0, 2)] - g[offset(2, 0)]) +
        z * (g[offset(1, 2)] + g[offset(2, 1)]) -
        2.0 * y * (g[offset(0, 0)] + g[offset(2, 2)])));
    quaternion_adjoint[3] = static_cast<float>(2.0 * (
        w * (g[offset(1, 0)] - g[offset(0, 1)]) +
        x * (g[offset(0, 2)] + g[offset(2, 0)]) +
        y * (g[offset(1, 2)] + g[offset(2, 1)]) -
        2.0 * z * (g[offset(0, 0)] + g[offset(1, 1)])));
}

__global__ void unsafe_full_shape_backward_kernel(
    const int64_t* unsafe_indices,
    int64_t unsafe_count,
    const float* scales,
    const float* quaternion,
    const float* density,
    const float* jacobian,
    const double* inverse_factor64,
    const float* compensation,
    const double* root64,
    const double* rotation64,
    const double* eigenvectors64,
    const double* roots64,
    const float* grad_covariance,
    const float* grad_factor,
    const float* grad_inverse_factor,
    const float* grad_compensation,
    const float* grad_filtered_density,
    bool mass_preserving,
    float* grad_scales,
    float* grad_quaternion,
    float* grad_density,
    int* status) {
    const int64_t compact =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact >= unsafe_count) {
        return;
    }
    const int64_t row = unsafe_indices[compact];
    const float* scales_row_float = scales + row * 3;
    const float* quaternion_row = quaternion + row * 4;
    const double scales_row[3] = {
        static_cast<double>(scales_row_float[0]),
        static_cast<double>(scales_row_float[1]),
        static_cast<double>(scales_row_float[2])};
    const double* inverse_row = inverse_factor64 + compact * 9;
    const double* root_row = root64 + compact * 9;
    const double* rotation_row = rotation64 + compact * 9;
    const double* vectors = eigenvectors64 + compact * 9;
    const double* roots_row = roots64 + compact * 3;
    double gc[9];
    double gb[9];
    double gk[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            gc[offset(i, j)] = 0.5 * (
                static_cast<double>(grad_covariance[row * 9 + offset(i, j)]) +
                static_cast<double>(grad_covariance[row * 9 + offset(j, i)]));
            gb[offset(i, j)] =
                static_cast<double>(grad_factor[row * 9 + offset(i, j)]);
            gk[offset(i, j)] = static_cast<double>(
                grad_inverse_factor[row * 9 + offset(i, j)]);
        }
    }
    const double grad_filtered =
        static_cast<double>(grad_filtered_density[row]);
    const double coefficient = static_cast<double>(compensation[row]);
    const double combined_upstream = mass_preserving
        ? static_cast<double>(grad_compensation[row]) +
            static_cast<double>(density[row]) * grad_filtered
        : 0.0;
    grad_density[row] = static_cast<float>(
        mass_preserving ? coefficient * grad_filtered : grad_filtered);

    double inverse_transpose[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            inverse_transpose[offset(i, j)] = inverse_row[offset(j, i)];
        }
    }
    double temporary[9];
    double inverse_pullback[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += inverse_transpose[offset(i, k)] * gk[offset(k, j)];
            }
            temporary[offset(i, j)] = value;
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += temporary[offset(i, k)] *
                    inverse_transpose[offset(k, j)];
            }
            inverse_pullback[offset(i, j)] = value;
            gb[offset(i, j)] -= value;
        }
    }

    double root_adjoint_raw[9];
    double rotation_adjoint[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double h_value = 0.0;
            double r_value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                h_value += gb[offset(i, k)] * rotation_row[offset(j, k)];
                r_value += root_row[offset(k, i)] * gb[offset(k, j)];
            }
            root_adjoint_raw[offset(i, j)] = h_value;
            rotation_adjoint[offset(i, j)] = r_value;
        }
    }
    double root_adjoint[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            root_adjoint[offset(i, j)] = 0.5 *
                (root_adjoint_raw[offset(i, j)] +
                 root_adjoint_raw[offset(j, i)]);
        }
    }
    double local[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double numerator = 0.0;
#pragma unroll
            for (int a = 0; a < 3; ++a) {
#pragma unroll
                for (int b = 0; b < 3; ++b) {
                    numerator += eigenvector_value64(vectors, a, i) *
                        root_adjoint[offset(a, b)] *
                        eigenvector_value64(vectors, b, j);
                }
            }
            const double denominator = roots_row[i] + roots_row[j];
            if (!(isfinite(denominator) && denominator > 0.0)) {
                atomicOr(status, kInvalidSylvesterDenominator);
                return;
            }
            local[offset(i, j)] = numerator / denominator;
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int a = 0; a < 3; ++a) {
#pragma unroll
                for (int b = 0; b < 3; ++b) {
                    value += eigenvector_value64(vectors, i, a) *
                        local[offset(a, b)] *
                        eigenvector_value64(vectors, j, b);
                }
            }
            gc[offset(i, j)] += value;
        }
    }
    double a_matrix[9];
    double a_adjoint[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            a_matrix[offset(i, j)] =
                rotation_row[offset(i, j)] * scales_row[j];
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            double value = 0.0;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                value += (gc[offset(i, k)] + gc[offset(k, i)]) *
                    a_matrix[offset(k, j)];
            }
            a_adjoint[offset(i, j)] = value;
        }
    }
    if (mass_preserving && combined_upstream != 0.0) {
        double jacobian64[6];
#pragma unroll
        for (int index = 0; index < 6; ++index) {
            jacobian64[index] = static_cast<double>(jacobian[row * 6 + index]);
        }
        AnalyticPCompensationState<double> compensation_state;
        if (!robust_analytic_p_compensation_state(
                scales_row, rotation_row, jacobian64,
                &compensation_state) ||
            !add_analytic_p_compensation_vjp(
                compensation_state, jacobian64,
                compensation_state.coefficient * combined_upstream,
                a_adjoint)) {
            atomicOr(status, kInvalidGradient);
            return;
        }
    }
    double scale_adjoint[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        double value = 0.0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            value += a_adjoint[offset(i, j)] * rotation_row[offset(i, j)];
            rotation_adjoint[offset(i, j)] +=
                a_adjoint[offset(i, j)] * scales_row[j];
        }
        scale_adjoint[j] = value;
        grad_scales[row * 3 + j] = static_cast<float>(value);
    }
    quaternion_rotation_vjp64(
        quaternion_row, rotation_adjoint, grad_quaternion + row * 4);
    if (!(isfinite(grad_scales[row * 3]) &&
          isfinite(grad_scales[row * 3 + 1]) &&
          isfinite(grad_scales[row * 3 + 2]) &&
          isfinite(grad_density[row]) &&
          isfinite(grad_quaternion[row * 4]) &&
          isfinite(grad_quaternion[row * 4 + 1]) &&
          isfinite(grad_quaternion[row * 4 + 2]) &&
          isfinite(grad_quaternion[row * 4 + 3]))) {
        atomicOr(status, kInvalidGradient);
    }
}

__global__ void all_observed_compensation_backward_kernel(
    const float* scales,
    const float* quaternion,
    const float* density,
    const float* jacobian,
    const bool* observed,
    const bool* escalated,
    const float* grad_compensation,
    const float* grad_filtered_density,
    int64_t count,
    float* grad_scales,
    float* grad_quaternion,
    int* status) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= count || !observed[row] || escalated[row]) return;
    const double combined_upstream =
        static_cast<double>(grad_compensation[row]) +
        static_cast<double>(density[row]) *
            static_cast<double>(grad_filtered_density[row]);
    if (combined_upstream == 0.0) return;
    double scales64[3], rotation64[9], jacobian64[6];
    promote_projected_state64(
        scales + row * 3, quaternion + row * 4, jacobian + row * 6,
        scales64, rotation64, jacobian64);
    AnalyticPCompensationState<double> state;
    if (!robust_analytic_p_compensation_state(
            scales64, rotation64, jacobian64, &state)) {
        atomicOr(status, kInvalidGradient);
        return;
    }
    double a_adjoint[9] = {0.0};
    const double gamma = state.coefficient * combined_upstream;
    if (!add_analytic_p_compensation_vjp(
            state, jacobian64, gamma, a_adjoint)) {
        atomicOr(status, kInvalidGradient);
        return;
    }
    double rotation_adjoint[9];
#pragma unroll
    for (int index = 0; index < 9; ++index) rotation_adjoint[index] = 0.0;
    double scale_add[3] = {0.0, 0.0, 0.0};
#pragma unroll
    for (int world = 0; world < 3; ++world) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            scale_add[axis] +=
                a_adjoint[offset(world, axis)] * rotation64[offset(world, axis)];
            rotation_adjoint[offset(world, axis)] +=
                a_adjoint[offset(world, axis)] * scales64[axis];
        }
    }
    float quaternion_add[4];
    quaternion_rotation_vjp64(
        quaternion + row * 4, rotation_adjoint, quaternion_add);
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float value = static_cast<float>(scale_add[axis]);
        grad_scales[row * 3 + axis] += value;
    }
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        grad_quaternion[row * 4 + index] += quaternion_add[index];
    }
    if (!(isfinite(grad_scales[row * 3]) &&
          isfinite(grad_scales[row * 3 + 1]) &&
          isfinite(grad_scales[row * 3 + 2]) &&
          isfinite(grad_quaternion[row * 4]) &&
          isfinite(grad_quaternion[row * 4 + 1]) &&
          isfinite(grad_quaternion[row * 4 + 2]) &&
          isfinite(grad_quaternion[row * 4 + 3]))) {
        atomicOr(status, kInvalidGradient);
    }
}

void throw_for_status(int status, const char* phase) {
    TORCH_CHECK(
        status == 0,
        "footprint cuSOLVER full-shape ",
        phase,
        " rejected its state; status=",
        status,
        " (scale=1 quaternion=2 density=4 q=8 solver=16 spd=32 "
        "float32_resolution=64 output=128 gradient=256 sylvester=512)");
}

int synchronize_status(
    const torch::Tensor& status, cudaStream_t stream, const char* phase) {
    int host_status = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_status,
        status.data_ptr<int>(),
        sizeof(host_status),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    throw_for_status(host_status, phase);
    return host_status;
}

template <typename T, size_t N>
std::array<T, N> copy_device_array(const T* source) {
    std::array<T, N> result{};
    C10_CUDA_CHECK(cudaMemcpy(
        result.data(), source, sizeof(T) * N, cudaMemcpyDeviceToHost));
    return result;
}

const char* invalid_output_subtype_name(int subtype) {
    switch (subtype) {
        case kFloat32FactorInverse: return "float32_factor_inverse";
        case kFloat32ProjectedM2: return "float32_projected_m2";
        case kFloat32FinalOutput: return "float32_final_output";
        case kEscalatedSpectralFactorInverse:
            return "escalated_spectral_factor_inverse";
        case kEscalatedProjectedM2: return "escalated_projected_m2";
        case kProjectedOnlyProjectedM2:
            return "projected_only_projected_m2";
        case kEscalatedNarrowedOutput:
            return "escalated_narrowed_output";
        default: return "unknown";
    }
}

struct HostProjectedTerms {
    std::array<double, 3> s0{};
    std::array<double, 3> s1{};
    double det0 = std::numeric_limits<double>::quiet_NaN();
    double det1 = std::numeric_limits<double>::quiet_NaN();
    double logdet0 = std::numeric_limits<double>::quiet_NaN();
    double logdet1 = std::numeric_limits<double>::quiet_NaN();
    double coefficient = std::numeric_limits<double>::quiet_NaN();
    bool factors_finite = false;
    bool determinants_positive_finite = false;
    bool logs_finite = false;
    bool coefficient_valid = false;
};

HostProjectedTerms host_projected_terms(
    const std::array<float, 3>& scales,
    const std::array<float, 9>& rotation,
    const std::array<double, 9>& eigenvectors,
    const std::array<double, 3>& eigenvalues,
    const std::array<float, 6>& jacobian) {
    HostProjectedTerms result;
    double f0[6] = {};
    double f1[6] = {};
    bool finite = true;
    for (int pixel = 0; pixel < 2; ++pixel) {
        for (int axis = 0; axis < 3; ++axis) {
            const double root = eigenvalues[axis] > 0.0
                ? std::sqrt(eigenvalues[axis])
                : std::numeric_limits<double>::quiet_NaN();
            for (int world = 0; world < 3; ++world) {
                f0[pixel * 3 + axis] +=
                    static_cast<double>(jacobian[pixel * 3 + world]) *
                    static_cast<double>(rotation[world * 3 + axis]) *
                    static_cast<double>(scales[axis]);
                f1[pixel * 3 + axis] +=
                    static_cast<double>(jacobian[pixel * 3 + world]) *
                    eigenvectors[world + axis * 3] * root;
            }
            finite = finite && std::isfinite(f0[pixel * 3 + axis]) &&
                std::isfinite(f1[pixel * 3 + axis]);
        }
    }
    result.factors_finite = finite;
    auto fill = [](const double* factor, std::array<double, 3>* state,
                   double* determinant) {
        for (int axis = 0; axis < 3; ++axis) {
            (*state)[0] += factor[axis] * factor[axis];
            (*state)[1] += factor[axis] * factor[3 + axis];
            (*state)[2] += factor[3 + axis] * factor[3 + axis];
        }
        *determinant = 0.0;
        for (int first = 0; first < 3; ++first) {
            for (int second = first + 1; second < 3; ++second) {
                const double minor = factor[first] * factor[3 + second] -
                    factor[second] * factor[3 + first];
                *determinant += minor * minor;
            }
        }
    };
    fill(f0, &result.s0, &result.det0);
    fill(f1, &result.s1, &result.det1);
    result.determinants_positive_finite =
        std::isfinite(result.det0) && result.det0 > 0.0 &&
        std::isfinite(result.det1) && result.det1 > 0.0;
    if (result.determinants_positive_finite) {
        result.logdet0 = std::log(result.det0);
        result.logdet1 = std::log(result.det1);
        result.coefficient = std::exp(
            0.5 * (result.logdet0 - result.logdet1));
    }
    result.logs_finite = std::isfinite(result.logdet0) &&
        std::isfinite(result.logdet1);
    result.coefficient_valid = std::isfinite(result.coefficient) &&
        result.coefficient > 0.0 && result.coefficient <= 1.0 &&
        result.logdet1 >= result.logdet0;
    return result;
}

HostProjectedTerms host_projected_terms_analytic_p(
    const std::array<float, 3>& scales,
    const std::array<float, 4>& quaternion,
    const std::array<float, 6>& jacobian) {
    HostProjectedTerms result;
    const double w = quaternion[0], x = quaternion[1];
    const double y = quaternion[2], z = quaternion[3];
    const double rotation[9] = {
        1.0-2.0*(y*y+z*z), 2.0*(x*y-z*w), 2.0*(x*z+y*w),
        2.0*(x*y+z*w), 1.0-2.0*(x*x+z*z), 2.0*(y*z-x*w),
        2.0*(x*z-y*w), 2.0*(y*z+x*w), 1.0-2.0*(x*x+y*y)};
    double factor[6] = {};
    for (int pixel = 0; pixel < 2; ++pixel) {
        for (int axis = 0; axis < 3; ++axis) {
            for (int world = 0; world < 3; ++world) {
                factor[pixel * 3 + axis] +=
                    static_cast<double>(jacobian[pixel * 3 + world]) *
                    rotation[world * 3 + axis] * scales[axis];
            }
        }
    }
    result.factors_finite = true;
    for (double value : factor) {
        result.factors_finite = result.factors_finite && std::isfinite(value);
    }
    for (int axis = 0; axis < 3; ++axis) {
        result.s0[0] += factor[axis] * factor[axis];
        result.s0[1] += factor[axis] * factor[3 + axis];
        result.s0[2] += factor[3 + axis] * factor[3 + axis];
    }
    for (int first = 0; first < 3; ++first) {
        for (int second = first + 1; second < 3; ++second) {
            const double minor = factor[first] * factor[3 + second] -
                factor[second] * factor[3 + first];
            result.det0 += minor * minor;
        }
    }
    const double projected_q[3] = {GOFISH_PIXEL_FOOTPRINT_VARIANCE, 0.0, GOFISH_PIXEL_FOOTPRINT_VARIANCE};
    result.s1 = {result.s0[0] + projected_q[0],
                 result.s0[1] + projected_q[1],
                 result.s0[2] + projected_q[2]};
    const double determinant_q = projected_q[0] * projected_q[2] -
        projected_q[1] * projected_q[1];
    const double mixed = result.s0[0] * projected_q[2] +
        result.s0[2] * projected_q[0] -
        2.0 * result.s0[1] * projected_q[1];
    result.det1 = result.det0 + mixed + determinant_q;
    if (std::isfinite(result.det1) && result.det1 > 0.0 &&
        result.det0 > 0.0) {
        result.logdet0 = std::log(result.det0);
        result.logdet1 = std::log(result.det1);
        result.coefficient = std::exp(
            0.5 * (result.logdet0 - result.logdet1));
    }
    result.determinants_positive_finite =
        std::isfinite(result.det0) && result.det0 > 0.0 &&
        std::isfinite(result.det1) && result.det1 > 0.0;
    result.logs_finite = std::isfinite(result.logdet0) &&
        std::isfinite(result.logdet1);
    result.coefficient_valid = std::isfinite(result.coefficient) &&
        result.coefficient > 0.0 && result.coefficient <= 1.0 &&
        result.logdet1 >= result.logdet0;
    return result;
}

template <size_t N>
std::string format_values(const std::array<double, N>& values) {
    std::ostringstream stream;
    stream << std::setprecision(17) << "[";
    for (size_t index = 0; index < N; ++index) {
        if (index) stream << ",";
        stream << values[index];
    }
    stream << "]";
    return stream.str();
}

template <size_t N>
std::string format_float_hex(const std::array<float, N>& values) {
    std::ostringstream stream;
    stream << "[" << std::hexfloat;
    for (size_t index = 0; index < N; ++index) {
        if (index) stream << ",";
        stream << values[index];
    }
    stream << "]";
    return stream.str();
}

template <size_t N>
std::string format_double_hex(const std::array<double, N>& values) {
    std::ostringstream stream;
    stream << "[" << std::hexfloat;
    for (size_t index = 0; index < N; ++index) {
        if (index) stream << ",";
        stream << values[index];
    }
    stream << "]";
    return stream.str();
}

int synchronize_forward_status(
    const torch::Tensor& status,
    const torch::Tensor& failure_subtype,
    const torch::Tensor& first_failure_row,
    const torch::Tensor& scales,
    const torch::Tensor& quaternion,
    const torch::Tensor& density,
    const torch::Tensor& q6,
    const torch::Tensor& jacobian,
    const torch::Tensor& observed,
    const torch::Tensor& escalated,
    const torch::Tensor& projected_only_unsafe,
    const torch::Tensor& rotation,
    const torch::Tensor& eigenvectors,
    const torch::Tensor& eigenvalues,
    const torch::Tensor& unsafe_indices,
    const torch::Tensor& eigenvectors64,
    const torch::Tensor& eigenvalues64,
    cudaStream_t stream) {
    int host_status = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &host_status, status.data_ptr<int>(), sizeof(host_status),
        cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!(host_status & kInvalidOutput)) {
        throw_for_status(host_status, "forward");
        return host_status;
    }

    unsigned long long row64 = 0;
    C10_CUDA_CHECK(cudaMemcpy(
        &row64, first_failure_row.data_ptr<int64_t>(), sizeof(row64),
        cudaMemcpyDeviceToHost));
    TORCH_CHECK(row64 < static_cast<unsigned long long>(scales.size(0)),
                "footprint output failure did not record a valid row");
    const int64_t row = static_cast<int64_t>(row64);
    const int subtype = copy_device_array<int, 1>(
        failure_subtype.data_ptr<int>() + row)[0];
    const bool row_observed = copy_device_array<bool, 1>(
        observed.data_ptr<bool>() + row)[0];
    const bool row_escalated = copy_device_array<bool, 1>(
        escalated.data_ptr<bool>() + row)[0];
    const bool row_projected_only = copy_device_array<bool, 1>(
        projected_only_unsafe.data_ptr<bool>() + row)[0];
    const auto scales_row = copy_device_array<float, 3>(
        scales.data_ptr<float>() + row * 3);
    const auto quaternion_row = copy_device_array<float, 4>(
        quaternion.data_ptr<float>() + row * 4);
    const float density_row = copy_device_array<float, 1>(
        density.data_ptr<float>() + row)[0];
    const auto q6_row = copy_device_array<double, 6>(
        q6.data_ptr<double>() + row * 6);
    const auto jacobian_row = copy_device_array<float, 6>(
        jacobian.data_ptr<float>() + row * 6);
    const auto rotation_row = copy_device_array<float, 9>(
        rotation.data_ptr<float>() + row * 9);
    const auto eigenvalues_row_float = copy_device_array<float, 3>(
        eigenvalues.data_ptr<float>() + row * 3);
    const auto eigenvectors_row_float = copy_device_array<float, 9>(
        eigenvectors.data_ptr<float>() + row * 9);
    std::array<double, 3> eigenvalues_row = {
        eigenvalues_row_float[0], eigenvalues_row_float[1],
        eigenvalues_row_float[2]};
    std::array<double, 9> eigenvectors_row{};
    for (int index = 0; index < 9; ++index) {
        eigenvectors_row[index] = eigenvectors_row_float[index];
    }
    int64_t compact = -1;
    if (row_escalated && unsafe_indices.numel() > 0) {
        std::vector<int64_t> host_indices(
            static_cast<size_t>(unsafe_indices.numel()));
        C10_CUDA_CHECK(cudaMemcpy(
            host_indices.data(), unsafe_indices.data_ptr<int64_t>(),
            host_indices.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
        const auto found = std::lower_bound(
            host_indices.begin(), host_indices.end(), row);
        if (found != host_indices.end() && *found == row) {
            compact = static_cast<int64_t>(found - host_indices.begin());
            eigenvalues_row = copy_device_array<double, 3>(
                eigenvalues64.data_ptr<double>() + compact * 3);
            eigenvectors_row = copy_device_array<double, 9>(
                eigenvectors64.data_ptr<double>() + compact * 9);
        }
    }
    const HostProjectedTerms projected = host_projected_terms_analytic_p(
        scales_row, quaternion_row, jacobian_row);
    std::array<double, 3> scales_double = {
        scales_row[0], scales_row[1], scales_row[2]};
    std::array<double, 4> quaternion_double{};
    std::array<double, 6> q6_double{};
    std::array<double, 6> jacobian_double{};
    std::array<double, 9> rotation_double{};
    for (int index = 0; index < 4; ++index) {
        quaternion_double[index] = quaternion_row[index];
    }
    for (int index = 0; index < 6; ++index) {
        q6_double[index] = q6_row[index];
        jacobian_double[index] = jacobian_row[index];
    }
    for (int index = 0; index < 9; ++index) {
        rotation_double[index] = rotation_row[index];
    }
    std::ostringstream message;
    message << std::setprecision(17)
        << "footprint cuSOLVER full-shape forward rejected its state; status="
        << host_status << " FOOTPRINT_DIAGNOSTIC local_row=" << row
        << " subtype_code=" << subtype
        << " subtype=" << invalid_output_subtype_name(subtype)
        << " observed=" << static_cast<int>(row_observed)
        << " escalated=" << static_cast<int>(row_escalated)
        << " q_owner=stable_float64_q_analytic_projected_p"
        << " projected_only=" << static_cast<int>(row_projected_only)
        << " compact_row=" << compact
        << " scales=" << format_values(scales_double)
        << " quaternion=" << format_values(quaternion_double)
        << " quaternion_hex=" << format_float_hex(quaternion_row)
        << " q6=" << format_values(q6_double)
        << " q6_hex=" << format_double_hex(q6_row)
        << " jacobian=" << format_values(jacobian_double)
        << " jacobian_hex=" << format_float_hex(jacobian_row)
        << " rotation=" << format_values(rotation_double)
        << " rotation_hex=" << format_float_hex(rotation_row)
        << " density=" << static_cast<double>(density_row)
        << " eigenvalues=" << format_values(eigenvalues_row)
        << " projected_s0=" << format_values(projected.s0)
        << " projected_s1=" << format_values(projected.s1)
        << " projected_det=[" << projected.det0 << "," << projected.det1
        << "] projected_logdet=[" << projected.logdet0 << ","
        << projected.logdet1 << "] projected_coefficient="
        << projected.coefficient
        << " finite_flags={scales:"
        << static_cast<int>(std::isfinite(scales_double[0]) &&
            std::isfinite(scales_double[1]) && std::isfinite(scales_double[2]))
        << ",density:" << static_cast<int>(std::isfinite(density_row))
        << ",eigenvalues:"
        << static_cast<int>(std::isfinite(eigenvalues_row[0]) &&
            std::isfinite(eigenvalues_row[1]) &&
            std::isfinite(eigenvalues_row[2]))
        << ",projected_factors:" << static_cast<int>(projected.factors_finite)
        << ",projected_determinants:"
        << static_cast<int>(projected.determinants_positive_finite)
        << ",projected_logs:" << static_cast<int>(projected.logs_finite)
        << ",projected_coefficient:"
        << static_cast<int>(projected.coefficient_valid) << "}";
    TORCH_CHECK(false, message.str());
    return host_status;
}

}  // namespace

std::vector<torch::Tensor> footprint_full_shape_forward_cuda(
    const torch::Tensor& scales,
    const torch::Tensor& quaternion,
    const torch::Tensor& density,
    const torch::Tensor& q6,
    const torch::Tensor& jacobian,
    const torch::Tensor& observed,
    bool mass_preserving) {
    const c10::cuda::CUDAGuard guard(scales.device());
    const int64_t count = scales.size(0);
    auto covariance = torch::empty({count, 3, 3}, scales.options());
    auto factor = torch::empty_like(covariance);
    auto inverse_factor = torch::empty_like(covariance);
    auto compensation = torch::empty({count}, scales.options());
    auto filtered_density = torch::empty_like(density);
    auto root = torch::empty_like(covariance);
    auto rotation = torch::empty_like(covariance);
    auto eigenvectors = torch::empty_like(covariance);
    auto roots = torch::empty({count, 3}, scales.options());
    auto escalated = torch::zeros(
        {count}, scales.options().dtype(torch::kBool));
    auto projected_only_unsafe = torch::zeros_like(escalated);
    auto q_roundoff_projected = torch::zeros_like(escalated);
    auto failure_subtype = torch::zeros(
        {count}, scales.options().dtype(torch::kInt32));
    auto first_failure_row = torch::full(
        {1}, count, scales.options().dtype(torch::kInt64));
    auto unsafe_indices = torch::empty(
        {0}, scales.options().dtype(torch::kInt64));
    const auto double_options = scales.options().dtype(torch::kFloat64);
    auto covariance64 = torch::empty({0, 3, 3}, double_options);
    auto inverse_factor64 = torch::empty({0, 3, 3}, double_options);
    auto compensation64 = torch::empty({0}, double_options);
    auto root64 = torch::empty({0, 3, 3}, double_options);
    auto rotation64 = torch::empty({0, 3, 3}, double_options);
    auto eigenvectors64 = torch::empty({0, 3, 3}, double_options);
    auto eigenvalues64 = torch::empty({0, 3}, double_options);
    auto roots64 = torch::empty({0, 3}, double_options);
    auto projected_q64 = torch::empty({0, 3}, double_options);
    if (count == 0) {
        return {
            covariance,
            factor,
            inverse_factor,
            compensation,
            filtered_density,
            root,
            rotation,
            eigenvectors,
            roots,
            escalated,
            projected_only_unsafe,
            q_roundoff_projected,
            unsafe_indices,
            covariance64,
            inverse_factor64,
            compensation64,
            root64,
            rotation64,
            eigenvectors64,
            roots64,
            projected_q64};
    }

    auto eigenvalues = torch::empty({count, 3}, scales.options());
    auto solver_info = torch::empty(
        {count}, scales.options().dtype(torch::kInt32));
    auto status = torch::zeros({1}, scales.options().dtype(torch::kInt32));
    const auto current_stream = at::cuda::getCurrentCUDAStream();
    const cudaStream_t stream = current_stream.stream();
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    assemble_covariance_kernel<<<blocks, kThreads, 0, stream>>>(
        scales.data_ptr<float>(),
        quaternion.data_ptr<float>(),
        density.data_ptr<float>(),
        q6.data_ptr<double>(),
        observed.data_ptr<bool>(),
        count,
        covariance.data_ptr<float>(),
        eigenvectors.data_ptr<float>(),
        rotation.data_ptr<float>(),
        status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::unique_ptr<SolverHandle> local_solver;
    std::unique_ptr<SyevjParameters> local_syevj;
    SolverHandle* solver_holder = nullptr;
    SyevjParameters* syevj_holder = nullptr;
    if (g_persistent_solver) {
        const int device = scales.get_device();
        if (!g_solver || g_solver_device != device) {
            g_syevj.reset();
            g_solver.reset();
            g_solver = std::make_unique<SolverHandle>();
            g_syevj = std::make_unique<SyevjParameters>();
            g_solver_device = device;
        }
        solver_holder = g_solver.get();
        syevj_holder = g_syevj.get();
    } else {
        local_solver = std::make_unique<SolverHandle>();
        local_syevj = std::make_unique<SyevjParameters>();
        solver_holder = local_solver.get();
        syevj_holder = local_syevj.get();
    }
    const cusolverDnHandle_t solver = solver_holder->get();
    GOFISH_CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));
    const SyevjParameters& parameters = *syevj_holder;
    const int maximum_batch = static_cast<int>(
        std::min<int64_t>(count, kMaxSolverBatch));
    int workspace_elements = 0;
    GOFISH_CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(
        solver,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        kMatrixOrder,
        eigenvectors.data_ptr<float>(),
        kMatrixOrder,
        eigenvalues.data_ptr<float>(),
        &workspace_elements,
        parameters.get(),
        maximum_batch));
    auto workspace = torch::empty({workspace_elements}, scales.options());
    for (int64_t start = 0; start < count; start += kMaxSolverBatch) {
        const int batch = static_cast<int>(
            std::min<int64_t>(count - start, kMaxSolverBatch));
        GOFISH_CUSOLVER_CHECK(cusolverDnSsyevjBatched(
            solver,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            kMatrixOrder,
            eigenvectors.data_ptr<float>() + start * 9,
            kMatrixOrder,
            eigenvalues.data_ptr<float>() + start * 3,
            workspace.data_ptr<float>(),
            workspace_elements,
            solver_info.data_ptr<int>() + start,
            parameters.get(),
            batch));
    }

    classify_float32_rows_kernel<<<blocks, kThreads, 0, stream>>>(
        observed.data_ptr<bool>(), eigenvalues.data_ptr<float>(),
        solver_info.data_ptr<int>(), count, escalated.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // nonzero returns ascending, duplicate-free global row ownership. The
    // compact float64 branch scatters exclusively through these indices.
    unsafe_indices = torch::nonzero(escalated).reshape({-1}).contiguous();
    const int64_t unsafe_count = unsafe_indices.size(0);

    finish_full_shape_kernel<<<blocks, kThreads, 0, stream>>>(
        scales.data_ptr<float>(),
        density.data_ptr<float>(),
        observed.data_ptr<bool>(),
        escalated.data_ptr<bool>(),
        covariance.data_ptr<float>(),
        eigenvectors.data_ptr<float>(),
        eigenvalues.data_ptr<float>(),
        rotation.data_ptr<float>(),
        count,
        mass_preserving,
        root.data_ptr<float>(),
        factor.data_ptr<float>(),
        inverse_factor.data_ptr<float>(),
        compensation.data_ptr<float>(),
        filtered_density.data_ptr<float>(),
        roots.data_ptr<float>(),
        failure_subtype.data_ptr<int>(),
        reinterpret_cast<unsigned long long*>(
            first_failure_row.data_ptr<int64_t>()),
        status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (unsafe_count > 0) {
        const int unsafe_blocks = static_cast<int>(
            (unsafe_count + kThreads - 1) / kThreads);
        auto q_eigenvectors64 = torch::empty(
            {unsafe_count, 3, 3}, double_options);
        auto q_eigenvalues64 = torch::empty(
            {unsafe_count, 3}, double_options);
        auto q_solver_info = torch::empty(
            {unsafe_count}, scales.options().dtype(torch::kInt32));
        rotation64 = torch::empty({unsafe_count, 3, 3}, double_options);
        gather_unsafe_q_kernel<<<unsafe_blocks, kThreads, 0, stream>>>(
            unsafe_indices.data_ptr<int64_t>(),
            unsafe_count,
            quaternion.data_ptr<float>(),
            q6.data_ptr<double>(),
            q_eigenvectors64.data_ptr<double>(),
            rotation64.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int maximum_unsafe_batch = static_cast<int>(
            std::min<int64_t>(unsafe_count, kMaxSolverBatch));
        int q_workspace_elements = 0;
        GOFISH_CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
            solver,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            kMatrixOrder,
            q_eigenvectors64.data_ptr<double>(),
            kMatrixOrder,
            q_eigenvalues64.data_ptr<double>(),
            &q_workspace_elements,
            parameters.get(),
            maximum_unsafe_batch));
        auto q_workspace = torch::empty(
            {q_workspace_elements}, double_options);
        for (int64_t start = 0; start < unsafe_count;
             start += kMaxSolverBatch) {
            const int batch = static_cast<int>(
                std::min<int64_t>(
                    unsafe_count - start, kMaxSolverBatch));
            GOFISH_CUSOLVER_CHECK(cusolverDnDsyevjBatched(
                solver,
                CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,
                kMatrixOrder,
                q_eigenvectors64.data_ptr<double>() + start * 9,
                kMatrixOrder,
                q_eigenvalues64.data_ptr<double>() + start * 3,
                q_workspace.data_ptr<double>(),
                q_workspace_elements,
                q_solver_info.data_ptr<int>() + start,
                parameters.get(),
                batch));
        }

        covariance64 = torch::empty(
            {unsafe_count, 3, 3}, double_options);
        eigenvectors64 = torch::empty_like(covariance64);
        projected_q64 = torch::empty({unsafe_count, 3}, double_options);
        build_unsafe_covariance_kernel<<<
            unsafe_blocks, kThreads, 0, stream>>>(
                unsafe_indices.data_ptr<int64_t>(),
                unsafe_count,
                scales.data_ptr<float>(),
                q6.data_ptr<double>(),
                q_eigenvalues64.data_ptr<double>(),
                q_solver_info.data_ptr<int>(),
                rotation64.data_ptr<double>(),
                covariance64.data_ptr<double>(),
                eigenvectors64.data_ptr<double>(),
                projected_q64.data_ptr<double>(),
                q_roundoff_projected.data_ptr<bool>(),
                status.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        eigenvalues64 = torch::empty({unsafe_count, 3}, double_options);
        auto covariance_solver_info = torch::empty(
            {unsafe_count}, scales.options().dtype(torch::kInt32));
        int covariance_workspace_elements = 0;
        GOFISH_CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
            solver,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            kMatrixOrder,
            eigenvectors64.data_ptr<double>(),
            kMatrixOrder,
            eigenvalues64.data_ptr<double>(),
            &covariance_workspace_elements,
            parameters.get(),
            maximum_unsafe_batch));
        auto covariance_workspace = torch::empty(
            {covariance_workspace_elements}, double_options);
        for (int64_t start = 0; start < unsafe_count;
             start += kMaxSolverBatch) {
            const int batch = static_cast<int>(
                std::min<int64_t>(
                    unsafe_count - start, kMaxSolverBatch));
            GOFISH_CUSOLVER_CHECK(cusolverDnDsyevjBatched(
                solver,
                CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,
                kMatrixOrder,
                eigenvectors64.data_ptr<double>() + start * 9,
                kMatrixOrder,
                eigenvalues64.data_ptr<double>() + start * 3,
                covariance_workspace.data_ptr<double>(),
                covariance_workspace_elements,
                covariance_solver_info.data_ptr<int>() + start,
                parameters.get(),
                batch));
        }

        inverse_factor64 = torch::empty_like(covariance64);
        compensation64 = torch::empty({unsafe_count}, double_options);
        root64 = torch::empty_like(covariance64);
        roots64 = torch::empty({unsafe_count, 3}, double_options);
        finish_unsafe_full_shape_kernel<<<
            unsafe_blocks, kThreads, 0, stream>>>(
                unsafe_indices.data_ptr<int64_t>(),
                unsafe_count,
                scales.data_ptr<float>(),
                density.data_ptr<float>(),
                covariance64.data_ptr<double>(),
                eigenvectors64.data_ptr<double>(),
                eigenvalues64.data_ptr<double>(),
                covariance_solver_info.data_ptr<int>(),
                rotation64.data_ptr<double>(),
                jacobian.data_ptr<float>(),
                mass_preserving,
                covariance.data_ptr<float>(),
                root.data_ptr<float>(),
                factor.data_ptr<float>(),
                inverse_factor.data_ptr<float>(),
                compensation.data_ptr<float>(),
                filtered_density.data_ptr<float>(),
                roots.data_ptr<float>(),
                root64.data_ptr<double>(),
                inverse_factor64.data_ptr<double>(),
                compensation64.data_ptr<double>(),
                roots64.data_ptr<double>(),
                failure_subtype.data_ptr<int>(),
                reinterpret_cast<unsigned long long*>(
                    first_failure_row.data_ptr<int64_t>()),
                status.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (mass_preserving) {
        finish_all_observed_compensation_kernel<<<blocks, kThreads, 0, stream>>>(
            scales.data_ptr<float>(), quaternion.data_ptr<float>(),
            density.data_ptr<float>(), jacobian.data_ptr<float>(),
            observed.data_ptr<bool>(),
            escalated.data_ptr<bool>(),
            count, compensation.data_ptr<float>(),
            filtered_density.data_ptr<float>(), failure_subtype.data_ptr<int>(),
            reinterpret_cast<unsigned long long*>(
                first_failure_row.data_ptr<int64_t>()),
            status.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    synchronize_forward_status(
        status, failure_subtype, first_failure_row, scales, quaternion,
        density, q6, jacobian,
        observed, escalated, projected_only_unsafe, rotation, eigenvectors,
        eigenvalues, unsafe_indices, eigenvectors64, eigenvalues64, stream);
    return {
        covariance,
        factor,
        inverse_factor,
        compensation,
        filtered_density,
        root,
        rotation,
        eigenvectors,
        roots,
        escalated,
        projected_only_unsafe,
        q_roundoff_projected,
        unsafe_indices,
        covariance64,
        inverse_factor64,
        compensation64,
        root64,
        rotation64,
        eigenvectors64,
        roots64,
        projected_q64};
}

void footprint_set_persistent_solver_cuda(bool enabled) {
    g_persistent_solver = enabled;
    if (!enabled) {
        g_syevj.reset();
        g_solver.reset();
        g_solver_device = -1;
    }
}

std::vector<torch::Tensor> footprint_full_shape_backward_cuda(
    const torch::Tensor& scales,
    const torch::Tensor& quaternion,
    const torch::Tensor& density,
    const torch::Tensor& q6,
    const torch::Tensor& jacobian,
    const torch::Tensor& observed,
    const torch::Tensor& covariance,
    const torch::Tensor& inverse_factor,
    const torch::Tensor& compensation,
    const torch::Tensor& root,
    const torch::Tensor& rotation,
    const torch::Tensor& eigenvectors,
    const torch::Tensor& roots,
    const torch::Tensor& escalated,
    const torch::Tensor& projected_only_unsafe,
    const torch::Tensor& unsafe_indices,
    const torch::Tensor& covariance64,
    const torch::Tensor& inverse_factor64,
    const torch::Tensor& compensation64,
    const torch::Tensor& root64,
    const torch::Tensor& rotation64,
    const torch::Tensor& eigenvectors64,
    const torch::Tensor& roots64,
    const torch::Tensor& projected_q64,
    const torch::Tensor& grad_covariance,
    const torch::Tensor& grad_factor,
    const torch::Tensor& grad_inverse_factor,
    const torch::Tensor& grad_compensation,
    const torch::Tensor& grad_filtered_density,
    bool mass_preserving) {
    (void)projected_only_unsafe;
    (void)projected_q64;
    const c10::cuda::CUDAGuard guard(scales.device());
    const int64_t count = scales.size(0);
    auto grad_scales = torch::empty_like(scales);
    auto grad_quaternion = torch::empty_like(quaternion);
    auto grad_density = torch::empty_like(density);
    if (count == 0) {
        return {grad_scales, grad_quaternion, grad_density};
    }
    auto status = torch::zeros({1}, scales.options().dtype(torch::kInt32));
    const auto current_stream = at::cuda::getCurrentCUDAStream();
    const cudaStream_t stream = current_stream.stream();
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    full_shape_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        scales.data_ptr<float>(),
        quaternion.data_ptr<float>(),
        density.data_ptr<float>(),
        jacobian.data_ptr<float>(),
        observed.data_ptr<bool>(),
        escalated.data_ptr<bool>(),
        covariance.data_ptr<float>(),
        inverse_factor.data_ptr<float>(),
        compensation.data_ptr<float>(),
        root.data_ptr<float>(),
        rotation.data_ptr<float>(),
        eigenvectors.data_ptr<float>(),
        roots.data_ptr<float>(),
        grad_covariance.data_ptr<float>(),
        grad_factor.data_ptr<float>(),
        grad_inverse_factor.data_ptr<float>(),
        grad_compensation.data_ptr<float>(),
        grad_filtered_density.data_ptr<float>(),
        count,
        mass_preserving,
        grad_scales.data_ptr<float>(),
        grad_quaternion.data_ptr<float>(),
        grad_density.data_ptr<float>(),
        status.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t unsafe_count = unsafe_indices.size(0);
    if (unsafe_count > 0) {
        const int unsafe_blocks = static_cast<int>(
            (unsafe_count + kThreads - 1) / kThreads);
        unsafe_full_shape_backward_kernel<<<
            unsafe_blocks, kThreads, 0, stream>>>(
                unsafe_indices.data_ptr<int64_t>(),
                unsafe_count,
                scales.data_ptr<float>(),
                quaternion.data_ptr<float>(),
                density.data_ptr<float>(),
                jacobian.data_ptr<float>(),
                inverse_factor64.data_ptr<double>(),
                compensation.data_ptr<float>(),
                root64.data_ptr<double>(),
                rotation64.data_ptr<double>(),
                eigenvectors64.data_ptr<double>(),
                roots64.data_ptr<double>(),
                grad_covariance.data_ptr<float>(),
                grad_factor.data_ptr<float>(),
                grad_inverse_factor.data_ptr<float>(),
                grad_compensation.data_ptr<float>(),
                grad_filtered_density.data_ptr<float>(),
                mass_preserving,
                grad_scales.data_ptr<float>(),
                grad_quaternion.data_ptr<float>(),
                grad_density.data_ptr<float>(),
                status.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (mass_preserving) {
        all_observed_compensation_backward_kernel<<<
            blocks, kThreads, 0, stream>>>(
                scales.data_ptr<float>(), quaternion.data_ptr<float>(),
                density.data_ptr<float>(), jacobian.data_ptr<float>(),
                observed.data_ptr<bool>(),
                escalated.data_ptr<bool>(),
                grad_compensation.data_ptr<float>(),
                grad_filtered_density.data_ptr<float>(), count,
                grad_scales.data_ptr<float>(),
                grad_quaternion.data_ptr<float>(), status.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    synchronize_status(status, stream, "backward");
    return {grad_scales, grad_quaternion, grad_density};
}
