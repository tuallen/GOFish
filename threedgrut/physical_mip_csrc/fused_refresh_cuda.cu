// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

#include "camera_local_validity.cuh"

namespace {

constexpr double kFisheyeCenterEpsilon = 1.0e-12;
constexpr int32_t kSimplePinhole = 0;
constexpr int32_t kPinhole = 1;
constexpr int32_t kOpenCvFisheye = 2;
constexpr int32_t kSimpleRadial = 3;
constexpr int32_t kRadial = 4;
constexpr int32_t kOpenCv = 5;
constexpr int32_t kFullOpenCv = 6;
constexpr int32_t kPackedParameterCount = 12;

__device__ inline bool finite6(const double value[6]) {
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        if (!isfinite(value[i])) return false;
    }
    return true;
}

__global__ void best_rate_kernel(
    const double* __restrict__ positions,
    int64_t point_count,
    const double* __restrict__ rotations,
    const double* __restrict__ centers,
    const int32_t* __restrict__ frame_camera,
    const int32_t* __restrict__ frame_index,
    const int64_t* __restrict__ frame_rank,
    int64_t frame_count,
    const int32_t* __restrict__ camera_model,
    const int32_t* __restrict__ camera_width,
    const int32_t* __restrict__ camera_height,
    const double* __restrict__ camera_params,
    const double* __restrict__ camera_max_angle,
    const int32_t* __restrict__ row_lo,
    const int32_t* __restrict__ row_hi,
    int64_t row_stride,
    double* __restrict__ best_rate,
    int32_t* __restrict__ owner,
    uint8_t* __restrict__ observed,
    double* __restrict__ owner_direction) {
    const int64_t point = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= point_count) return;

    const double px = positions[3 * point + 0];
    const double py = positions[3 * point + 1];
    const double pz = positions[3 * point + 2];
    double winner = 0.0;
    int64_t winner_rank = LLONG_MAX;
    int32_t winner_frame = -1;
    double winner_dx = 0.0, winner_dy = 0.0, winner_dz = 0.0;

    // Frames are packed in camera-id/tie-rank order.  The explicit rank test
    // makes the result independent of that traversal order.
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        const int32_t camera = frame_camera[frame];
        const double dx = px - centers[3 * frame + 0];
        const double dy = py - centers[3 * frame + 1];
        const double dz = pz - centers[3 * frame + 2];
        const double* r = rotations + 9 * frame;
        // Reference: einsum("bpi,bij->bpj", delta, rotations).
        const double x = dx * r[0] + dy * r[3] + dz * r[6];
        const double y = dx * r[1] + dy * r[4] + dz * r[7];
        const double z = dx * r[2] + dy * r[5] + dz * r[8];
        const double distance = sqrt(x * x + y * y + z * z);
        const double rho = hypot(x, y);
        const double theta = atan2(rho, z);
        const double* params = camera_params + kPackedParameterCount * camera;
        double u = 0.0, v = 0.0;
        double jc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        bool model_valid = false;

        const int32_t model = camera_model[camera];
        if (model == kSimplePinhole || model == kPinhole) {
            const double fx = params[0];
            const double fy = params[1];
            const double cx = params[2];
            const double cy = params[3];
            const double safe_z = fabs(z) > DBL_MIN ? z : 1.0;
            u = fx * x / safe_z + cx;
            v = fy * y / safe_z + cy;
            jc[0] = fx / safe_z;
            jc[2] = -fx * x / (safe_z * safe_z);
            jc[4] = fy / safe_z;
            jc[5] = -fy * y / (safe_z * safe_z);
            model_valid = z > 0.0;
        } else if (model == kSimpleRadial || model == kRadial
                   || model == kOpenCv || model == kFullOpenCv) {
            const double fx = params[0], fy = params[1];
            const double cx = params[2], cy = params[3];
            const double k1 = params[4];
            const double k2 = model >= kRadial ? params[5] : 0.0;
            const double p1 = model >= kOpenCv ? params[6] : 0.0;
            const double p2 = model >= kOpenCv ? params[7] : 0.0;
            const double k3 = model == kFullOpenCv ? params[8] : 0.0;
            const double k4 = model == kFullOpenCv ? params[9] : 0.0;
            const double k5 = model == kFullOpenCv ? params[10] : 0.0;
            const double k6 = model == kFullOpenCv ? params[11] : 0.0;
            const double safe_z = fabs(z) > DBL_MIN ? z : 1.0;
            const double xn = x / safe_z, yn = y / safe_z;
            const double r2 = xn * xn + yn * yn;
            const double r4 = r2 * r2, r6 = r4 * r2;
            const double numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            const double denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
            const double safe_denominator = fabs(denominator) > DBL_MIN
                ? denominator : 1.0;
            const double radial = numerator / safe_denominator;
            const double radial_derivative = (
                (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4) * safe_denominator
                - numerator * (k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4))
                / (safe_denominator * safe_denominator);
            const double xd = xn * radial + 2.0 * p1 * xn * yn
                + p2 * (r2 + 2.0 * xn * xn);
            const double yd = yn * radial + p1 * (r2 + 2.0 * yn * yn)
                + 2.0 * p2 * xn * yn;
            u = fx * xd + cx;
            v = fy * yd + cy;

            const double dxd_x = radial + 2.0 * xn * xn * radial_derivative
                + 2.0 * p1 * yn + 6.0 * p2 * xn;
            const double dxd_y = 2.0 * xn * yn * radial_derivative
                + 2.0 * p1 * xn + 2.0 * p2 * yn;
            const double dyd_x = 2.0 * xn * yn * radial_derivative
                + 2.0 * p1 * xn + 2.0 * p2 * yn;
            const double dyd_y = radial + 2.0 * yn * yn * radial_derivative
                + 6.0 * p1 * yn + 2.0 * p2 * xn;
            const double inverse_z = 1.0 / safe_z;
            jc[0] = fx * dxd_x * inverse_z;
            jc[1] = fx * dxd_y * inverse_z;
            jc[2] = -fx * (dxd_x * xn + dxd_y * yn) * inverse_z;
            jc[3] = fy * dyd_x * inverse_z;
            jc[4] = fy * dyd_y * inverse_z;
            jc[5] = -fy * (dyd_x * xn + dyd_y * yn) * inverse_z;
            model_valid = z > 0.0 && fabs(denominator) > DBL_MIN;
        } else if (model == kOpenCvFisheye) {
            const double fx = params[0], fy = params[1];
            const double cx = params[2], cy = params[3];
            const double k1 = params[4], k2 = params[5];
            const double k3 = params[6], k4 = params[7];
            const double t2 = theta * theta;
            const double q = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
            const double qp = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
            const double safe_rho = fmax(rho, sqrt(DBL_MIN));
            const double safe_dist2 = fmax(distance * distance, DBL_MIN);
            const double cosine = x / safe_rho;
            const double sine = y / safe_rho;
            u = fx * q * cosine + cx;
            v = fy * q * sine + cy;
            const double gt[3] = {
                z * x / (safe_rho * safe_dist2),
                z * y / (safe_rho * safe_dist2),
                -rho / safe_dist2};
            const double gp[3] = {
                -y / (safe_rho * safe_rho),
                x / (safe_rho * safe_rho), 0.0};
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                jc[axis] = fx * (qp * cosine * gt[axis] - q * sine * gp[axis]);
                jc[3 + axis] = fy * (qp * sine * gt[axis] + q * cosine * gp[axis]);
            }
            const bool center = rho <= kFisheyeCenterEpsilon && z > 0.0;
            if (center) {
                u = cx; v = cy;
#pragma unroll
                for (int item = 0; item < 6; ++item) jc[item] = 0.0;
                jc[0] = fx / distance;
                jc[4] = fy / distance;
            }
            model_valid = (rho > kFisheyeCenterEpsilon || z > 0.0) && qp > 0.0;
        } else {
            continue;
        }

        // Camera-local near test, then the raster and calibrated-domain
        // tests on the projected centre.
        bool valid = gofish::cameraLocalNearValid(camera_model[camera], z, distance)
                  && model_valid && finite6(jc)
                  && isfinite(u) && isfinite(v);
        valid = valid && u >= 0.0 && u < camera_width[camera]
                      && v >= 0.0 && v < camera_height[camera];
        const double cap = camera_max_angle[camera];
        if (cap >= 0.0) valid = valid && theta < cap;
        if (valid) {
            const int64_t ix = static_cast<int64_t>(floor(u));
            const int64_t iy = static_cast<int64_t>(floor(v));
            valid = iy >= 0 && iy < camera_height[camera]
                 && ix >= row_lo[camera * row_stride + iy]
                 && ix < row_hi[camera * row_stride + iy];
        }
        if (!valid) continue;

        // Reference: J_world[i,k] = sum_j J_camera[i,j] * rotations[k,j].
        double jw0[3], jw1[3];
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            jw0[axis] = jc[0] * r[3 * axis + 0]
                      + jc[1] * r[3 * axis + 1]
                      + jc[2] * r[3 * axis + 2];
            jw1[axis] = jc[3] * r[3 * axis + 0]
                      + jc[4] * r[3 * axis + 1]
                      + jc[5] * r[3 * axis + 2];
        }
        const double a = jw0[0] * jw0[0] + jw0[1] * jw0[1] + jw0[2] * jw0[2];
        const double b = jw0[0] * jw1[0] + jw0[1] * jw1[1] + jw0[2] * jw1[2];
        const double c = jw1[0] * jw1[0] + jw1[1] * jw1[1] + jw1[2] * jw1[2];
        const double lambda_plus = 0.5 * (a + c + hypot(a - c, 2.0 * b));
        const double determinant = a * c - b * b;
        const double lambda_minus = determinant / lambda_plus;
        if (!isfinite(lambda_plus) || !isfinite(lambda_minus)
            || !(lambda_plus > 0.0) || !(lambda_minus > 0.0)) continue;
        const double rate = sqrt(lambda_minus);
        const int64_t rank = frame_rank[frame];
        if (rate > winner || (rate == winner && rank < winner_rank)) {
            winner = rate;
            winner_rank = rank;
            winner_frame = frame_index[frame];
            winner_dx = dx; winner_dy = dy; winner_dz = dz;
        }
    }

    best_rate[point] = winner;
    owner[point] = winner_frame;
    observed[point] = winner_frame >= 0 ? 1u : 0u;
    if (winner_frame >= 0) {
        const double norm = sqrt(winner_dx * winner_dx + winner_dy * winner_dy
                               + winner_dz * winner_dz);
        owner_direction[3 * point + 0] = winner_dx / norm;
        owner_direction[3 * point + 1] = winner_dy / norm;
        owner_direction[3 * point + 2] = winner_dz / norm;
    }
}

}  // namespace

std::vector<torch::Tensor> mip3d_fused_best_rate_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& rotations,
    const torch::Tensor& centers,
    const torch::Tensor& frame_camera,
    const torch::Tensor& frame_index,
    const torch::Tensor& frame_rank,
    const torch::Tensor& camera_model,
    const torch::Tensor& camera_width,
    const torch::Tensor& camera_height,
    const torch::Tensor& camera_params,
    const torch::Tensor& camera_max_angle,
    const torch::Tensor& row_lo,
    const torch::Tensor& row_hi,
    bool canonical_pixel_chart) {
    TORCH_CHECK(
        canonical_pixel_chart,
        "physical Mip requires the COLMAP corner-coordinate pixel chart");
    const c10::cuda::CUDAGuard device_guard(positions.device());
    const auto count = positions.size(0);
    auto best = torch::zeros({count}, positions.options());
    auto owner = torch::full({count}, -1, positions.options().dtype(torch::kInt32));
    auto observed = torch::zeros({count}, positions.options().dtype(torch::kUInt8));
    auto direction = torch::zeros({count, 3}, positions.options());
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    best_rate_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        positions.data_ptr<double>(), count,
        rotations.data_ptr<double>(), centers.data_ptr<double>(),
        frame_camera.data_ptr<int32_t>(), frame_index.data_ptr<int32_t>(),
        frame_rank.data_ptr<int64_t>(), rotations.size(0),
        camera_model.data_ptr<int32_t>(), camera_width.data_ptr<int32_t>(),
        camera_height.data_ptr<int32_t>(), camera_params.data_ptr<double>(),
        camera_max_angle.data_ptr<double>(), row_lo.data_ptr<int32_t>(),
        row_hi.data_ptr<int32_t>(), row_lo.size(1),
        best.data_ptr<double>(), owner.data_ptr<int32_t>(),
        observed.data_ptr<uint8_t>(), direction.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {best, owner, observed, direction};
}
