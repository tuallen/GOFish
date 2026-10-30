// SPDX-License-Identifier: Apache-2.0
//
// GOFish (Inference, approximate shape): one fused kernel per frame that forms
// the per-view full-shape factors C = Sigma + Q_v, a square-root factor B with
// B B^T = C, B^{-1}, and the mass-preserving opacity m, and writes them into
// the renderer's packed particle buffer.
//
// It evaluates the same quantities as the reference operators
// (view_observation_filter.current_view_q6 + the footprint full-shape operator) in
// float64 per row, with two deliberate differences:
//   * B is the Cholesky factor of C instead of C^{1/2} R.  The rendered
//     response depends on B only through B B^T = C; the UT sigma points (used
//     by approximate admission) rotate.
//   * no cuSOLVER Jacobi, no float64 escalation path, no host synchronization;
//     a row whose C is not numerically SPD falls back to the stored shape.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace {

constexpr int kPacked = 32;  // position 3, density 1, rotation 4, scale 3, intrinsic density 1, B 9, B^-1 9, pad 2

struct Camera {
    double R[9];      // camera-to-world rotation, row-major
    double c[3];      // camera centre
    double fx, fy;
    double k[4];      // Kannala-Brandt radial coefficients (fisheye)
    double maxAngle;
    int fisheye;      // 1: OPENCV_FISHEYE, 0: ideal pinhole
};

__device__ __forceinline__ void quaternionRotation(const float* q, double* r) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1.0 - 2.0 * (y * y + z * z); r[1] = 2.0 * (x * y - z * w);       r[2] = 2.0 * (x * z + y * w);
    r[3] = 2.0 * (x * y + z * w);       r[4] = 1.0 - 2.0 * (x * x + z * z); r[5] = 2.0 * (y * z - x * w);
    r[6] = 2.0 * (x * z - y * w);       r[7] = 2.0 * (y * z + x * w);       r[8] = 1.0 - 2.0 * (x * x + y * y);
}

// Camera Jacobian d(pixel)/d(world) (2x3, row-major) and the chart validity of
// current_view_q6.  J is rounded to float32 as in the reference.
__device__ bool viewJacobian(const Camera& cam, const float* pos, double* J) {
    double d[3] = {pos[0] - cam.c[0], pos[1] - cam.c[1], pos[2] - cam.c[2]};
    double p[3];
    for (int j = 0; j < 3; ++j) p[j] = d[0] * cam.R[0 * 3 + j] + d[1] * cam.R[1 * 3 + j] + d[2] * cam.R[2 * 3 + j];
    const double x = p[0], y = p[1], z = p[2];
    double Jc[6];
    if (cam.fisheye) {
        const double distance = sqrt(x * x + y * y + z * z);
        const double rho = hypot(x, y);
        const double theta = atan2(rho, z);
        const double t2 = theta * theta;
        const double rt = theta * (1.0 + t2 * (cam.k[0] + t2 * (cam.k[1] + t2 * (cam.k[2] + t2 * cam.k[3]))));
        const double rd = 1.0 + t2 * (3.0 * cam.k[0] + t2 * (5.0 * cam.k[1] + t2 * (7.0 * cam.k[2] + t2 * 9.0 * cam.k[3])));
        if (!(distance > 0.2 && (rho > 1.0e-12 || z > 0.0) && theta < cam.maxAngle && rd > 0.0)) return false;
        if (rho <= 1.0e-12 && z > 0.0) {
            Jc[0] = cam.fx / distance; Jc[1] = 0.0; Jc[2] = 0.0;
            Jc[3] = 0.0; Jc[4] = cam.fy / distance; Jc[5] = 0.0;
        } else {
            const double d2 = distance * distance;
            const double cp = x / rho, sp = y / rho;
            const double gt[3] = {z * x / (rho * d2), z * y / (rho * d2), -rho / d2};
            const double gp[3] = {-y / (rho * rho), x / (rho * rho), 0.0};
            for (int a = 0; a < 3; ++a) {
                Jc[a] = cam.fx * (rd * cp * gt[a] - rt * sp * gp[a]);
                Jc[3 + a] = cam.fy * (rd * sp * gt[a] + rt * cp * gp[a]);
            }
        }
    } else {
        if (!(z > 0.2)) return false;
        Jc[0] = cam.fx / z; Jc[1] = 0.0; Jc[2] = -cam.fx * x / (z * z);
        Jc[3] = 0.0; Jc[4] = cam.fy / z; Jc[5] = -cam.fy * y / (z * z);
    }
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k) {
            double v = 0.0;
            for (int j = 0; j < 3; ++j) v += Jc[i * 3 + j] * cam.R[k * 3 + j];
            const float f = static_cast<float>(v);
            if (!isfinite(f)) return false;
            J[i * 3 + k] = static_cast<double>(f);
        }
    return true;
}

// Q = gamma J^+ J^+^T by column-pivoted modified Gram-Schmidt of J^T (as the reference).
__device__ bool pullbackQ(const double* J, double gamma, double* Q) {
    double a[3] = {J[0], J[1], J[2]}, b[3] = {J[3], J[4], J[5]};
    const double na = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], nb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    const double* lead = nb > na ? b : a;
    const double* trail = nb > na ? a : b;
    const double p0 = sqrt(lead[0] * lead[0] + lead[1] * lead[1] + lead[2] * lead[2]);
    if (!(isfinite(p0) && p0 > 0.0)) return false;
    double u0[3], r[3];
    for (int i = 0; i < 3; ++i) u0[i] = lead[i] / p0;
    double coef = u0[0] * trail[0] + u0[1] * trail[1] + u0[2] * trail[2];
    for (int i = 0; i < 3; ++i) r[i] = trail[i] - coef * u0[i];
    const double corr = u0[0] * r[0] + u0[1] * r[1] + u0[2] * r[2];
    for (int i = 0; i < 3; ++i) r[i] -= corr * u0[i];
    coef += corr;
    const double p1 = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (!(isfinite(p1) && p1 > 0.0 && isfinite(coef))) return false;
    double u1[3], c0[3], c1[3];
    for (int i = 0; i < 3; ++i) u1[i] = r[i] / p1;
    const double i00 = 1.0 / p0, i10 = -coef / (p0 * p1), i11 = 1.0 / p1;
    for (int i = 0; i < 3; ++i) { c0[i] = u0[i] * i00 + u1[i] * i10; c1[i] = u1[i] * i11; }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Q[i * 3 + j] = gamma * (c0[i] * c0[j] + c1[i] * c1[j]);
            if (!isfinite(Q[i * 3 + j])) return false;
        }
    return true;
}

// m = sqrt(det P0 / det(P0 + gamma I)), P0 = (J R S)(J R S)^T (reference analytic form).
__device__ bool compensation(const double* J, const double* R, const double* s, double gamma, double* m) {
    double f[6];
    for (int px = 0; px < 2; ++px)
        for (int ax = 0; ax < 3; ++ax) {
            double v = 0.0;
            for (int w = 0; w < 3; ++w) v += J[px * 3 + w] * R[w * 3 + ax] * s[ax];
            f[px * 3 + ax] = v;
        }
    const double m0 = f[1] * f[5] - f[2] * f[4], m1 = f[2] * f[3] - f[0] * f[5], m2 = f[0] * f[4] - f[1] * f[3];
    const double det0 = m0 * m0 + m1 * m1 + m2 * m2;
    if (!(isfinite(det0) && det0 > 0.0)) return false;
    const double sxx = f[0] * f[0] + f[1] * f[1] + f[2] * f[2], syy = f[3] * f[3] + f[4] * f[4] + f[5] * f[5];
    const double det1 = det0 + gamma * (sxx + syy) + gamma * gamma;
    if (!(isfinite(det1) && det1 > 0.0)) return false;
    *m = exp(0.5 * (log(det0) - log(det1)));
    return isfinite(*m);
}

__global__ void fastShapeKernel(int64_t n, const float* __restrict__ positions, const float* __restrict__ quaternions,
                                const float* __restrict__ scales, const float* __restrict__ density, Camera cam,
                                double gamma, float* __restrict__ packed) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* q = quaternions + row * 4;
    double R[9];
    quaternionRotation(q, R);
    const double s[3] = {scales[row * 3 + 0], scales[row * 3 + 1], scales[row * 3 + 2]};
    float* out = packed + row * kPacked;
    double B[9], Bi[9];
    double m = 1.0;
    bool shaped = false;
    double J[6], Q[9];
    if (gamma > 0.0 && viewJacobian(cam, positions + row * 3, J) && pullbackQ(J, gamma, Q)
        && compensation(J, R, s, gamma, &m)) {
        double C[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double v = Q[i * 3 + j];
                for (int k = 0; k < 3; ++k) v += R[i * 3 + k] * s[k] * s[k] * R[j * 3 + k];
                C[i * 3 + j] = v;
            }
        // Cholesky C = L L^T.
        const double l00 = C[0] > 0.0 ? sqrt(C[0]) : -1.0;
        if (l00 > 0.0) {
            const double l10 = C[3] / l00, l20 = C[6] / l00;
            const double t11 = C[4] - l10 * l10;
            const double l11 = t11 > 0.0 ? sqrt(t11) : -1.0;
            if (l11 > 0.0) {
                const double l21 = (C[7] - l20 * l10) / l11;
                const double t22 = C[8] - l20 * l20 - l21 * l21;
                const double l22 = t22 > 0.0 ? sqrt(t22) : -1.0;
                if (l22 > 0.0) {
                    B[0] = l00; B[1] = 0.0; B[2] = 0.0;
                    B[3] = l10; B[4] = l11; B[5] = 0.0;
                    B[6] = l20; B[7] = l21; B[8] = l22;
                    const double a00 = 1.0 / l00, a11 = 1.0 / l11, a22 = 1.0 / l22;
                    const double a10 = -l10 * a00 * a11;
                    const double a21 = -l21 * a11 * a22;
                    const double a20 = -(l20 * a00 + l21 * a10) * a22;
                    Bi[0] = a00; Bi[1] = 0.0; Bi[2] = 0.0;
                    Bi[3] = a10; Bi[4] = a11; Bi[5] = 0.0;
                    Bi[6] = a20; Bi[7] = a21; Bi[8] = a22;
                    shaped = true;
                }
            }
        }
    }
    if (!shaped) {  // stored shape: B = R S, B^-1 = S^-1 R^T, m = 1
        m = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                B[i * 3 + j] = R[i * 3 + j] * s[j];
                Bi[i * 3 + j] = R[j * 3 + i] / s[i];
            }
    }
    out[3] = static_cast<float>(static_cast<double>(density[row]) * m);
    for (int i = 0; i < 9; ++i) {
        out[12 + i] = static_cast<float>(B[i]);
        out[21 + i] = static_cast<float>(Bi[i]);
    }
}

__device__ void shapeRowF64(int64_t row, const float* __restrict__ positions, const float* __restrict__ quaternions,
                            const float* __restrict__ scales, const float* __restrict__ density, const Camera& cam,
                            double gamma, float* __restrict__ packed) {
    const float* q = quaternions + row * 4;
    double R[9];
    quaternionRotation(q, R);
    const double s[3] = {scales[row * 3 + 0], scales[row * 3 + 1], scales[row * 3 + 2]};
    float* out = packed + row * kPacked;
    double B[9], Bi[9];
    double m = 1.0;
    bool shaped = false;
    double J[6], Q[9];
    if (gamma > 0.0 && viewJacobian(cam, positions + row * 3, J) && pullbackQ(J, gamma, Q)
        && compensation(J, R, s, gamma, &m)) {
        double C[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double v = Q[i * 3 + j];
                for (int k = 0; k < 3; ++k) v += R[i * 3 + k] * s[k] * s[k] * R[j * 3 + k];
                C[i * 3 + j] = v;
            }
        // Cholesky C = L L^T.
        const double l00 = C[0] > 0.0 ? sqrt(C[0]) : -1.0;
        if (l00 > 0.0) {
            const double l10 = C[3] / l00, l20 = C[6] / l00;
            const double t11 = C[4] - l10 * l10;
            const double l11 = t11 > 0.0 ? sqrt(t11) : -1.0;
            if (l11 > 0.0) {
                const double l21 = (C[7] - l20 * l10) / l11;
                const double t22 = C[8] - l20 * l20 - l21 * l21;
                const double l22 = t22 > 0.0 ? sqrt(t22) : -1.0;
                if (l22 > 0.0) {
                    B[0] = l00; B[1] = 0.0; B[2] = 0.0;
                    B[3] = l10; B[4] = l11; B[5] = 0.0;
                    B[6] = l20; B[7] = l21; B[8] = l22;
                    const double a00 = 1.0 / l00, a11 = 1.0 / l11, a22 = 1.0 / l22;
                    const double a10 = -l10 * a00 * a11;
                    const double a21 = -l21 * a11 * a22;
                    const double a20 = -(l20 * a00 + l21 * a10) * a22;
                    Bi[0] = a00; Bi[1] = 0.0; Bi[2] = 0.0;
                    Bi[3] = a10; Bi[4] = a11; Bi[5] = 0.0;
                    Bi[6] = a20; Bi[7] = a21; Bi[8] = a22;
                    shaped = true;
                }
            }
        }
    }
    if (!shaped) {  // stored shape: B = R S, B^-1 = S^-1 R^T, m = 1
        m = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                B[i * 3 + j] = R[i * 3 + j] * s[j];
                Bi[i * 3 + j] = R[j * 3 + i] / s[i];
            }
    }
    out[3] = static_cast<float>(static_cast<double>(density[row]) * m);
    for (int i = 0; i < 9; ++i) {
        out[12 + i] = static_cast<float>(B[i]);
        out[21 + i] = static_cast<float>(Bi[i]);
    }
}


// ---- real-time renderer: float32 shape with float64 retry -------------------------------
// The same quantities in float32.  A row is recomputed by shapeRowF64 (the float64 path above) whenever the
// float32 evaluation is not trustworthy: Jacobian/Q/m invalid, or a Cholesky pivot below kRel of its
// diagonal (cancellation), so ill-conditioned rim / surfel rows keep float64 factors.
constexpr float kRel = 1.0e-2f;

struct CameraF {  // float32 copy of Camera (no float64 conversions in the float32 kernel)
    float R[9], c[3], fx, fy, k[4], maxAngle;
    int fisheye;
};


__device__ int viewJacobianF(const CameraF& cam, const float* pos, float* J) {
    float d[3] = {pos[0] - cam.c[0], pos[1] - cam.c[1], pos[2] - cam.c[2]};
    float p[3];
    for (int j = 0; j < 3; ++j) p[j] = d[0] * cam.R[0 * 3 + j] + d[1] * cam.R[1 * 3 + j] + d[2] * cam.R[2 * 3 + j];
    const float x = p[0], y = p[1], z = p[2];
    float Jc[6];
    if (cam.fisheye) {
        const float distance = sqrtf(x * x + y * y + z * z);
        const float rho = hypotf(x, y);
        const float theta = atan2f(rho, z);
        const float t2 = theta * theta;
        const float k0 = cam.k[0], k1 = cam.k[1], k2 = cam.k[2], k3 = cam.k[3];
        const float rt = theta * (1.f + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * k3))));
        const float rd = 1.f + t2 * (3.f * k0 + t2 * (5.f * k1 + t2 * (7.f * k2 + t2 * 9.f * k3)));
        // clearly outside the float64 chart (distance <= 0.2, theta >= maxAngle, rd <= 0 with margins): -1;
        // near any boundary: 0, decided by the float64 row
        if (distance < 0.19f || theta > cam.maxAngle + 1.0e-4f || rd < -1.0e-3f) return -1;
        if (!(distance > 0.21f && rho > 1.0e-6f * distance && theta < cam.maxAngle - 1.0e-4f && rd > 1.0e-3f)) return 0;
        const float d2 = distance * distance;
        const float cp = x / rho, sp = y / rho;
        const float gt[3] = {z * x / (rho * d2), z * y / (rho * d2), -rho / d2};
        const float gp[3] = {-y / (rho * rho), x / (rho * rho), 0.f};
        const float fx = cam.fx, fy = cam.fy;
        for (int a = 0; a < 3; ++a) {
            Jc[a] = fx * (rd * cp * gt[a] - rt * sp * gp[a]);
            Jc[3 + a] = fy * (rd * sp * gt[a] + rt * cp * gp[a]);
        }
    } else {
        if (z < 0.19f) return -1;
        if (!(z > 0.21f)) return 0;
        const float fx = cam.fx, fy = cam.fy;
        Jc[0] = fx / z; Jc[1] = 0.f; Jc[2] = -fx * x / (z * z);
        Jc[3] = 0.f; Jc[4] = fy / z; Jc[5] = -fy * y / (z * z);
    }
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k) {
            float v = 0.f;
            for (int j = 0; j < 3; ++j) v += Jc[i * 3 + j] * cam.R[k * 3 + j];
            if (!isfinite(v)) return 0;
            J[i * 3 + k] = v;
        }
    return 1;
}

__device__ bool pullbackQF(const float* J, float gamma, float* Q) {
    float a[3] = {J[0], J[1], J[2]}, b[3] = {J[3], J[4], J[5]};
    const float na = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], nb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    const float* lead = nb > na ? b : a;
    const float* trail = nb > na ? a : b;
    const float p0 = sqrtf(lead[0] * lead[0] + lead[1] * lead[1] + lead[2] * lead[2]);
    if (!(isfinite(p0) && p0 > 0.f)) return false;
    float u0[3], r[3];
    for (int i = 0; i < 3; ++i) u0[i] = lead[i] / p0;
    float coef = u0[0] * trail[0] + u0[1] * trail[1] + u0[2] * trail[2];
    for (int i = 0; i < 3; ++i) r[i] = trail[i] - coef * u0[i];
    const float corr = u0[0] * r[0] + u0[1] * r[1] + u0[2] * r[2];
    for (int i = 0; i < 3; ++i) r[i] -= corr * u0[i];
    coef += corr;
    const float p1 = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    // near-parallel rows of J lose digits in float32: retry in float64
    if (!(isfinite(p1) && p1 > kRel * p0 && isfinite(coef))) return false;
    float u1[3], c0[3], c1[3];
    for (int i = 0; i < 3; ++i) u1[i] = r[i] / p1;
    const float i00 = 1.f / p0, i10 = -coef / (p0 * p1), i11 = 1.f / p1;
    for (int i = 0; i < 3; ++i) { c0[i] = u0[i] * i00 + u1[i] * i10; c1[i] = u1[i] * i11; }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Q[i * 3 + j] = gamma * (c0[i] * c0[j] + c1[i] * c1[j]);
            if (!isfinite(Q[i * 3 + j])) return false;
        }
    return true;
}

__device__ bool compensationF(const float* J, const float* R, const float* s, float gamma, float* m) {
    float f[6];
    for (int px = 0; px < 2; ++px)
        for (int ax = 0; ax < 3; ++ax) {
            float v = 0.f;
            for (int w = 0; w < 3; ++w) v += J[px * 3 + w] * R[w * 3 + ax] * s[ax];
            f[px * 3 + ax] = v;
        }
    const float m0 = f[1] * f[5] - f[2] * f[4], m1 = f[2] * f[3] - f[0] * f[5], m2 = f[0] * f[4] - f[1] * f[3];
    const float det0 = m0 * m0 + m1 * m1 + m2 * m2;
    const float sxx = f[0] * f[0] + f[1] * f[1] + f[2] * f[2], syy = f[3] * f[3] + f[4] * f[4] + f[5] * f[5];
    // cross-product cancellation (nearly rank-1 projected footprint): retry in float64
    if (!(isfinite(det0) && det0 > 1.0e-4f * sxx * syy && det0 > 1.0e-30f)) return false;
    const float det1 = det0 + gamma * (sxx + syy) + gamma * gamma;
    if (!(isfinite(det1) && det1 > 0.f)) return false;
    *m = expf(0.5f * (logf(det0) - logf(det1)));
    return isfinite(*m);
}

__global__ void fastShapeKernel32(int64_t n, const float* __restrict__ positions, const float* __restrict__ quaternions,
                                  const float* __restrict__ scales, const float* __restrict__ density, Camera cam,
                                  double gamma, CameraF camf, float g, float* __restrict__ packed, unsigned int* __restrict__ retries,
                                  unsigned int* __restrict__ retryRows) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const float* qq = quaternions + row * 4;
    float R[9];
    {
        const float w = qq[0], x = qq[1], y = qq[2], z = qq[3];
        R[0] = 1.f - 2.f * (y * y + z * z); R[1] = 2.f * (x * y - z * w);       R[2] = 2.f * (x * z + y * w);
        R[3] = 2.f * (x * y + z * w);       R[4] = 1.f - 2.f * (x * x + z * z); R[5] = 2.f * (y * z - x * w);
        R[6] = 2.f * (x * z - y * w);       R[7] = 2.f * (y * z + x * w);       R[8] = 1.f - 2.f * (x * x + y * y);
    }
    const float s[3] = {scales[row * 3 + 0], scales[row * 3 + 1], scales[row * 3 + 2]};
    float J[6], Q[9], m = 1.f;
    const int chart = g > 0.f ? viewJacobianF(camf, positions + row * 3, J) : -1;
    float B[9], Bi[9];
    if (chart < 0) {  // clearly no view chart (or no footprint): stored shape, as the float64 path
        float* o = packed + row * kPacked;
        o[3] = density[row];
        float Bs[9], Bis[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) { Bs[i * 3 + j] = R[i * 3 + j] * s[j]; Bis[i * 3 + j] = R[j * 3 + i] / s[i]; }
        float4* o4 = reinterpret_cast<float4*>(o + 12);
        o4[0] = make_float4(Bs[0], Bs[1], Bs[2], Bs[3]);
        o4[1] = make_float4(Bs[4], Bs[5], Bs[6], Bs[7]);
        o4[2] = make_float4(Bs[8], Bis[0], Bis[1], Bis[2]);
        o4[3] = make_float4(Bis[3], Bis[4], Bis[5], Bis[6]);
        *reinterpret_cast<float2*>(o + 28) = make_float2(Bis[7], Bis[8]);
        return;
    }
    bool ok = chart > 0 && pullbackQF(J, g, Q) && compensationF(J, R, s, g, &m);
    if (ok) {
        float C[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                float v = Q[i * 3 + j];
                for (int k = 0; k < 3; ++k) v += R[i * 3 + k] * s[k] * s[k] * R[j * 3 + k];
                C[i * 3 + j] = v;
            }
        ok = false;
        if (C[0] > 0.f) {
            const float l00 = sqrtf(C[0]);
            const float l10 = C[3] / l00, l20 = C[6] / l00;
            const float t11 = C[4] - l10 * l10;
            if (t11 > kRel * C[4]) {
                const float l11 = sqrtf(t11);
                const float l21 = (C[7] - l20 * l10) / l11;
                const float t22 = C[8] - l20 * l20 - l21 * l21;
                if (t22 > kRel * C[8]) {
                    const float l22 = sqrtf(t22);
                    B[0] = l00; B[1] = 0.f; B[2] = 0.f;
                    B[3] = l10; B[4] = l11; B[5] = 0.f;
                    B[6] = l20; B[7] = l21; B[8] = l22;
                    const float a00 = 1.f / l00, a11 = 1.f / l11, a22 = 1.f / l22;
                    const float a10 = -l10 * a00 * a11;
                    const float a21 = -l21 * a11 * a22;
                    const float a20 = -(l20 * a00 + l21 * a10) * a22;
                    Bi[0] = a00; Bi[1] = 0.f; Bi[2] = 0.f;
                    Bi[3] = a10; Bi[4] = a11; Bi[5] = 0.f;
                    Bi[6] = a20; Bi[7] = a21; Bi[8] = a22;
                    ok = true;
                }
            }
        }
    }
    if (!ok) {  // invalid view chart, stored-shape fallback, or ill-conditioned: deferred to the float64 pass
        const unsigned int slot = atomicAdd(retries, 1u);
        retryRows[slot] = static_cast<unsigned int>(row);
        return;
    }
    float* out = packed + row * kPacked;
    out[3] = density[row] * m;
    // slots 12..29 = B (9) then B^-1 (9): four float4 + one float2 stores (row base is 128-byte aligned)
    float4* o4 = reinterpret_cast<float4*>(out + 12);
    o4[0] = make_float4(B[0], B[1], B[2], B[3]);
    o4[1] = make_float4(B[4], B[5], B[6], B[7]);
    o4[2] = make_float4(B[8], Bi[0], Bi[1], Bi[2]);
    o4[3] = make_float4(Bi[3], Bi[4], Bi[5], Bi[6]);
    *reinterpret_cast<float2*>(out + 28) = make_float2(Bi[7], Bi[8]);
}


// Second pass: the deferred rows through the float64 path (grid-stride over the device-side count).
__global__ void fastShapeRetryKernel(const unsigned int* __restrict__ retries, const unsigned int* __restrict__ retryRows,
                                     const float* __restrict__ positions, const float* __restrict__ quaternions,
                                     const float* __restrict__ scales, const float* __restrict__ density, Camera cam,
                                     double gamma, float* __restrict__ packed) {
    const unsigned int count = *retries;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        shapeRowF64(int64_t(retryRows[i]), positions, quaternions, scales, density, cam, gamma, packed);
}

// Mixed precision: the view Jacobian (transcendentals) in float32, everything after it in float64 exactly as
// fastShapeKernel (the reference also rounds J to float32); borderline chart validity is decided by the full
// float64 row.
__global__ void fastShapeKernelMixed(int64_t n, const float* __restrict__ positions, const float* __restrict__ quaternions,
                                     const float* __restrict__ scales, const float* __restrict__ density, Camera cam,
                                     double gamma, CameraF camf, float* __restrict__ packed, unsigned int* __restrict__ retries) {
    const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float Jf[6];
    if (!(gamma > 0.0 && viewJacobianF(camf, positions + row * 3, Jf) > 0)) {
        shapeRowF64(row, positions, quaternions, scales, density, cam, gamma, packed);
        if (retries) atomicAdd(retries, 1u);
        return;
    }
    const float* q = quaternions + row * 4;
    double R[9];
    quaternionRotation(q, R);
    const double s[3] = {scales[row * 3 + 0], scales[row * 3 + 1], scales[row * 3 + 2]};
    float* out = packed + row * kPacked;
    double B[9], Bi[9];
    double m = 1.0;
    bool shaped = false;
    double J[6], Q[9];
    for (int i = 0; i < 6; ++i) J[i] = static_cast<double>(Jf[i]);
    if (pullbackQ(J, gamma, Q) && compensation(J, R, s, gamma, &m)) {
        double C[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double v = Q[i * 3 + j];
                for (int k = 0; k < 3; ++k) v += R[i * 3 + k] * s[k] * s[k] * R[j * 3 + k];
                C[i * 3 + j] = v;
            }
        const double l00 = C[0] > 0.0 ? sqrt(C[0]) : -1.0;
        if (l00 > 0.0) {
            const double l10 = C[3] / l00, l20 = C[6] / l00;
            const double t11 = C[4] - l10 * l10;
            const double l11 = t11 > 0.0 ? sqrt(t11) : -1.0;
            if (l11 > 0.0) {
                const double l21 = (C[7] - l20 * l10) / l11;
                const double t22 = C[8] - l20 * l20 - l21 * l21;
                const double l22 = t22 > 0.0 ? sqrt(t22) : -1.0;
                if (l22 > 0.0) {
                    B[0] = l00; B[1] = 0.0; B[2] = 0.0;
                    B[3] = l10; B[4] = l11; B[5] = 0.0;
                    B[6] = l20; B[7] = l21; B[8] = l22;
                    const double a00 = 1.0 / l00, a11 = 1.0 / l11, a22 = 1.0 / l22;
                    const double a10 = -l10 * a00 * a11;
                    const double a21 = -l21 * a11 * a22;
                    const double a20 = -(l20 * a00 + l21 * a10) * a22;
                    Bi[0] = a00; Bi[1] = 0.0; Bi[2] = 0.0;
                    Bi[3] = a10; Bi[4] = a11; Bi[5] = 0.0;
                    Bi[6] = a20; Bi[7] = a21; Bi[8] = a22;
                    shaped = true;
                }
            }
        }
    }
    if (!shaped) {
        m = 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                B[i * 3 + j] = R[i * 3 + j] * s[j];
                Bi[i * 3 + j] = R[j * 3 + i] / s[i];
            }
    }
    out[3] = static_cast<float>(static_cast<double>(density[row]) * m);
    for (int i = 0; i < 9; ++i) {
        out[12 + i] = static_cast<float>(B[i]);
        out[21 + i] = static_cast<float>(Bi[i]);
    }
}

}  // namespace

void fast_shape(torch::Tensor positions, torch::Tensor quaternions, torch::Tensor scales, torch::Tensor density,
                torch::Tensor camera_to_world, std::vector<double> intrinsics, bool fisheye, double gamma,
                torch::Tensor packed) {
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kFloat32 && positions.is_contiguous());
    TORCH_CHECK(quaternions.is_contiguous() && scales.is_contiguous() && density.is_contiguous());
    TORCH_CHECK(packed.is_contiguous() && packed.size(1) == kPacked && packed.scalar_type() == torch::kFloat32);
    TORCH_CHECK(intrinsics.size() == 7, "intrinsics = fx, fy, k1..k4, max_angle");
    const c10::cuda::CUDAGuard guard(positions.device());
    const int64_t n = positions.size(0);
    auto pose = camera_to_world.to(torch::kCPU, torch::kFloat64).contiguous();
    const double* P = pose.data_ptr<double>();
    Camera cam{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) cam.R[i * 3 + j] = P[i * 4 + j];
        cam.c[i] = P[i * 4 + 3];
    }
    cam.fx = intrinsics[0]; cam.fy = intrinsics[1];
    for (int i = 0; i < 4; ++i) cam.k[i] = intrinsics[2 + i];
    cam.maxAngle = intrinsics[6];
    cam.fisheye = fisheye ? 1 : 0;
    if (n == 0) return;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    fastShapeKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n, positions.data_ptr<float>(), quaternions.data_ptr<float>(), scales.data_ptr<float>(),
        density.data_ptr<float>(), cam, gamma, packed.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void fast_shape32(torch::Tensor positions, torch::Tensor quaternions, torch::Tensor scales, torch::Tensor density,
                  torch::Tensor camera_to_world, std::vector<double> intrinsics, bool fisheye, double gamma,
                  torch::Tensor packed, torch::Tensor retries, int64_t mode) {
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kFloat32 && positions.is_contiguous());
    TORCH_CHECK(quaternions.is_contiguous() && scales.is_contiguous() && density.is_contiguous());
    TORCH_CHECK(packed.is_contiguous() && packed.size(1) == kPacked && packed.scalar_type() == torch::kFloat32);
    TORCH_CHECK(intrinsics.size() == 7, "intrinsics = fx, fy, k1..k4, max_angle");
    const c10::cuda::CUDAGuard guard(positions.device());
    const int64_t n = positions.size(0);
    auto pose = camera_to_world.to(torch::kCPU, torch::kFloat64).contiguous();
    const double* P = pose.data_ptr<double>();
    Camera cam{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) cam.R[i * 3 + j] = P[i * 4 + j];
        cam.c[i] = P[i * 4 + 3];
    }
    cam.fx = intrinsics[0]; cam.fy = intrinsics[1];
    for (int i = 0; i < 4; ++i) cam.k[i] = intrinsics[2 + i];
    cam.maxAngle = intrinsics[6];
    cam.fisheye = fisheye ? 1 : 0;
    if (n == 0) return;
    TORCH_CHECK(retries.is_cuda() && retries.scalar_type() == torch::kInt32 && retries.numel() >= n + 1,
                "retries must be an int32 CUDA buffer of n + 1 entries (count, rows)");
    unsigned int* rp = reinterpret_cast<unsigned int*>(retries.data_ptr<int32_t>());
    if (mode != 1) C10_CUDA_CHECK(cudaMemsetAsync(rp, 0, sizeof(unsigned int), at::cuda::getCurrentCUDAStream()));
    CameraF camf{};
    for (int i = 0; i < 9; ++i) camf.R[i] = static_cast<float>(cam.R[i]);
    for (int i = 0; i < 3; ++i) camf.c[i] = static_cast<float>(cam.c[i]);
    camf.fx = static_cast<float>(cam.fx); camf.fy = static_cast<float>(cam.fy);
    for (int i = 0; i < 4; ++i) camf.k[i] = static_cast<float>(cam.k[i]);
    camf.maxAngle = static_cast<float>(cam.maxAngle); camf.fisheye = cam.fisheye;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (mode == 1)
        fastShapeKernelMixed<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            n, positions.data_ptr<float>(), quaternions.data_ptr<float>(), scales.data_ptr<float>(),
            density.data_ptr<float>(), cam, gamma, camf, packed.data_ptr<float>(), rp);
    else
    fastShapeKernel32<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n, positions.data_ptr<float>(), quaternions.data_ptr<float>(), scales.data_ptr<float>(),
        density.data_ptr<float>(), cam, gamma, camf, static_cast<float>(gamma), packed.data_ptr<float>(), rp, rp + 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (mode != 1) {
        fastShapeRetryKernel<<<128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            rp, rp + 1, positions.data_ptr<float>(), quaternions.data_ptr<float>(), scales.data_ptr<float>(),
            density.data_ptr<float>(), cam, gamma, packed.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fast_shape", &fast_shape, "the forward-only exact renderer fused per-view full-shape factors");
    module.def("fast_shape32", &fast_shape32, "float32 (mode 0) / mixed float32-Jacobian (mode 1) per-view full-shape factors (real-time renderer)");
}
