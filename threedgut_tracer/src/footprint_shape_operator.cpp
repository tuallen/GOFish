// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> footprint_full_shape_forward_cuda(
    const torch::Tensor& scales,
    const torch::Tensor& quaternion,
    const torch::Tensor& density,
    const torch::Tensor& q6,
    const torch::Tensor& jacobian,
    const torch::Tensor& observed,
    bool mass_preserving);

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
    bool mass_preserving);

namespace {

void require_cuda_float_contiguous(
    const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        value.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void require_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& value,
    const char* name) {
    TORCH_CHECK(value.device() == reference.device(), name, " device differs");
}

void require_cuda_double_contiguous(
    const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        value.scalar_type() == torch::kFloat64, name, " must be float64");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void require_matrix(
    const torch::Tensor& value, int64_t count, const char* name) {
    require_cuda_float_contiguous(value, name);
    TORCH_CHECK(
        value.dim() == 3 && value.size(0) == count &&
            value.size(1) == 3 && value.size(2) == 3,
        name,
        " must be [N,3,3]");
}

void require_observed(
    const torch::Tensor& observed,
    int64_t count,
    const torch::Tensor& reference) {
    TORCH_CHECK(observed.is_cuda(), "observed must be CUDA");
    TORCH_CHECK(
        observed.scalar_type() == torch::kBool,
        "observed must be bool");
    TORCH_CHECK(observed.is_contiguous(), "observed must be contiguous");
    TORCH_CHECK(
        observed.dim() == 1 && observed.size(0) == count,
        "observed must be [N]");
    require_same_device(reference, observed, "observed");
    TORCH_CHECK(
        !observed.requires_grad(),
        "observed must be detached lifecycle state");
}

void require_jacobian(
    const torch::Tensor& jacobian,
    int64_t count,
    const torch::Tensor& reference) {
    require_cuda_float_contiguous(jacobian, "jacobian");
    TORCH_CHECK(
        jacobian.dim() == 3 && jacobian.size(0) == count &&
            jacobian.size(1) == 2 && jacobian.size(2) == 3,
        "jacobian must be [N,2,3]");
    require_same_device(reference, jacobian, "jacobian");
    TORCH_CHECK(!jacobian.requires_grad(),
                "jacobian must be detached observation state");
}

void require_density(
    const torch::Tensor& density, int64_t count, const char* name) {
    require_cuda_float_contiguous(density, name);
    TORCH_CHECK(
        (density.dim() == 1 && density.size(0) == count) ||
            (density.dim() == 2 && density.size(0) == count &&
             density.size(1) == 1),
        name,
        " must be [N] or [N,1]");
}

}  // namespace

std::vector<torch::Tensor> footprint_full_shape_forward(
    const torch::Tensor& scales,
    const torch::Tensor& quaternion,
    const torch::Tensor& density,
    const torch::Tensor& q6,
    const torch::Tensor& jacobian,
    const torch::Tensor& observed,
    bool mass_preserving) {
    require_cuda_float_contiguous(scales, "scales");
    require_cuda_float_contiguous(quaternion, "quaternion");
    const int64_t count = scales.size(0);
    TORCH_CHECK(
        scales.dim() == 2 && scales.size(1) == 3,
        "scales must be [N,3]");
    TORCH_CHECK(
        quaternion.dim() == 2 && quaternion.size(0) == count &&
            quaternion.size(1) == 4,
        "quaternion must be [N,4]");
    require_density(density, count, "density");
    require_cuda_double_contiguous(q6, "q6");
    TORCH_CHECK(
        q6.dim() == 2 && q6.size(0) == count && q6.size(1) == 6,
        "q6 must be [N,6]");
    require_jacobian(jacobian, count, scales);
    require_observed(observed, count, scales);
    for (const auto& value : {quaternion, density, q6, jacobian}) {
        require_same_device(scales, value, "forward input");
    }
    TORCH_CHECK(!q6.requires_grad(), "q6 must be detached lifecycle state");
    return footprint_full_shape_forward_cuda(
        scales, quaternion, density, q6, jacobian, observed, mass_preserving);
}

std::vector<torch::Tensor> footprint_full_shape_backward(
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
    require_cuda_float_contiguous(scales, "scales");
    require_cuda_float_contiguous(quaternion, "quaternion");
    const int64_t count = scales.size(0);
    TORCH_CHECK(
        scales.dim() == 2 && scales.size(1) == 3,
        "scales must be [N,3]");
    TORCH_CHECK(
        quaternion.dim() == 2 && quaternion.size(0) == count &&
            quaternion.size(1) == 4,
        "quaternion must be [N,4]");
    require_density(density, count, "density");
    require_cuda_double_contiguous(q6, "q6");
    TORCH_CHECK(q6.dim() == 2 && q6.size(0) == count && q6.size(1) == 6,
                "q6 must be [N,6]");
    require_same_device(scales, q6, "q6");
    TORCH_CHECK(!q6.requires_grad(), "q6 must be detached lifecycle state");
    require_jacobian(jacobian, count, scales);
    require_observed(observed, count, scales);
    for (const auto& pair : std::vector<std::pair<torch::Tensor, const char*>>{
             {covariance, "covariance"},
             {inverse_factor, "inverse_factor"},
             {root, "root"},
             {rotation, "rotation"},
             {eigenvectors, "eigenvectors"},
             {grad_covariance, "grad_covariance"},
             {grad_factor, "grad_factor"},
             {grad_inverse_factor, "grad_inverse_factor"}}) {
        require_matrix(pair.first, count, pair.second);
    }
    require_cuda_float_contiguous(compensation, "compensation");
    require_cuda_float_contiguous(roots, "roots");
    require_cuda_float_contiguous(grad_compensation, "grad_compensation");
    require_density(
        grad_filtered_density, count, "grad_filtered_density");
    TORCH_CHECK(
        compensation.dim() == 1 && compensation.size(0) == count,
        "compensation must be [N]");
    TORCH_CHECK(
        roots.dim() == 2 && roots.size(0) == count && roots.size(1) == 3,
        "roots must be [N,3]");
    require_observed(escalated, count, scales);
    require_observed(projected_only_unsafe, count, scales);
    TORCH_CHECK(
        unsafe_indices.is_cuda() && unsafe_indices.is_contiguous() &&
            unsafe_indices.scalar_type() == torch::kInt64 &&
            unsafe_indices.dim() == 1,
        "unsafe_indices must be contiguous CUDA int64 [M]");
    const int64_t unsafe_count = unsafe_indices.size(0);
    for (const auto& pair : std::vector<std::pair<torch::Tensor, const char*>>{
             {covariance64, "covariance64"},
             {inverse_factor64, "inverse_factor64"},
             {root64, "root64"},
             {rotation64, "rotation64"},
             {eigenvectors64, "eigenvectors64"}}) {
        require_cuda_double_contiguous(pair.first, pair.second);
        TORCH_CHECK(
            pair.first.dim() == 3 && pair.first.size(0) == unsafe_count &&
                pair.first.size(1) == 3 && pair.first.size(2) == 3,
            pair.second,
            " must be [M,3,3]");
    }
    require_cuda_double_contiguous(compensation64, "compensation64");
    require_cuda_double_contiguous(roots64, "roots64");
    require_cuda_double_contiguous(projected_q64, "projected_q64");
    TORCH_CHECK(
        compensation64.dim() == 1 && compensation64.size(0) == unsafe_count,
        "compensation64 must be [M]");
    TORCH_CHECK(
        roots64.dim() == 2 && roots64.size(0) == unsafe_count &&
            roots64.size(1) == 3,
        "roots64 must be [M,3]");
    TORCH_CHECK(
        projected_q64.dim() == 2 && projected_q64.size(0) == unsafe_count &&
            projected_q64.size(1) == 3,
        "projected_q64 must be [M,3]");
    TORCH_CHECK(
        grad_compensation.sizes() == compensation.sizes(),
        "grad_compensation shape differs");
    TORCH_CHECK(
        grad_filtered_density.sizes() == density.sizes(),
        "grad_filtered_density shape differs");
    for (const auto& value : {
             quaternion,
             density,
             q6,
             jacobian,
             observed,
             covariance,
             inverse_factor,
             compensation,
             root,
             rotation,
             eigenvectors,
             roots,
             escalated,
             projected_only_unsafe,
             unsafe_indices,
             covariance64,
             inverse_factor64,
             compensation64,
             root64,
             rotation64,
             eigenvectors64,
             roots64,
             projected_q64,
             grad_covariance,
             grad_factor,
             grad_inverse_factor,
             grad_compensation,
             grad_filtered_density}) {
        require_same_device(scales, value, "backward input");
    }
    return footprint_full_shape_backward_cuda(
        scales,
        quaternion,
        density,
        q6,
        jacobian,
        observed,
        covariance,
        inverse_factor,
        compensation,
        root,
        rotation,
        eigenvectors,
        roots,
        escalated,
        projected_only_unsafe,
        unsafe_indices,
        covariance64,
        inverse_factor64,
        compensation64,
        root64,
        rotation64,
        eigenvectors64,
        roots64,
        projected_q64,
        grad_covariance,
        grad_factor,
        grad_inverse_factor,
        grad_compensation,
        grad_filtered_density,
        mass_preserving);
}

void footprint_set_persistent_solver_cuda(bool enabled);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "set_persistent_solver",
        &footprint_set_persistent_solver_cuda,
        "Reuse one cuSOLVER handle across calls (inference renderers)");
    module.def(
        "forward",
        &footprint_full_shape_forward,
        "Full-shape factor forward using cuSOLVER (CUDA)");
    module.def(
        "backward",
        &footprint_full_shape_backward,
        "Full-shape factor analytic VJP (CUDA)");
}
