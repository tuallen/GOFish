// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <torch/extension.h>

#include <utility>
#include <vector>

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
    bool canonical_pixel_chart);

namespace {

void require_cuda_contiguous(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> mip3d_fused_best_rate(
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
    for (const auto& pair : {
             std::pair<const torch::Tensor*, const char*>{&positions, "positions"},
             {&rotations, "rotations"}, {&centers, "centers"},
             {&frame_camera, "frame_camera"}, {&frame_index, "frame_index"},
             {&frame_rank, "frame_rank"}, {&camera_model, "camera_model"},
             {&camera_width, "camera_width"}, {&camera_height, "camera_height"},
             {&camera_params, "camera_params"}, {&camera_max_angle, "camera_max_angle"},
             {&row_lo, "row_lo"}, {&row_hi, "row_hi"}}) {
        require_cuda_contiguous(*pair.first, pair.second);
        TORCH_CHECK(pair.first->device() == positions.device(), pair.second,
                    " must be on the positions device");
    }
    TORCH_CHECK(positions.scalar_type() == torch::kFloat64 && positions.dim() == 2
                && positions.size(1) == 3, "positions must be contiguous CUDA float64 [N,3]");
    TORCH_CHECK(rotations.scalar_type() == torch::kFloat64 && rotations.dim() == 3
                && rotations.size(1) == 3 && rotations.size(2) == 3,
                "rotations must be float64 [F,3,3]");
    TORCH_CHECK(centers.scalar_type() == torch::kFloat64 && centers.dim() == 2
                && centers.size(0) == rotations.size(0) && centers.size(1) == 3,
                "centers must be float64 [F,3]");
    TORCH_CHECK(positions.size(0) > 0 && rotations.size(0) > 0,
                "point and frame inventories must be nonempty");
    TORCH_CHECK(frame_camera.scalar_type() == torch::kInt32,
                "frame_camera must be int32");
    TORCH_CHECK(frame_index.scalar_type() == torch::kInt32, "frame_index must be int32");
    TORCH_CHECK(frame_rank.scalar_type() == torch::kInt64, "frame_rank must be int64");
    TORCH_CHECK(camera_model.scalar_type() == torch::kInt32, "camera_model must be int32");
    TORCH_CHECK(camera_width.scalar_type() == torch::kInt32, "camera_width must be int32");
    TORCH_CHECK(camera_height.scalar_type() == torch::kInt32, "camera_height must be int32");
    TORCH_CHECK(camera_params.scalar_type() == torch::kFloat64 && camera_params.dim() == 2
                && camera_params.size(1) == 12, "camera_params must be float64 [C,12]");
    TORCH_CHECK(camera_max_angle.scalar_type() == torch::kFloat64,
                "camera_max_angle must be float64");
    TORCH_CHECK(row_lo.scalar_type() == torch::kInt32 && row_hi.scalar_type() == torch::kInt32
                && row_lo.sizes() == row_hi.sizes(), "row runs must be matching int32 [C,Hmax]");
    TORCH_CHECK(frame_camera.numel() == rotations.size(0)
                && frame_index.numel() == rotations.size(0)
                && frame_rank.numel() == rotations.size(0), "frame arrays differ");
    TORCH_CHECK(camera_model.numel() == camera_params.size(0)
                && camera_width.numel() == camera_params.size(0)
                && camera_height.numel() == camera_params.size(0)
                && camera_max_angle.numel() == camera_params.size(0), "camera arrays differ");
    TORCH_CHECK(row_lo.dim() == 2 && row_lo.size(0) == camera_params.size(0),
                "row runs must be [C,Hmax]");
    TORCH_CHECK(row_lo.size(1) > 0, "row-run stride must be positive");
    TORCH_CHECK(frame_camera.dim() == 1 && frame_index.dim() == 1
                && frame_rank.dim() == 1, "frame arrays must be one-dimensional");
    TORCH_CHECK(camera_model.dim() == 1 && camera_width.dim() == 1
                && camera_height.dim() == 1 && camera_max_angle.dim() == 1,
                "camera arrays must be one-dimensional");
    return mip3d_fused_best_rate_cuda(
        positions, rotations, centers, frame_camera, frame_index, frame_rank,
        camera_model, camera_width, camera_height, camera_params,
        camera_max_angle, row_lo, row_hi, canonical_pixel_chart);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("best_rate", &mip3d_fused_best_rate,
               "Persistent float64 physical-Mip owner search (CUDA)");
}
