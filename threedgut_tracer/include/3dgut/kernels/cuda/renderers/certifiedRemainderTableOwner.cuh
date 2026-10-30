// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <3dgut/kernels/cuda/renderers/certifiedRemainderAdmission.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderPackedKbTable.cuh>
#include <3dgut/kernels/cuda/renderers/certifiedRemainderProvider.cuh>
#include <3dgut/sensors/sensors.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace gofish_certified_remainder {

using ProviderCameraDescriptor =
    ::gofish_certified_remainder_provider::SelectedKbCameraDescriptor;

class PackedKbTableOwner {
public:
    PackedKbTableOwner() = default;
    PackedKbTableOwner(const PackedKbTableOwner&) = delete;
    PackedKbTableOwner& operator=(const PackedKbTableOwner&) = delete;
    ~PackedKbTableOwner() { clear(); }

    cudaError_t initialize(cudaStream_t stream) {
        if (initialized_) return cudaSuccess;
        if (!validateHostBlob(kPackedKbTableBytes, kPackedKbTableSize))
            return cudaErrorInvalidValue;
        host_.assign(kPackedKbTableBytes,
                     kPackedKbTableBytes + kPackedKbTableSize);
        cudaError_t status = cudaMalloc(&device_, kPackedKbTableSize);
        if (status != cudaSuccess) { clear(); return status; }
        status = cudaMemcpyAsync(device_, kPackedKbTableBytes,
                                 kPackedKbTableSize,
                                 cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) { clear(); return status; }
        initialized_ = true;
        return cudaSuccess;
    }

    void clear() {
        if (device_ != nullptr) cudaFree(device_);
        device_ = nullptr;
        host_.clear();
        initialized_ = false;
        std::memset(&header_, 0, sizeof(header_));
    }

    bool bind(const threedgut::TSensorModel& sensorModel,
              std::uint32_t activeWidth, std::uint32_t activeHeight,
              CameraThetaBandHessianTable* output,
              ProviderCameraDescriptor* providerCamera) const {
        using namespace gofish::kb_hessian_abi;
        if (!initialized_ || output == nullptr || providerCamera == nullptr
            || activeWidth == 0u || activeHeight == 0u
            || (sensorModel.modelType
                    != threedgut::TSensorModel::OpenCVFisheyeModel
                && sensorModel.modelType
                    != threedgut::TSensorModel::OpenCVPinholeModel)
            || sensorModel.shutterType
                != threedgut::TSensorModel::GlobalShutter) return false;
        std::uint32_t selected = header_.camera_count;
        const PackedKbCameraSelector* selector = nullptr;
        CameraDescriptor descriptor{};
        for (std::size_t candidate = 0;
             candidate < kPackedKbCameraSelectorCount; ++candidate) {
            const auto& row = kPackedKbCameraSelectors[candidate];
            if (row.descriptorIndex >= header_.camera_count) return false;
            std::memcpy(&descriptor,
                        host_.data() + header_.cameras_offset
                            + std::size_t(row.descriptorIndex)
                                * header_.camera_stride,
                        sizeof(descriptor));
            if (descriptor.native_width != row.nativeWidth
                || descriptor.native_height != row.nativeHeight
                || descriptor.model_id != row.modelId
                || row.chartMode > kSelectorActiveCornerUnitScale) return false;
            const bool activeUnit =
                row.chartMode == kSelectorActiveCornerUnitScale;
            if (activeUnit && (activeWidth != row.nativeWidth
                               || activeHeight != row.nativeHeight))
                continue;
            const float xScale = activeUnit
                ? 1.0f : float(activeWidth) / float(row.nativeWidth);
            const float yScale = activeUnit
                ? 1.0f : float(activeHeight) / float(row.nativeHeight);
            if (row.modelId == kModelOpenCvFisheyeKb4) {
                if (sensorModel.modelType
                        != threedgut::TSensorModel::OpenCVFisheyeModel) continue;
                const auto& active = sensorModel.ocvFisheyeParams;
                if (active.maxAngle != row.maxAngle
                    || std::memcmp(&active.radialCoeffs[0], &row.parameters[4],
                                4 * sizeof(float)) != 0
                    || active.focalLength.x != row.parameters[0] * xScale
                    || active.focalLength.y != row.parameters[1] * yScale
                    || active.principalPoint.x != (activeUnit
                        ? row.parameters[2]
                        : row.parameters[2] * xScale)
                    || active.principalPoint.y != (activeUnit
                        ? row.parameters[3]
                        : row.parameters[3] * yScale))
                    continue;
            } else if (row.modelId == kModelIdealPinhole) {
                if (sensorModel.modelType
                        != threedgut::TSensorModel::OpenCVPinholeModel) continue;
                const auto& active = sensorModel.ocvPinholeParams;
                bool distortionFree = true;
                for (int index = 0; index < 6; ++index)
                    distortionFree = distortionFree
                        && active.radialCoeffs[index] == 0.0f;
                for (int index = 0; index < 2; ++index)
                    distortionFree = distortionFree
                        && active.tangentialCoeffs[index] == 0.0f;
                for (int index = 0; index < 4; ++index)
                    distortionFree = distortionFree
                        && active.thinPrismCoeffs[index] == 0.0f;
                if (!distortionFree
                    || active.focalLength.x != row.parameters[0] * xScale
                    || active.focalLength.y != row.parameters[1] * yScale
                    || active.principalPoint.x != (activeUnit
                        ? row.parameters[2]
                        : row.parameters[2] * xScale)
                    || active.principalPoint.y != (activeUnit
                        ? row.parameters[3]
                        : row.parameters[3] * yScale))
                    continue;
            } else {
                return false;
            }
            selected = row.descriptorIndex;
            selector = &row;
            break;
        }
        if (selected == header_.camera_count || selector == nullptr) return false;
        const auto* deviceBytes = static_cast<const std::uint8_t*>(device_);
        output->view = TableView{
            reinterpret_cast<const Header*>(deviceBytes),
            reinterpret_cast<const CameraDescriptor*>(
                deviceBytes + header_.cameras_offset),
            reinterpret_cast<const Band*>(deviceBytes + header_.bands_offset)};
        output->camera = output->view.cameras + selected;
        output->activeWidth = activeWidth;
        output->activeHeight = activeHeight;
        std::memcpy(output->activeCameraIdentitySha256,
                    descriptor.camera_identity_sha256, 32);
        std::memcpy(output->activeCanonicalDomainSha256,
                    descriptor.active_canonical_domain_sha256, 32);
        output->initializedCertified = 1u;
        auto& provider = *providerCamera;
        provider = {};
        provider.descriptor = output->camera;
        provider.modelId = descriptor.model_id;
        if (descriptor.model_id == kModelOpenCvFisheyeKb4) {
            provider.nativeProjection.focalLength = tcnn::vec2{
                selector->parameters[0], selector->parameters[1]};
            provider.nativeProjection.principalPoint = tcnn::vec2{
                selector->parameters[2], selector->parameters[3]};
            provider.nativeProjection.radialCoeffs = tcnn::vec4{
                selector->parameters[4], selector->parameters[5],
                selector->parameters[6], selector->parameters[7]};
            provider.nativeProjection.maxAngle = selector->maxAngle;
            provider.activeProjection = sensorModel.ocvFisheyeParams;
        } else {
            provider.nativePinholeProjection.focalLength = tcnn::vec2{
                selector->parameters[0], selector->parameters[1]};
            provider.nativePinholeProjection.principalPoint = tcnn::vec2{
                selector->parameters[2], selector->parameters[3]};
            provider.activePinholeProjection = sensorModel.ocvPinholeParams;
        }
        provider.activeResolution = tcnn::ivec2{
            int(activeWidth), int(activeHeight)};
        const auto bindPhysicalSampleOffset = [](
            float& lower, float& upper) {
            // Runtime rays are evaluated at corner-chart pixel centres. The
            // native point (i+0.5)/a maps exactly back to active i+0.5 under
            // c_active=a*c_native.
            lower = 0.5f;
            upper = 0.5f;
        };
        bindPhysicalSampleOffset(provider.physicalSampleOffsetLower.x,
            provider.physicalSampleOffsetUpper.x);
        bindPhysicalSampleOffset(provider.physicalSampleOffsetLower.y,
            provider.physicalSampleOffsetUpper.y);
        std::memcpy(provider.linearizationContract,
                    header_.table_json_sha256, 4 * sizeof(std::uint64_t));
        provider.hostSelectionValidated = 1u;
        provider.chartMode = selector->chartMode;
        return true;
    }

private:
    bool validateHostBlob(const std::uint8_t* bytes, std::size_t size) {
        using namespace gofish::kb_hessian_abi;
        if (bytes == nullptr || size < sizeof(Header)) return false;
        std::memcpy(&header_, bytes, sizeof(header_));
        static const std::uint8_t magic[8] = {
            'G','F','K','B','H','0','2','\0'};
        if (std::memcmp(header_.magic, magic, 8) != 0
            || header_.abi_version != kVersion
            || header_.endian_tag != kEndianTag
            || header_.header_bytes != kHeaderBytes
            || header_.camera_stride != kCameraBytes
            || header_.band_stride != kBandBytes
            || header_.flags != kRequiredFlags
            || header_.payload_bytes != size
            || header_.cameras_offset != kHeaderBytes
            || header_.bands_offset != header_.cameras_offset
                + std::uint64_t(header_.camera_count) * kCameraBytes
            || header_.payload_bytes != header_.bands_offset
                + std::uint64_t(header_.band_count) * kBandBytes) return false;
        for (std::uint32_t index = 0; index < header_.camera_count; ++index) {
            CameraDescriptor descriptor{};
            std::memcpy(&descriptor,
                        bytes + header_.cameras_offset
                            + std::size_t(index) * header_.camera_stride,
                        sizeof(descriptor));
            if ((descriptor.model_id != kModelOpenCvFisheyeKb4
                    && descriptor.model_id != kModelIdealPinhole)
                || descriptor.native_width == 0u
                || descriptor.native_height == 0u
                || descriptor.reserved != 0u
                || descriptor.first_band + descriptor.band_count
                    > header_.band_count) return false;
        }
        return true;
    }

    std::vector<std::uint8_t> host_;
    void* device_ = nullptr;
    gofish::kb_hessian_abi::Header header_{};
    bool initialized_ = false;
};

} // namespace gofish_certified_remainder

