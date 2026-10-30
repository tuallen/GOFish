// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: CC-BY-NC-4.0

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gofish_physical_candidate {

// Immutable view of the context-owned physical tile mask. Invalid
// metadata fails open; the host renderer independently rejects it before any
// launch. Original tile IDs are never remapped.
struct PhysicalTileView {
    const std::uint8_t* mask;
    std::uint32_t count;
    std::uint32_t gridX;
    std::uint32_t gridY;

    inline __host__ __device__ bool metadataValid() const {
        return mask != nullptr && gridX > 0u && gridY > 0u
            && std::uint64_t(gridX) * gridY == count;
    }

    inline __device__ bool retainTile(std::uint32_t tile) const {
        if (!metadataValid()) return true;
        return tile < count && mask[tile] != 0u;
    }

    inline __device__ bool retainXY(std::uint32_t x, std::uint32_t y) const {
        if (!metadataValid()) return true;
        if (x >= gridX || y >= gridY) return false;
        return mask[y * gridX + x] != 0u;
    }
};

} // namespace gofish_physical_candidate

