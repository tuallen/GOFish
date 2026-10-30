// Portions derived from SOF (https://github.com/r4dl/SOF),
// distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
// research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md
// Scheduled SOF hierarchy (render.splat.sof_schedule, GOFISH_SOF_SCHEDULE).
// The same three-level SOF cascade as sofMid8Hierarchy.inl, with identical
// keys, ranks, emission order and per-pixel feed order.  The cascade runs once
// per forward pass in the sofMid8Schedule kernel and writes each MID8 group's
// emission sequence to device memory; render and renderBackward only consume
// it.
// Compile-gated, three-level SOF hierarchy for inclusion inside
// GUTKBufferRenderer<Particles, Params, Backward>.
//
// Port of the hierarchical sort in SOF's released
// hierarchical_render.cuh.  The port preserves SOF's spatial/window contract:
//   per-pixel HEAD4 <- per-2x2 MID8 <- per-4x4 TAIL64
// and its 32-new + 32-retained tail merge, 16-at-a-time tail emission,
// 4-new + 4-retained mid merge, and 4-at-a-time mid emission.
// It deliberately reuses GUT's densityHit/processHitParticle path.

#if GAUSSIAN_SOF_MID8_HIERARCHY

    static_assert(GAUSSIAN_HEAD_SIZE == 4,
                  "released SOF hierarchy requires per-pixel HEAD4");
    static_assert(GAUSSIAN_MID_SIZE == 8,
                  "released SOF hierarchy requires per-2x2 MID8");
    static_assert(GAUSSIAN_TAIL_SIZE == 64,
                  "released SOF hierarchy requires per-4x4 TAIL64");
    static_assert(threedgut::GUTParameters::Tiling::BlockX == 16 &&
                  threedgut::GUTParameters::Tiling::BlockY == 16 &&
                  threedgut::GUTParameters::Tiling::BlockSize == 256,
                  "SOF MID8 spatial mapping requires a 16x16 tile");

    static inline __device__ bool sofMid8Less(float lhsKey,
                                               uint32_t lhsOrdinal,
                                               float rhsKey,
                                               uint32_t rhsOrdinal) {
        return (lhsKey < rhsKey) || ((lhsKey == rhsKey) && (lhsOrdinal < rhsOrdinal));
    }


    // Exact cooperative rank of one item in retained[0:R] + new[0:N].
    // Every item has a unique original candidate ordinal. Therefore the rank
    // under sofMid8Less is unique, and writing by rank reproduces the serial
    // stable-insertion result bit-for-bit without changing key arithmetic.
    static inline __device__ int sofTail64Rank(
        float itemKey, uint32_t itemOrdinal,
        const float* retainedKeys, const uint32_t* retainedOrdinals,
        int retainedCount,
        float newKey0, uint32_t newOrdinal0,
        float newKey1, uint32_t newOrdinal1,
        int newCount, unsigned subgroupMask) {
        // The retained window is sorted under the strict total order
        // sofMid8Less (unique ordinals), so the count of retained items that
        // precede the item is a lower bound: binary search == linear count.
        int lo = 0, hi = retainedCount;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (sofMid8Less(retainedKeys[mid], retainedOrdinals[mid], itemKey, itemOrdinal)) lo = mid + 1; else hi = mid;
        }
        int rank = lo;
        for (int sourceLane = 0; sourceLane < 16; ++sourceLane) {
            if (sourceLane < newCount) {
                const float key = __shfl_sync(
                    subgroupMask, newKey0, sourceLane, 16);
                const uint32_t ordinal = __shfl_sync(
                    subgroupMask, newOrdinal0, sourceLane, 16);
                rank += sofMid8Less(key, ordinal, itemKey, itemOrdinal) ? 1 : 0;
            }
            if (sourceLane + 16 < newCount) {
                const float key = __shfl_sync(
                    subgroupMask, newKey1, sourceLane, 16);
                const uint32_t ordinal = __shfl_sync(
                    subgroupMask, newOrdinal1, sourceLane, 16);
                rank += sofMid8Less(key, ordinal, itemKey, itemOrdinal) ? 1 : 0;
            }
        }
        return rank;
    }

    static inline __device__ int sofMid8Rank(
        float itemKey, uint32_t itemOrdinal,
        const float* retainedKeys, const uint32_t* retainedOrdinals,
        int retainedCount,
        float newKey, uint32_t newOrdinal,
        int newCount, unsigned subgroupMask) {
        // The retained window is sorted under the strict total order
        // sofMid8Less (unique ordinals), so the count of retained items that
        // precede the item is a lower bound: binary search == linear count.
        int lo = 0, hi = retainedCount;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (sofMid8Less(retainedKeys[mid], retainedOrdinals[mid], itemKey, itemOrdinal)) lo = mid + 1; else hi = mid;
        }
        int rank = lo;
        for (int sourceLane = 0; sourceLane < 4; ++sourceLane) {
            if (sourceLane < newCount) {
                const float key = __shfl_sync(
                    subgroupMask, newKey, sourceLane, 4);
                const uint32_t ordinal = __shfl_sync(
                    subgroupMask, newOrdinal, sourceLane, 4);
                rank += sofMid8Less(key, ordinal, itemKey, itemOrdinal) ? 1 : 0;
            }
        }
        return rank;
    }

    static inline __device__ float3 sofMid8Normalize(float3 value) {
        const float n2 = dot(value, value);
        if (!(n2 > 0.0f) || !isfinite(n2)) {
            return make_float3(0.0f, 0.0f, 1.0f);
        }
        return value * rsqrtf(n2);
    }

    static inline __device__ float sofMid8RepresentativeDepth(
        Particles& particles,
        uint32_t particleIdx,
        const float3& origin,
        const float3& direction,
        unsigned long long* fallbackHitCounter) {
        const DensityRawParameters raw = particles.fetchDensityRawParameters(particleIdx);
        // The representative sort key must use the same effective Gaussian as
        // response and compositing. For standard row-major B^-1, CUDA's
        // `vector * matrix` evaluates B^-1*vector.
        const float3 canonicalOrigin = (origin - raw.position) * raw.inverseFactor;
        const float3 canonicalDirection = direction * raw.inverseFactor;
        auto radialFallback = [&]() -> float {
            if (fallbackHitCounter != nullptr) {
                atomicAdd(fallbackHitCounter, 1ULL);
            }
            const float3 delta = raw.position - origin;
            const float radialSquared = dot(delta, delta);
            if (!(radialSquared >= 0.0f) || !isfinite(radialSquared)) {
                // A finite Gaussian and calibrated representative origin must
                // have a finite radial distance. Do not hide corrupt state in
                // a sort-key fallback.
                asm volatile("trap;");
                return 0.0f;
            }
            return sqrtf(radialSquared);
        };
        const float denominator = dot(canonicalDirection, canonicalDirection);
        if (!(denominator > 0.0f) || !isfinite(denominator)) {
            return radialFallback();
        }
        const float depth = -dot(canonicalOrigin, canonicalDirection) / denominator;
        return isfinite(depth) ? fmaxf(0.0f, depth) : radialFallback();
    }

    static inline __device__ void sofMid8Insert(
        float* keys,
        uint32_t* ids,
        uint32_t* ordinals,
        int& count,
        int capacity,
        float key,
        uint32_t id,
        uint32_t ordinal) {
        // The caller proves count < capacity. Stable tie resolution by original
        // candidate ordinal makes forward and backward replay deterministic.
        int position = count;
        while (position > 0 &&
               sofMid8Less(key, ordinal, keys[position - 1], ordinals[position - 1])) {
            keys[position] = keys[position - 1];
            ids[position] = ids[position - 1];
            ordinals[position] = ordinals[position - 1];
            --position;
        }
        keys[position] = key;
        ids[position] = id;
        ordinals[position] = ordinal;
        ++count;
    }

    template <typename TRay>
    static inline __device__ void evalSOFMid8Schedule(
        const threedgut::RenderParameters& params,
        TRay& ray,
        Particles& particles,
        const tcnn::uvec2& tileParticleRangeIndices,
        uint32_t tileNumBlocksToProcess,
        uint32_t tileNumParticlesToProcess,
        const uint32_t tileThreadIdx,
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const float* __restrict__ particlesGlobalDepthPtr,
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
        float* __restrict__ particlesPositionAbsGradPtr = nullptr,
        unsigned long long* __restrict__ sofDepthFallbackHitCounter = nullptr) {

        (void)tileNumBlocksToProcess;
        (void)tileNumParticlesToProcess;
        (void)particlesGlobalDepthPtr; // view-z is never a generic-camera fallback

        constexpr uint32_t Invalid = threedgut::GUTParameters::InvalidParticleIdx;
        constexpr int TailGroups = 16;   // 4x4 groups in a 16x16 tile
        constexpr int MidGroups = 64;    // 2x2 groups in a 16x16 tile
        constexpr int TailRetained = 32; // SOF steady-state half of TAIL64

        __shared__ float3 sharedRayOrigin[256];
        __shared__ float3 sharedRayDirection[256];
        __shared__ uint8_t sharedRayValid[256];

        __shared__ float3 tailOrigin[TailGroups];
        __shared__ float3 tailDirection[TailGroups];
        __shared__ float3 midOrigin[MidGroups];
        __shared__ float3 midDirection[MidGroups];

        __shared__ float tailKeys[TailGroups][GAUSSIAN_TAIL_SIZE];
        __shared__ uint32_t tailIds[TailGroups][GAUSSIAN_TAIL_SIZE];
        __shared__ uint32_t tailOrdinals[TailGroups][GAUSSIAN_TAIL_SIZE];
        __shared__ int tailCounts[TailGroups];
        __shared__ uint32_t tailEmitIdsA[TailGroups][16];
        __shared__ uint32_t tailEmitOrdinalsA[TailGroups][16];
        __shared__ int tailEmitCountsA[TailGroups];
        __shared__ uint32_t tailEmitIdsB[TailGroups][16];
        __shared__ uint32_t tailEmitOrdinalsB[TailGroups][16];
        __shared__ int tailEmitCountsB[TailGroups];

        __shared__ float midKeys[MidGroups][GAUSSIAN_MID_SIZE];
        __shared__ uint32_t midIds[MidGroups][GAUSSIAN_MID_SIZE];
        __shared__ uint32_t midOrdinals[MidGroups][GAUSSIAN_MID_SIZE];
        __shared__ int midCounts[MidGroups];

        // Reused between TAIL and MID phases. TAIL needs 16x64 slots and MID
        // needs only the first 64x8 slots, so one 1024-item workspace avoids
        // duplicating shared storage.

        const int localX = static_cast<int>(threadIdx.x);
        const int localY = static_cast<int>(threadIdx.y);
        const int tailGroup = (localY >> 2) * 4 + (localX >> 2);
        const int midGroup = (localY >> 1) * 8 + (localX >> 1);
        const uint32_t pixelX = blockIdx.x * threedgut::GUTParameters::Tiling::BlockX + threadIdx.x;
        const uint32_t pixelY = blockIdx.y * threedgut::GUTParameters::Tiling::BlockY + threadIdx.y;
        const bool pixelInBounds = pixelX < static_cast<uint32_t>(params.resolution.x) &&
                                   pixelY < static_cast<uint32_t>(params.resolution.y);

        sharedRayValid[tileThreadIdx] = pixelInBounds ? 1 : 0;
        sharedRayOrigin[tileThreadIdx] = pixelInBounds
            ? reinterpret_cast<const float3&>(ray.origin)
            : make_float3(0.0f, 0.0f, 0.0f);
        sharedRayDirection[tileThreadIdx] = pixelInBounds
            ? reinterpret_cast<const float3&>(ray.direction)
            : make_float3(0.0f, 0.0f, 0.0f);
        if (tileThreadIdx < TailGroups) {
            tailCounts[tileThreadIdx] = 0;
            tailEmitCountsA[tileThreadIdx] = 0;
            tailEmitCountsB[tileThreadIdx] = 0;
        }
        if (tileThreadIdx < MidGroups) {
            midCounts[tileThreadIdx] = 0;
        }
        __syncthreads();

        // Camera-native counterpart of SOF's fractional region-center rays:
        // average the four calibrated rays surrounding each 4x4/2x2 center.
        // For global-shutter pinhole this converges to SOF's center ray; unlike
        // selecting one representative pixel it remains symmetric for fisheye.
        if (tileThreadIdx < TailGroups) {
            const int group = static_cast<int>(tileThreadIdx);
            const int baseX = (group & 3) * 4;
            const int baseY = (group >> 2) * 4;
            float3 origin = make_float3(0.0f);
            float3 direction = make_float3(0.0f);
            int count = 0;
            for (int oy = 1; oy <= 2; ++oy) {
                for (int ox = 1; ox <= 2; ++ox) {
                    const int index = (baseY + oy) * 16 + baseX + ox;
                    if (sharedRayValid[index]) {
                        origin = origin + sharedRayOrigin[index];
                        direction = direction + sharedRayDirection[index];
                        ++count;
                    }
                }
            }
            if (count == 0) {
                for (int oy = 0; oy < 4; ++oy) {
                    for (int ox = 0; ox < 4; ++ox) {
                        const int index = (baseY + oy) * 16 + baseX + ox;
                        if (sharedRayValid[index]) {
                            origin = origin + sharedRayOrigin[index];
                            direction = direction + sharedRayDirection[index];
                            ++count;
                        }
                    }
                }
            }
            tailOrigin[group] = count ? origin * (1.0f / static_cast<float>(count))
                                      : make_float3(0.0f);
            tailDirection[group] = sofMid8Normalize(direction);
        }
        if (tileThreadIdx < MidGroups) {
            const int group = static_cast<int>(tileThreadIdx);
            const int baseX = (group & 7) * 2;
            const int baseY = (group >> 3) * 2;
            float3 origin = make_float3(0.0f);
            float3 direction = make_float3(0.0f);
            int count = 0;
            for (int oy = 0; oy < 2; ++oy) {
                for (int ox = 0; ox < 2; ++ox) {
                    const int index = (baseY + oy) * 16 + baseX + ox;
                    if (sharedRayValid[index]) {
                        origin = origin + sharedRayOrigin[index];
                        direction = direction + sharedRayDirection[index];
                        ++count;
                    }
                }
            }
            midOrigin[group] = count ? origin * (1.0f / static_cast<float>(count))
                                     : make_float3(0.0f);
            midDirection[group] = sofMid8Normalize(direction);
        }
        __syncthreads();


        // Same cascade as sofMid8Hierarchy.inl with reorganised bookkeeping.  Every TAIL64/MID8 merge is private to its 16- or 4-lane
        // subgroup and fenced by __syncwarp, merges place each item straight at
        // its rank, emissions are written from the merged list by all subgroup
        // lanes, and each MID8 group's emission sequence for a TAIL chunk is
        // produced warp-locally into a per-group queue.
        constexpr int QueueCapacity = 36;  // 2 emissions x 4 batches x 4, + final MID4 drain
        const int mWorkerGroup = static_cast<int>(tileThreadIdx >> 2);
        const int mWorkerLane = static_cast<int>(tileThreadIdx & 3u);
        const unsigned mSubgroupMask =
            0xFu << ((static_cast<unsigned>(tileThreadIdx) & 31u) & ~3u);
        const int mParent = ((mWorkerGroup >> 3) >> 1) * 4 + ((mWorkerGroup & 7) >> 1);
        int mRetained = 0;   // identical in the four lanes of the group
        int mQueued = 0;     // identical in the four lanes of the group

        // MID emissions go straight to this group's global schedule.
        const uint32_t schedTile = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t schedLength = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        uint32_t* __restrict__ schedIds = gSofSchedIds
            + static_cast<size_t>(tileParticleRangeIndices.x - gSofSchedBase) * MidGroups
            + static_cast<size_t>(mWorkerGroup) * schedLength;

        // Insert one 16-wide TAIL emission into this MID8 in four four-wide
        // batches, exactly matching SOF pushPullThroughMid; emitted IDs are
        // appended to this group's queue.
        auto routeTailIntoQueue = [&](uint32_t (*emitIds)[16], uint32_t (*emitOrdinals)[16], const int* emitCounts) {
            const int available = emitCounts[mParent];
            for (int batch = 0; batch < 4; ++batch) {
                const int retainedCount = mRetained;
                const int firstSource = batch * 4;
                const int newCount = max(0, min(4, available - firstSource));
                const int source = firstSource + mWorkerLane;
                uint32_t newId = Invalid;
                uint32_t newOrdinal = 0xffffffffu;
                float newKey = CUDART_INF_F;
                if (mWorkerLane < newCount) {
                    newId = emitIds[mParent][source];
                    newOrdinal = emitOrdinals[mParent][source];
                    if (newId != Invalid) {
                        newKey = sofMid8RepresentativeDepth(
                            particles, newId, midOrigin[mWorkerGroup],
                            midDirection[mWorkerGroup], sofDepthFallbackHitCounter);
                    }
                }
                const unsigned validMask = __ballot_sync(
                    mSubgroupMask, mWorkerLane < newCount && newId != Invalid);
                const int validNewCount = __popc(validMask & mSubgroupMask);
                const bool hasRetained = mWorkerLane < retainedCount;
                const float retainedKey = hasRetained
                    ? midKeys[mWorkerGroup][mWorkerLane] : CUDART_INF_F;
                const uint32_t retainedOrdinal = hasRetained
                    ? midOrdinals[mWorkerGroup][mWorkerLane] : 0xffffffffu;
                const uint32_t retainedId = hasRetained
                    ? midIds[mWorkerGroup][mWorkerLane] : Invalid;
                const int retainedRank = sofMid8Rank(
                    retainedKey, retainedOrdinal,
                    midKeys[mWorkerGroup], midOrdinals[mWorkerGroup], retainedCount,
                    newKey, newOrdinal, newCount, mSubgroupMask);
                const int newRank = sofMid8Rank(
                    newKey, newOrdinal,
                    midKeys[mWorkerGroup], midOrdinals[mWorkerGroup], retainedCount,
                    newKey, newOrdinal, newCount, mSubgroupMask);
                const int mergedCount = retainedCount + validNewCount;
                const int emitCount = mergedCount > 4 ? 4 : 0;
                __syncwarp(mSubgroupMask);  // all rank reads of the MID window done
                auto placeMid = [&](int rank, float key, uint32_t id, uint32_t ordinal) {
                    if (rank < emitCount) {
                        schedIds[mQueued + rank] = id;
                    } else if (rank < mergedCount) {
                        midKeys[mWorkerGroup][rank - emitCount] = key;
                        midIds[mWorkerGroup][rank - emitCount] = id;
                        midOrdinals[mWorkerGroup][rank - emitCount] = ordinal;
                    }
                };
                if (hasRetained) placeMid(retainedRank, retainedKey, retainedId, retainedOrdinal);
                if (mWorkerLane < newCount && newId != Invalid) placeMid(newRank, newKey, newId, newOrdinal);
                __syncwarp(mSubgroupMask);  // window writes visible to the next batch
                mRetained = mergedCount - emitCount;
                mQueued += emitCount;
            }
        };

        // SOF ingests 32 coarse-sorted IDs at a time. Each
        // 16-lane subgroup owns one TAIL64 and ranks two new candidates per
        // lane against the retained TAIL32 and all 32 new candidates.
        for (uint32_t chunk = tileParticleRangeIndices.x;
             chunk < tileParticleRangeIndices.y;
             chunk += 32u) {
            const int workerGroup = static_cast<int>(tileThreadIdx >> 4);
            const int workerLane = static_cast<int>(tileThreadIdx & 15u);
            const unsigned subgroupMask = (static_cast<unsigned>(tileThreadIdx) & 16u)
                ? 0xffff0000u : 0x0000ffffu;
            const int retainedCount = tailCounts[workerGroup];
            const uint32_t stop = min(chunk + 32u, tileParticleRangeIndices.y);
            const int newCount = static_cast<int>(stop - chunk);

            const uint32_t cursor0 = chunk + static_cast<uint32_t>(workerLane);
            const uint32_t cursor1 = cursor0 + 16u;
            uint32_t newId0 = Invalid;
            uint32_t newId1 = Invalid;
            float newKey0 = CUDART_INF_F;
            float newKey1 = CUDART_INF_F;
            if (workerLane < newCount) {
                newId0 = sortedTileParticleIdxPtr[cursor0];
                if (newId0 != Invalid) {
                    newKey0 = sofMid8RepresentativeDepth(
                        particles, newId0, tailOrigin[workerGroup],
                        tailDirection[workerGroup], sofDepthFallbackHitCounter);
                }
            }
            if (workerLane + 16 < newCount) {
                newId1 = sortedTileParticleIdxPtr[cursor1];
                if (newId1 != Invalid) {
                    newKey1 = sofMid8RepresentativeDepth(
                        particles, newId1, tailOrigin[workerGroup],
                        tailDirection[workerGroup], sofDepthFallbackHitCounter);
                }
            }
            const unsigned valid0 = __ballot_sync(
                subgroupMask, workerLane < newCount && newId0 != Invalid);
            const unsigned valid1 = __ballot_sync(
                subgroupMask, workerLane + 16 < newCount && newId1 != Invalid);
            const int validNewCount = __popc(valid0 & subgroupMask)
                                    + __popc(valid1 & subgroupMask);

            const int retainedSlot0 = workerLane;
            const int retainedSlot1 = workerLane + 16;
            const bool hasRetained0 = retainedSlot0 < retainedCount;
            const bool hasRetained1 = retainedSlot1 < retainedCount;
            const float retainedKey0 = hasRetained0
                ? tailKeys[workerGroup][retainedSlot0] : CUDART_INF_F;
            const float retainedKey1 = hasRetained1
                ? tailKeys[workerGroup][retainedSlot1] : CUDART_INF_F;
            const uint32_t retainedOrdinal0 = hasRetained0
                ? tailOrdinals[workerGroup][retainedSlot0] : 0xffffffffu;
            const uint32_t retainedOrdinal1 = hasRetained1
                ? tailOrdinals[workerGroup][retainedSlot1] : 0xffffffffu;
            const uint32_t retainedId0 = hasRetained0
                ? tailIds[workerGroup][retainedSlot0] : Invalid;
            const uint32_t retainedId1 = hasRetained1
                ? tailIds[workerGroup][retainedSlot1] : Invalid;
            const int rankRetained0 = sofTail64Rank(
                retainedKey0, retainedOrdinal0,
                tailKeys[workerGroup], tailOrdinals[workerGroup], retainedCount,
                newKey0, cursor0, newKey1, cursor1, newCount, subgroupMask);
            const int rankRetained1 = sofTail64Rank(
                retainedKey1, retainedOrdinal1,
                tailKeys[workerGroup], tailOrdinals[workerGroup], retainedCount,
                newKey0, cursor0, newKey1, cursor1, newCount, subgroupMask);
            const int rankNew0 = sofTail64Rank(
                newKey0, cursor0,
                tailKeys[workerGroup], tailOrdinals[workerGroup], retainedCount,
                newKey0, cursor0, newKey1, cursor1, newCount, subgroupMask);
            const int rankNew1 = sofTail64Rank(
                newKey1, cursor1,
                tailKeys[workerGroup], tailOrdinals[workerGroup], retainedCount,
                newKey0, cursor0, newKey1, cursor1, newCount, subgroupMask);
            // Emit 16 while more than 32 remain, at most twice.
            const int mergedCount = retainedCount + validNewCount;
            const int emitA = mergedCount > TailRetained ? 16 : 0;
            const int emitB = (mergedCount - emitA) > TailRetained ? 16 : 0;
            const int emitted = emitA + emitB;
            __syncwarp(subgroupMask);  // all rank reads of the TAIL window done
            auto placeTail = [&](int rank, float key, uint32_t id, uint32_t ordinal) {
                if (rank < emitA) {
                    tailEmitIdsA[workerGroup][rank] = id;
                    tailEmitOrdinalsA[workerGroup][rank] = ordinal;
                } else if (rank < emitted) {
                    tailEmitIdsB[workerGroup][rank - emitA] = id;
                    tailEmitOrdinalsB[workerGroup][rank - emitA] = ordinal;
                } else if (rank < mergedCount) {
                    tailKeys[workerGroup][rank - emitted] = key;
                    tailIds[workerGroup][rank - emitted] = id;
                    tailOrdinals[workerGroup][rank - emitted] = ordinal;
                }
            };
            if (hasRetained0) placeTail(rankRetained0, retainedKey0, retainedId0, retainedOrdinal0);
            if (hasRetained1) placeTail(rankRetained1, retainedKey1, retainedId1, retainedOrdinal1);
            if (workerLane < newCount && newId0 != Invalid) placeTail(rankNew0, newKey0, newId0, cursor0);
            if (workerLane + 16 < newCount && newId1 != Invalid) placeTail(rankNew1, newKey1, newId1, cursor1);
            if (workerLane == 0) {
                tailCounts[workerGroup] = mergedCount - emitted;
                tailEmitCountsA[workerGroup] = emitA;
                tailEmitCountsB[workerGroup] = emitB;
            }
            // Publish both TAIL emissions to the MID workers (other warps).
            // This barrier also orders the previous chunk's queue consumption
            // before this chunk's queue writes.
            if (__syncthreads_or(emitA > 0)) {
                routeTailIntoQueue(tailEmitIdsA, tailEmitOrdinalsA, tailEmitCountsA);
                routeTailIntoQueue(tailEmitIdsB, tailEmitOrdinalsB, tailEmitCountsB);
                __syncthreads();  // MID reads of the TAIL emissions done before reuse
            }
        }

        // Drain the retained TAIL32 in two 16-wide batches.
        {
            const int workerGroup = static_cast<int>(tileThreadIdx >> 4);
            const int workerLane = static_cast<int>(tileThreadIdx & 15u);
            const int count = tailCounts[workerGroup];
            const int drainA = min(16, count);
            const int drainB = min(16, count - drainA);
            for (int slot = workerLane; slot < drainA + drainB; slot += 16) {
                if (slot < drainA) {
                    tailEmitIdsA[workerGroup][slot] = tailIds[workerGroup][slot];
                    tailEmitOrdinalsA[workerGroup][slot] = tailOrdinals[workerGroup][slot];
                } else {
                    tailEmitIdsB[workerGroup][slot - drainA] = tailIds[workerGroup][slot];
                    tailEmitOrdinalsB[workerGroup][slot - drainA] = tailOrdinals[workerGroup][slot];
                }
            }
            __syncwarp((static_cast<unsigned>(tileThreadIdx) & 16u) ? 0xffff0000u : 0x0000ffffu);
            if (workerLane == 0) {
                tailEmitCountsA[workerGroup] = drainA;
                tailEmitCountsB[workerGroup] = drainB;
                tailCounts[workerGroup] = count - drainA - drainB;
            }
        }
        __syncthreads();
        routeTailIntoQueue(tailEmitIdsA, tailEmitOrdinalsA, tailEmitCountsA);
        routeTailIntoQueue(tailEmitIdsB, tailEmitOrdinalsB, tailEmitCountsB);

        // Drain the retained MID4 behind them, in window order.
        for (int i = mWorkerLane; i < mRetained; i += 4) {
            schedIds[mQueued + i] = midIds[mWorkerGroup][i];
        }
        mQueued += mRetained;
        mRetained = 0;
        if (mWorkerLane == 0) {
            gSofSchedCounts[static_cast<size_t>(schedTile) * MidGroups + mWorkerGroup] = static_cast<uint32_t>(mQueued);
        }

    }

    // Consumer: each pixel feeds its MID group's precomputed SOF emission
    // sequence (written by evalSOFMid8Schedule for this exact tile list)
    // through its exact HEAD4, in the identical order the cascade emitted it.
    template <typename TRay>
    static inline __device__ void evalSOFMid8Hierarchy(
        const threedgut::RenderParameters& params,
        TRay& ray,
        Particles& particles,
        const tcnn::uvec2& tileParticleRangeIndices,
        uint32_t tileNumBlocksToProcess,
        uint32_t tileNumParticlesToProcess,
        const uint32_t tileThreadIdx,
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const float* __restrict__ particlesGlobalDepthPtr,
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
        float* __restrict__ particlesPositionAbsGradPtr = nullptr,
        unsigned long long* __restrict__ sofDepthFallbackHitCounter = nullptr) {
        (void)params; (void)tileNumBlocksToProcess; (void)tileNumParticlesToProcess; (void)tileThreadIdx;
        (void)sortedTileParticleIdxPtr; (void)particlesGlobalDepthPtr; (void)sofDepthFallbackHitCounter;
        constexpr int MidGroups = 64;
        const int midGroup = (static_cast<int>(threadIdx.y) >> 1) * 8 + (static_cast<int>(threadIdx.x) >> 1);
        const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t length = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t* __restrict__ ids = gSofSchedIds
            + static_cast<size_t>(tileParticleRangeIndices.x - gSofSchedBase) * MidGroups
            + static_cast<size_t>(midGroup) * length;
        const uint32_t count = (length > 0u)
            ? gSofSchedCounts[static_cast<size_t>(tile) * MidGroups + midGroup] : 0u;
        HitParticleHeadBuffer hitParticleHead;
        for (uint32_t k = 0; k < count && ray.isAlive(); ++k) {
            const uint32_t id = ids[k];
            feedHead(ray, hitParticleHead, particles, id,
                     particles.fetchDensityParameters(id),
                     particleFeaturesBuffer, particleFeaturesGradientBuffer,
                     particlesPositionAbsGradPtr);
        }
        for (int i = 0; ray.isAlive() && i < static_cast<int>(hitParticleHead.numHits()); ++i) {
            processHitParticle(
                ray,
                hitParticleHead[GAUSSIAN_HEAD_SIZE - hitParticleHead.numHits() + i],
                particles,
                particleFeaturesBuffer,
                particleFeaturesGradientBuffer,
                particlesPositionAbsGradPtr);
        }
    }

#endif  // GAUSSIAN_SOF_MID8_HIERARCHY
