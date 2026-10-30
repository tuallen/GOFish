# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Deterministic memory plan for the per-particle full-shape factor operator.

All constants are derived from the native tensor inventory, cuSOLVER's 16,384
row batch, CUDA's signed launch-index limit, or current allocator accounting.
They are implementation limits, not method hyperparameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


SOLVER_BATCH_ROWS = 16384
CUDA_THREADS = 256
CUDA_SIGNED_BLOCK_LIMIT = (1 << 31) - 1
CUDA_INDEX_ROW_LIMIT = CUDA_THREADS * CUDA_SIGNED_BLOCK_LIMIT

# Native forward retained state (float32 matrices/vectors, flags and indices),
# the five concatenated public outputs, and alignment are below 320 bytes/row
# when the rare float64 escalation path is absent. Float64 escalation remains
# chunk-local and is covered by the 1 KiB transient envelope below.
CHUNKED_RETAINED_AND_CONCAT_BYTES_PER_ROW = 320
# Native forward fixed outputs, eigensystem work arrays, the all-unsafe float64
# branch, backward products and contiguous-input copies total < 883 bytes/row.
# Round upward to one binary KiB for allocator granularity and library work.
NATIVE_PEAK_BYTES_PER_CHUNK_ROW = 1024
# cuSOLVER/context and caching-allocator headroom: one thirty-second of device
# capacity, never less than 64 MiB. This is device-derived and scene-independent.
MIN_LIBRARY_HEADROOM_BYTES = 64 << 20
DEVICE_HEADROOM_DIVISOR = 32


@dataclass(frozen=True)
class FootprintChunkPlan:
    mode: str
    particle_count: int
    chunk_rows: int
    chunk_count: int
    driver_free_bytes: int
    reusable_reserved_bytes: int
    effective_available_bytes: int
    device_total_bytes: int
    library_headroom_bytes: int
    retained_and_concat_reservation_bytes: int
    transient_budget_bytes: int
    native_peak_bytes_per_chunk_row: int
    solver_batch_rows: int
    decision: str

    def record(self) -> dict[str, object]:
        return asdict(self)


def _aligned_rows(raw_rows: int, count: int) -> int:
    bounded = min(int(raw_rows), int(count), CUDA_INDEX_ROW_LIMIT)
    if count <= SOLVER_BATCH_ROWS:
        return count if bounded >= count else 0
    if bounded < SOLVER_BATCH_ROWS:
        return 0
    return (bounded // SOLVER_BATCH_ROWS) * SOLVER_BATCH_ROWS


def plan_footprint_chunks(
    particle_count: int,
    *,
    driver_free_bytes: int,
    device_total_bytes: int,
    allocator_reserved_bytes: int,
    allocator_allocated_bytes: int,
) -> FootprintChunkPlan:
    count = int(particle_count)
    values = (
        count, int(driver_free_bytes), int(device_total_bytes),
        int(allocator_reserved_bytes), int(allocator_allocated_bytes))
    if any(value < 0 for value in values) or values[2] <= 0:
        raise ValueError("footprint memory-accounting values must be nonnegative")
    if values[4] > values[3]:
        raise ValueError("allocated CUDA bytes exceed reserved bytes")
    reusable = values[3] - values[4]
    effective = values[1] + reusable
    headroom = max(MIN_LIBRARY_HEADROOM_BYTES,
                   values[2] // DEVICE_HEADROOM_DIVISOR)
    retained = count * CHUNKED_RETAINED_AND_CONCAT_BYTES_PER_ROW
    full_peak = count * NATIVE_PEAK_BYTES_PER_CHUNK_ROW

    if count == 0:
        chunk_rows, mode, decision = 0, "single_call", "empty_identity"
    elif count <= CUDA_INDEX_ROW_LIMIT and effective >= headroom + full_peak:
        chunk_rows, mode, decision = (
            count, "single_call", "one_call_fits_accounted_peak")
    else:
        transient = effective - headroom - retained
        raw_rows = transient // NATIVE_PEAK_BYTES_PER_CHUNK_ROW
        chunk_rows = _aligned_rows(raw_rows, count)
        if chunk_rows <= 0:
            raise MemoryError(
                "footprint retained outputs plus device-derived library headroom "
                "do not fit current CUDA memory")
        mode, decision = (
            "contiguous_chunks",
            "full_call_exceeds_accounted_peak_or_cuda_index_limit")
    chunk_count = (0 if count == 0 else
                   int(math.ceil(count / max(1, chunk_rows))))
    transient_budget = max(0, effective - headroom - retained)
    return FootprintChunkPlan(
        mode=mode,
        particle_count=count,
        chunk_rows=chunk_rows,
        chunk_count=chunk_count,
        driver_free_bytes=values[1],
        reusable_reserved_bytes=reusable,
        effective_available_bytes=effective,
        device_total_bytes=values[2],
        library_headroom_bytes=headroom,
        retained_and_concat_reservation_bytes=retained,
        transient_budget_bytes=transient_budget,
        native_peak_bytes_per_chunk_row=NATIVE_PEAK_BYTES_PER_CHUNK_ROW,
        solver_batch_rows=SOLVER_BATCH_ROWS,
        decision=decision,
    )


__all__ = [
    "CHUNKED_RETAINED_AND_CONCAT_BYTES_PER_ROW",
    "CUDA_INDEX_ROW_LIMIT",
    "FootprintChunkPlan",
    "NATIVE_PEAK_BYTES_PER_CHUNK_ROW",
    "SOLVER_BATCH_ROWS",
    "plan_footprint_chunks",
]
