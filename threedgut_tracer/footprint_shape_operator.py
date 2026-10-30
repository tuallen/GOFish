# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Autograd boundary of the native full-shape factor operator.

For each Gaussian and view this forms C = Sigma + Q_v, its square-root factor
and inverse, and the mass-preserving ray-marginal factor
m = sqrt(det S0 / det(S0 + gamma I)), S0 = J Sigma J^T, with an analytic
VJP back to the intrinsic scale, rotation and opacity.  The torch references
are ``view_observation_filter.footprint_covariance`` and
``mass_preserving_factor``.  Rows are independent,
so large scenes are processed in contiguous chunks that fit device memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from threedgut_tracer.footprint_chunk_planner import plan_footprint_chunks


_NATIVE = None
_LOCAL_ROW = re.compile(r"\blocal_row=(\d+)\b")


def bind_native(module) -> None:
    """Bind the JIT-built operator (see setup_3dgut.load_full_shape_operator)."""
    global _NATIVE
    _NATIVE = module


def _native():
    if _NATIVE is None:
        raise RuntimeError("the full-shape operator is not built; construct the Tracer first")
    return _NATIVE


class FootprintNumericalError(RuntimeError):
    """Native factorization failure with the offending global row."""

    def __init__(self, message: str, *, global_row: int | None, completed_native_chunks: int):
        super().__init__(message)
        self.global_row = global_row
        self.completed_native_chunks = int(completed_native_chunks)


class _FootprintFullShape(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scales,
        quaternion,
        density,
        q6,
        jacobian,
        observed,
        mass_preserving,
    ):
        native = _native()
        scales_c = scales.contiguous()
        quaternion_c = quaternion.contiguous()
        density_c = density.contiguous()
        for name, value in (
            ("q6", q6), ("jacobian", jacobian), ("observed", observed)
        ):
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous before detach")
        q6_c = q6.detach()
        jacobian_c = jacobian.detach()
        observed_c = observed.detach()
        outputs = native.forward(
            scales_c,
            quaternion_c,
            density_c,
            q6_c,
            jacobian_c,
            observed_c,
            bool(mass_preserving),
        )
        if len(outputs) != 21:
            raise RuntimeError("native footprint full-shape forward returned wrong arity")
        (
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
            projected_q64,
        ) = outputs
        ctx.native = native
        ctx.mass_preserving = bool(mass_preserving)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            scales_c,
            quaternion_c,
            density_c,
            q6_c,
            jacobian_c,
            observed_c,
            covariance,
            inverse_factor,
            compensation,
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
            projected_q64,
        )
        return (
            covariance,
            factor,
            inverse_factor,
            compensation,
            filtered_density,
        )

    @staticmethod
    def backward(
        ctx,
        grad_covariance,
        grad_factor,
        grad_inverse_factor,
        grad_compensation,
        grad_filtered_density,
    ):
        (
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
            _q_roundoff_projected,
            unsafe_indices,
            covariance64,
            inverse_factor64,
            compensation64,
            root64,
            rotation64,
            eigenvectors64,
            roots64,
            projected_q64,
        ) = ctx.saved_tensors
        grad_covariance = (
            torch.zeros_like(covariance)
            if grad_covariance is None
            else grad_covariance.contiguous()
        )
        grad_factor = (
            torch.zeros_like(covariance)
            if grad_factor is None
            else grad_factor.contiguous()
        )
        grad_inverse_factor = (
            torch.zeros_like(inverse_factor)
            if grad_inverse_factor is None
            else grad_inverse_factor.contiguous()
        )
        grad_compensation = (
            torch.zeros_like(compensation)
            if grad_compensation is None
            else grad_compensation.contiguous()
        )
        grad_filtered_density = (
            torch.zeros_like(density)
            if grad_filtered_density is None
            else grad_filtered_density.contiguous()
        )
        grad_scales, grad_quaternion, grad_density = ctx.native.backward(
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
            ctx.mass_preserving,
        )
        # Q and validity are detached lifecycle state; the boolean mode is not
        # differentiable.
        return grad_scales, grad_quaternion, grad_density, None, None, None, None


def _native_failure(error: RuntimeError, *, row_offset: int, completed_native_chunks: int) -> FootprintNumericalError:
    match = _LOCAL_ROW.search(str(error))
    global_row = None if match is None else row_offset + int(match.group(1))
    suffix = " global_row=unavailable" if global_row is None else f" global_row={global_row}"
    return FootprintNumericalError(str(error) + suffix, global_row=global_row,
                             completed_native_chunks=completed_native_chunks)


def _is_native_numerical_failure(error: RuntimeError) -> bool:
    return "footprint cuSOLVER full-shape" in str(error)


def _apply_footprint_chunks(
    scales,
    quaternion,
    density,
    q6,
    jacobian,
    observed,
    mass_preserving,
    chunk_rows,
):
    """Exact per-particle partition used at runtime."""
    count = int(scales.shape[0])
    rows = int(chunk_rows)
    if rows <= 0 and count:
        raise ValueError("positive footprint chunk size required")
    if count == 0 or rows >= count:
        try:
            return _FootprintFullShape.apply(
                scales, quaternion, density, q6, jacobian, observed,
                bool(mass_preserving))
        except RuntimeError as error:
            if not _is_native_numerical_failure(error):
                raise
            raise _native_failure(
                error, row_offset=0, completed_native_chunks=0) from error
    chunks = []
    for chunk_index, start in enumerate(range(0, count, rows)):
        stop = min(count, start + rows)
        try:
            chunks.append(_FootprintFullShape.apply(
                scales[start:stop],
                quaternion[start:stop],
                density[start:stop],
                q6[start:stop],
                jacobian[start:stop],
                observed[start:stop],
                bool(mass_preserving),
            ))
        except RuntimeError as error:
            if not _is_native_numerical_failure(error):
                raise
            raise _native_failure(
                error, row_offset=start,
                completed_native_chunks=chunk_index) from error
    return tuple(torch.cat([chunk[index] for chunk in chunks], dim=0)
                 for index in range(5))


def _runtime_chunk_plan(scales):
    free_bytes, total_bytes = torch.cuda.mem_get_info(scales.device)
    reserved = torch.cuda.memory_reserved(scales.device)
    allocated = torch.cuda.memory_allocated(scales.device)
    return plan_footprint_chunks(
        int(scales.shape[0]),
        driver_free_bytes=int(free_bytes),
        device_total_bytes=int(total_bytes),
        allocator_reserved_bytes=int(reserved),
        allocator_allocated_bytes=int(allocated),
    )


@dataclass(frozen=True)
class FullShape:
    covariance: torch.Tensor
    factor: torch.Tensor
    inverse_factor: torch.Tensor
    compensation: torch.Tensor
    density: torch.Tensor
    observed: torch.Tensor


def build_full_shape(
    scales: torch.Tensor,
    quaternion: torch.Tensor,
    density: torch.Tensor,
    q6: torch.Tensor,
    jacobian: torch.Tensor,
    observed: torch.Tensor,
    *,
    mass_preserving: bool = True,
) -> FullShape:
    """Full-shape factors for detached Q (float64 [N,6]) and camera Jacobian (float32 [N,2,3])."""
    if q6.requires_grad or q6.dtype != torch.float64 or q6.shape != (scales.shape[0], 6):
        raise ValueError("q6 must be detached float64 with shape [N,6]")
    if jacobian.requires_grad or jacobian.dtype != torch.float32 or jacobian.shape != (scales.shape[0], 2, 3):
        raise ValueError("jacobian must be detached float32 with shape [N,2,3]")
    if observed.requires_grad or observed.dtype != torch.bool:
        raise ValueError("observed must be a detached bool tensor")
    plan = _runtime_chunk_plan(scales)
    covariance, factor, inverse_factor, compensation, filtered_density = _apply_footprint_chunks(
        scales, quaternion, density, q6, jacobian, observed, bool(mass_preserving), plan.chunk_rows)
    return FullShape(covariance=covariance, factor=factor, inverse_factor=inverse_factor,
                     compensation=compensation, density=filtered_density, observed=observed)


__all__ = ["FootprintNumericalError", "FullShape", "bind_native", "build_full_shape"]
