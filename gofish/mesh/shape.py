# Portions derived from GOF (https://github.com/autonomousvision/gaussian-opacity-fields),
# distributed under the Gaussian-Splatting License (Inria / MPII; non-commercial
# research and evaluation use): third_party/licenses/GAUSSIAN_SPLATTING_LICENSE.md

"""Effective Gaussian shape used by mesh extraction.

The mesh is extracted from the intrinsic scene: each Gaussian's scale is
combined with its training-camera Mip 3-D filter, Sbar^2 = S^2 + eps^2 I, and
the rotation is kept, B = R Sbar. No per-view footprint is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEAD_OPACITY = 1.0 / 255.0


@dataclass
class EffectiveShape:
    positions: np.ndarray  # [N, 3] float32
    density: np.ndarray  # [N, 1] float32, sigmoid opacity
    factor: np.ndarray  # [N, 3, 3] float32, B = R Sbar
    inverse_factor: np.ndarray  # [N, 3, 3] float32, B^-1 = Sbar^-1 R^T
    sigma_max: np.ndarray  # [N, 1] float32, largest filtered scale
    retained: np.ndarray  # [N] bool, opacity > 1/255

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def packed_density(self) -> np.ndarray:
        """[N, 32] float32 in the renderer's full-shape ParticleDensity layout."""
        count = self.count
        packed = np.concatenate((
            self.positions,
            self.density,
            np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (count, 1)),
            np.ones((count, 3), np.float32),
            np.zeros((count, 1), np.float32),
            self.factor.reshape(count, 9),
            self.inverse_factor.reshape(count, 9),
            np.zeros((count, 2), np.float32),
        ), axis=1, dtype=np.float32)
        assert packed.shape == (count, 32) and packed.flags.c_contiguous
        return packed


def _mip_filter(checkpoint: dict) -> torch.Tensor:
    """The checkpoint's training-camera Mip 3-D filter, [N, 1] (zero if trained without it)."""
    from threedgrut.model.model import load_checkpoint_filter

    filt, _ = load_checkpoint_filter(checkpoint)
    if filt is None:
        return torch.zeros_like(checkpoint["positions"][:, :1])
    return filt


def _to_numpy(value: torch.Tensor, dtype: str = "<f4") -> np.ndarray:
    return value.detach().cpu().contiguous().numpy().astype(np.dtype(dtype), copy=False)


def effective_shape(checkpoint: dict, device: str = "cuda") -> EffectiveShape:
    """Filtered shape of every Gaussian in a geometry checkpoint."""
    from threedgrut.utils.misc import quaternion_to_so3

    config = checkpoint["config"]
    if str(config.model.scale_activation) != "exp" or str(config.model.density_activation) != "sigmoid":
        raise ValueError("meshing expects exp scale and sigmoid density activations")

    positions = checkpoint["positions"].to(device).detach().contiguous()
    raw_scales = torch.exp(checkpoint["scale"].to(device).detach()).contiguous()
    rotations = quaternion_to_so3(checkpoint["rotation"].to(device).detach()).contiguous()
    alpha = torch.sigmoid(checkpoint["density"].to(device).detach()).contiguous()
    filt = _mip_filter(checkpoint).detach().to(device=positions.device, dtype=positions.dtype).contiguous()
    count = int(positions.shape[0])
    if tuple(filt.shape) != (count, 1):
        raise ValueError("the checkpoint Mip filter does not match the Gaussian count")

    scales_eff = torch.sqrt(raw_scales.square() + filt.square()).contiguous()
    diagonal = torch.diag_embed(scales_eff)
    inverse_diagonal = torch.diag_embed(scales_eff.reciprocal())
    factor = (rotations @ diagonal).contiguous()
    inverse = (inverse_diagonal @ rotations.transpose(-1, -2)).contiguous()
    sigma_max = scales_eff.max(dim=1, keepdim=True).values.contiguous()
    retained = alpha[:, 0] > DEAD_OPACITY
    if not bool(torch.isfinite(factor).all() and torch.isfinite(inverse).all()):
        raise ValueError("the filtered Gaussian shape is not finite")

    return EffectiveShape(
        positions=_to_numpy(positions),
        density=_to_numpy(alpha),
        factor=_to_numpy(factor),
        inverse_factor=_to_numpy(inverse),
        sigma_max=_to_numpy(sigma_max),
        retained=_to_numpy(retained.to(torch.uint8), "u1") != 0,
    )
