# SPDX-License-Identifier: Apache-2.0
"""Scene extents for the SOF world-size prune (``strategy.prune_big.extent``).

``median``
    the trainer's scene extent: 1.1 x the median distance of the training
    cameras from their coordinatewise-median centre (passed in by the trainer).
``sof_max_radius``
    SOF's extent (``getNerfppNorm`` of its released code): 1.1 x the largest
    distance of the training cameras from their mean.
"""

import numpy as np

WORLD_SIZE_EXTENTS = ("median", "sof_max_radius")


def sof_max_radius_extent(observer_points) -> float:
    """1.1 x the largest training-camera distance from the camera mean, computed in float64."""
    centres = np.asarray(observer_points, dtype=np.float64).reshape(-1, 3)
    return float(1.1 * np.linalg.norm(centres - centres.mean(0), axis=1).max())
