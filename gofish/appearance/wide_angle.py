# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Wide-Angle Appearance Modeling: PPISP's camera model extended to the full fisheye aperture.

Released PPISP models vignetting with a radial polynomial that cannot follow
the steep fall-off near a fisheye's rim.  Wide-Angle Appearance Modeling multiplies the
PPISP vignetting of each camera by ``W = w(rho) exp(-a(u))``, the product of

* a nonincreasing radial profile ``W_r(rho) = w(rho)`` (``radial_attenuation``): K
  piecewise-linear log-knots over ``rho in [rho_ap / 2, rho_ap]``, with
  ``rho = |p - pp| / max(W, H)`` and ``rho_ap`` the calibrated-domain radius
  (``geometry_from``), identity inside ``rho_ap / 2``;
* a flat-field attenuation ``W_ff(p) = exp(-a(p))`` (``flatfield_attenuation``): a G x G grid of
  non-negative log-attenuations whose cells inside ``rho_ap / 2`` stay at
  identity (``flatfield_gauge``), so the map cannot trade a global scale
  with exposure.

``post_processing.radial_attenuation_knots`` = K and ``flatfield_grid`` = G
(50 and 50 in the FIORD, FullCircle and ScanNet++ recipes; 0, released PPISP with
``W = 1``, in the Tanks & Temples recipe for its pinhole cameras).  The
factors are applied after the controller with PPISP's vignetting
(``gofish_ppisp.PPISP``).  ``ppisp_config_from`` is the single place where a
GOFish config becomes a ``PPISPConfig``.  These functions need torch only;
PPISP's CUDA kernels load on first use.
"""

from __future__ import annotations

from gofish_ppisp.wide_angle import (
    flatfield_attenuation,
    flatfield_gauge,
    geometry_from,
    pixel_centres,
    radial_attenuation,
)


def ppisp_config_from(conf, *, training: bool = False, scene_steps: int | None = None):
    """``(PPISPConfig, distillation start step or -1)`` of a GOFish config.

    The controller distils during the last ``post_processing.n_distillation_steps``
    steps; ``scene_steps`` defaults to ``n_iterations - n_distillation_steps``.
    With ``training`` the config also carries the scheduler horizon and the
    Wide-Angle Appearance Modeling sizes; otherwise ``PPISP.from_state_dict`` takes the
    sizes from the checkpoint.
    """
    from gofish_ppisp import PPISPConfig

    post = conf.post_processing
    use_controller = post.get("use_controller", True)
    n_distillation_steps = post.get("n_distillation_steps", 5000)
    if scene_steps is None:
        scene_steps = int(conf.n_iterations) - int(n_distillation_steps)
    if use_controller and n_distillation_steps > 0:
        ratio, distillation, start = scene_steps / conf.n_iterations, True, scene_steps
    elif use_controller:
        ratio, distillation, start = 0.8, False, -1
    else:
        ratio, distillation, start = 0.0, False, -1
    if not training:
        return PPISPConfig(use_controller=use_controller, controller_distillation=distillation,
                           controller_activation_ratio=ratio), start
    config = PPISPConfig(
        use_controller=use_controller,
        controller_distillation=distillation,
        controller_activation_ratio=ratio,
        scheduler_decay_max_steps=scene_steps,
        radial_attenuation_knots=int(post.get("radial_attenuation_knots", 0)),
        flatfield_grid=int(post.get("flatfield_grid", 0)),
    )
    return config, start


__all__ = ["flatfield_attenuation", "flatfield_gauge", "geometry_from", "pixel_centres", "ppisp_config_from",
           "radial_attenuation"]
