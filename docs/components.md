# Using the components

Both components need only torch, plus their CUDA extension for the parts marked
so; see THIRD-PARTY-LICENSES for the license of each file.

## Wide-Angle Appearance Modeling

| File | Contents |
|---|---|
| `gofish_ppisp/wide_angle.py` | `radial_attenuation`, `flatfield_attenuation` (the factors w(ρ) and exp(−a(u)) of W), `geometry_from`, `flatfield_gauge`, `pixel_centres` |
| `gofish_ppisp/__init__.py` | `PPISP` with the factors applied after its vignetting; CUDA kernels load on first use |
| `gofish/appearance/wide_angle.py` | the model description, re-exports, `ppisp_config_from(conf)` |
| `threedgrut/datasets/wide_angle.py` | the same inputs computed from a GOFish dataset |

Config keys: `post_processing.radial_attenuation_knots` (K, 50 or 0 = off) and
`post_processing.flatfield_grid` (G, 50 or 0 = off; needs K > 0).

```python
import torch
from gofish_ppisp import PPISP, PPISPConfig
from gofish_ppisp.wide_angle import flatfield_gauge, geometry_from, pixel_centres

W, H = 816, 816
geometry = geometry_from([408.0, 408.0], [W, H], pixel_centres(W, H))   # pass only the valid pixels of a fisheye
ppisp = PPISP(num_cameras=1, num_frames=100, config=PPISPConfig(radial_attenuation_knots=50, flatfield_grid=50),
              radial_attenuation_geometry=torch.tensor([geometry]))
ppisp.flatfield_free[0] = flatfield_gauge(geometry, [W, H], 50).cuda()
rgb = ppisp(render, camera_idx=0, frame_idx=7)   # render: [H, W, 3] on CUDA; pixel centres are derived
```

## Calibrated Mip Filtering

| File | Contents |
|---|---|
| `threedgrut/physical_mip.py` (+ `physical_mip_csrc/`, CUDA) | the view-independent 3D filter ε: `pack_camera_table`, `compute_filter` |
| `threedgut_tracer/view_observation_filter.py` | the per-view footprint: torch references `footprint_covariance` (γJ⁺J⁺ᵀ) and `mass_preserving_factor` (m), the renderer's `current_view_q6` |
| `threedgut_tracer/footprint_shape_operator.py` (+ `src/footprint_shape_operator*`, CUDA) | C = Σ + γJ⁺J⁺ᵀ, its factors and m with their VJP |

Config keys: `render.gofish.scene_sensor_factorization` (on/off),
`render.gofish.mip_filter_variance` (0.2 px²) and
`render.gofish.pixel_footprint_variance` (gamma, 0.3 px²).

```python
from threedgut_tracer.view_observation_filter import footprint_covariance, mass_preserving_factor

q = footprint_covariance(J)            # J: [N, 2, 3] pixel Jacobians of the Gaussian centres in one view
m = mass_preserving_factor(Sigma, J)   # Sigma: [N, 3, 3]; draw Sigma + q with opacity m * o
```

`pack_camera_table` documents its camera and frame records with a runnable example.
