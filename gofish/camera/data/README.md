# Calibrated camera domains

Each calibrated domain is matched to a camera by its native COLMAP record
(`model`, `width`, `height`, `params`):

* `kind: aperture` — measured lens aperture, a circle `aperture_center`,
  `aperture_radius` in native pixels, plus the first monotone branch of the
  distortion polynomial (`theta_limit`, and `rho_limit = rho(theta_limit)`).
* `kind: max_angle` — full-raster camera with the angular cap `max_angle`.

`fiord.json` holds the two Insta360 lenses for every FIORD paper scene, and
`scannetpp.json` holds the DSLR calibrations used for ScanNet++. Create records
for new data with `python -m gofish.camera.make_domain`.
