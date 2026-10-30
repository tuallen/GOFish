# Opacity-field extension

`gofish.mesh` evaluates a trained scene's opacity field at tetrahedral sites and
along bisection rays. It uses the 3DGUT particle model, unscented projection,
and native tile admission from the training renderer, together with entry
points specific to mesh extraction: the opacity field
(`opacityField.cuh`, `opacityFieldBinding.inl`), the bisection table and the
support enumerator of the calibrated camera domain.

This directory contains only the renderer headers needed by mesh extraction,
without the training forward or backward paths. It compiles into the separate
`gofish_opacity_field` extension through `gofish/mesh/build.py`. It remains
separate from `threedgut_tracer/` because the mesh extractor is a forward-only
consumer with its own buffers and kernels. Changes to the camera model,
particle model, or projector must be applied to both copies.
