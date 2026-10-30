#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# GOFish on ScanNet++ DSLR: train, test-render, FPS, mesh and evaluation of every paper scene.
#
#   bash benchmark/scannetpp.sh --data /data/scannetpp [options]
#
#   --data   ScanNet++ data root, <scene>/dslr/{colmap, resized_images, nerfstudio, undistorted_images}
#            and <scene>/scans/mesh_aligned_0.05.ply; dslr/resized_images_2 is created when missing.
set -euo pipefail
DATASET=scannetpp; SCENES="0a5c013435 8d563fc2cc bb87c292ad d415cc449b e8ea9b4da8 fe1733741f"
source "$(dirname "$0")/common.sh"
for scene in ${SCENES}; do
  dslr="${DATA}/${scene}/dslr"
  [ "${DRY}" = 1 ] || [ -d "${dslr}/resized_images_2" ] || python -m gofish.data.scannetpp "${dslr}"
  train "${scene}" scannetpp "${dslr}"
  [ "${DRY}" = 1 ] && continue
  run="${OUT}/${scene}"
  render_and_mesh "${run}"
  python -m gofish.eval.scannetpp.render_metrics --scene "${DATA}/${scene}" --pred "${run}/heldout/pred" \
    --region full --out "${run}/eval_full.json"
  python -m gofish.eval.scannetpp.render_metrics --scene "${DATA}/${scene}" --pred "${run}/heldout/pred" \
    --region common --device cuda --out "${run}/eval_common.json"
  python -m gofish.eval.scannetpp.mesh_metrics --mesh "${run}/mesh/mesh.ply" \
    --gt "${DATA}/${scene}/scans/mesh_aligned_0.05.ply" --out "${run}/eval_mesh.json"
done
[ "${DRY}" = 1 ] || python -m gofish.eval.summary --scannetpp "${OUT}" --scenes "${SCENES}"
