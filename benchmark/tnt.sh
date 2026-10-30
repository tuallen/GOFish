#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# GOFish on Tanks & Temples: train, test-render, FPS, mesh and evaluation of every paper scene.
#
#   bash benchmark/tnt.sh --data /data/tnt [--tnt-eval /data/tnt_eval] [options]
#
#   --data      GOF-preprocessed TrainingSet, <scene>/{sparse/0, images_2}
#   --tnt-eval  official evaluation folders, <scene>/{<scene>.ply, <scene>_COLMAP_SfM.log, ...};
#               geometry is evaluated when given
set -euo pipefail
DATASET=tnt; SCENES="Barn Caterpillar Courthouse Ignatius Meetingroom Truck"; TNT_EVAL=""
source "$(dirname "$0")/common.sh"
for scene in ${SCENES}; do
  train "${scene}" tnt "${DATA}/${scene}"
  [ "${DRY}" = 1 ] && continue
  run="${OUT}/${scene}"
  render_and_mesh "${run}"
  python -m gofish.eval.tnt.render_metrics --pred "${run}/heldout/pred" --gt "${run}/heldout/gt" --out "${run}/eval_render.json"
  if [ -n "${TNT_EVAL}" ]; then
    python -m gofish.eval.tnt.mesh_metrics --mesh "${run}/mesh/mesh.ply" --gt-dir "${TNT_EVAL}/${scene}" \
      --traj "${TNT_EVAL}/${scene}/${scene}_COLMAP_SfM.log" --out "${run}/eval_mesh"
  fi
done
[ "${DRY}" = 1 ] || python -m gofish.eval.summary --tnt "${OUT}" --scenes "${SCENES}"
