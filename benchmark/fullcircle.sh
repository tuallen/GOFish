#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# GOFish on FullCircle: prepare, train, test-render, FPS, mesh and rendering evaluation of every paper scene.
#
#   bash benchmark/fullcircle.sh --data /data/FullCircle [options]
#
#   --data   the FullCircle release, <scene>/{sparse/0, images_4, masks_4} and masking/mask_test.png.
#            Each scene is prepared in OUT/_data/<scene>; FullCircle has no geometry ground truth.
set -euo pipefail
DATASET=fullcircle; SCENES="dark flat1 flat2 lab lounge persons room1 room2 room3"
source "$(dirname "$0")/common.sh"
[ "${DRY}" = 1 ] || [ -f "${DATA}/masking/mask_test.png" ] || { echo "${DATA}/masking/mask_test.png not found" >&2; exit 2; }
for scene in ${SCENES}; do
  data="${OUT}/_data/${scene}"
  [ "${DRY}" = 1 ] || [ -s "${data}/test_list.txt" ] || python -m gofish.data.fullcircle "${DATA}/${scene}" "${data}"
  train "${scene}" fullcircle "${data}"
  [ "${DRY}" = 1 ] && continue
  run="${OUT}/${scene}"
  # The mesh is the digital twin; FullCircle has no geometry ground truth, so it is not scored.
  render_and_mesh "${run}"
  for region in full common; do
    python -m gofish.eval.fullcircle.render_metrics --scene "${data}" --pred "${run}/heldout/pred" \
      --mask-test "${DATA}/masking/mask_test.png" --region ${region} --out "${run}/eval_${region}.json"
  done
done
[ "${DRY}" = 1 ] || python -m gofish.eval.summary --fullcircle "${OUT}" --scenes "${SCENES}"
