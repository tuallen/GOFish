#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# GOFish on FIORD: train, test-render, FPS, mesh and evaluation of every paper scene.
#
#   bash benchmark/fiord.sh --data /data/fiord [--lidar /data/fiord_lidar/%s.npy] [options]
#
#   Scenes are the paper labels Kitchen_In Corridor_Out MeetingRoom_In Road_Out Bridge_Out Building_In
#   Hall_In (also for --scenes and the outputs OUT/<label>); gofish.data.fiord maps them to the release folders.
#   --data   FIORD root: the unzipped release, <folder>/{model, images}, prepared into OUT/_data/<label> by
#            gofish.data.fiord, or prepared scenes <label>/{sparse/0, images_4}
#   --lidar  printf pattern of each scene's LiDAR cloud; %s receives the paper label
#            (.npy from gofish.data.fiord lidar, or .ply); geometry is evaluated when given
set -euo pipefail
DATASET=fiord; SCENES="Kitchen_In Corridor_Out MeetingRoom_In Road_Out Bridge_Out Building_In Hall_In"; LIDAR=""
source "$(dirname "$0")/common.sh"
for scene in ${SCENES}; do
  # Camera sampler seed of the paper's runs: 1234 for Kitchen_In, 771234 for every other scene.
  seed=771234; [ "${scene}" = "Kitchen_In" ] && seed=1234
  data="${DATA}/${scene}"
  if [ ! -d "${data}/sparse/0" ]; then
    folder=$(python -m gofish.data.fiord folder "${scene}")
    data="${OUT}/_data/${scene}"
    [ "${DRY}" = 1 ] || [ -d "${data}/images_4" ] || python -m gofish.data.fiord scene "${DATA}/${folder}" "${data}"
  fi
  train "${scene}" fiord "${data}" dataset.camera_sampler_seed=${seed}
  [ "${DRY}" = 1 ] && continue
  run="${OUT}/${scene}"
  render_and_mesh "${run}"
  rect="${OUT}/_rectified/${scene}"
  [ -d "${rect}" ] || python -m gofish.eval.fiord.rectify --scene "${data}" --out "${rect}"
  # Full and Common are scored from the saved 8-bit PNGs, as the paper's tables; Full over each lens's aperture.
  python -m gofish.eval.common.dataset_cameras --checkpoint "${run}/ckpt_last.pt" --data "${data}" \
    --no-frames --out "${run}/cameras_test"
  python -m gofish.eval.fiord.render_metrics --scene "${data}" --pred "${run}/heldout/pred" \
    --region full --masks "${run}/cameras_test/masks" --device cuda --out "${run}/eval_full.json"
  python -m gofish.eval.fiord.render_metrics --scene "${data}" --pred "${run}/heldout/pred" \
    --region common --common-gt "${rect}" --device cuda --out "${run}/eval_common.json"
  if [ -n "${LIDAR}" ] && [ ! -f "gofish/eval/fiord/alignments/${scene}.json" ]; then
    echo "${scene}: no SfM-to-LiDAR alignment; scored for rendering only, as in the paper" >&2
  elif [ -n "${LIDAR}" ]; then
    python -m gofish.eval.fiord.training_cameras --checkpoint "${run}/ckpt_last.pt" \
      --data "${data}" --out "${run}/cameras"
    python -m gofish.eval.fiord.mesh_metrics --mesh "${run}/mesh/mesh.ply" --gt "$(printf "${LIDAR}" "${scene}")" \
      --alignment "gofish/eval/fiord/alignments/${scene}.json" --cameras "${run}/cameras/training_cameras.json" \
      --cache "${OUT}/_observed_gt/${scene}" --out "${run}/eval_mesh.json"
  fi
done
[ "${DRY}" = 1 ] || python -m gofish.eval.summary --fiord "${OUT}" --scenes "${SCENES}"
