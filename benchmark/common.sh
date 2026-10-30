# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Shared option parsing of the dataset scripts; sourced, not run.
#   --data DIR      dataset root (required)
#   --out DIR       output root, by default runs/<dataset>[_geometry] in the current directory
#   --geometry      train + Geometry Objective instead of GOFish
#   --exact         render the test set with the exact training renderer, as for the paper's metrics;
#                   by default the real-time renderer is used for the renders and the FPS
#   --scenes "..."  subset of the paper's scenes
#   --dry-run       print each scene's resolved training configuration and exit
#   -- KEY=VALUE    further train.py overrides
GEOMETRY=0; EXACT=0; DRY=0; DATA=""; OUT=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --geometry) GEOMETRY=1 ;;
    --exact) EXACT=1 ;;
    --scenes) SCENES="$2"; shift ;;
    --dry-run) DRY=1 ;;
    --lidar) LIDAR="$2"; shift ;;
    --tnt-eval) TNT_EVAL="$2"; shift ;;
    -h|--help) { sed -n '2,/^set -euo/p' "$0" | sed '$d'; sed -n '/^#   --data DIR/,/^[^#]/p' "$(dirname "$0")/common.sh" | sed '$d'; } \
                 | grep -v '^# SPDX' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -n "${DATA}" ] || { echo "--data is required (see --help)" >&2; exit 2; }
[ "${GEOMETRY}" = 1 ] && EXTRA=(geometry=sof "${EXTRA[@]+"${EXTRA[@]}"}")
SUFFIX=""; [ "${GEOMETRY}" = 1 ] && SUFFIX=_geometry
OUT=${OUT:-runs/${DATASET}${SUFFIX}}
RENDERER=realtime; [ "${EXACT}" = 1 ] && RENDERER=exact
[ -n "${LIDAR:-}" ] && [ "${DATASET}" != fiord ] && echo "warning: --lidar is used by fiord.sh only; ignored" >&2
[ -n "${TNT_EVAL:-}" ] && [ "${DATASET}" != tnt ] && echo "warning: --tnt-eval is used by tnt.sh only; ignored" >&2
# Paths are made absolute before the scripts run from the repository root.
abspath() { case "$1" in /*) echo "$1" ;; *) echo "${PWD}/$1" ;; esac; }
DATA=$(abspath "${DATA}"); OUT=$(abspath "${OUT}")
[ -n "${LIDAR:-}" ] && LIDAR=$(abspath "${LIDAR}")
[ -n "${TNT_EVAL:-}" ] && TNT_EVAL=$(abspath "${TNT_EVAL}")
cd "$(dirname "$0")/.."

# train SCENE CONFIG PATH [overrides...]: train one scene (or print its configuration with --dry-run).
train() {
  local scene=$1 config=$2 path=$3; shift 3
  local args=(--config-name "apps/${config}" path="${path}" out_dir="${OUT}" experiment_name="${scene}" "$@"
              "${EXTRA[@]+"${EXTRA[@]}"}")
  if [ "${DRY}" = 1 ]; then python train.py "${args[@]}" --cfg job --resolve; return; fi
  python train.py "${args[@]}"
}

# render_and_mesh RUN: test-set renders and Full-aperture metrics, real-time FPS, and the mesh of the geometry
# checkpoint (end of the joint scene + PPISP steps, before controller distillation).
render_and_mesh() {
  local run=$1
  python render.py --checkpoint "${run}/ckpt_last.pt" --out "${run}/heldout" --renderer "${RENDERER}"
  python -m threedgrut.benchmark --checkpoint "${run}/ckpt_last.pt" --renderer realtime --out "${run}/fps.json"
  python mesh.py --checkpoint "$(geometry_checkpoint "${run}")" --out "${run}/mesh"
}

# geometry_checkpoint RUN: path of the run's geometry checkpoint.
geometry_checkpoint() {
  local steps; steps=$(python -c "from omegaconf import OmegaConf; c = OmegaConf.load('$1/parsed.yaml'); \
print(c.n_iterations - c.post_processing.n_distillation_steps)")
  echo "$1/ours_${steps}/ckpt_${steps}.pt"
}
