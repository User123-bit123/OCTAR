#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-configs/person/octar_occluded_duke.yml}"
DATA_ROOT="${DATA_ROOT:-./data}"
GPU_ID="${GPU_ID:-0}"
WEIGHT="${WEIGHT:-}"

if [ -z "$WEIGHT" ]; then
  echo "Set WEIGHT to an OCTAR checkpoint." >&2
  exit 1
fi

cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" python test_octar.py \
  --config_file "$CONFIG_FILE" \
  DATASETS.ROOT_DIR "$DATA_ROOT" \
  TEST.WEIGHT "$WEIGHT" \
  "$@"
