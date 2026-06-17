#!/usr/bin/env bash
set -euo pipefail

SPMAMBA_ROOT=/mnt/d/project/prjANS/src/AVSE/SPMamba
ECHO2MIX_ROOT=/mnt/e/data/Echo2Mix
OUT_ROOT="$SPMAMBA_ROOT/DataPreProcess"

if [ "$#" -gt 0 ]; then
  SPLITS=("$@")
else
  SPLITS=(test val train)
fi

python "$SPMAMBA_ROOT/DataPreProcess/Echo2Mix/create_echo2mix_json.py" \
  --src_root "$ECHO2MIX_ROOT" \
  --out_root "$OUT_ROOT" \
  --splits "${SPLITS[@]}"
