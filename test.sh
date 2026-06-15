#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SPMAMBA_ENV=/home/marky/miniconda3/envs/spmamba
TORCH_LIB="$SPMAMBA_ENV/lib/python3.9/site-packages/torch/lib"

if [ ! -e "$SPMAMBA_ENV/lib/libnvrtc.so" ] && [ -e "$SPMAMBA_ENV/lib/libnvrtc.so.11.2" ]; then
  ln -s "$SPMAMBA_ENV/lib/libnvrtc.so.11.2" "$SPMAMBA_ENV/lib/libnvrtc.so"
fi

export LD_LIBRARY_PATH="$SPMAMBA_ENV/lib:$TORCH_LIB"

# test with librimix
"$SPMAMBA_ENV/bin/python" audio_test.py --conf_dir configs/spmamba-librimix.yml