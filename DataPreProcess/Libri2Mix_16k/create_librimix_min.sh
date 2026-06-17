#!/usr/bin/env bash
set -euo pipefail

SPMAMBA_PYTHON=/home/marky/miniconda3/envs/spmamba/bin/python

metadata_src=/mnt/d/project/prjANS/src/AVSE/LibriMix/metadata/Libri2Mix
metadata_tmp=$(mktemp -d)
trap 'rm -rf "$metadata_tmp"' EXIT

cp "$metadata_src/libri2mix_dev-clean.csv" "$metadata_tmp/"
cp "$metadata_src/libri2mix_test-clean.csv" "$metadata_tmp/"
cp "$metadata_src/libri2mix_train-clean-100.csv" "$metadata_tmp/"

"$SPMAMBA_PYTHON" /mnt/d/project/prjANS/src/AVSE/LibriMix/scripts/create_librimix_from_metadata.py \
  --librispeech_dir /mnt/e/data/LibriMix/LibriSpeech \
  --wham_dir /mnt/e/data/LibriMix/wham_noise \
  --metadata_dir "$metadata_tmp" \
  --librimix_outdir /mnt/e/data/LibriMix \
  --n_src 2 \
  --freqs 16k \
  --modes min \
  --types mix_clean